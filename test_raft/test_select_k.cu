#include <raft/core/device_resources.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/matrix/select_k.cuh>

#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <iostream>
#include <vector>
#include <random>
#include <ctime>

namespace raft::matrix {

// Simple struct to hold test inputs and expected outputs
template <typename KeyT, typename IdxT>
struct SelectKInputs {
  std::vector<KeyT> input_dists;   // Input distances
  std::vector<IdxT> input_ids;     // Input IDs
  std::vector<KeyT> output_dists;  // Expected output distances
  std::vector<IdxT> output_ids;    // Expected output IDs
  int batch_size;                  // Number of rows
  int len;                         // Number of columns in input
  int k;                           // Number of columns in output (k neighbors)
  bool select_min;                 // Whether to select minimum values
  bool use_index_input;            // Whether to use index input
};

// Generate random test inputs
template <typename KeyT, typename IdxT>
std::vector<SelectKInputs<KeyT, IdxT>> generateRandomInputs(int input_num, int batch_size, int len, int k, bool select_min) {
  std::vector<SelectKInputs<KeyT, IdxT>> results;
  results.reserve(input_num);
  
  // Setup random number generator
  std::mt19937 gen(static_cast<unsigned int>(std::time(nullptr)));
  std::uniform_real_distribution<KeyT> dis(0.0, 100.0);
  
  for (int n = 0; n < input_num; n++) {
    // Create input distances and IDs
    std::vector<KeyT> input_dists(batch_size * len);
    std::vector<IdxT> input_ids(batch_size * len);
    
    // Generate random distances
    for (int i = 0; i < batch_size * len; i++) {
      input_dists[i] = dis(gen);
      input_ids[i] = i % len;  // Each ID is column index
    }
    
    // Calculate expected outputs (using CPU sorting)
    std::vector<KeyT> output_dists(batch_size * k);
    std::vector<IdxT> output_ids(batch_size * k);
    
    for (int i = 0; i < batch_size; i++) {
      // Create a vector of pairs (distance, id) for this row
      std::vector<std::pair<KeyT, IdxT>> row_data;
      for (int j = 0; j < len; j++) {
        row_data.push_back({input_dists[i * len + j], input_ids[i * len + j]});
      }
      
      // Sort based on distance
      if (select_min) {
        std::sort(row_data.begin(), row_data.end());
      } else {
        std::sort(row_data.begin(), row_data.end(), 
                 [](const std::pair<KeyT, IdxT>& a, const std::pair<KeyT, IdxT>& b) {
                   return a.first > b.first;
                 });
      }
      
      // Take top k
      for (int j = 0; j < k; j++) {
        output_dists[i * k + j] = row_data[j].first;
        output_ids[i * k + j] = row_data[j].second;
      }
    }
    
    results.push_back({input_dists, input_ids, output_dists, output_ids, 
                      batch_size, len, k, select_min, true});
  }
  
  return results;
}

// Simple test function to run select_k and check results
template <typename KeyT, typename IdxT>
bool runSelectKTest(const SelectKInputs<KeyT, IdxT>& params, SelectAlgo algo) {
  raft::device_resources handle;
  auto stream = resource::get_cuda_stream(handle);
  
  std::cout << "Testing with algorithm " << static_cast<int>(algo) 
            << ", batch_size=" << params.batch_size 
            << ", len=" << params.len
            << ", k=" << params.k
            << ", select_min=" << params.select_min << std::endl;
  
  // Create device arrays
  rmm::device_uvector<KeyT> d_input_dists(params.batch_size * params.len, stream);
  rmm::device_uvector<IdxT> d_input_ids(params.batch_size * params.len, stream);
  rmm::device_uvector<KeyT> d_output_dists(params.batch_size * params.k, stream);
  rmm::device_uvector<IdxT> d_output_ids(params.batch_size * params.k, stream);
  
  // Copy input data to device
  raft::update_device(d_input_dists.data(), params.input_dists.data(), params.input_dists.size(), stream);
  raft::update_device(d_input_ids.data(), params.input_ids.data(), params.input_ids.size(), stream);
  
  // Create input views
  auto input_dists_view = raft::make_device_matrix_view<const KeyT, int64_t>(
    d_input_dists.data(), params.batch_size, params.len);

  std::optional<raft::device_matrix_view<const IdxT, int64_t, row_major>> input_ids_view;
  if (params.use_index_input) {
    input_ids_view = raft::make_device_matrix_view<const IdxT, int64_t>(
      d_input_ids.data(), params.batch_size, params.len);
  }
  
  // Create output views
  auto output_dists_view = raft::make_device_matrix_view<KeyT, int64_t>(
    d_output_dists.data(), params.batch_size, params.k);
  auto output_ids_view = raft::make_device_matrix_view<IdxT, int64_t>(
    d_output_ids.data(), params.batch_size, params.k);
  
  // Create CUDA events for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  // Record start event
  cudaEventRecord(start, stream);
  
  // Run select_k
  matrix::select_k<KeyT, IdxT>(
    handle,
    input_dists_view,
    input_ids_view,
    output_dists_view,
    output_ids_view,
    params.select_min,
    false,  // not in-place
    algo
  );
  
  // Record stop event
  cudaEventRecord(stop, stream);
  
  // Wait for operation to complete
  cudaEventSynchronize(stop);
  
  // Calculate elapsed time
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  
  std::cout << "select_k execution time: " << milliseconds << " ms" << std::endl;
  
  // Clean up events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  
  // Copy results back to host
  std::vector<KeyT> h_output_dists(params.batch_size * params.k);
  std::vector<IdxT> h_output_ids(params.batch_size * params.k);
  
  raft::update_host(h_output_dists.data(), d_output_dists.data(), h_output_dists.size(), stream);
  raft::update_host(h_output_ids.data(), d_output_ids.data(), h_output_ids.size(), stream);
  
  resource::sync_stream(handle);
  
  // Compare results
  bool passed = true;
  int max_errors_to_print = 10;
  int errors_printed = 0;
  
  // Check distances
  for (int i = 0; i < params.batch_size; i++) {
    // Sort each row for comparison as algorithms might return values in different order
    std::vector<std::pair<KeyT, IdxT>> expected_row;
    std::vector<std::pair<KeyT, IdxT>> actual_row;
    
    for (int j = 0; j < params.k; j++) {
      expected_row.push_back({params.output_dists[i * params.k + j], params.output_ids[i * params.k + j]});
      actual_row.push_back({h_output_dists[i * params.k + j], h_output_ids[i * params.k + j]});
    }
    
    // Sort by distance then by ID
    auto compare = [](const std::pair<KeyT, IdxT>& a, const std::pair<KeyT, IdxT>& b) {
      return (a.first < b.first) || (a.first == b.first && a.second < b.second);
    };
    
    std::sort(expected_row.begin(), expected_row.end(), compare);
    std::sort(actual_row.begin(), actual_row.end(), compare);
    
    // Compare sorted results
    for (int j = 0; j < params.k; j++) {
      if (std::abs(expected_row[j].first - actual_row[j].first) > 1e-5 ||
          expected_row[j].second != actual_row[j].second) {
        if (errors_printed < max_errors_to_print) {
          std::cout << "Mismatch at batch " << i << ", index " << j 
                    << ": expected=(" << expected_row[j].first << "," << expected_row[j].second
                    << "), got=(" << actual_row[j].first << "," << actual_row[j].second << ")" << std::endl;
          errors_printed++;
        }
        passed = false;
      }
    }
  }
  
  return passed;
}

} // namespace raft::matrix

// Define available algorithms
std::vector<raft::matrix::SelectAlgo> get_algorithms() {
  return {
    raft::matrix::SelectAlgo::kAuto,
    raft::matrix::SelectAlgo::kRadix8bits,
    raft::matrix::SelectAlgo::kRadix11bits,
    // raft::matrix::SelectAlgo::kWarpImmediate,
    // raft::matrix::SelectAlgo::kWarpFiltered,
    // raft::matrix::SelectAlgo::kWarpDistributed
  };
}

int main(int argc, char** argv) {
  int batch_size = 5;
  int len = 20;
  int k = 5;
  
  // Parse command line arguments
  if (argc > 1) batch_size = std::atoi(argv[1]);
  if (argc > 2) len = std::atoi(argv[2]);
  if (argc > 3) k = std::atoi(argv[3]);
  
  if (batch_size <= 0 || len <= 0 || k <= 0 || k > len) {
    std::cerr << "Usage: " << argv[0] << " [batch_size] [len] [k]" << std::endl;
    std::cerr << "batch_size, len, and k must be positive integers, and k must not exceed len" << std::endl;
    return 1;
  }
  
  std::cout << "Starting select_k tests with batch_size=" << batch_size 
            << ", len=" << len << ", k=" << k << std::endl;
  
  // Generate test inputs
  int num_inputs = 5;
  auto inputs_min = raft::matrix::generateRandomInputs<float, uint32_t>(
    num_inputs, batch_size, len, k, true);  // select min
  auto inputs_max = raft::matrix::generateRandomInputs<float, uint32_t>(
    num_inputs, batch_size, len, k, false); // select max
    
  // Get algorithms
  auto algorithms = get_algorithms();
  
  int total_tests = 0;
  int passed_tests = 0;
  
  // Run all tests with all algorithms
  for (const auto& input : inputs_min) {
    for (const auto& algo : algorithms) {
      total_tests++;
      if (raft::matrix::runSelectKTest(input, algo)) {
        passed_tests++;
      }
    }
  }
  
  for (const auto& input : inputs_max) {
    for (const auto& algo : algorithms) {
      total_tests++;
      if (raft::matrix::runSelectKTest(input, algo)) {
        passed_tests++;
      }
    }
  }
  
  std::cout << "\nTest summary: " << passed_tests << " of " << total_tests << " tests passed." << std::endl;
  
  return (passed_tests == total_tests) ? 0 : 1;
}
