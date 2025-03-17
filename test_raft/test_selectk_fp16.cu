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
#include <cuda_fp16.h>  // Include for half precision

namespace raft::matrix {

// Simple struct to hold test inputs and expected outputs, specialized for half precision
template <typename IdxT>
struct SelectKInputsFP16 {
  std::vector<half> input_dists;   // Input distances in FP16
  std::vector<IdxT> input_ids;     // Input IDs
  std::vector<half> output_dists;  // Expected output distances in FP16
  std::vector<IdxT> output_ids;    // Expected output IDs
  int batch_size;                  // Number of rows
  int len;                         // Number of columns in input
  int k;                           // Number of columns in output (k neighbors)
  bool select_min;                 // Whether to select minimum values
  bool use_index_input;            // Whether to use index input
};

// Generate random test inputs with half precision
template <typename IdxT>
std::vector<SelectKInputsFP16<IdxT>> generateRandomInputsFP16(int input_num, int batch_size, int len, int k, bool select_min) {
  std::vector<SelectKInputsFP16<IdxT>> results;
  results.reserve(input_num);
  
  // Setup random number generator
  std::mt19937 gen(static_cast<unsigned int>(std::time(nullptr)));
  std::uniform_real_distribution<float> dis(0.0, 100.0);
  
  for (int n = 0; n < input_num; n++) {
    // Create input distances and IDs
    std::vector<half> input_dists(batch_size * len);
    std::vector<IdxT> input_ids(batch_size * len);
    
    // Generate random distances (convert from float to half)
    for (int i = 0; i < batch_size * len; i++) {
      input_dists[i] = __float2half(dis(gen));
      input_ids[i] = i % len;  // Each ID is column index
    }
    
    // Calculate expected outputs (using CPU sorting)
    std::vector<half> output_dists(batch_size * k);
    std::vector<IdxT> output_ids(batch_size * k);
    
    for (int i = 0; i < batch_size; i++) {
      // Create a vector of pairs (distance as float for sorting, id) for this row
      std::vector<std::pair<float, IdxT>> row_data;
      for (int j = 0; j < len; j++) {
        row_data.push_back({__half2float(input_dists[i * len + j]), input_ids[i * len + j]});
      }
      
      // Sort based on distance
      if (select_min) {
        std::sort(row_data.begin(), row_data.end());
      } else {
        std::sort(row_data.begin(), row_data.end(), 
                 [](const std::pair<float, IdxT>& a, const std::pair<float, IdxT>& b) {
                   return a.first > b.first;
                 });
      }
      
      // Take top k and convert back to half
      for (int j = 0; j < k; j++) {
        output_dists[i * k + j] = __float2half(row_data[j].first);
        output_ids[i * k + j] = row_data[j].second;
      }
    }
    
    results.push_back({input_dists, input_ids, output_dists, output_ids, 
                      batch_size, len, k, select_min, true});
  }
  
  return results;
}

// Simple test function to run select_k and check results for half precision
template <typename IdxT>
bool runSelectKTestFP16(const SelectKInputsFP16<IdxT>& params, SelectAlgo algo) {
  raft::device_resources handle;
  auto stream = resource::get_cuda_stream(handle);
  
  std::cout << "Testing with half precision, algorithm " << static_cast<int>(algo) 
            << ", batch_size=" << params.batch_size 
            << ", len=" << params.len
            << ", k=" << params.k
            << ", select_min=" << params.select_min << std::endl;
  
  // Create device arrays
  rmm::device_uvector<half> d_input_dists(params.batch_size * params.len, stream);
  rmm::device_uvector<IdxT> d_input_ids(params.batch_size * params.len, stream);
  rmm::device_uvector<half> d_output_dists(params.batch_size * params.k, stream);
  rmm::device_uvector<IdxT> d_output_ids(params.batch_size * params.k, stream);
  
  // Copy input data to device
  raft::update_device(d_input_dists.data(), params.input_dists.data(), params.input_dists.size(), stream);
  raft::update_device(d_input_ids.data(), params.input_ids.data(), params.input_ids.size(), stream);
  
  // Create input views
  auto input_dists_view = raft::make_device_matrix_view<const half, int64_t>(
    d_input_dists.data(), params.batch_size, params.len);

  std::optional<raft::device_matrix_view<const IdxT, int64_t, row_major>> input_ids_view;
  if (params.use_index_input) {
    input_ids_view = raft::make_device_matrix_view<const IdxT, int64_t>(
      d_input_ids.data(), params.batch_size, params.len);
  }
  
  // Create output views
  auto output_dists_view = raft::make_device_matrix_view<half, int64_t>(
    d_output_dists.data(), params.batch_size, params.k);
  auto output_ids_view = raft::make_device_matrix_view<IdxT, int64_t>(
    d_output_ids.data(), params.batch_size, params.k);
  
  // Create CUDA events for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  // Record start event
  cudaEventRecord(start, stream);
  
  // Run select_k with half precision
  matrix::select_k<half, IdxT>(
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
  std::vector<half> h_output_dists(params.batch_size * params.k);
  std::vector<IdxT> h_output_ids(params.batch_size * params.k);
  
  raft::update_host(h_output_dists.data(), d_output_dists.data(), h_output_dists.size(), stream);
  raft::update_host(h_output_ids.data(), d_output_ids.data(), h_output_ids.size(), stream);
  
  resource::sync_stream(handle);
  
  // Compare results
  bool passed = true;
  int max_errors_to_print = 20;
  int errors_printed = 0;
  
  // Check distances
  for (int i = 0; i < params.batch_size; i++) {
    // Sort each row for comparison as algorithms might return values in different order
    std::vector<std::pair<float, IdxT>> expected_row;
    std::vector<std::pair<float, IdxT>> actual_row;
    
    for (int j = 0; j < params.k; j++) {
      expected_row.push_back({__half2float(params.output_dists[i * params.k + j]), params.output_ids[i * params.k + j]});
      actual_row.push_back({__half2float(h_output_dists[i * params.k + j]), h_output_ids[i * params.k + j]});
    }
    
    // Sort by distance then by ID
    auto compare = [](const std::pair<float, IdxT>& a, const std::pair<float, IdxT>& b) {
      return (a.first < b.first) || (a.first == b.first && a.second < b.second);
    };
    
    std::sort(expected_row.begin(), expected_row.end(), compare);
    std::sort(actual_row.begin(), actual_row.end(), compare);
    
    // Compare sorted results (with slightly higher tolerance for FP16)
    for (int j = 0; j < params.k; j++) {
      if (std::abs(expected_row[j].first - actual_row[j].first) > 1e-2 ||  // Higher tolerance for FP16
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
    // raft::matrix::SelectAlgo::kRadix8bits,
    // raft::matrix::SelectAlgo::kRadix11bits,
    // Commented out as in the original
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
  
  std::cout << "Starting select_k FP16 tests with batch_size=" << batch_size 
            << ", len=" << len << ", k=" << k << std::endl;
  
  // Generate test inputs for FP16
  int num_inputs = 5;
  auto inputs_min = raft::matrix::generateRandomInputsFP16<uint32_t>(
    num_inputs, batch_size, len, k, true);  // select min
  // auto inputs_max = raft::matrix::generateRandomInputsFP16<uint32_t>(
  //   num_inputs, batch_size, len, k, false); // select max
    
  // Get algorithms
  auto algorithms = get_algorithms();
  
  int total_tests = 0;
  int passed_tests = 0;
  
  // Run all tests with all algorithms
  for (const auto& input : inputs_min) {
    for (const auto& algo : algorithms) {
      total_tests++;
      if (raft::matrix::runSelectKTestFP16(input, algo)) {
        passed_tests++;
      }
    }
  }
  
  // for (const auto& input : inputs_max) {
  //   for (const auto& algo : algorithms) {
  //     total_tests++;
  //     if (raft::matrix::runSelectKTestFP16(input, algo)) {
  //       passed_tests++;
  //     }
  //   }
  // }
  
  std::cout << "\nTest summary: " << passed_tests << " of " << total_tests << " tests passed." << std::endl;
  
  return (passed_tests == total_tests) ? 0 : 1;
}