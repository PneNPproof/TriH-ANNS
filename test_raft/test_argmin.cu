#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/argmin.cuh>
#include <raft/util/cudart_utils.hpp>
#include <iostream>
#include <vector>

// support random generate ArgMinInputs based on n_rows and n_cols specify via command-line arguments
#include <random>
#include <limits>
#include <ctime>

#include <cuda_fp16.h>



namespace raft {
namespace matrix {

template <typename T, typename IdxT>
struct ArgMinInputs {
    std::vector<T> input_matrix;
    std::vector<IdxT> output_matrix;
    std::size_t n_rows;
    std::size_t n_cols;
};

// Function to generate random test inputs
template<typename T>
std::vector<raft::matrix::ArgMinInputs<T, int>> generateRandomInputs(std::size_t input_num, std::size_t n_rows, std::size_t n_cols) {
    std::vector<raft::matrix::ArgMinInputs<T, int>> results;
    results.reserve(input_num);
    
    // Setup random number generator
    std::mt19937 gen(static_cast<unsigned int>(std::time(nullptr)));
    std::uniform_real_distribution<T> dis(0.0, 1.0);
    
    for (std::size_t k = 0; k < input_num; ++k) {
        std::vector<T> input_matrix(n_rows * n_cols);
        std::vector<int> output_matrix(n_rows);
        
        // Generate random matrix and calculate expected argmin
        for (std::size_t i = 0; i < n_rows; ++i) {
            T min_val = std::numeric_limits<T>::max();
            int min_idx = 0;
            
            for (std::size_t j = 0; j < n_cols; ++j) {
                T val = dis(gen);
                input_matrix[i * n_cols + j] = val;
                
                if (val < min_val) {
                    min_val = val;
                    min_idx = j;
                }
            }
            
            output_matrix[i] = min_idx;
        }
        
        results.push_back({input_matrix, output_matrix, n_rows, n_cols});
    }
    
    return results;
}


// Specialized version for half precision
std::vector<raft::matrix::ArgMinInputs<half, int>> generateRandomInputsFP16(std::size_t input_num, std::size_t n_rows, std::size_t n_cols) {
    std::vector<raft::matrix::ArgMinInputs<half, int>> results;
    results.reserve(input_num);
    
    // Setup random number generator
    std::mt19937 gen(static_cast<unsigned int>(std::time(nullptr)));
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    for (std::size_t k = 0; k < input_num; ++k) {
        std::vector<half> input_matrix(n_rows * n_cols);
        std::vector<int> output_matrix(n_rows);
        
        // Generate random matrix and calculate expected argmin
        for (std::size_t i = 0; i < n_rows; ++i) {
            float min_val = std::numeric_limits<float>::max();
            int min_idx = 0;
            
            for (std::size_t j = 0; j < n_cols; ++j) {
                // Generate float and convert to half
                float val_float = dis(gen);
                half val = __float2half(val_float);
                input_matrix[i * n_cols + j] = val;
                
                // Convert back to float for comparison
                float val_as_float = __half2float(val);
                if (val_as_float < min_val) {
                    min_val = val_as_float;
                    min_idx = j;
                }
            }
            
            output_matrix[i] = min_idx;
        }
        
        results.push_back({input_matrix, output_matrix, n_rows, n_cols});
    }
    
    return results;
}

template <typename T, typename IdxT>
bool runArgMinTest(const ArgMinInputs<T, IdxT>& params) {
    raft::resources handle;
    
    auto input = raft::make_device_matrix<T, std::uint32_t, row_major>(
        handle, params.n_rows, params.n_cols);
    auto output = raft::make_device_vector<IdxT, std::uint32_t>(handle, params.n_rows);
    // auto expected = raft::make_device_vector<IdxT, std::uint32_t>(handle, params.n_rows);

    raft::update_device(input.data_handle(),
                                            params.input_matrix.data(),
                                            params.input_matrix.size(),
                                            resource::get_cuda_stream(handle));
    // raft::update_device(expected.data_handle(),
    //                                         params.output_matrix.data(),
    //                                         params.output_matrix.size(),
    //                                         resource::get_cuda_stream(handle));

    auto input_const_view = raft::make_device_matrix_view<const T, std::uint32_t, row_major>(
        input.data_handle(), input.extent(0), input.extent(1));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record start event
    cudaEventRecord(start, resource::get_cuda_stream(handle));
    
    raft::matrix::argmin(handle, input_const_view, output.view());
    
    // Record stop event
    cudaEventRecord(stop, resource::get_cuda_stream(handle));
    
    // Wait for the operation to complete
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    std::cout << "argmin execution time: " << milliseconds << " ms" << std::endl;
    
    // Clean up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    resource::sync_stream(handle);

    // Compare results
    std::vector<IdxT> h_output(params.n_rows);
    raft::update_host(h_output.data(),
                     output.data_handle(),
                     params.n_rows,
                     resource::get_cuda_stream(handle));
    
    resource::sync_stream(handle);
    
    bool passed = true;
    for (std::uint32_t i = 0; i < params.n_rows; ++i) {
        if (h_output[i] != params.output_matrix[i]) {
            passed = false;
            std::cout << "Mismatch at position " << i << ": expected=" 
                     << params.output_matrix[i] << ", got=" << h_output[i] << std::endl;
        }
    }
    
    return passed;
}

}  // namespace matrix
}  // namespace raft

int main(int argc, char** argv) {

    int input_num = 5;

    int r = 10;  // Default values
    int c = 12;
    
    if (argc > 1) {
        r = std::atoi(argv[1]);
        if (argc > 2) {
            c = std::atoi(argv[2]);
        }
    }
    
    if (r <= 0 || c <= 0) {
        std::cerr << "Usage: " << argv[0] << " [n_rows] [n_cols]" << std::endl;
        std::cerr << "n_rows and n_cols must be positive integers" << std::endl;
        return 1;
    }

    auto inputs = raft::matrix::generateRandomInputsFP16(input_num, r, c);
    for (size_t i = 0; i < inputs.size(); i++) {
        bool result = raft::matrix::runArgMinTest(inputs[i]);
        std::cout << "Random test " << i << ": " << (result ? "PASSED" : "FAILED") << std::endl;
    }


    // // Define test inputs
    // const std::vector<raft::matrix::ArgMinInputs<float, int>> inputsf = {
    //     {{0.1f, 0.2f, 0.3f, 0.4f, 0.4f, 0.3f, 0.2f, 0.1f, 0.2f, 0.3f, 0.5f, 0.0f}, {0, 3, 3}, 3, 4}
    // };

    // const std::vector<raft::matrix::ArgMinInputs<double, int>> inputsd = {
    //     {{0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1, 0.2, 0.3, 0.5, 0.0}, {0, 3, 3}, 3, 4}
    // };

    // Run float tests
    // std::cout << "Running float tests..." << std::endl;
    // for (size_t i = 0; i < inputs.size(); i++) {
    //     bool result = raft::matrix::runArgMinTest(inputs[i]);
    //     std::cout << "Float test " << i << ": " << (result ? "PASSED" : "FAILED") << std::endl;
    // }

    // // Run double tests
    // std::cout << "Running double tests..." << std::endl;
    // for (size_t i = 0; i < inputsd.size(); i++) {
    //     bool result = raft::matrix::runArgMinTest(inputsd[i]);
    //     std::cout << "Double test " << i << ": " << (result ? "PASSED" : "FAILED") << std::endl;
    // }

    return 0;
}
