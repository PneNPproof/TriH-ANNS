#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <stdexcept> // Required for std::runtime_error
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <device_launch_parameters.h> // For blockIdx, threadIdx etc.

// Thrust library for device-side sorting and algorithms
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/functional.h> // for thrust::greater
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/transform.h>        // Include explicitly if needed
#include <thrust/execution_policy.h> // for thrust::device

#include "pca.cuh"

using std::cout;
using std::endl;
using std::vector;

// Define MAX_SAMPLE if it's not defined elsewhere
#ifndef MAX_SAMPLE
#define MAX_SAMPLE 3000000 // Example value, adjust as needed
#endif

// Simple CUDA error checking macro
#define CHECK_CUDA_ERROR(call)                                                               \
    do                                                                                       \
    {                                                                                        \
        cudaError_t err = call;                                                              \
        if (err != cudaSuccess)                                                              \
        {                                                                                    \
            fprintf(stderr, "CUDA error in file '%s' in line %d : %s.\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));                            \
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err))); \
        }                                                                                    \
    } while (0)

// --- CUDA Error Checking Macro ---
// (Keep the enhanced CHECK_CUDA, CHECK_CUBLAS, CHECK_CUSOLVER macros from the previous version)
#define CHECK_CUDA(call)                                                                                \
    do                                                                                                  \
    {                                                                                                   \
        cudaError_t err = call;                                                                         \
        if (err != cudaSuccess)                                                                         \
        {                                                                                               \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            throw std::runtime_error(cudaGetErrorString(err));                                          \
        }                                                                                               \
    } while (0)

#define CHECK_CUBLAS(call)                                                                                    \
    do                                                                                                        \
    {                                                                                                         \
        cublasStatus_t status = call;                                                                         \
        if (status != CUBLAS_STATUS_SUCCESS)                                                                  \
        {                                                                                                     \
            const char *err_str = "";                                                                         \
            switch (status)                                                                                   \
            {                                                                                                 \
            case CUBLAS_STATUS_NOT_INITIALIZED:                                                               \
                err_str = "CUBLAS_STATUS_NOT_INITIALIZED";                                                    \
                break;                                                                                        \
            case CUBLAS_STATUS_ALLOC_FAILED:                                                                  \
                err_str = "CUBLAS_STATUS_ALLOC_FAILED";                                                       \
                break;                                                                                        \
            case CUBLAS_STATUS_INVALID_VALUE:                                                                 \
                err_str = "CUBLAS_STATUS_INVALID_VALUE";                                                      \
                break;                                                                                        \
            case CUBLAS_STATUS_ARCH_MISMATCH:                                                                 \
                err_str = "CUBLAS_STATUS_ARCH_MISMATCH";                                                      \
                break;                                                                                        \
            case CUBLAS_STATUS_MAPPING_ERROR:                                                                 \
                err_str = "CUBLAS_STATUS_MAPPING_ERROR";                                                      \
                break;                                                                                        \
            case CUBLAS_STATUS_EXECUTION_FAILED:                                                              \
                err_str = "CUBLAS_STATUS_EXECUTION_FAILED";                                                   \
                break;                                                                                        \
            case CUBLAS_STATUS_INTERNAL_ERROR:                                                                \
                err_str = "CUBLAS_STATUS_INTERNAL_ERROR";                                                     \
                break;                                                                                        \
            case CUBLAS_STATUS_NOT_SUPPORTED:                                                                 \
                err_str = "CUBLAS_STATUS_NOT_SUPPORTED";                                                      \
                break;                                                                                        \
            case CUBLAS_STATUS_LICENSE_ERROR:                                                                 \
                err_str = "CUBLAS_STATUS_LICENSE_ERROR";                                                      \
                break;                                                                                        \
            default:                                                                                          \
                err_str = "Unknown cuBLAS status";                                                            \
                break;                                                                                        \
            }                                                                                                 \
            fprintf(stderr, "cuBLAS Error at %s:%d - %s (Status %d)\n", __FILE__, __LINE__, err_str, status); \
            throw std::runtime_error("cuBLAS error");                                                         \
        }                                                                                                     \
    } while (0)

#define CHECK_CUSOLVER(call)                                                                                    \
    do                                                                                                          \
    {                                                                                                           \
        cusolverStatus_t status = call;                                                                         \
        if (status != CUSOLVER_STATUS_SUCCESS)                                                                  \
        {                                                                                                       \
            const char *err_str = "";                                                                           \
            switch (status)                                                                                     \
            {                                                                                                   \
            case CUSOLVER_STATUS_NOT_INITIALIZED:                                                               \
                err_str = "CUSOLVER_STATUS_NOT_INITIALIZED";                                                    \
                break;                                                                                          \
            case CUSOLVER_STATUS_ALLOC_FAILED:                                                                  \
                err_str = "CUSOLVER_STATUS_ALLOC_FAILED";                                                       \
                break;                                                                                          \
            case CUSOLVER_STATUS_INVALID_VALUE:                                                                 \
                err_str = "CUSOLVER_STATUS_INVALID_VALUE";                                                      \
                break;                                                                                          \
            case CUSOLVER_STATUS_ARCH_MISMATCH:                                                                 \
                err_str = "CUSOLVER_STATUS_ARCH_MISMATCH";                                                      \
                break;                                                                                          \
            case CUSOLVER_STATUS_MAPPING_ERROR:                                                                 \
                err_str = "CUSOLVER_STATUS_MAPPING_ERROR";                                                      \
                break;                                                                                          \
            case CUSOLVER_STATUS_EXECUTION_FAILED:                                                              \
                err_str = "CUSOLVER_STATUS_EXECUTION_FAILED";                                                   \
                break;                                                                                          \
            case CUSOLVER_STATUS_INTERNAL_ERROR:                                                                \
                err_str = "CUSOLVER_STATUS_INTERNAL_ERROR";                                                     \
                break;                                                                                          \
            case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:                                                     \
                err_str = "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";                                          \
                break;                                                                                          \
            case CUSOLVER_STATUS_NOT_SUPPORTED:                                                                 \
                err_str = "CUSOLVER_STATUS_NOT_SUPPORTED ";                                                     \
                break;                                                                                          \
            case CUSOLVER_STATUS_ZERO_PIVOT:                                                                    \
                err_str = "CUSOLVER_STATUS_ZERO_PIVOT";                                                         \
                break;                                                                                          \
            case CUSOLVER_STATUS_INVALID_LICENSE:                                                               \
                err_str = "CUSOLVER_STATUS_INVALID_LICENSE";                                                    \
                break;                                                                                          \
            default:                                                                                            \
                err_str = "Unknown cuSOLVER status";                                                            \
            }                                                                                                   \
            fprintf(stderr, "cuSOLVER Error at %s:%d - %s (Status %d)\n", __FILE__, __LINE__, err_str, status); \
            throw std::runtime_error("cuSOLVER error");                                                         \
        }                                                                                                       \
    } while (0)

// Kernel to project data onto the first 'column_num' principal components
__global__ void pcaProjectKernel(const float *data,     // Input data (record_num x dim)
                                 const float *pca_data, // Transposed PCA components (dim x dim)
                                 float *trans_data,     // Output projected data (record_num x column_num)
                                 int record_num,
                                 int dim,
                                 int column_num)
{
    // Calculate the global row and column index for this thread
    int r = blockIdx.y * blockDim.y + threadIdx.y; // Global row index (record index)
    int c = blockIdx.x * blockDim.x + threadIdx.x; // Global column index (component index)

    // Boundary check: Ensure the thread is within the bounds of the output matrix
    if (r < record_num && c < column_num)
    {
        float sum = 0.0f;
        // Perform the dot product: row 'r' of data with column 'c' of pca_data
        for (int d = 0; d < dim; ++d)
        {
            // data[r * dim + d] accesses element in row r, column d of data matrix
            // pca_data[d * dim + c] accesses element in row d, column c of pca_data matrix (transposed)
            sum += data[r * dim + d] * pca_data[d * dim + c];
        }
        // Store the result in the output matrix
        trans_data[r * column_num + c] = sum;
    }
}

// Kernel to project data onto the remaining principal components
__global__ void pcaProjectRemainKernel(const float *data,        // Input data (record_num x dim)
                                       const float *pca_data,    // Transposed PCA components (dim x dim)
                                       float *trans_data_remain, // Output remaining projected data (record_num x (dim - column_num))
                                       int record_num,
                                       int dim,
                                       int column_num) // Start column index for remaining components
{
    // Calculate the global row and *local* column index for this thread
    int r = blockIdx.y * blockDim.y + threadIdx.y;       // Global row index (record index)
    int c_local = blockIdx.x * blockDim.x + threadIdx.x; // Local column index (0 to dim-column_num-1)

    // Calculate the *global* column index within the original pca_data matrix
    int c_global = column_num + c_local;

    // Boundary check: Ensure the thread is within the bounds of the output matrix
    // The number of columns in trans_data_remain is (dim - column_num)
    if (r < record_num && c_local < (dim - column_num))
    {
        float sum = 0.0f;
        // Perform the dot product: row 'r' of data with column 'c_global' of pca_data
        for (int d = 0; d < dim; ++d)
        {
            // data[r * dim + d] accesses element in row r, column d of data matrix
            // pca_data[d * dim + c_global] accesses element in row d, column c_global of pca_data matrix
            sum += data[r * dim + d] * pca_data[d * dim + c_global];
        }
        // Store the result in the output matrix
        // Index calculation: r * (width) + c_local
        trans_data_remain[r * (dim - column_num) + c_local] = sum;
    }
}

// --- Custom CUDA Kernels (Same as before) ---
// Kernel to calculate the sum for each dimension (feature)
__global__ void calculate_dim_sums(const float *data, int N, int D, float *dim_sums)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int dim_idx = blockIdx.x;
    int block_dim = blockDim.x;
    float my_sum = 0.0f;
    for (int i = tid; i < N; i += block_dim)
    {
        my_sum += data[i * D + dim_idx];
    }
    sdata[tid] = my_sum;
    __syncthreads();
    for (int s = block_dim / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        dim_sums[dim_idx] = sdata[0];
    }
}

// Kernel to subtract mean from each element and center the data
__global__ void center_data(float *data, const float *mean, int N, int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int elem_idx = idx; elem_idx < N * D; elem_idx += stride)
    {
        int col = elem_idx % D; // Dimension index
        data[elem_idx] -= mean[col];
    }
}

// Kernel to reorder eigenvector columns based on sorted indices
__global__ void reorder_eigenvectors(const float *original_eigenvectors, // From cuSOLVER (column-major)
                                     const int *sorted_indices,          // Indices mapping sorted position to original position
                                     float *pca_data,                    // Output (column-major eigenvectors)
                                     int D)
{
    int col_out = blockIdx.x * blockDim.x + threadIdx.x; // Index of the principal component (0 = highest eigenvalue)
    if (col_out < D)
    {
        int col_in = sorted_indices[col_out]; // Original column index of this principal component
        for (int row = 0; row < D; ++row)
        {
            // original_eigenvectors[row + col_in * D] -> element (row, col_in) col-major
            // pca_data[row + col_out * D]             -> element (row, col_out) col-major
            pca_data[row + col_out * D] = original_eigenvectors[row + col_in * D];
        }
    }
}

// --- Main PCA Function (CUDA Version) ---
void PCA_CUDA(const float *src, const int N0, const int D, float &ratio, int &d, float *&pca_data_host)
{

    // --- 1. Initialization & Input Validation ---
    int N = N0;
    if (N > MAX_SAMPLE)
    {
        std::cout << "Warning: N0 (" << N0 << ") > MAX_SAMPLE (" << MAX_SAMPLE
                  << "). Truncating N to " << MAX_SAMPLE << "." << std::endl;
        N = MAX_SAMPLE;
    }
    if (D <= 0)
    {
        std::cerr << "Error: Invalid dimension D=" << D << ". Must be positive." << std::endl;
        pca_data_host = nullptr;
        d = 0;
        ratio = 0.0f;
        throw std::invalid_argument("D must be positive");
    }
    if (N <= 1)
    {
        std::cerr << "Error: Invalid number of samples N=" << N << ". Must be greater than 1." << std::endl;
        pca_data_host = nullptr;
        d = 0;
        ratio = 0.0f;
        throw std::invalid_argument("N must be greater than 1");
    }

    std::cout << "CUDA PCA running with N=" << N << ", D=" << D << std::endl;

    // Initialize handles and pointers
    cublasHandle_t cublas_handle = nullptr;
    cusolverDnHandle_t cusolver_handle = nullptr;
    float *d_data = nullptr, *d_mean = nullptr, *d_cov_matrix = nullptr;
    float *d_W = nullptr, *d_pca_data = nullptr, *d_work = nullptr;
    int *d_info = nullptr, *d_indices = nullptr, *d_sorted_indices = nullptr;
    float *d_data_colmajor_temp = nullptr; // Temporary buffer

    try
    {
        CHECK_CUBLAS(cublasCreate(&cublas_handle));
        CHECK_CUSOLVER(cusolverDnCreate(&cusolver_handle));

        // --- 2. Device Memory Allocation ---
        size_t data_size = (size_t)N * D * sizeof(float);
        size_t mean_size = D * sizeof(float);
        size_t cov_matrix_size = (size_t)D * D * sizeof(float);
        size_t eigenvalues_size = D * sizeof(float);
        size_t indices_size = D * sizeof(int);
        size_t pca_data_size = (size_t)D * D * sizeof(float);

        // --- DEBUG: Print memory requirements ---
        size_t freeMem = 0, totalMem = 0;
        CHECK_CUDA(cudaMemGetInfo(&freeMem, &totalMem));
        std::cout << "Memory Info: Free = " << freeMem / (1024.0 * 1024.0) << " MiB, Total = " << totalMem / (1024.0 * 1024.0) << " MiB" << std::endl;
        std::cout << "Required: d_data ~ " << data_size / (1024.0 * 1024.0) << " MiB" << std::endl;
        if (N > D)
        {
            std::cout << "Required: d_data_colmajor_temp ~ " << data_size / (1024.0 * 1024.0) << " MiB" << std::endl;
        }
        std::cout << "Required: d_cov_matrix ~ " << cov_matrix_size / (1024.0 * 1024.0) << " MiB" << std::endl;

        CHECK_CUDA(cudaMalloc((void **)&d_data, data_size));
        CHECK_CUDA(cudaMalloc((void **)&d_mean, mean_size));
        CHECK_CUDA(cudaMalloc((void **)&d_cov_matrix, cov_matrix_size));
        CHECK_CUDA(cudaMalloc((void **)&d_W, eigenvalues_size));
        CHECK_CUDA(cudaMalloc((void **)&d_info, sizeof(int)));
        CHECK_CUDA(cudaMalloc((void **)&d_indices, indices_size));
        CHECK_CUDA(cudaMalloc((void **)&d_sorted_indices, indices_size));
        CHECK_CUDA(cudaMalloc((void **)&d_pca_data, pca_data_size));

        // --- 3. Copy Input Data Host -> Device ---
        std::cout << "Copying data H->D..." << std::endl;
        CHECK_CUDA(cudaMemcpy(d_data, src, data_size, cudaMemcpyHostToDevice));
        std::cout << "Data copy done." << std::endl;

        // --- 4. Calculate Mean Vector ---
        std::cout << "Calculating mean..." << std::endl;
        {
            float *d_dim_sums = nullptr;
            CHECK_CUDA(cudaMalloc((void **)&d_dim_sums, mean_size));
            int threads_per_block = std::min(1024, N);
            threads_per_block = (threads_per_block <= 0) ? 1 : threads_per_block;
            int num_blocks = D;
            size_t shared_mem_size = threads_per_block * sizeof(float);

            calculate_dim_sums<<<num_blocks, threads_per_block, shared_mem_size>>>(d_data, N, D, d_dim_sums);
            CHECK_CUDA(cudaGetLastError());      // Check kernel launch
            CHECK_CUDA(cudaDeviceSynchronize()); // Wait for kernel

            thrust::device_ptr<float> d_dim_sums_ptr(d_dim_sums);
            thrust::device_ptr<float> d_mean_ptr(d_mean);
            thrust::transform(thrust::device, d_dim_sums_ptr, d_dim_sums_ptr + D, d_mean_ptr, thrust::placeholders::_1 / (float)N);
            CHECK_CUDA(cudaDeviceSynchronize()); // Sync after thrust kernel

            CHECK_CUDA(cudaFree(d_dim_sums));
        }
        std::cout << "Mean calculation done." << std::endl;

        // --- 5. Center Data (Subtract Mean) ---
        std::cout << "Centering data..." << std::endl;
        {
            int block_size = 256;
            int grid_size = (N * D + block_size - 1) / block_size;
            center_data<<<grid_size, block_size>>>(d_data, d_mean, N, D);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
        }
        std::cout << "Data centering done." << std::endl;

        // --- 6. Calculate Covariance Matrix: (X^T * X) / (N-1) ---
        std::cout << "Calculating covariance matrix..." << std::endl;
        {
            float alpha = 1.0f / (float)(N - 1);
            float beta = 0.0f;

            if (d_data == nullptr || d_cov_matrix == nullptr)
            {
                throw std::runtime_error("Device data pointer is null before cuBLAS call");
            }

            // --- Conditional Handling for N vs D ---
            if (N > D)
            {
                std::cout << "Info: N > D detected. Transposing data to temporary column-major buffer for cuBLAS SGEMM." << std::endl;
                size_t colmajor_size = (size_t)N * D * sizeof(float);

                // --- DEBUG: Check memory AGAIN before allocating temp buffer ---
                CHECK_CUDA(cudaMemGetInfo(&freeMem, &totalMem));
                std::cout << "Memory before temp alloc: Free = " << freeMem / (1024.0 * 1024.0) << " MiB" << std::endl;
                std::cout << "Attempting to allocate " << colmajor_size / (1024.0 * 1024.0) << " MiB for temp buffer..." << std::endl;
                if (freeMem < colmajor_size)
                {
                    std::cerr << "Error: Insufficient free memory for temporary column-major buffer." << std::endl;
                    throw std::runtime_error("Insufficient GPU memory for transpose buffer");
                }
                // --- End Debug ---

                CHECK_CUDA(cudaMalloc((void **)&d_data_colmajor_temp, colmajor_size));
                std::cout << "Temporary buffer allocated." << std::endl;

                // Transpose using cublasSgeam: C = alpha*op(A) + beta*op(B)
                const float geam_alpha = 1.0f;
                const float geam_beta = 0.0f;
                std::cout << "Performing transpose (cublasSgeam)..." << std::endl;
                CHECK_CUBLAS(cublasSgeam(cublas_handle,
                                         CUBLAS_OP_T,            // Transpose A
                                         CUBLAS_OP_N,            // Don't transpose B (not used)
                                         N,                      // Rows of C (output matrix = N)
                                         D,                      // Cols of C (output matrix = D)
                                         &geam_alpha,            // alpha
                                         d_data, D,              // A matrix (N x D row-major), lda=D
                                         &geam_beta,             // beta
                                         nullptr, N,             // B matrix (not used), ldb=N (must be >= max(1,m=N))
                                         d_data_colmajor_temp, N // C matrix (N x D col-major), ldc=N (must be >= max(1,m=N))
                                         ));
                // --- DEBUG: Sync and check error immediately after sgeam ---
                std::cout << "Transpose (cublasSgeam) call returned. Synchronizing..." << std::endl;
                CHECK_CUDA(cudaDeviceSynchronize());
                CHECK_CUDA(cudaGetLastError()); // Check for async errors from sgeam
                std::cout << "Transpose synchronization complete." << std::endl;
                // --- End Debug ---

                std::cout << "Performing matrix multiplication (cublasSgemm)..." << std::endl;
                CHECK_CUBLAS(cublasSgemm(cublas_handle,
                                         CUBLAS_OP_T,          // Transpose A (X_cm)
                                         CUBLAS_OP_N,          // No transpose B (X_cm)
                                         D,                    // m: Rows of A^T, C = D
                                         D,                    // n: Cols of B, C = D
                                         N,                    // k: Cols of A^T, Rows of B = N
                                         &alpha,               // Scaling factor
                                         d_data_colmajor_temp, // A = X_cm (N x D col-major)
                                         N,                    // lda must be >= max(1, k)=max(1, N). OK.
                                         d_data_colmajor_temp, // B = X_cm (N x D col-major)
                                         N,                    // ldb must be >= max(1, k)=max(1, N). OK.
                                         &beta,                // Beta factor
                                         d_cov_matrix,         // C = Covariance Matrix (D x D col-major)
                                         D                     // ldc must be >= max(1, m)=max(1, D). OK.
                                         ));
                std::cout << "Matrix multiplication (cublasSgemm) call returned. Synchronizing..." << std::endl;
                CHECK_CUDA(cudaDeviceSynchronize()); // Ensure cuBLAS call is finished
                CHECK_CUDA(cudaGetLastError());      // Check for async errors from sgemm
                std::cout << "Matrix multiplication synchronization complete." << std::endl;

                std::cout << "Freeing temporary buffer..." << std::endl;
                CHECK_CUDA(cudaFree(d_data_colmajor_temp));
                d_data_colmajor_temp = nullptr; // Avoid double free in finally block
                std::cout << "Temporary buffer freed." << std::endl;
            }
            else
            { // N <= D
                std::cout << "Info: N <= D detected. Using standard cuBLAS SGEMM call with row-major input interpretation." << std::endl;
                CHECK_CUBLAS(cublasSgemm(cublas_handle,
                                         CUBLAS_OP_T,  // Transpose A (X_rm)
                                         CUBLAS_OP_N,  // No transpose B (X_rm)
                                         D,            // m = D
                                         D,            // n = D
                                         N,            // k = N
                                         &alpha,       // Scaling factor
                                         d_data,       // A = X_rm (N x D row-major)
                                         D,            // lda must be >= max(1, k)=max(1, N). VALID since N<=D.
                                         d_data,       // B = X_rm (N x D row-major)
                                         D,            // ldb must be >= max(1, k)=max(1, N). VALID since N<=D.
                                         &beta,        // Beta factor
                                         d_cov_matrix, // C = Covariance Matrix (D x D col-major)
                                         D             // ldc must be >= max(1, m)=max(1, D). OK.
                                         ));
                CHECK_CUDA(cudaDeviceSynchronize()); // Ensure cuBLAS call is finished
                CHECK_CUDA(cudaGetLastError());      // Check for async errors from sgemm
            }
        } // End scope for covariance calculation
        std::cout << "Covariance calculation done." << std::endl;

        // --- DEBUG: Check d_cov_matrix for NaN/Inf ---
        std::cout << "DEBUG: Checking d_cov_matrix for NaN/Inf..." << std::endl;
        size_t cov_matrix_bytes = (size_t)D * D * sizeof(float);
        float *h_cov_matrix_check = new float[D * D];
        CHECK_CUDA(cudaMemcpy(h_cov_matrix_check, d_cov_matrix, cov_matrix_bytes, cudaMemcpyDeviceToHost));
        bool found_nan_inf = false;
        for (size_t i = 0; i < (size_t)D * D; ++i)
        {
            if (std::isnan(h_cov_matrix_check[i]) || std::isinf(h_cov_matrix_check[i]))
            {
                size_t row = i % D; // Assuming column-major for cuSOLVER/cuBLAS output
                size_t col = i / D;
                std::cerr << "ERROR: Found NaN/Inf in d_cov_matrix at index " << i
                          << " (col=" << col << ", row=" << row << ")"
                          << " value=" << h_cov_matrix_check[i] << std::endl;
                found_nan_inf = true;
                // Optional: break early if you just need to know if any exist
                // break;
            }
        }
        delete[] h_cov_matrix_check;
        if (found_nan_inf)
        {
            throw std::runtime_error("NaN/Inf detected in covariance matrix before cuSOLVER call.");
        }
        std::cout << "DEBUG: d_cov_matrix check complete. No NaN/Inf found." << std::endl;
        // --- END DEBUG ---

        // --- 7. Eigen Decomposition using cuSOLVER ---
        std::cout << "Performing Eigen Decomposition (cusolverDnSsyevd)..." << std::endl;
        {
            int lwork = 0;
            CHECK_CUSOLVER(cusolverDnSsyevd_bufferSize(cusolver_handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, D, d_cov_matrix, D, d_W, &lwork));
            std::cout << "Eigen decomp workspace size: " << lwork * sizeof(float) / (1024.0 * 1024.0) << " MiB" << std::endl;

            // --- DEBUG: Check memory before workspace alloc ---
            CHECK_CUDA(cudaMemGetInfo(&freeMem, &totalMem));
            std::cout << "Memory before syevd workspace alloc: Free = " << freeMem / (1024.0 * 1024.0) << " MiB" << std::endl;
            if (freeMem < (size_t)lwork * sizeof(float))
            {
                std::cerr << "Error: Insufficient free memory for cuSOLVER workspace." << std::endl;
                throw std::runtime_error("Insufficient GPU memory for cuSOLVER workspace");
            }
            // --- End Debug ---

            CHECK_CUDA(cudaMalloc((void **)&d_work, lwork * sizeof(float)));

            // Add try-catch block specifically for the cusolverDnSsyevd call
            try
            {
                CHECK_CUSOLVER(cusolverDnSsyevd(cusolver_handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, D, d_cov_matrix, D, d_W, d_work, lwork, d_info));
            }
            catch (const std::exception &e)
            {
                // First free d_work to prevent double free later
                if (d_work)
                {
                    cudaFree(d_work);
                    d_work = nullptr;
                }
                throw; // Re-throw the exception
            }

            int h_info = 0;
            CHECK_CUDA(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
            if (h_info < 0)
            {
                fprintf(stderr, "cuSOLVER Error: Argument %d of ssyevd had an illegal value.\n", -h_info);
                // Free d_work before throwing
                if (d_work)
                {
                    cudaFree(d_work);
                    d_work = nullptr;
                }
                throw std::runtime_error("Eigen decomposition failed (illegal argument).");
            }
            else if (h_info > 0)
            {
                fprintf(stderr, "cuSOLVER Warning: ssyevd failed to converge for %d eigenvalues.\n", h_info);
            }

            CHECK_CUDA(cudaFree(d_work));
            d_work = nullptr;                    // Avoid double free
            CHECK_CUDA(cudaDeviceSynchronize()); // Ensure solver is finished
        }
        std::cout << "Eigen decomposition done." << std::endl;

        // --- 8. Sort Eigenpairs (Descending by Eigenvalue) ---
        std::cout << "Sorting eigenpairs..." << std::endl;
        {
            thrust::device_ptr<float> d_W_ptr(d_W);
            thrust::device_ptr<int> d_indices_ptr(d_indices);
            thrust::device_ptr<int> d_sorted_indices_ptr(d_sorted_indices);

            thrust::sequence(thrust::device, d_indices_ptr, d_indices_ptr + D);
            thrust::copy(thrust::device, d_indices_ptr, d_indices_ptr + D, d_sorted_indices_ptr);

            try
            {
                thrust::sort_by_key(thrust::device, d_W_ptr, d_W_ptr + D, d_sorted_indices_ptr, thrust::greater<float>());
                thrust::sort(thrust::device, d_W_ptr, d_W_ptr + D, thrust::greater<float>());
            }
            catch (const thrust::system_error &e)
            { // Catch Thrust exceptions specifically
                std::cerr << "Thrust Error during sort: " << e.what() << std::endl;
                // cudaError_t err = cudaGetLastError(); // Check if there was an underlying CUDA error
                // if(err != cudaSuccess) fprintf(stderr, "Underlying CUDA error: %s\n", cudaGetErrorString(err));
                throw; // Re-throw
            }
            catch (const std::exception &e)
            {
                std::cerr << "Standard exception during Thrust sort: " << e.what() << std::endl;
                throw;
            }
            CHECK_CUDA(cudaDeviceSynchronize()); // Sync after sort
        }
        std::cout << "Sorting done." << std::endl;

        // --- 9. Determine d (Number of Components) ---
        std::cout << "Determining number of components (d)..." << std::endl;
        thrust::host_vector<float> h_W(D);
        CHECK_CUDA(cudaMemcpy(h_W.data(), d_W, eigenvalues_size, cudaMemcpyDeviceToHost));
        // ... (rest of d/ratio calculation logic remains the same) ...
        float sum_eigenvalues = 0;
        for (float val : h_W)
        {
            sum_eigenvalues += (val > 0) ? val : 0; // Only sum non-negative eigenvalues
        }

        if (sum_eigenvalues <= 1e-9)
        {
            std::cerr << "Warning: Sum of non-negative eigenvalues is close to zero. Setting d=0." << std::endl;
            d = 0;
            ratio = 0.0f;
            if (pca_data_host == nullptr)
                pca_data_host = new float[D * D];
            std::fill_n(pca_data_host, D * D, 0.0f);
        }
        else
        {
            if (d <= 0)
            {
                if (ratio <= 0 || ratio > 1.0f)
                {
                    std::cerr << "Warning: Invalid target ratio (" << ratio << "). Resetting ratio to 0.95." << endl;
                    ratio = 0.95f;
                }
                float cumulative_sum = 0;
                int calculated_d = 0;
                for (; calculated_d < D; ++calculated_d)
                {
                    float current_eig = (h_W[calculated_d] > 0) ? h_W[calculated_d] : 0;
                    cumulative_sum += current_eig;
                    if (cumulative_sum >= sum_eigenvalues * ratio)
                        break;
                }
                calculated_d += 1;
                d = std::min(calculated_d, D);
                cumulative_sum = 0;
                for (int i = 0; i < d; ++i)
                    cumulative_sum += (h_W[i] > 0) ? h_W[i] : 0;
                ratio = (sum_eigenvalues > 1e-9) ? (cumulative_sum / sum_eigenvalues) : 0.0f;
            }
            else
            {
                if (d > D)
                {
                    std::cerr << "Warning: Requested d > D. Clamping d to D." << endl;
                    d = D;
                }
                float cumulative_sum = 0;
                for (int i = 0; i < d; ++i)
                    cumulative_sum += (h_W[i] > 0) ? h_W[i] : 0;
                ratio = (sum_eigenvalues > 1e-9) ? (cumulative_sum / sum_eigenvalues) : 0.0f;
            }
        }
        std::cout << "Determination of d done. d=" << d << ", ratio=" << ratio << std::endl;

        // --- 10. Reorder Eigenvector Columns on GPU ---
        std::cout << "Reordering eigenvectors..." << std::endl;
        if (d > 0 || sum_eigenvalues <= 1e-9)
        {
            int block_size = 32;
            int grid_size = (D + block_size - 1) / block_size;
            reorder_eigenvectors<<<grid_size, block_size>>>(d_cov_matrix, d_sorted_indices, d_pca_data, D);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());

            // --- 11. Allocate Host Output Memory & Copy Result Device -> Host ---
            // if (pca_data_host == nullptr) {
            std::cout << "Allocating host output memory..." << std::endl;
            pca_data_host = new float[D * D];
            // }
            std::cout << "Copying final PCA data D->H..." << std::endl;
            CHECK_CUDA(cudaMemcpy(pca_data_host, d_pca_data, pca_data_size, cudaMemcpyDeviceToHost));
        }
        else if (pca_data_host == nullptr)
        {
            pca_data_host = new float[D * D];
            std::fill_n(pca_data_host, D * D, 0.0f);
        }
        std::cout << "Eigenvector reordering and final copy done." << std::endl;

        // --- 12. Cleanup (Moved outside try-catch is safer) ---
        std::cout << "CUDA PCA finished successfully. Cleaning up..." << std::endl;
    }
    catch (const std::exception &e)
    {
        // Error occurred, cleanup allocated resources
        std::cerr << "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
        std::cerr << "Exception during CUDA PCA: " << e.what() << std::endl;
        std::cerr << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;

        // delete[] pca_data_host; // Delete if allocated by us
        // pca_data_host = nullptr;
        d = 0;
        ratio = 0.0f;

        // Cleanup CUDA resources (best effort in case of error)
        if (d_data_colmajor_temp)
            cudaFree(d_data_colmajor_temp);
        if (d_work)
            cudaFree(d_work);
        if (d_pca_data)
            cudaFree(d_pca_data);
        if (d_sorted_indices)
            cudaFree(d_sorted_indices);
        if (d_indices)
            cudaFree(d_indices);
        if (d_info)
            cudaFree(d_info);
        if (d_W)
            cudaFree(d_W);
        if (d_cov_matrix)
            cudaFree(d_cov_matrix);
        if (d_mean)
            cudaFree(d_mean);
        if (d_data)
            cudaFree(d_data);
        if (cusolver_handle)
            cusolverDnDestroy(cusolver_handle); // Ignore errors during cleanup after error
        if (cublas_handle)
            cublasDestroy(cublas_handle); // Ignore errors during cleanup after error

        throw; // Re-throw the exception
    }

    // --- Normal Cleanup ---
    // Free all device memory allocated at the start
    if (d_data_colmajor_temp)
        cudaFree(d_data_colmajor_temp);
    if (d_work)
        cudaFree(d_work);
    if (d_pca_data)
        cudaFree(d_pca_data);
    if (d_sorted_indices)
        cudaFree(d_sorted_indices);
    if (d_indices)
        cudaFree(d_indices);
    if (d_info)
        cudaFree(d_info);
    if (d_W)
        cudaFree(d_W);
    if (d_cov_matrix)
        cudaFree(d_cov_matrix);
    if (d_mean)
        cudaFree(d_mean);
    if (d_data)
        cudaFree(d_data);

    // Destroy handles
    if (cusolver_handle)
        CHECK_CUSOLVER(cusolverDnDestroy(cusolver_handle));
    if (cublas_handle)
        CHECK_CUBLAS(cublasDestroy(cublas_handle));
    std::cout << "CUDA cleanup complete." << std::endl;
}

#define CHECK_CUBLAS_ERROR(call)                                                  \
    do                                                                            \
    {                                                                             \
        cublasStatus_t status = call;                                             \
        if (status != CUBLAS_STATUS_SUCCESS)                                      \
        {                                                                         \
            fprintf(stderr, "cuBLAS error in file '%s' in line %d: Status %d.\n", \
                    __FILE__, __LINE__, status);                                  \
            /* You might want a function to convert cublasStatus_t to string */   \
            throw std::runtime_error("cuBLAS error");                             \
        }                                                                         \
    } while (0)

void projectPCA_CUDA_cuBLAS(const float *h_data,        // Host input data (row-major, record_num x dim)
                            const float *h_pca_data,    // Host PCA matrix (col-major components, dim x dim)
                            float *h_trans_data,        // Host output projected data (row-major, record_num x column_num)
                            float *h_trans_data_remain, // Host output remaining projected data (row-major, record_num x (dim-col))
                            int record_num,
                            int dim,
                            int column_num)
{
    if (!h_data || !h_pca_data || !h_trans_data)
    {
        throw std::invalid_argument("Received null pointer for mandatory args in projectPCA_CUDA_cuBLAS");
    }
    if (dim - column_num > 0 && !h_trans_data_remain)
    {
        throw std::invalid_argument("Received null pointer for h_trans_data_remain when needed");
    }
    if (record_num <= 0 || dim <= 0 || column_num <= 0 || column_num > dim)
    {
        throw std::invalid_argument("Invalid dimensions provided to projectPCA_CUDA_cuBLAS");
    }

    std::cout << "Starting cuBLAS PCA projection..." << std::endl;

    // Device pointers
    float *d_data = nullptr;
    float *d_pca_data = nullptr;
    float *d_trans_data = nullptr;
    float *d_trans_data_remain = nullptr;

    // cuBLAS handle
    cublasHandle_t cublasHandle = nullptr;

    // Calculate sizes
    size_t data_size = (size_t)record_num * dim * sizeof(float);
    size_t pca_data_size = (size_t)dim * dim * sizeof(float); // Components are columns
    size_t trans_data_size = (size_t)record_num * column_num * sizeof(float);
    size_t trans_data_remain_size = 0;
    int remain_cols = dim - column_num;
    if (remain_cols > 0)
    {
        trans_data_remain_size = (size_t)record_num * remain_cols * sizeof(float);
    }

    try
    {
        // 1. Initialize cuBLAS
        std::cout << "  Initializing cuBLAS..." << std::endl;
        CHECK_CUBLAS_ERROR(cublasCreate(&cublasHandle));

        // 2. Allocate memory on the GPU device
        std::cout << "  Allocating GPU memory..." << std::endl;
        CHECK_CUDA_ERROR(cudaMalloc((void **)&d_data, data_size));
        CHECK_CUDA_ERROR(cudaMalloc((void **)&d_pca_data, pca_data_size));
        CHECK_CUDA_ERROR(cudaMalloc((void **)&d_trans_data, trans_data_size));
        if (remain_cols > 0)
        {
            CHECK_CUDA_ERROR(cudaMalloc((void **)&d_trans_data_remain, trans_data_remain_size));
        }
        // No need to cudaMemset outputs to 0 if using beta=0 in sgemm

        // 3. Copy data from Host (CPU) to Device (GPU)
        std::cout << "  Copying data H->D..." << std::endl;
        CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_pca_data, h_pca_data, pca_data_size, cudaMemcpyHostToDevice));

        // 4. Define constants for cuBLAS Sgemm
        const float alpha = 1.0f;
        const float beta = 0.0f;

        // 5. Perform first matrix multiplication: C1^T = B1^T * A^T
        // C1 = trans_data, B1 = pca_data[:, 0:column_num], A = data
        std::cout << "  Performing first Sgemm..." << std::endl;
        // Parameters based on C1_orig^T = B1_orig^T * A_orig^T
        cublasOperation_t transa = CUBLAS_OP_N; // Treat row-major A_orig as col-major A_orig^T
        cublasOperation_t transb = CUBLAS_OP_T; // Need B1_orig^T, B1_orig is col-major
        int m1 = column_num;                    // Rows of C1^T
        int n1 = record_num;                    // Cols of C1^T
        int k1 = dim;                           // Inner dimension
        int lda1 = dim;                         // Leading dim of A_orig^T (col-major)
        int ldb1 = dim;                         // Leading dim of B1_orig (col-major)
        int ldc1 = column_num;                  // Leading dim of C1^T (col-major)

        CHECK_CUBLAS_ERROR(cublasSgemm(cublasHandle,
                                       transb, transa, // Note: cuBLAS computes C = op(A)*op(B). We want C1^T = B1^T * A^T. So A=B1, B=A
                                       m1, n1, k1,     // m, n, k for C1^T
                                       &alpha,
                                       d_pca_data, ldb1, // B1_orig (as A in cublas call)
                                       d_data, lda1,     // A_orig (as B in cublas call)
                                       &beta,
                                       d_trans_data, ldc1)); // C1^T (as C in cublas call)

        // 6. Perform second matrix multiplication if needed: C2^T = B2^T * A^T
        // C2 = trans_data_remain, B2 = pca_data[:, column_num:dim], A = data
        if (remain_cols > 0)
        {
            std::cout << "  Performing second Sgemm..." << std::endl;
            // Parameters based on C2_orig^T = B2_orig^T * A_orig^T
            cublasOperation_t transa2 = CUBLAS_OP_N; // Treat row-major A_orig as col-major A_orig^T
            cublasOperation_t transb2 = CUBLAS_OP_T; // Need B2_orig^T, B2_orig is col-major
            int m2 = remain_cols;                    // Rows of C2^T
            int n2 = record_num;                     // Cols of C2^T
            int k2 = dim;                            // Inner dimension
            int lda2 = dim;                          // Leading dim of A_orig^T (col-major)
            int ldb2 = dim;                          // Leading dim of B2_orig (col-major part of pca_data)
            int ldc2 = remain_cols;                  // Leading dim of C2^T (col-major)

            // Pointer to the start of the (column_num)-th column in pca_data
            const float *d_pca_data_offset = d_pca_data + (size_t)column_num * dim;

            CHECK_CUBLAS_ERROR(cublasSgemm(cublasHandle,
                                           transb2, transa2, // A=B2, B=A
                                           m2, n2, k2,       // m, n, k for C2^T
                                           &alpha,
                                           d_pca_data_offset, ldb2, // B2_orig (as A)
                                           d_data, lda2,            // A_orig (as B)
                                           &beta,
                                           d_trans_data_remain, ldc2)); // C2^T (as C)
        }
        else
        {
            std::cout << "  Skipping second Sgemm (dim == column_num)." << std::endl;
        }

        // Optional: Synchronize device if subsequent operations depend on results immediately
        // Often implicitly synchronized by D->H copy, but good practice if unsure
        // CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // 7. Copy results from Device (GPU) back to Host (CPU)
        std::cout << "  Copying results D->H..." << std::endl;
        CHECK_CUDA_ERROR(cudaMemcpy(h_trans_data, d_trans_data, trans_data_size, cudaMemcpyDeviceToHost));
        if (remain_cols > 0)
        {
            CHECK_CUDA_ERROR(cudaMemcpy(h_trans_data_remain, d_trans_data_remain, trans_data_remain_size, cudaMemcpyDeviceToHost));
        }

        // 8. Clean up GPU memory and cuBLAS handle
        std::cout << "  Freeing GPU memory and cuBLAS handle..." << std::endl;
        CHECK_CUDA_ERROR(cudaFree(d_data));
        CHECK_CUDA_ERROR(cudaFree(d_pca_data));
        CHECK_CUDA_ERROR(cudaFree(d_trans_data));
        if (d_trans_data_remain)
        { // Only free if it was allocated
            CHECK_CUDA_ERROR(cudaFree(d_trans_data_remain));
        }
        CHECK_CUBLAS_ERROR(cublasDestroy(cublasHandle));

        std::cout << "cuBLAS PCA projection finished." << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error during cuBLAS PCA projection: " << e.what() << std::endl;
        // Clean up resources in case of error
        if (d_data)
            cudaFree(d_data);
        if (d_pca_data)
            cudaFree(d_pca_data);
        if (d_trans_data)
            cudaFree(d_trans_data);
        if (d_trans_data_remain)
            cudaFree(d_trans_data_remain);
        if (cublasHandle)
            cublasDestroy(cublasHandle);
        // Re-throw the exception or handle as appropriate
        throw;
    }
}

void projectPCA_CUDA(const float *h_data,        // Host input data
                     const float *h_pca_data,    // Host transposed PCA matrix
                     float *h_trans_data,        // Host output projected data
                     float *h_trans_data_remain, // Host output remaining projected data
                     int record_num,
                     int dim,
                     int column_num)
{
    if (!h_data || !h_pca_data || !h_trans_data || !h_trans_data_remain)
    {
        throw std::invalid_argument("Received null pointer in projectPCA_CUDA");
    }
    if (record_num <= 0 || dim <= 0 || column_num <= 0 || column_num > dim)
    {
        throw std::invalid_argument("Invalid dimensions provided to projectPCA_CUDA");
    }

    std::cout << "Starting CUDA PCA projection..." << std::endl;

    // Device pointers
    float *d_data = nullptr;
    float *d_pca_data = nullptr;
    float *d_trans_data = nullptr;
    float *d_trans_data_remain = nullptr;

    // Calculate sizes
    size_t data_size = (size_t)record_num * dim * sizeof(float);
    size_t pca_data_size = (size_t)dim * dim * sizeof(float);
    size_t trans_data_size = (size_t)record_num * column_num * sizeof(float);
    size_t trans_data_remain_size = (size_t)record_num * (dim - column_num) * sizeof(float);

    try
    {
        // 1. Allocate memory on the GPU device
        std::cout << "  Allocating GPU memory..." << std::endl;
        CHECK_CUDA_ERROR(cudaMalloc((void **)&d_data, data_size));
        CHECK_CUDA_ERROR(cudaMalloc((void **)&d_pca_data, pca_data_size));
        CHECK_CUDA_ERROR(cudaMalloc((void **)&d_trans_data, trans_data_size));
        // Initialize device memory to 0, as the kernels use += logic (though strictly not needed here as we assign directly)
        CHECK_CUDA_ERROR(cudaMemset(d_trans_data, 0, trans_data_size));
        if (dim - column_num > 0)
        { // Only allocate if there are remaining components
            CHECK_CUDA_ERROR(cudaMalloc((void **)&d_trans_data_remain, trans_data_remain_size));
            CHECK_CUDA_ERROR(cudaMemset(d_trans_data_remain, 0, trans_data_remain_size));
        }

        // 2. Copy data from Host (CPU) to Device (GPU)
        std::cout << "  Copying data H->D..." << std::endl;
        CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_pca_data, h_pca_data, pca_data_size, cudaMemcpyHostToDevice));
        // Note: We don't need to copy trans_data or trans_data_remain H->D as they are outputs

        // 3. Configure and Launch Kernels

        // Define block dimensions (e.g., 16x16 threads per block)
        // Tweak these based on your GPU architecture and problem size for best performance
        const int BLOCK_DIM_X = 16;
        const int BLOCK_DIM_Y = 64;
        dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);

        // --- Launch First Kernel (pcaProjectKernel) ---
        std::cout << "  Launching projection kernel..." << std::endl;
        // Calculate grid dimensions needed to cover the output matrix (trans_data)
        int gridDimX1 = (column_num + BLOCK_DIM_X - 1) / BLOCK_DIM_X;
        int gridDimY1 = (record_num + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y;
        dim3 gridDim1(gridDimX1, gridDimY1);

        pcaProjectKernel<<<gridDim1, blockDim>>>(d_data, d_pca_data, d_trans_data, record_num, dim, column_num);
        // Check for kernel launch errors (asynchronous, so check after potential sync point)
        CHECK_CUDA_ERROR(cudaPeekAtLastError());

        // --- Launch Second Kernel (pcaProjectRemainKernel) ---
        if (dim - column_num > 0)
        { // Only launch if there are remaining components
            std::cout << "  Launching remaining projection kernel..." << std::endl;
            int remain_cols = dim - column_num;
            // Calculate grid dimensions needed to cover the output matrix (trans_data_remain)
            int gridDimX2 = (remain_cols + BLOCK_DIM_X - 1) / BLOCK_DIM_X;
            int gridDimY2 = (record_num + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y; // Same Y dimension
            dim3 gridDim2(gridDimX2, gridDimY2);

            pcaProjectRemainKernel<<<gridDim2, blockDim>>>(d_data, d_pca_data, d_trans_data_remain, record_num, dim, column_num);
            CHECK_CUDA_ERROR(cudaPeekAtLastError());
        }
        else
        {
            std::cout << "  Skipping remaining projection kernel (dim == column_num)." << std::endl;
        }

        // 4. Synchronize device to ensure kernels have finished before copying back
        std::cout << "  Synchronizing device..." << std::endl;
        CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Wait for all GPU work to complete

        // 5. Copy results from Device (GPU) back to Host (CPU)
        std::cout << "  Copying results D->H..." << std::endl;
        CHECK_CUDA_ERROR(cudaMemcpy(h_trans_data, d_trans_data, trans_data_size, cudaMemcpyDeviceToHost));
        if (dim - column_num > 0)
        {
            CHECK_CUDA_ERROR(cudaMemcpy(h_trans_data_remain, d_trans_data_remain, trans_data_remain_size, cudaMemcpyDeviceToHost));
        }

        // 6. Clean up GPU memory
        std::cout << "  Freeing GPU memory..." << std::endl;
        CHECK_CUDA_ERROR(cudaFree(d_data));
        CHECK_CUDA_ERROR(cudaFree(d_pca_data));
        CHECK_CUDA_ERROR(cudaFree(d_trans_data));
        if (d_trans_data_remain)
        { // Only free if it was allocated
            CHECK_CUDA_ERROR(cudaFree(d_trans_data_remain));
        }

        std::cout << "CUDA PCA projection finished." << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error during CUDA PCA projection: " << e.what() << std::endl;
        // Clean up any allocated memory before re-throwing or exiting
        if (d_data)
            cudaFree(d_data);
        if (d_pca_data)
            cudaFree(d_pca_data);
        if (d_trans_data)
            cudaFree(d_trans_data);
        if (d_trans_data_remain)
            cudaFree(d_trans_data_remain);
        // Re-throw the exception or handle as appropriate
        throw;
    }
}