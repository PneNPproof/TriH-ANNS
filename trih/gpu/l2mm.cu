#include <cublas_v2.h>
#include <stdexcept>
#include <cuda_fp16.h>

#include "l2mm.cuh"

// CUDA kernel to add vector B to vector A and store the result in A
__global__ void addVectorsKernel(half* A, const half* B, int n) {
  // Calculate the global thread index
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Process elements within bounds
  if (idx < n) {
      A[idx] = __hadd(A[idx], B[idx]); // Use CUDA's intrinsic for FP16 addition
  }
}

void addVectors(half* d_A, const half* d_B, int n, cudaStream_t stream) {
  // Define block and grid sizes
  int blockSize = 256; // Threads per block
  int gridSize = (n + blockSize - 1) / blockSize; // Grid size to cover all elements

  // Launch the kernel
  addVectorsKernel<<<gridSize, blockSize, 0, stream>>>(d_A, d_B, n);
}

inline void checkCublasStatus(cublasStatus_t status)
{
  if (status != CUBLAS_STATUS_SUCCESS)
  {
    printf("cuBLAS API failed with status %d\n", status);
    throw std::logic_error("cuBLAS API failed");
  }
  // else {
  //   printf("success cublas call\n");
  // }
}


/**
 * @brief Performs matrix multiplication with L2 normalization using cuBLAS
 *
 * This function computes C = alpha*A^T*B + beta*C, where alpha=-2.0f and beta=1.0f,
 * which is a part of L2 distance computation between vectors.
 *
 * @param m Number of columns in matrix A
 * @param n Number of columns in matrix B
 * @param k Number of rows in both matrices A and B
 * @param A Pointer to matrix A in device memory
 * @param B Pointer to matrix B in device memory
 * @param C Pointer to result matrix C in device memory
 * @param workspace Pointer to workspace memory (currently unused)
 * @param workspaceSize Size of workspace in bytes (currently unused)
 * @param stream CUDA stream to execute the operation in
 * 
 * @throw std::logic_error if there's a cuBLAS error
 */
void l2mm(
    int m,
    int n,
    int k,
    const float *A,
    const float *B,
    float *C,
    // void *workspace,
    // size_t workspaceSize,
    cudaStream_t stream)
{
  // Create and initialize cublas handle
  cublasHandle_t handle;
  checkCublasStatus(cublasCreate(&handle));
  
  // Set the stream
  checkCublasStatus(cublasSetStream(handle, stream));
  
  // Optional: Enable Tensor Cores if available
  // checkCublasStatus(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
  
  // Define constants for the operation C = alpha*A*B + beta*C
  float alpha = -2.0f;
  float beta = 1.0f;

  printf("before cublasGemmEx \n");
  
  // Use cublasGemmEx similar to bench.cu
  // We're using CUBLAS_OP_T for A and CUBLAS_OP_N for B
  checkCublasStatus(cublasGemmEx(
    handle,
    CUBLAS_OP_T,              // op_A
    CUBLAS_OP_N,              // op_B
    m, n, k,                 // n, m, k dimensions
    &alpha,                  // alpha = -2.0f
    A, CUDA_R_32F, k,        // A matrix, data type, leading dimension
    B, CUDA_R_32F, k,        // B matrix, data type, leading dimension
    &beta,                   // beta = 1.0f
    C, CUDA_R_32F, m,        // C matrix (output), data type, leading dimension
    CUDA_R_32F,              // Computation type
    CUBLAS_GEMM_DFALT        // Algorithm selection
  ));

  printf("after cublasGemmEx \n");
  
  // Clean up
  checkCublasStatus(cublasDestroy(handle));
}

/**
 * @brief Performs matrix multiplication with L2 normalization using cuBLAS with FP16 precision
 *
 * This function computes C = alpha*A^T*B + beta*C, where alpha=-2.0f and beta=1.0f,
 * which is a part of L2 distance computation between vectors in half precision.
 *
 * @param m Number of columns in matrix A
 * @param n Number of columns in matrix B
 * @param k Number of rows in both matrices A and B
 * @param A Pointer to matrix A in device memory (half precision)
 * @param B Pointer to matrix B in device memory (half precision)
 * @param C Pointer to result matrix C in device memory (half precision)
 * @param stream CUDA stream to execute the operation in
 * 
 * @throw std::logic_error if there's a cuBLAS error
 */
void l2mm_fp16(
    int m,
    int n,
    int k,
    const half *A,
    const half *B,
    half *C,
    // void *workspace,
    // size_t workspaceSize,
    cudaStream_t stream)
{
  // Create and initialize cublas handle
  cublasHandle_t handle;
  checkCublasStatus(cublasCreate(&handle));
  
  // Set the stream
  checkCublasStatus(cublasSetStream(handle, stream));
  
  // Optional: Enable Tensor Cores for FP16 operations
  checkCublasStatus(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
  
  // Define constants for the operation C = alpha*A*B + beta*C
  half alpha = __float2half(-2.0f);
  half beta = __float2half(1.0f);

  // printf("before cublasGemmEx (FP16) \n");
  
  // Use cublasGemmEx for FP16 computation
  checkCublasStatus(cublasGemmEx(
    handle,
    CUBLAS_OP_T,              // op_A
    CUBLAS_OP_N,              // op_B
    m, n, k,                  // n, m, k dimensions
    &alpha,                   // alpha = -2.0f (half)
    A, CUDA_R_16F, k,         // A matrix, data type, leading dimension
    B, CUDA_R_16F, k,         // B matrix, data type, leading dimension
    &beta,                    // beta = 1.0f (half)
    C, CUDA_R_16F, m,         // C matrix (output), data type, leading dimension
    CUDA_R_16F,               // Computation type
    CUBLAS_GEMM_DEFAULT_TENSOR_OP // Algorithm using Tensor Cores when possible
  ));

  // printf("after cublasGemmEx (FP16) \n");
  
  // Clean up
  checkCublasStatus(cublasDestroy(handle));
}


