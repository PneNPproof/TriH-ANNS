/**
 * @file pca.cuh
 * @brief CUDA-accelerated PCA computation and projection kernels
 * 
 * This header provides GPU-accelerated functions for Principal Component Analysis
 * operations used in TriH-ANNS index building and query processing:
 * 
 * - PCA computation from training data using GPU eigensolvers
 * - Batch projection of data to PCA space using optimized GEMM
 * - Separation of projected data into primary and remaining dimensions
 * 
 * The GPU implementation significantly accelerates PCA operations for large
 * datasets, especially during index building where millions of vectors
 * need to be projected to the PCA space.
 * 
 * Dependencies:
 * - CUDA runtime for GPU execution
 * - cuBLAS for optimized matrix operations
 * - cuSOLVER for eigenvalue decomposition
 */

#ifndef PCA_CUH
#define PCA_CUH

#include <cuda_runtime.h>

/**
 * @brief GPU-accelerated PCA computation from training data
 * 
 * Computes Principal Component Analysis using GPU eigensolvers for efficient
 * processing of large training datasets. Determines the optimal number of
 * components based on variance ratio or explicit component count.
 * 
 * @param src Training data matrix (N0 x D)
 * @param N0 Number of training vectors
 * @param D Original dimensionality
 * @param ratio Target variance ratio to preserve (in/out parameter)
 * @param d Number of components to retain (in/out parameter)
 * @param pca_data_host Output PCA transformation matrix (allocated by function)
 * 
 * @note Either ratio or d is used depending on input values
 * @note pca_data_host memory is allocated by function - caller must free
 * @note Uses cuSOLVER for efficient eigenvalue decomposition on GPU
 */
void PCA_CUDA(const float *src, const int N0, const int D, float &ratio, int &d, float *&pca_data_host);

/**
 * @brief GPU-accelerated PCA projection with custom kernels
 * 
 * Projects dataset to PCA space using custom CUDA kernels. Separates
 * the projection into primary components (PCA space) and remaining
 * dimensions for the two-phase search approach.
 * 
 * @param h_data Input data vectors (record_num x dim)
 * @param h_pca_data PCA transformation matrix (transposed for efficiency)
 * @param h_trans_data Output projected data in PCA space
 * @param h_trans_data_remain Output projected data in remaining dimensions
 * @param record_num Number of data vectors
 * @param dim Original dimensionality
 * @param column_num Number of PCA components
 * 
 * @note Uses custom kernels optimized for specific access patterns
 * @note May be faster than cuBLAS for certain matrix sizes
 */
void projectPCA_CUDA(const float *h_data,        // Host input data
                     const float *h_pca_data,    // Host transposed PCA matrix
                     float *h_trans_data,        // Host output projected data
                     float *h_trans_data_remain, // Host output remaining projected data
                     int record_num,
                     int dim,
                     int column_num);

/**
 * @brief GPU-accelerated PCA projection using cuBLAS GEMM
 * 
 * High-performance implementation using cuBLAS optimized GEMM operations
 * for PCA projection. Generally provides better performance than custom
 * kernels for large matrices due to highly optimized BLAS routines.
 * 
 * @param h_data Input data vectors (row-major, record_num x dim)
 * @param h_pca_data PCA transformation matrix (col-major, dim x dim)
 * @param h_trans_data Output projected data (row-major, record_num x column_num)
 * @param h_trans_data_remain Output remaining projected data (row-major, record_num x (dim-col))
 * @param record_num Number of data vectors
 * @param dim Original dimensionality
 * @param column_num Number of PCA components to retain
 * 
 * @note Uses cuBLAS SGEMM for optimal performance on modern GPUs
 * @note Handles matrix layout conversions automatically
 * @note Recommended for large-scale batch projections
 */
void projectPCA_CUDA_cuBLAS(const float *h_data,        // Host input data (row-major, record_num x dim)
                            const float *h_pca_data,    // Host PCA matrix (col-major components, dim x dim)
                            float *h_trans_data,        // Host output projected data (row-major, record_num x column_num)
                            float *h_trans_data_remain, // Host output remaining projected data (row-major, record_num x (dim-col))
                            int record_num,
                            int dim,
                            int column_num);

#endif // PCA_CUH