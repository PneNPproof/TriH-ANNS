/**
 * @file l2mm.cuh
 * @brief CUDA kernels for L2 distance matrix computation
 * 
 * This header provides GPU-accelerated functions for computing L2 (Euclidean) distance
 * matrices between query vectors and database vectors. The implementation supports:
 * - Both FP32 and FP16 precision for memory/performance tradeoffs
 * - Streaming for overlapped computation
 * - Vector addition utilities for distance computation
 * 
 * The L2 distance matrix computation is a critical component in the first phase
 * of TriH-ANNS, where queries are compared against the PCA-projected database
 * to identify candidate nearest neighbors.
 * 
 * Dependencies:
 * - CUDA runtime for GPU execution
 * - CUDA FP16 library for half-precision operations
 */

#pragma once

#include <cuda_runtime.h>  
#include <stddef.h>
#include <cuda_fp16.h>

/**
 * @brief Element-wise vector addition for half-precision vectors
 * 
 * Performs A = A + B for half-precision vectors, typically used in distance
 * computation pipelines where bias terms need to be added.
 * 
 * @param d_A Input/output vector A (modified in-place)
 * @param d_B Input vector B to add
 * @param n Number of elements in vectors
 * @param stream CUDA stream for asynchronous execution
 */
void addVectors(half* d_A, const half* d_B, int n, cudaStream_t stream);

/**
 * @brief Compute L2 distance matrix in single precision
 * 
 * Computes the L2 (Euclidean) distance matrix between m query vectors and n database vectors.
 * For each query-database pair (i,j), computes ||A[i] - B[j]||_2.
 * 
 * @param m Number of query vectors
 * @param n Number of database vectors
 * @param k Dimensionality of vectors
 * @param A Query vectors matrix (m x k)
 * @param B Database vectors matrix (n x k)
 * @param C Output distance matrix (m x n)
 * @param stream CUDA stream for asynchronous execution
 * 
 * @note Uses optimized CUDA kernels for high throughput
 * @note Memory layout: row-major for optimal coalescing
 */
void l2mm(
  int m,
  int n,
  int k,
  const float *A,
  const float *B,
  float *C,
  cudaStream_t stream);

/**
 * @brief Compute L2 distance matrix in half precision
 * 
 * Half-precision version of L2 distance matrix computation for improved
 * memory bandwidth and throughput on modern GPUs with tensor cores.
 * 
 * @param m Number of query vectors
 * @param n Number of database vectors  
 * @param k Dimensionality of vectors
 * @param A Query vectors matrix in FP16 (m x k)
 * @param B Database vectors matrix in FP16 (n x k)
 * @param C Output distance matrix in FP16 (m x n)
 * @param stream CUDA stream for asynchronous execution
 * 
 * @note Leverages tensor cores for accelerated computation when available
 * @note May have slightly reduced numerical precision compared to FP32 version
 * @note Approximately 2x memory bandwidth improvement over FP32
 */
void l2mm_fp16(
  int m,
  int n,
  int k,
  const half *A,
  const half *B,
  half *C,
  cudaStream_t stream);