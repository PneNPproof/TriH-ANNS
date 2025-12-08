/**
 * @file reduce_min.cuh
 * @brief CUDA kernels for parallel minimum finding and top-k selection
 * 
 * This header provides GPU-accelerated functions for finding minimum distances
 * and selecting top-k candidates from large distance matrices. These operations
 * are critical for the first phase of TriH-ANNS where candidates are selected
 * from PCA-projected search results.
 * 
 * Key features:
 * - Half-precision (FP16) support for memory bandwidth optimization
 * - Segmented reduction for per-query minimum finding
 * - Segmented sorting for top-k selection
 * - Optimized for modern GPU architectures with warp-level primitives
 * 
 * Dependencies:
 * - CUDA runtime for GPU execution
 * - CUDA FP16 library for half-precision operations
 * - CUB library for optimized primitives (segmented sort)
 */

#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

/**
 * @brief Compute group-wise minima for distance matrix (FP32 version)
 * 
 * Legacy function for finding minimum distances within groups. Primarily
 * used for debugging and comparison with half-precision versions.
 * 
 * @param d_input Input distance matrix
 * @param n Number of rows (queries)
 * @param m Number of columns (database size)
 * @param group_size Size of each group for reduction
 * @param d_output_min Output minimum distances per group
 * @param d_output_idx Output indices of minimum elements
 * @param block_size CUDA block size for kernel launch
 */
void launchComputeGroupMinimaKernel(
    const float *d_input,
    int n,
    int m,
    int group_size,
    float **d_output_min,
    int **d_output_idx,
    int block_size);

/**
 * @brief Half-precision segmented minimum reduction
 * 
 * Finds minimum distance and corresponding index within each segment
 * of the distance matrix. Each segment corresponds to a group of
 * database vectors, and this function finds the closest vector
 * in each group for each query.
 * 
 * @param dists_per_query Input distance matrix in FP16 (query_num * segment_size * segment_num)
 * @param reduced_dists_per_query Output minimum distances per segment
 * @param reduced_ids_per_query Output indices of minimum elements per segment
 * @param segment_size Number of elements per segment
 * @param segment_num Number of segments total
 * @param seg_num_per_query Number of segments per query
 * @param stream CUDA stream for asynchronous execution
 * 
 * @note Uses warp-level primitives for optimal performance on modern GPUs
 */
void half_matrix_reduce(
    const half *dists_per_query,
    half *reduced_dists_per_query,
    int *reduced_ids_per_query,
    int segment_size,
    int segment_num,
    int seg_num_per_query,
    cudaStream_t stream);

/**
 * @brief Optimized half-precision segmented minimum reduction (v2)
 * 
 * Enhanced version with improved memory access patterns and better
 * utilization of GPU memory hierarchy. Uses configurable warp count
 * for different GPU architectures.
 * 
 * @param dists_per_query Input distance matrix in FP16
 * @param reduced_dists_per_query Output minimum distances per segment
 * @param reduced_ids_per_query Output indices of minimum elements per segment  
 * @param segment_size Number of elements per segment
 * @param dists_num_per_query Total distances per query
 * @param query_batch_num Number of queries in batch
 * @param warp_num_per_block Number of warps per thread block
 * @param stream CUDA stream for asynchronous execution
 * 
 * @note Optimized for better cache utilization and reduced memory traffic
 */
void half_matrix_reduce_v2(
    const half *dists_per_query,
    half *reduced_dists_per_query,
    int *reduced_ids_per_query,
    size_t segment_size,
    size_t dists_num_per_query,
    size_t query_batch_num,
    int warp_num_per_block,
    cudaStream_t stream);

/**
 * @brief Segmented sorting for top-k selection in half precision
 * 
 * Performs segmented sort to extract top-k smallest distances from each
 * query's reduced candidate set. This is the final selection step in
 * Phase 1 of TriH-ANNS.
 * 
 * @param reduced_dists Input/output reduced distances in FP16 (modified in-place)
 * @param reduced_inds Input/output corresponding indices (modified in-place)
 * @param query_num Number of queries (number of segments)
 * @param group_num Length of each segment (candidates per query)
 * @param phase1_topk Number of top-k elements to extract per segment
 * @return cudaError_t Error code (cudaSuccess on success)
 * 
 * @note Uses CUB library's segmented radix sort for optimal performance
 * @note Results are sorted in ascending order (smallest distances first)
 */
cudaError_t segmented_sort_topk_pairs_fp16(
    half *reduced_dists, // [query_num * group_num] in/out - half precision
    int *reduced_inds,   // [query_num * group_num] in/out
    int query_num,       // number of segments
    int group_num,       // length of each segment
    int phase1_topk);    // how many elements to extract from each segment

/**
 * @brief Extract top-k results from sorted segments
 * 
 * Extracts the first 'topk' elements from each sorted segment into
 * a compact output array. This separates the top-k candidates for
 * further processing in Phase 2.
 * 
 * @param reduced_dists Input sorted distances per segment
 * @param reduced_ids Input sorted indices per segment
 * @param topk_dists Output top-k distances in compact format
 * @param topk_ids Output top-k indices in compact format
 * @param query_num Number of queries
 * @param group_num Length of each input segment
 * @param topk Number of top elements to extract per query
 * @param stream CUDA stream for asynchronous execution
 * @param enable_timing Whether to measure kernel execution time
 * @return cudaError_t Error code (cudaSuccess on success)
 * 
 * @note Output arrays must be pre-allocated: [query_num * topk] elements
 */
cudaError_t extract_topk(
    half *reduced_dists,
    int *reduced_ids,
    half *topk_dists,
    int *topk_ids,
    int query_num,
    int group_num,
    int topk,
    cudaStream_t stream = 0,
    bool enable_timing = false);