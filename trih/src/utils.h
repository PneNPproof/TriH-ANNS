/**
 * @file utils.h
 * @brief Utility functions and helpers for TriH-ANNS system
 * 
 * This header provides essential utility functions for:
 * - Index building and management
 * - Data shuffling and persistence
 * - Performance evaluation (recall calculation)
 * - Query batch preparation for multi-worker processing
 * - Memory allocation utilities for CUDA interoperability
 * 
 * Dependencies:
 * - CUDA runtime for GPU memory allocation helpers
 * - Standard C++ library for I/O operations
 * - Custom data structures (data, pca_index)
 */

#pragma once

#include <iostream>
#include <fstream>
#include <cstddef> // For size_t

#include <cuda_runtime.h>

#include "data.h"
#include "pca.h"

using namespace std;

namespace trih
{

/**
 * @brief Save shuffled dataset to binary file
 * 
 * Persists shuffled training data and neighbor indices to disk for later use.
 * This is used when data shuffling is applied to improve memory access patterns
 * during index building or search operations.
 * 
 * @param filename Output binary file path
 * @param gdata Dataset structure with shuffled data
 * 
 * @note File format: training_vectors + neighbor_indices (binary)
 */
void save_shuffled_data(const char* filename, data &gdata);

/**
 * @brief Load previously shuffled dataset from binary file
 * 
 * Overwrites current dataset with shuffled version from disk.
 * Must match the exact format created by save_shuffled_data.
 * 
 * @param filename Input binary file path
 * @param gdata Dataset structure to be overwritten with shuffled data
 * 
 * @note Assumes gdata has been allocated with correct dimensions
 */
void load_shuffled_data(const char* filename, data &gdata);

/**
 * @brief Build PCA-based search index from training data
 * 
 * Main index building function that:
 * 1. Computes PCA transformation from training vectors
 * 2. Projects training data to PCA space and remaining dimensions
 * 3. Saves complete index to binary file
 * 
 * @param base Training data vectors (size x dimension)
 * @param dimension Original feature dimensionality
 * @param size Number of training vectors
 * @param column_num Target number of PCA components (in/out parameter)
 * @param ratio Target variance ratio to preserve (in/out parameter)
 * @param ofs Output file stream for saving index
 * 
 * @note Either column_num or ratio is used depending on which is specified
 * @note Both column_num and ratio may be modified to reflect actual values used
 */
void build_index(float *base, int dimension, int size, int &column_num, float &ratio, ofstream &ofs);

/**
 * @brief Calculate Recall@K performance metric
 * 
 * Computes average recall across a batch of queries, handling:
 * - Duplicate IDs in ground truth or retrieved results
 * - Variable ground truth list sizes
 * - Batch processing for efficiency
 * 
 * Recall@K = (unique relevant items in top-K) / (unique relevant items in ground truth top-K)
 * 
 * @param gt Ground truth neighbor indices (batch_query_num x gt_neighbors_per_query)
 * @param topk_ids Retrieved top-K indices (batch_query_num x topk)
 * @param topk Number of top results to evaluate (K in Recall@K)
 * @param gt_neighbors_per_query Number of ground truth neighbors per query
 * @param batch_query_num Number of queries in batch
 * 
 * @note Prints recall statistics to stdout
 * @note Only considers first 'topk' entries from ground truth as relevant
 */
void cal_recall(const int *gt, const int *topk_ids, int topk, int gt_neighbors_per_query, int batch_query_num);

/**
 * @brief Prepare query batches for multi-worker processing
 * 
 * Organizes test queries into batches suitable for parallel processing by multiple workers.
 * Each worker processes multiple batches to balance load and maximize GPU utilization.
 * 
 * @param gdata Dataset containing test queries and ground truth
 * @param query_batch_size Number of queries per batch
 * @param worker_num Number of worker threads/GPUs
 * @param batch_num_per_worker Number of batches each worker will process
 * @param batch_query Output array for organized query batches
 * @param batch_query_groundtruth Output array for corresponding ground truth
 * 
 * @note Output arrays must be pre-allocated by caller
 * @note Query distribution ensures balanced workload across workers
 */
void prepare_query_batch(data &gdata, int query_batch_size, int worker_num, int batch_num_per_worker, float *batch_query, int *batch_query_groundtruth);

}

/**
 * @brief Allocate aligned host memory compatible with CUDA
 * 
 * Allocates host memory with specified alignment for optimal GPU transfer performance.
 * Use this instead of malloc/new when data will be transferred to/from GPU.
 * 
 * @param ptr_out Output pointer to allocated memory
 * @param size Size in bytes to allocate
 * @param alignment Memory alignment in bytes (typically 64 for SIMD/GPU efficiency)
 * @return cudaSuccess on success, error code on failure
 * 
 * @note Memory must be freed with cudaFreeHost() or free() depending on implementation
 * @note Aligned memory improves performance for SIMD operations and GPU transfers
 */
cudaError_t aligned_malloc_host(void** ptr_out, size_t size, size_t alignment);