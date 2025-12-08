/**
 * @file rerank.h
 * @brief CPU-based reranking algorithms for precise candidate filtering
 * 
 * This header provides the Phase 2 reranking functionality of TriH-ANNS.
 * After Phase 1 GPU search identifies candidate nearest neighbors using
 * PCA-projected features, these functions perform precise distance computation
 * using full-dimensional features to produce final results.
 * 
 * Key features:
 * - Multiple reranking strategies (exact and quantized)
 * - Heap-based top-k selection for efficiency
 * - Support for scalar quantization to reduce memory bandwidth
 * - Optimized distance computation using SIMD instructions
 * 
 * The reranking phase runs asynchronously on CPU threads while GPU
 * processes the next batch, enabling pipeline parallelism.
 * 
 * Dependencies:
 * - Scalar quantization support (sq.h)
 * - PCA structures for dimension information
 * - CUDA FP16 types for interface compatibility
 */

#pragma once
#include <cstdint>
#include "sq.h"
#include "pca.h"

/**
 * @brief Heap element for efficient top-k selection during reranking
 * 
 * Template structure used in priority queues/heaps for maintaining
 * the current top-k candidates during reranking. Stores both the
 * original index and computed distance for efficient comparisons.
 * 
 * @tparam T Distance type (typically float or half)
 */
template<typename T>
struct HeapElement{
    int index;      ///< Original dataset index
    T distance;     ///< Computed distance value (cached for efficiency)
} ;

/**
 * @brief Standard exact reranking using full-dimensional features
 * 
 * Performs precise L2 distance computation between query and Phase 1
 * candidates using complete feature vectors. This is the most accurate
 * reranking method but also the most computationally expensive.
 * 
 * @param dataset Complete dataset vectors for distance computation
 * @param dataset_squared_norms Precomputed squared norms for efficiency
 * @param query Query vector (full dimensionality)
 * @param phase1_topk_ids Candidate indices from Phase 1
 * @param phase1_topk Number of candidates to rerank
 * @param data_num Total number of vectors in dataset
 * @param dim Full feature dimensionality
 * @param phase2_topk Number of final results to return
 * @param phase2_topk_ids Output: final top-k indices
 * 
 * @note Uses optimized SIMD distance computation for performance
 * @note Results are guaranteed to be exact (no approximation error)
 */
void re_rank2(
  float* dataset, 
  float* dataset_squared_norms, 
  float *query, 
  int *phase1_topk_ids,
  int phase1_topk, 
  int data_num, 
  int dim, 
  int phase2_topk, 
  int *phase2_topk_ids
);

/**
 * @brief Memory-efficient reranking using scalar quantization
 * 
 * Performs approximate reranking using scalar-quantized vectors to
 * reduce memory bandwidth and improve cache performance. Uses precomputed
 * quantization statistics for fast distance approximation.
 * 
 * @param query Original query vector
 * @param quant_query Quantized query vector (8-bit per dimension)
 * @param query_remain Query projected to remaining dimensions
 * @param dim Full feature dimensionality
 * @param pca_dim PCA-reduced dimensionality
 * @param p_sq_info Scalar quantization info for each dataset vector
 * @param phase1_topk_dists Phase 1 distances (for tie-breaking)
 * @param phase1_topk_ids Candidate indices from Phase 1
 * @param phase1_topk Number of candidates to rerank
 * @param phase2_topk_ids Output: final top-k indices
 * @param phase2_topk Number of final results to return
 * @return Status code (0 = success)
 * 
 * @note Uses 8-bit quantization for ~4x memory bandwidth reduction
 * @note Slight accuracy loss compared to exact reranking
 * @note Significantly faster for memory-bound workloads
 */
int re_rank(
  float *query,
  uint8_t *quant_query,
  float *query_remain,
  int dim,
  int pca_dim,
  sq_info *p_sq_info,
  half *phase1_topk_dists,
  int *phase1_topk_ids,
  int phase1_topk,
  int *phase2_topk_ids,
  int phase2_topk
);