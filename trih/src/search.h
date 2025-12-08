/**
 * @file search.h
 * @brief Core search algorithms and distance computation functions for TriH-ANNS
 * 
 * This header provides the fundamental search algorithms used in TriH-ANNS:
 * - Distance computation utilities (Euclidean distance)
 * - Top-k selection algorithms
 * - Multi-phase search coordination
 * - Scalar quantization support for memory efficiency
 * 
 * The search system supports both exact and approximate modes with optional
 * scalar quantization for reduced memory footprint during reranking phase.
 * 
 * Dependencies:
 * - PCA index structures for transformation matrices
 * - Standard C++ library for algorithm utilities
 */

#ifndef __SEARCH_H__
#define __SEARCH_H__

#include "pca.h"

/**
 * @brief Compute Euclidean distance between two vectors
 * 
 * Basic L2 distance computation for exact reranking phase.
 * 
 * @param a First vector
 * @param b Second vector  
 * @param dimension Vector dimensionality
 * @return Euclidean distance between vectors
 */
float euclideanDistance(const float a[], const float b[], int dimension);

/**
 * @brief Compute distances from query to all base vectors
 * 
 * Batch distance computation for brute-force search or reranking.
 * 
 * @param query Query vector
 * @param base Database vectors (size x dimension)
 * @param dimension Vector dimensionality
 * @param size Number of database vectors
 * @param distances Output distance array
 */
void computeDistances(const float *query, const float *base, int dimension, int size, float distances[]);

/**
 * @brief Select top-k vectors with smallest distances within sections
 * 
 * Partitioned top-k selection that processes data in sections for memory efficiency.
 * 
 * @param distances Input distance array
 * @param size Total number of distances
 * @param sectionNum Number of sections to process
 * @param topk Number of top results per section
 * @param ids Output indices of top-k results
 */
void computeTopkDistances(const float *distances, int size, int sectionNum, int topk, int *ids);

/**
 * @brief Multi-phase search with multiple reranking strategies
 * 
 * Complete search pipeline that combines PCA-based filtering with multiple
 * reranking approaches for comparison and validation.
 * 
 * @param query Query vector
 * @param base Complete database vectors
 * @param index PCA index with transformation matrices
 * @param section_num Number of sections for processing
 * @param topk Phase 1 candidates
 * @param ids Phase 1 candidate indices
 * @param final_topk Final number of results
 * @param final_topk_ids0 Output: basic reranking results
 * @param final_topk_ids Output: standard reranking results
 * @param final_topk_ids_sq Output: scalar quantized reranking results
 * @param final_topk_ids_shuffle Output: shuffled reranking results
 */
void search_index(float *query, float *base, pca_index &index, int section_num, int topk, int *ids, int final_topk, int *final_topk_ids0, int *final_topk_ids, int *final_topk_ids_sq, int *final_topk_ids_shuffle);

/**
 * @brief Extended multi-phase search with additional scalar quantization options
 * 
 * Enhanced version supporting pre-quantized database and additional quantization parameters.
 * 
 * @param query Query vector
 * @param base Complete database vectors
 * @param index PCA index with transformation matrices
 * @param section_num Number of sections for processing
 * @param topk Phase 1 candidates
 * @param ids Phase 1 candidate indices
 * @param final_topk Final number of results
 * @param final_topk_ids0 Output: basic reranking results
 * @param final_topk_ids Output: standard reranking results
 * @param final_topk_ids_sq Output: scalar quantized reranking results
 * @param final_topk_ids_shuffle Output: shuffled reranking results
 * @param quant_base Pre-quantized database vectors
 * @param scale Quantization scale factor
 * @param zp Quantization zero point
 * @param b Quantization bit width
 * @param final_topk_ids_sq_all_vec Output: full vector quantization results
 */
void search_index(float *query, float *base, pca_index &index, int section_num, int topk, int *ids, int final_topk, int *final_topk_ids0, int *final_topk_ids, int *final_topk_ids_sq, int *final_topk_ids_shuffle,
    uint8_t *quant_base, float scale, float zp, int b, int *final_topk_ids_sq_all_vec);

/**
 * @brief Clip and quantize a single value
 * 
 * @param x Input floating point value
 * @param scale Quantization scale
 * @param zp Zero point
 * @param start Clipping range minimum
 * @param end Clipping range maximum
 * @return Quantized 8-bit value
 */
uint8_t clip(float x, float scale, float zp, int start, int end);

/**
 * @brief Quantize a complete vector
 * 
 * @param point Input floating point vector
 * @param quant_point Output quantized vector
 * @param scale Quantization scale
 * @param zp Zero point
 * @param dimension Vector dimensionality
 * @param b Bit width (typically 8)
 */
void quant_point(const float *point, uint8_t *quant_point, float scale, float zp, int dimension, int b);

/**
 * @brief Dequantize a vector back to floating point
 * 
 * @param quant_point Input quantized vector
 * @param point Output floating point vector
 * @param scale Quantization scale
 * @param zp Zero point
 * @param dimension Vector dimensionality
 */
void dequant_point(uint8_t *quant_point, float *point, float scale, float zp, int dimension);

/**
 * @brief Dequantize L2 distance
 * 
 * @param distance Quantized distance
 * @param scale Quantization scale
 * @return Dequantized distance
 */
float dequant_l2(float distance, float scale);

#endif