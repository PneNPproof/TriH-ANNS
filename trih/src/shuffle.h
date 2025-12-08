/**
 * @file shuffle.h
 * @brief Data shuffling utilities for memory access optimization
 * 
 * This header provides functions for shuffling dataset vectors and associated
 * metadata to improve memory access patterns during search operations.
 * Data shuffling can significantly improve cache performance by ensuring
 * that frequently accessed vectors are stored contiguously in memory.
 * 
 * The shuffling functions maintain consistency between:
 * - Training vectors and their indices
 * - Associated metadata (neighbor indices, etc.)
 * 
 * Dependencies:
 * - Standard C library for integer types
 */

#ifndef __SHUFFLE_H__
#define __SHUFFLE_H__

#include <stdint.h>

/**
 * @brief Shuffle vectors and corresponding IDs consistently
 * 
 * Applies the same random permutation to both vector data and their
 * associated ID array to maintain correspondence after shuffling.
 * 
 * @param vectors Input/output vector array (size x dim)
 * @param dim Dimensionality of each vector
 * @param size Number of vectors to shuffle
 * @param ids Input/output ID array corresponding to vectors
 * 
 * @note Both vectors and ids arrays are modified in-place
 * @note Uses consistent permutation to maintain vector-ID correspondence
 */
void myshuffle(float *vectors, int dim, int size, int *ids);

/**
 * @brief Shuffle vectors with additional associated data
 * 
 * Extended shuffling function that applies the same permutation to
 * training vectors and a separate array of associated data (e.g.,
 * neighbor indices, metadata). This ensures all related data
 * remains synchronized after shuffling.
 * 
 * @param vectors Input/output vector array (size x dim)
 * @param dim Dimensionality of each vector
 * @param size Number of vectors to shuffle
 * @param others Input/output associated data array
 * @param others_size Size of the associated data array
 * 
 * @note All arrays are modified in-place using the same permutation
 * @note Used when multiple data structures need to stay synchronized
 */
void myshuffle2(float *vectors, int dim, int size, int *others, int others_size);

#endif