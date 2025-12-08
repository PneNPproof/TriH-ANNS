/**
 * @file data.h
 * @brief Data structure definitions and loading interface for TriH-ANNS datasets
 * 
 * This header defines the core data structure used throughout the TriH-ANNS system
 * for representing datasets with training points, test queries, and ground truth neighbors.
 * Supports both HDF5 and binary vector file formats commonly used in ANNS benchmarks.
 * 
 * Dependencies:
 * - HDF5 library for reading .h5 dataset files
 * 
 * Supported file formats:
 * - HDF5 files with 'train', 'test', and 'neighbors' datasets
 * - Binary vector files (.fvecs, .ivecs) as used in SIFT, GIST, etc.
 */

#ifndef __GIST_H__
#define __GIST_H__

#include "hdf5.h"

namespace trih
{
    /**
     * @brief Core data structure representing an ANNS dataset
     * 
     * Contains training data for index building, test queries for evaluation,
     * and ground truth nearest neighbors for recall calculation.
     * Memory layout is optimized for both CPU and GPU processing.
     */
    typedef struct data_t
    {
        uint32_t dim;                    ///< Dimensionality of feature vectors

        float *train;                    ///< Training data points (record_count x dim)
        uint32_t train_point_count;      ///< Number of training vectors

        float *test;                     ///< Query vectors for search evaluation (test_count x dim)
        uint32_t test_point_count;       ///< Number of test/query vectors

        uint32_t *neighbors;             ///< Ground truth nearest neighbors (test_count x neighbors_per_test)
        uint32_t neighbors_per_test;     ///< Number of ground truth neighbors per query (typically 100)
    } data;

    /**
     * @brief Load dataset from file (auto-detects format)
     * 
     * Automatically determines file format and calls appropriate loader:
     * - If file exists as-is, assumes HDF5 format
     * - Otherwise, attempts to load as binary vector format using filename as prefix
     * 
     * @param filename Path to dataset file or filename prefix for .fvecs/.ivecs files
     * @return Loaded dataset structure with allocated memory
     * 
     * @note Caller is responsible for freeing allocated memory in returned structure
     * @note Training data uses 64-byte aligned allocation for SIMD optimization
     */
    trih::data load_data(const char *filename);

}
#endif