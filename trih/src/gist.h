/**
 * @file gist.h
 * @brief Legacy GIST dataset loading interface
 * 
 * This header provides a legacy interface for loading GIST datasets with
 * distance information. It appears to be an older version of the dataset
 * loading functionality that includes distance data in addition to neighbor
 * indices.
 * 
 * Note: This interface is largely superseded by the trih::data structure
 * defined in data.h, which provides more comprehensive dataset handling.
 * This may be retained for backward compatibility or specific GIST benchmarks.
 * 
 * Dependencies:
 * - HDF5 library for dataset file loading
 */

#ifndef __GIST_H__
#define __GIST_H__

#include "hdf5.h"

/**
 * @brief Legacy GIST dataset structure with distance information
 * 
 * Extended dataset structure that includes both neighbor indices and
 * corresponding distances. This provides additional ground truth
 * information that may be useful for certain evaluation scenarios.
 */
typedef struct gist_t
{
    uint32_t dim;                 ///< Feature dimensionality

    float *train;                 ///< Training vectors
    uint32_t train_point_count;   ///< Number of training vectors

    float *test;                  ///< Test/query vectors
    uint32_t test_point_count;    ///< Number of test vectors

    uint32_t *neighbors;          ///< Ground truth neighbor indices
    float *distances;             ///< Ground truth distances (additional info)
    uint32_t neighbors_per_test;  ///< Number of neighbors per test query
} gist;

/**
 * @brief Load GIST dataset with distance information
 * 
 * Legacy function for loading GIST datasets that include both neighbor
 * indices and their corresponding distances in the HDF5 file.
 * 
 * @param filename Path to HDF5 file containing GIST dataset
 * @return Loaded GIST dataset structure
 * 
 * @note This is a legacy interface - consider using trih::load_data() instead
 * @note Caller is responsible for freeing allocated memory
 */
gist load_gist(const char *filename);

#endif