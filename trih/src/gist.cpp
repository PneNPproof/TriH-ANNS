/**
 * @file gist.cpp
 * @brief Implementation of GIST dataset loading functionality
 * 
 * This file provides functions to load the GIST dataset from HDF5 format.
 * The GIST dataset is a standard benchmark dataset for approximate nearest
 * neighbor search, containing 1M training vectors and 1K test queries in
 * 960-dimensional space.
 * 
 * Features:
 * - Memory-aligned allocation for optimal SIMD performance
 * - HDF5-based data loading for efficient storage format
 * - Ground truth neighbors and distances for evaluation
 */

#include <fstream>

#include "hdf5.h"
#include "gist.h"

/**
 * @brief Load GIST dataset from HDF5 file
 * 
 * Loads the complete GIST dataset including training vectors, test queries,
 * ground truth neighbors, and exact distances. Memory is allocated with
 * 64-byte alignment for optimal SIMD performance.
 * 
 * Dataset specifications:
 * - Training set: 1,000,000 vectors × 960 dimensions
 * - Test set: 1,000 query vectors × 960 dimensions  
 * - Ground truth: 100 nearest neighbors per query
 * - Data type: 32-bit floating point
 * 
 * @param filename Path to the HDF5 file containing GIST dataset
 * @return gist structure containing all loaded data
 * 
 * @note Memory must be freed by caller using aligned_free()
 * @note Requires HDF5 library for file I/O operations
 */
gist load_gist(const char *filename)
{
    gist data;
    
    // Initialize dataset parameters
    data.dim = 960;
    data.train_point_count = 1000000;
    data.test_point_count = 1000;
    data.neighbors_per_test = 100;
    
    // Allocate memory with 64-byte alignment for SIMD optimization
    data.train = static_cast<float*>(aligned_alloc(64, 1000000 * 960 * sizeof(float)));
    data.test = static_cast<float*>(aligned_alloc(64, 1000 * 960 * sizeof(float)));
    data.neighbors = static_cast<uint32_t*>(aligned_alloc(64, 1000 * 100 * sizeof(uint32_t)));
    data.distances = static_cast<float*>(aligned_alloc(64, 1000 * 100 * sizeof(float)));

    // Open HDF5 file for reading
    hid_t h5f = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);

    // Load training data (1M × 960)
    hid_t ds_train = H5Dopen(h5f, "train", H5P_DEFAULT);
    H5Dread(ds_train, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.train);
    H5Dclose(ds_train);

    // Load test queries (1K × 960)
    hid_t ds_test = H5Dopen(h5f, "test", H5P_DEFAULT);
    H5Dread(ds_test, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.test);
    H5Dclose(ds_test);

    // Load ground truth neighbors (1K × 100)
    hid_t ds_neighbors = H5Dopen(h5f, "neighbors", H5P_DEFAULT);
    H5Dread(ds_neighbors, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.neighbors);
    H5Dclose(ds_neighbors);

    // Load exact distances to ground truth neighbors (1K × 100)
    hid_t ds_distances = H5Dopen(h5f, "distances", H5P_DEFAULT);
    H5Dread(ds_distances, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.distances);
    H5Dclose(ds_distances);

    // Close HDF5 file
    H5Fclose(h5f);

    return data;
}