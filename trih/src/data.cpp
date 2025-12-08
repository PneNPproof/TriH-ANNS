/**
 * @file data.cpp
 * @brief Implementation of dataset loading functions for TriH-ANNS
 * 
 * Provides support for loading datasets in two primary formats:
 * 1. HDF5 format (.h5 files) - commonly used in modern ANNS benchmarks
 * 2. Binary vector format (.fvecs/.ivecs) - traditional format used in SIFT, GIST datasets
 * 
 * The implementation automatically detects file format and uses appropriate loader.
 * Memory allocation is optimized for both CPU SIMD operations and GPU transfers.
 * 
 * Dependencies:
 * - HDF5 library for .h5 file support
 * - Standard C library for binary file operations
 * - POSIX stat for file existence checking
 */

#include <fstream>
#include <iostream>
#include <stdio.h>
#include <sys/stat.h>
#include "hdf5.h"

#include "data.h"

namespace trih
{

/**
 * @brief Load dataset from HDF5 format file
 * 
 * Reads datasets named 'train', 'test', and 'neighbors' from HDF5 file.
 * This is the preferred format for large-scale datasets due to compression
 * and metadata support.
 * 
 * @param filename Path to HDF5 (.h5) file
 * @return Loaded dataset with allocated memory
 * 
 * @note Training data uses 64-byte aligned allocation for SIMD optimization
 * @note Assumes datasets are stored as floating point (train/test) and unsigned int (neighbors)
 */
data load_h5(const char *filename)
{
    data d;
    hid_t file_id, dataset_id, dataspace_id;
    hsize_t dims[2];

    // Open HDF5 file in read-only mode
    file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);

    // Load training data
    dataset_id = H5Dopen(file_id, "train", H5P_DEFAULT);
    dataspace_id = H5Dget_space(dataset_id);
    H5Sget_simple_extent_dims(dataspace_id, dims, NULL);

    d.dim = dims[1];                    // Feature dimensionality
    d.train_point_count = dims[0];      // Number of training points
    
    // Use 64-byte aligned allocation for optimal SIMD performance
    d.train = (float *)aligned_alloc(64, dims[0]*dims[1]*sizeof(float));
    H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d.train);
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);

    // Load test/query data
    dataset_id = H5Dopen(file_id, "test", H5P_DEFAULT);
    dataspace_id = H5Dget_space(dataset_id);
    H5Sget_simple_extent_dims(dataspace_id, dims, NULL);

    d.test_point_count = dims[0];       // Number of test queries
    d.test = new float[dims[0]*dims[1]]; // Standard allocation for test data
    H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d.test);
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);

    // Load ground truth nearest neighbors
    dataset_id = H5Dopen(file_id, "neighbors", H5P_DEFAULT);
    // Alternative: dataset_id = H5Dopen(file_id, "neighbor_indices", H5P_DEFAULT);
    dataspace_id = H5Dget_space(dataset_id);
    H5Sget_simple_extent_dims(dataspace_id, dims, NULL);

    d.neighbors_per_test = dims[1];     // Number of neighbors per query (typically 100)
    d.neighbors = new uint32_t[dims[0]*dims[1]];
    H5Dread(dataset_id, H5T_NATIVE_UINT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d.neighbors);
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);

    H5Fclose(file_id);

    return d;
}

/**
 * @brief Load dataset from binary vector format files
 * 
 * Loads traditional .fvecs/.ivecs format files commonly used in ANNS benchmarks.
 * Expects three files with given prefix:
 * - {prefix}_base.fvecs: Training vectors
 * - {prefix}_query.fvecs: Query vectors  
 * - {prefix}_groundtruth.ivecs: Ground truth nearest neighbors
 * 
 * @param filename_prefix Prefix for .fvecs/.ivecs files
 * @return Loaded dataset with allocated memory
 * 
 * @note Binary format: each vector prefixed with dimension as 4-byte integer
 * @note All vectors in a file must have same dimensionality
 */
data load_vecs(const char *filename_prefix)
{
    data d;
    char filename[100];
    FILE *file;
    long file_size;

    // 1. Load training data from base.fvecs
    sprintf(filename, "%s_base.fvecs", filename_prefix);
    file = fopen(filename, "rb");  // Open in binary mode

    // Read dimensionality from first vector header
    if(  fread(&d.dim, sizeof(int), 1, file) != 1) {
        perror("fread test vector");
        throw std::runtime_error("Failed to read test vector from data file");
    }
  

    // Calculate number of vectors based on file size
    fseek(file, 0, SEEK_END);
    file_size = ftell(file);
    d.train_point_count = file_size / (d.dim * sizeof(float) + sizeof(int));  // (dim+1)*4 bytes per vector

    // Read all training vectors
    rewind(file);  // Return to file beginning
    d.train = new float[d.dim*d.train_point_count];
    for(int i=0; i<d.train_point_count; i++){
        fseek(file, sizeof(int), SEEK_CUR);     // Skip dimension header
        // Read vector data
        if(  fread(d.train+d.dim*i, sizeof(float)*d.dim, 1, file) != 1) {
            perror("fread test vector");
            throw std::runtime_error("Failed to read test vector from data file");
        }
    }
    fclose(file);

    // 2. Load test/query data from query.fvecs
    sprintf(filename, "%s_query.fvecs", filename_prefix);
    file = fopen(filename, "rb");  // Open in binary mode

    // Calculate number of query vectors
    fseek(file, 0, SEEK_END);
    file_size = ftell(file);
    d.test_point_count = file_size / (d.dim * sizeof(float) + sizeof(int));  // (dim+1)*4 bytes per vector

    // Read all query vectors
    rewind(file);  // Return to file beginning
    d.test = new float[d.dim*d.test_point_count];
    for(int i=0; i<d.test_point_count; i++){
        fseek(file, sizeof(int), SEEK_CUR);     // Skip dimension header
        // Read vector data
        if( fread(d.test+d.dim*i, sizeof(float)*d.dim, 1, file) != 1) {
            perror("fread test vector");
            throw std::runtime_error("Failed to read test vector from data file");
        }
    }
    fclose(file);

    // 3. Load ground truth neighbors from groundtruth.ivecs
    sprintf(filename, "%s_groundtruth.ivecs", filename_prefix);
    file = fopen(filename, "rb");  // Open in binary mode

    // Read number of neighbors per query from first header
    if( fread(&d.neighbors_per_test, sizeof(int), 1, file) != 1) {
        perror("fread test vector");
        throw std::runtime_error("Failed to read test vector from data file");
    }
   

    // Calculate number of ground truth vectors
    fseek(file, 0, SEEK_END);
    file_size = ftell(file);
    int vec_count = file_size / (d.neighbors_per_test * sizeof(int) + sizeof(int));  // (neighbors+1)*4 bytes per entry

    // Read all ground truth neighbor lists
    rewind(file);  // Return to file beginning
    d.neighbors = new uint32_t[d.neighbors_per_test*vec_count];
    for(int i=0; i<vec_count; i++){
        fseek(file, sizeof(int), SEEK_CUR);     // Skip neighbor count header
        if( fread(d.neighbors+d.neighbors_per_test*i, sizeof(uint32_t)*d.neighbors_per_test, 1, file) != 1) {
            perror("fread test vector");
            throw std::runtime_error("Failed to read test vector from data file");
        }
    }
    fclose(file);

    return d;
}

/**
 * @brief Auto-detecting dataset loader
 * 
 * Determines file format based on file existence and calls appropriate loader:
 * - If filename exists as a file, assumes HDF5 format
 * - Otherwise, treats as prefix for .fvecs/.ivecs format
 * 
 * @param filename Path to dataset file or filename prefix
 * @return Loaded dataset structure
 * 
 * @note Prints dataset statistics when DETAILED_LOG is defined
 * @note This is the primary interface used by the main application
 */
data load_data(const char *filename) {
    struct stat buffer;
    data d;
    
    // Check if file exists to determine format
    if(stat(filename, &buffer) == 0) {
        // File exists - assume HDF5 format
        d = load_h5(filename);
    }
    else {
        // File doesn't exist - assume .fvecs/.ivecs prefix
        d = load_vecs(filename);
    }
    
    #ifdef DETAILED_LOG
    std::cout << "\tData info: " << std::endl;
    std::cout << "\tdim: " << d.dim << std::endl;
    std::cout << "\ttrain: " << d.train_point_count << std::endl;
    std::cout << "\ttest: " << d.test_point_count << std::endl;
    std::cout << "\tneigbors per test: " << d.neighbors_per_test << std::endl;
    #endif

    return d;
}

}