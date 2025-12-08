/**
 * @file dataset.h
 * @brief Alternative dataset wrapper class for ANNS benchmarking
 * 
 * This header provides an alternative dataset representation with RAII
 * memory management. While the main system uses the trih::data structure
 * (defined in data.h), this class provides a more object-oriented interface
 * with automatic memory cleanup.
 * 
 * Key differences from trih::data:
 * - RAII-style memory management with destructor cleanup
 * - Object-oriented interface vs. C-style struct
 * - May be used in specific testing or benchmarking contexts
 * 
 * Dependencies:
 * - Standard C++ library for uint32_t type definitions
 */

#pragma once

#include <cstdint>

/**
 * @brief RAII wrapper for ANNS dataset with automatic memory management
 * 
 * Encapsulates training data, test queries, and ground truth neighbors
 * with automatic memory cleanup. Provides a safer alternative to manual
 * memory management for dataset handling.
 */
class DataSet
{
public:
  
  float *train_set;                 ///< Training vectors (train_set_size x data_dim)
  float *test_set;                  ///< Test/query vectors (test_set_size x data_dim)
  uint32_t *ground_truth_set;       ///< Ground truth neighbors (test_set_size x neighbors_per_test)

  uint32_t train_set_size;          ///< Number of training vectors
  uint32_t test_set_size;           ///< Number of test/query vectors
  uint32_t data_dim;                ///< Feature dimensionality
  uint32_t neighbors_per_test;      ///< Number of ground truth neighbors per query

  /**
   * @brief Constructor - takes ownership of provided memory
   * 
   * Initializes dataset with provided arrays. The DataSet object takes
   * ownership of the memory and will free it in the destructor.
   * 
   * @param train_set_ Training data array
   * @param test_set_ Test/query data array
   * @param ground_truth_set_ Ground truth neighbor indices
   * @param train_set_size_ Number of training vectors
   * @param test_set_size_ Number of test vectors
   * @param data_dim_ Feature dimensionality
   * @param neighbors_per_test_ Ground truth neighbors per query
   * 
   * @warning Caller must not free the provided arrays - DataSet takes ownership
   */
  DataSet(float *train_set_, float *test_set_, uint32_t *ground_truth_set_,
          uint32_t train_set_size_, uint32_t test_set_size_,
          uint32_t data_dim_, uint32_t neighbors_per_test_)
      : train_set(train_set_),
        test_set(test_set_),
        ground_truth_set(ground_truth_set_),
        train_set_size(train_set_size_),
        test_set_size(test_set_size_),
        data_dim(data_dim_),
        neighbors_per_test(neighbors_per_test_)
  {
  };
  
  /**
   * @brief Destructor - automatically frees all owned memory
   * 
   * Ensures proper cleanup of all allocated arrays to prevent memory leaks.
   * This is the key advantage over C-style manual memory management.
   */
  ~DataSet()
  {
    delete[] train_set;
    delete[] test_set;
    delete[] ground_truth_set;
  };
};

