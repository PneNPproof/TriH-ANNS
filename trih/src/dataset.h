#pragma once

#include <cstdint>

class DataSet
{
public:
  
  float *train_set; // test_set_size x data_dim
  float *test_set; // train_set_size x data_dim
  uint32_t *ground_truth_set; // test_set_size x neighbors_per_test

  uint32_t train_set_size;
  uint32_t test_set_size;
  uint32_t data_dim;
  uint32_t neighbors_per_test;

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
  ~DataSet()
  {
    delete[] train_set;
    delete[] test_set;
    delete[] ground_truth_set;
  };
};

