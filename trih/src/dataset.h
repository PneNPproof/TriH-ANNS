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

  DataSet(/* args */);
  ~DataSet();
};

