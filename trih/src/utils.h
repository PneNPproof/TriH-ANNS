#pragma once

#include <iostream>
#include <fstream>
#include <cstddef> // For size_t

#include <cuda_runtime.h>




#include "data.h"
#include "pca.h"

using namespace std;

namespace trih
{

void save_shuffled_data(const char* filename, data &gdata);
void load_shuffled_data(const char* filename, data &gdata);
void build_index(float *base, int dimension, int size, int &column_num, float &ratio, ofstream &ofs);

void cal_recall(const int *gt, const int *topk_ids, int topk, int gt_neighbors_per_query, int batch_query_num);
void prepare_query_batch(data &gdata, int query_batch_size, int worker_num, int batch_num_per_worker, float *batch_query, int *batch_query_groundtruth);
}

cudaError_t aligned_malloc_host(void** ptr_out, size_t size, size_t alignment);