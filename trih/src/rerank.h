#pragma once
#include <cstdint>

template<typename T>
struct HeapElement{
    int index;     // 原始数组索引
    T distance; // 缓存的距离值
} ;

int re_rank(
  float* dataset, 
  float* dataset_squared_norms, 
  float *query, 
  int *phase1_topk_ids,
  int phase1_topk, 
  int data_num, 
  int dim, 
  int phase2_topk, 
  int *phase2_topk_ids
);