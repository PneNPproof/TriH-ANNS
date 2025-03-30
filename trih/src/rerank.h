#pragma once
#include <cstdint>
#include "sq.h"
#include "pca.h"

template<typename T>
struct HeapElement{
    int index;     // 原始数组索引
    T distance; // 缓存的距离值
} ;

int re_rank2(
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


int re_rank(
  float *query, 
  pca_index &index,
  sq_info *p_sq_info,
  float *distances,
  int *ids,
  int size,
  int final_topk,
  int *final_topk_ids
);