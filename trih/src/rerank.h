#pragma once
#include <cstdint>
#include "sq.h"
#include "pca.h"

template<typename T>
struct HeapElement{
    int index;     // 原始数组索引
    T distance; // 缓存的距离值
} ;

void re_rank2(
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
  uint8_t *quant_query,
  float *query_remain,
  int dim,
  int pca_dim,
  sq_info *p_sq_info,
  half *phase1_topk_dists,
  int *phase1_topk_ids,
  int phase1_topk,
  int *phase2_topk_ids,
  int phase2_topk
);