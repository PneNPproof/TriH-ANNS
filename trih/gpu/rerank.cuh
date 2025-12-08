#pragma once

void compute_exact_distances(
  float** pdis_per_query,
  int** pidx_per_query,
  const float* src_data,
  const float* queries,
  int query_num,
  int record_num,
  int dim,
  int phase1_topk,
  int block_size);