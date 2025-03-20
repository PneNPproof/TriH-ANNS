#pragma once
#include "pca.h"
#include <cuda_fp16.h>
#include <cublas_v2.h>

void gpu_anns
(
  float *query,
  int query_num,
  float *src_data,
  pca_index &index,
  float *distances,
  int *neighbors,
  int reduce_group_size,
  int phase1_topk,
  int phase2_topk,
  int *ground_truth_neighbors
);

class TrihAnnsWorker
{
public:

/// for query project
  float* full_dim_pca_data_d;
  float* full_dim_pca_data_h;
  float* pca_dim_pca_data_d;
  float* batch_query_d;
  float* pca_batch_query_d;
  float* alpha0_d;
  float* beta0_d;
  // half* half_batch_query_d;

/// for query project

/// fro l2mm
  half* half_pca_dataset_d;
  half* half_pca_queries_d;
  half* half_pca_dataset_norms_d;
  half* half_dists_d;
  half* alpha_d;
  half* beta_d;
  half* alpha1_d;
  cublasHandle_t handle;
/// fro l2mm

/// for reduce_min
  half* reduced_dists_per_query_d;
  int* reduced_ids_per_query_d;
/// for reduce_min

/// for segment sort
  half *phase1_distances_d;
  int *phase1_ids_d;
  int *segments_offsets_d;
  void *temp_storage_d;
  size_t temp_storage_bytes;
/// for segment sort

/// for re-rank
  half *phase1_distances_h;
  int *phase1_ids_h;
  float* base_dataset_h;
  float* pca_dataset_h;
  float* base_dataset_norms_h;
  int *phase2_ids_h;
/// for re-rank

  int data_num;
  int max_queries_num;
  int dim;
  int pca_dim;
  int reduce_group_size;
  int reduce_group_num;
  int phase1_topk;
  int phase2_topk;

  cudaStream_t work_stream;

  TrihAnnsWorker
  (
    float *full_dim_pca_data,
    float *base_dataset,
    float *pca_dataset,
    int data_num,
    int max_queries_num,
    int dim,
    int pca_dim,
    int reduce_group_size,
    int reduce_group_num,
    int phase1_topk,
    int phase2_topk,
    cudaStream_t work_stream_
  );

  int* batch_query_search
  (
    float *batch_query,
    int batch_query_num,
    int *ground_truth_neighbors
  );

};