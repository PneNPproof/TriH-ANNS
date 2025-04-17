#pragma once

#include <mutex>
#include <memory>

#include <cuda_fp16.h>
#include <cublas_v2.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/device_memory.h"

#include "pca.h"
#include "thread_pool_v2.h"
#include "l2mm.cuh"
#include "sq.h"

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

/// for cutlass gemm
  using ElementOutput = cutlass::half_t;
  // using ElementAccumulator = cutlass::half_t;
  using ElementAccumulator = float;
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::ColumnMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 256, 64>,
      cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementAccumulator>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

  // using Gemm = cutlass::gemm::device::Gemm<
  //     cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
  //     cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::ColumnMajor,
  //     ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
  //     cutlass::gemm::GemmShape<64, 64, 32>,
  //     cutlass::gemm::GemmShape<32, 32, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
  //     cutlass::epilogue::thread::LinearCombination<
  //         ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
  //         ElementAccumulator, ElementAccumulator>,
  //     cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 10>;

  ElementOutput alpha;
  ElementOutput beta;
  ElementOutput* alpha_d;
  ElementOutput* beta_d;
  Gemm gemm_op;
  cutlass::device_memory::allocation<uint8_t> gemm_workspace;
///

/// for query project
  float* full_dim_pca_data_d;
  float* full_dim_pca_data_h;
  float* pca_dim_pca_data_d;
  float* remain_dim_pca_data_d;
  half* half_pca_dim_pca_data_d;
  float* batch_query_d;
  half* half_batch_query_d;
  float* pca_batch_query_d;
  float* remain_batch_query_d;
  // float* remain_batch_query_h;
  half* alpha0_d;
  half* beta0_d;
/// for query project

/// fro l2mm
  half* half_pca_dataset_d;
  half* half_pca_queries_d;
  half* half_pca_dataset_norms_d;
  half* half_dists_d;
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
  float* base_dataset_h;
  float* pca_dataset_h;
  float* base_dataset_norms_h;
  ThreadPool *rerank_thread_pool;
  static std::mutex thread_pool_mutex;
  // uint8_t* quant_queries_h;
  std::shared_ptr< std::vector<sq_info>> sq_info_h; 
/// for re-rank

  size_t data_num;
  size_t max_queries_num;
  size_t dim;
  size_t pca_dim;
  size_t reduce_group_size;
  size_t reduce_group_num;
  size_t phase1_topk;
  size_t phase2_topk;

  cudaStream_t work_stream;

  float *falpha_d;
  float *fbeta_d;

  TrihAnnsWorker
  (
    pca_index &index,
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
    cudaStream_t work_stream_,
    int re_rank_thread_pool_size_ = 12
  );

  TrihAnnsWorker
  (
    TrihAnnsWorker &other,
    cudaStream_t work_stream_
  );

  void batch_query_search(
    float *batch_query,
    int batch_query_num,
    half *phase1_distances_h,
    int *phase1_ids_h,
    int *phase2_ids_h,
    // cudaEvent_t syncEvent,
    float *remain_batch_query_h,
    uint8_t *quant_queries_h,
    bool verbose);
};

/// a search task which executes batch_query_search by one TrihAnnsWorker
void search_task(
  TrihAnnsWorker *worker,
  float *batch_query,
  int batch_query_num,
  half *phase1_distances_h,
  int *phase1_ids_h,
  int *phase2_ids_h,
  int query_batch_num,
  // , cudaEvent_t* syncEvent
  float *remain_batch_query_h,
  uint8_t *quant_queries_h
  );
///