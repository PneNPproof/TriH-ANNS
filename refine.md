## bug(repaired)

## about arguments of cublasGemmEx
The alpha and beta seems can't be a device pointer

### memory order of full_dim_pca_data_h
full_dim_pca_data_h is a dim*dim matrix, each column is a eigen vector, I assume it is column-majored, but it is row-majored which caused a bug.

### error use of cublasAxpyEx
The specific usage is still being researched, but replacing it with my own implemented kernel fixed the bug.

### error use of cub::DeviceSegmentedRadixSort::SortPairs
Two-stage invocation: the first stage invocation is not only for calculating temp_storage_bytes, so a sufficiently large temp_storage can be pre-allocated, but a two-stage invocation is still required during the actual invocation.

### about arguments of cub::DeviceSegmentedRadixSort::SortPairs

```
sizeof(half) * 8 get low recall
sizeof(float) * 8 get high recall
```
### about re-rank task submission

```cpp
for (int i=0; i<batch_query_num; i++)
    {
        rr_pool.detach_task(
          [this, batch_query, i, phase2_ids_h]{
            re_rank2(
              this->base_dataset_h, 
              this->base_dataset_norms_h,
              batch_query + i * this->dim,
              this->phase1_ids_h + i * this->phase1_topk,
              this->phase1_topk,
              this->data_num,
              this->dim,
              this->phase2_topk,
              phase2_ids_h + i * this->phase2_topk
            );
          }
        );
    }
```
To ensure that a worker can be immediately available for the next batch of queries after submitting a rerank task, it is necessary to guarantee that the previous rerank task no longer depends on a certain data structure of the worker. Here, an obvious dependency is `phase1_ids_h`, as the next batch of queries might modify the content of this structure, causing issues in the rerank process of the previous batch.

#### quant_queries_h and remain_batch_query_h
the same as last section