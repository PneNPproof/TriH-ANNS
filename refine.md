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
