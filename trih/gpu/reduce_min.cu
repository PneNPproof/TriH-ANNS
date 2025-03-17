// write a cuda kernel
// input1: a float pointer with a column-major matrix(n*m) stored in it
// each column is a distance vector
// input2: a group size, for a distance vector, we will find the minimum value in each group, if the rest elements are not enough to form a group, still find the minimum value in the rest elements
// output: a vector of DoubleBuffer<float> with the minimum value in each group
// output: a vector of DoubleBuffer<int> with the index of the minimum value in each group(the index is the index in the distance vector)
// the DoubleBuffer is passed by reference, so you need to modify the value in the DoubleBuffer directly

#include <cfloat>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <iostream>
#include <cassert>
#include <random>
#include <cuda_fp16.h> // Add include for half precision support
#include <cub/cub.cuh>

namespace cg = cooperative_groups;

__global__ void computeGroupMinimaKernel(
    const float* __restrict__ input,
    int n,
    int m,
    int group_size,
    float** output_min,
    int** output_idx)
{
    extern __shared__ char shared_mem[];

    int num_groups_per_col = (n + group_size - 1) / group_size;
    int subgroups_per_block = blockDim.x / group_size;
    int blocks_per_col = (num_groups_per_col + subgroups_per_block - 1) / subgroups_per_block;

    int col = blockIdx.x / blocks_per_col;
    int block_in_col = blockIdx.x % blocks_per_col;
    int group_start = block_in_col * subgroups_per_block;
    int subgroup_id = threadIdx.x / group_size;
    int group = group_start + subgroup_id;

    if (col >= m || group >= num_groups_per_col) return;

    int start = group * group_size;
    int end = start + group_size;
    if (end > n) end = n;
    int group_length = end - start;

    int tid_in_subgroup = threadIdx.x % group_size;

    float local_min = FLT_MAX;
    int local_idx = -1;
    if (tid_in_subgroup < group_length) {
        int idx = start + tid_in_subgroup;
        local_min = input[col * n + idx];
        local_idx = idx;
    }

    // Allocate shared memory for each subgroup
    size_t shared_size_per_group = group_size * (sizeof(float) + sizeof(int));
    float* s_min = (float*)(shared_mem + subgroup_id * shared_size_per_group);
    int* s_idx = (int*)(shared_mem + subgroup_id * shared_size_per_group + group_size * sizeof(float));

    s_min[tid_in_subgroup] = local_min;
    s_idx[tid_in_subgroup] = local_idx;

    __syncthreads();

    // Parallel reduction within subgroup
    for (int s = group_size / 2; s > 0; s >>= 1) {
        if (tid_in_subgroup < s) {
            if (s_min[tid_in_subgroup] > s_min[tid_in_subgroup + s]) {
                s_min[tid_in_subgroup] = s_min[tid_in_subgroup + s];
                s_idx[tid_in_subgroup] = s_idx[tid_in_subgroup + s];
            }
        }
        __syncthreads();
    }

    // Write result for the group
    if (tid_in_subgroup == 0) {
        output_min[col][group] = s_min[0];
        output_idx[col][group] = s_idx[0];
    }
}

void launchComputeGroupMinimaKernel(
    const float* d_input,
    int n,
    int m,
    int group_size,
    float **d_output_min,
    int **d_output_idx,
    int block_size)
{
    int num_groups_per_col = (n + group_size - 1) / group_size;
    int subgroups_per_block = block_size / group_size;
    int blocks_per_col = (num_groups_per_col + subgroups_per_block - 1) / subgroups_per_block;
    int total_blocks = m * blocks_per_col;
    size_t shared_mem_size = block_size * (sizeof(float) + sizeof(int));

    computeGroupMinimaKernel<<<total_blocks, block_size, shared_mem_size>>>(
        d_input, n, m, group_size,
        d_output_min, d_output_idx
    );
}

__device__ __forceinline__ void warp_reduce_index_only(int& idx, float& min_val) {
    cg::coalesced_group active = cg::coalesced_threads();
    for (int i = active.size() / 2; i > 0; i /= 2) {
        float other_val = active.shfl_down(min_val, i);
        int other_idx = active.shfl_down(idx, i);
        if (other_val < min_val || 
           (other_val == min_val && other_idx < idx)) {
            min_val = other_val;
            idx = other_idx;
        }
    }
}


// each warp process a segment, each thread process segment_size/32 elements
__global__ void segmented_argmin_kernel_half(
    const half* __restrict__ distances,
    half* __restrict__ reduced_dists_per_query,
    int* __restrict__ reduced_ids_per_query,
    int segment_size,
    int segment_num,
    int seg_num_per_query
)
{
    const int segment_id = blockIdx.x * blockDim.y + threadIdx.y;
    if (segment_id >= segment_num) return;

    // const int query_id = segment_id / SEGMENTS_PER_QUERY;
    const int local_segment = segment_id % seg_num_per_query;
    const int base_offset = segment_id * segment_size;
    const int base_offset_in_query = local_segment * segment_size;

    int VEC_SIZE = (segment_size + 31) / 32;
    float thread_min_val = INFINITY;
    int thread_min_idx = -1;

    #pragma unroll
    for (int i = 0; i < segment_size; i += blockDim.x * VEC_SIZE) {
        const int load_pos = i + threadIdx.x * VEC_SIZE;
        if (load_pos < segment_size) {
            // Process 4 elements at a time
            for (int v = 0; v < VEC_SIZE; ++v) {
                const int elem_pos = load_pos + v;
                if (elem_pos < segment_size) {
                    // Convert half to float for comparison
                    const float curr_dist = __half2float(distances[base_offset + elem_pos]);
                    const int curr_idx = base_offset_in_query + elem_pos;
                    if (curr_dist < thread_min_val) {
                        thread_min_val = curr_dist;
                        thread_min_idx = curr_idx;
                    }
                }
            }
        }
    }

    // Warp reduction - only care about index, but need to track value for comparison
    warp_reduce_index_only(thread_min_idx, thread_min_val);

    if (threadIdx.x == 0) {
        reduced_ids_per_query[segment_id] = thread_min_idx;
        reduced_dists_per_query[segment_id] = __float2half(thread_min_val);
    }
}


void half_matrix_reduce(
    const half* dists_per_query,
    half* reduced_dists_per_query,
    int* reduced_ids_per_query,
    int segment_size,
    int segment_num,
    int seg_num_per_query,
    cudaStream_t stream
)
{
    dim3 block(32, 32);  // 1024 threads per block
    dim3 grid((segment_num + block.y - 1) / block.y);
    
    segmented_argmin_kernel_half<<<grid, block, 0, stream>>>(dists_per_query, reduced_dists_per_query, reduced_ids_per_query, segment_size, segment_num, seg_num_per_query);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Half-precision kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

cudaError_t segmented_sort_topk_pairs_fp16(
    half* reduced_dists,    // [query_num * group_num] in/out - half precision
    int*  reduced_inds,     // [query_num * group_num] in/out
    int   query_num,        // number of segments
    int   group_num,        // length of each segment
    int   phase1_topk)      // how many elements to extract from each segment
{
    // Each query segment is of size group_num
    // so total elements = query_num * group_num
    const int total = query_num * group_num;

    // Prepare segment offsets
    std::vector<int> offsets(query_num + 1);
    for (int i = 0; i <= query_num; ++i) {
        offsets[i] = i * group_num;
    }

    int* d_offsets = nullptr;
    cudaMalloc(&d_offsets, (query_num + 1) * sizeof(int));
    cudaMemcpy(d_offsets, offsets.data(), (query_num + 1) * sizeof(int), cudaMemcpyHostToDevice);

    // Temporary storage requirements
    size_t temp_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairs(
        nullptr, temp_bytes,
        /* keysIn  */ reduced_dists, /* keysOut  */ reduced_dists,
        /* valsIn  */ reduced_inds,  /* valsOut  */ reduced_inds,
        /* numItems */ total,
        /* numSegments */ query_num,
        /* beginOffsets */ d_offsets,
        /* endOffsets   */ d_offsets + 1,
        /* beginBit, endBit = sort floats fully */ 0, sizeof(float)*8
    );

    void* d_temp = nullptr;
    cudaMalloc(&d_temp, temp_bytes);

    // In-place segmented sort by distance (ascending)
    // Create timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record start event
    cudaEventRecord(start);
    
    // Perform the sort
    cub::DeviceSegmentedRadixSort::SortPairs(
        d_temp, temp_bytes,
        reduced_dists, reduced_dists,
        reduced_inds,  reduced_inds,
        total,
        query_num,
        d_offsets,
        d_offsets + 1,
        0, sizeof(float)*8
    );
    
    // Record stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Segmented sort time: %.3f ms\n", milliseconds);
    
    // Clean up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaError_t err = cudaGetLastError();

    return err;
}
