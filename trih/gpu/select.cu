#include <cub/cub.cuh>
#include <iostream>
#include <vector>
#include <cuda_fp16.h>

__global__ void extract_topk_kernel(
    const float* sorted_input,
    float* output_topk,
    int segment_size,
    int top_k,
    int num_segments)
{
    const int segment_id = blockIdx.x;
    const int tid = threadIdx.x;
    const int elements_per_thread = (top_k + blockDim.x - 1) / blockDim.x;
    
    const float* segment_start = sorted_input + segment_id * segment_size;
    float* output_start = output_topk + segment_id * top_k;

    #pragma unroll
    for(int i = 0; i < elements_per_thread; ++i) {
        int idx = tid + i * blockDim.x;
        if (idx < top_k) {
            output_start[idx] = segment_start[idx];
        }
    }
}

// Kernel to extract the first top_k elements (distances + indices) from each segment
__global__ void extract_topk_pairs_kernel(
    const float* sorted_dists,
    const int*   sorted_inds,
    float*       output_dists,
    int*         output_inds,
    int          segment_size,
    int          top_k,
    int          num_segments)
{
    const int segment_id    = blockIdx.x;
    const int tid           = threadIdx.x;
    const int elems_per_thr = (top_k + blockDim.x - 1) / blockDim.x;

    const float* seg_dist_start = sorted_dists + segment_id * segment_size;
    const int*   seg_ind_start  = sorted_inds  + segment_id * segment_size;
    float*       out_dist_start = output_dists + segment_id * top_k;
    int*         out_ind_start  = output_inds  + segment_id * top_k;

    // Copy the first 'top_k' elements from sorted arrays
    for(int i = 0; i < elems_per_thr; ++i) {
        int idx = tid + i * blockDim.x;
        if (idx < top_k) {
            out_dist_start[idx] = seg_dist_start[idx];
            out_ind_start[idx]  = seg_ind_start[idx];
        }
    }
}

// Kernel to extract the first top_k elements (half-precision distances + indices) from each segment
__global__ void extract_topk_pairs_kernel_fp16(
    const half* sorted_dists,
    const int*  sorted_inds,
    half*       output_dists,
    int*        output_inds,
    int         segment_size,
    int         top_k,
    int         num_segments)
{
    const int segment_id    = blockIdx.x;
    const int tid           = threadIdx.x;
    const int elems_per_thr = (top_k + blockDim.x - 1) / blockDim.x;

    const half* seg_dist_start = sorted_dists + segment_id * segment_size;
    const int*  seg_ind_start  = sorted_inds  + segment_id * segment_size;
    half*       out_dist_start = output_dists + segment_id * top_k;
    int*        out_ind_start  = output_inds  + segment_id * top_k;

    // Copy the first 'top_k' elements from sorted arrays
    for(int i = 0; i < elems_per_thr; ++i) {
        int idx = tid + i * blockDim.x;
        if (idx < top_k) {
            out_dist_start[idx] = seg_dist_start[idx];
            out_ind_start[idx]  = seg_ind_start[idx];
        }
    }
}

cudaError_t segmented_sort_topk(
    float* d_input,
    float* d_output,
    int num_segments,
    int segment_size,
    int top_k)
{
    
    std::vector<int> offsets(num_segments + 1);
    for (int i = 0; i <= num_segments; ++i) {
        offsets[i] = i * segment_size;
    }
    
    int* d_offsets;
    cudaMalloc(&d_offsets, (num_segments + 1)*sizeof(int));
    cudaMemcpy(d_offsets, offsets.data(), 
              (num_segments + 1)*sizeof(int),
              cudaMemcpyHostToDevice);

    // 计算临时存储需求
    size_t temp_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortKeys(
        nullptr, temp_bytes,
        d_input, d_input, // 原地排序
        num_segments * segment_size,
        num_segments,
        d_offsets, d_offsets + 1);
    
    // 分配临时存储
    void* d_temp = nullptr;
    cudaMalloc(&d_temp, temp_bytes);

    // 执行分段排序（升序）
    cub::DeviceSegmentedRadixSort::SortKeys(
        d_temp, temp_bytes,
        d_input, d_input,
        num_segments * segment_size,
        num_segments,
        d_offsets, d_offsets + 1,
        0, sizeof(float)*8); // 所有bit参与排序

    // 提取前300元素
    const int block_size = 256;
    extract_topk_kernel<<<num_segments, block_size>>>(
        d_input, d_output,
        segment_size, top_k,
        num_segments);

    // 清理资源
    cudaFree(d_temp);
    cudaFree(d_offsets);
    return cudaGetLastError();
}

// Sorts (distance, index) pairs in ascending order per segment, then extracts top_k.
cudaError_t segmented_sort_topk_pairs(
    float* reduced_dists,   // [query_num * group_num] in/out
    int*   reduced_inds,    // [query_num * group_num] in/out
    int    query_num,       // number of segments
    int    group_num,       // length of each segment
    int    phase1_topk)     // how many elements to extract from each segment
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

    // Allocate space for top_k results
    // float* d_output_dists = nullptr;
    // int*   d_output_inds  = nullptr;
    // cudaMalloc(&d_output_dists, query_num * phase1_topk * sizeof(float));
    // cudaMalloc(&d_output_inds,  query_num * phase1_topk * sizeof(int));

    // // Extract top_k from each segment
    // const int block_size = 256;
    // extract_topk_pairs_kernel<<<query_num, block_size>>>(
    //     reduced_dists,
    //     reduced_inds,
    //     d_output_dists,
    //     d_output_inds,
    //     group_num,
    //     phase1_topk,
    //     query_num
    // );
    cudaError_t err = cudaGetLastError();

    // Copy the top_k results back to the original arrays (optional)
    // If you prefer to keep them separate, skip this step.
    // cudaMemcpy(reduced_dists, d_output_dists, query_num * phase1_topk * sizeof(float), cudaMemcpyDeviceToDevice);
    // cudaMemcpy(reduced_inds,  d_output_inds,  query_num * phase1_topk * sizeof(int),   cudaMemcpyDeviceToDevice);

    // // Clean up
    // cudaFree(d_temp);
    // cudaFree(d_offsets);
    // cudaFree(d_output_dists);
    // cudaFree(d_output_inds);

    return err;
}

// Sorts (half-precision distance, index) pairs in ascending order per segment
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
    cudaError_t err = cudaGetLastError();

    return err;
}

int main() {
    const int num_segments = 1000;
    const int segment_size = 8000;
    const int top_k = 300;
    const int total = num_segments * segment_size;

    // Test segmented_sort_topk_pairs
    const int query_num = num_segments;
    const int group_num = segment_size;
    const int phase1_topk = top_k;

    // Allocate memory
    float* reduced_dists;
    int* reduced_inds;
    cudaMalloc(&reduced_dists, query_num * group_num * sizeof(float));
    cudaMalloc(&reduced_inds, query_num * group_num * sizeof(int));

    // Setup timing events
    cudaEvent_t start_pairs, stop_pairs;
    cudaEventCreate(&start_pairs);
    cudaEventCreate(&stop_pairs);
    
    cudaEventRecord(start_pairs);
    segmented_sort_topk_pairs(reduced_dists, reduced_inds, query_num, group_num, phase1_topk);
    cudaEventRecord(stop_pairs);
    
    cudaEventSynchronize(stop_pairs);
    float ms_pairs = 0;
    cudaEventElapsedTime(&ms_pairs, start_pairs, stop_pairs);
    
    std::cout << "segmented_sort_topk_pairs execution time: " << ms_pairs << " ms" << std::endl;

    // Clean up
    cudaFree(reduced_dists);
    cudaFree(reduced_inds);

    // Test segmented_sort_topk_pairs_fp16
    half* reduced_dists_fp16;
    int* reduced_inds_fp16;
    cudaMalloc(&reduced_dists_fp16, query_num * group_num * sizeof(half));
    cudaMalloc(&reduced_inds_fp16, query_num * group_num * sizeof(int));

    // Setup timing events
    cudaEvent_t start_pairs_fp16, stop_pairs_fp16;
    cudaEventCreate(&start_pairs_fp16);
    cudaEventCreate(&stop_pairs_fp16);
    
    cudaEventRecord(start_pairs_fp16);
    segmented_sort_topk_pairs_fp16(reduced_dists_fp16, reduced_inds_fp16, query_num, group_num, phase1_topk);
    cudaEventRecord(stop_pairs_fp16);
    
    cudaEventSynchronize(stop_pairs_fp16);
    float ms_pairs_fp16 = 0;
    cudaEventElapsedTime(&ms_pairs_fp16, start_pairs_fp16, stop_pairs_fp16);
    
    std::cout << "segmented_sort_topk_pairs_fp16 execution time: " << ms_pairs_fp16 << " ms" << std::endl;

    // Clean up
    cudaFree(reduced_dists_fp16);
    cudaFree(reduced_inds_fp16);

    // 分配内存
    float *d_input, *d_output;
    cudaMalloc(&d_input, total * sizeof(float));
    cudaMalloc(&d_output, num_segments * top_k * sizeof(float));

    // 执行排序和提取
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    segmented_sort_topk(d_input, d_output, num_segments, segment_size, top_k);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    
    std::cout << "Execution time: " << ms << " ms" << std::endl;

    // 清理
    cudaFree(d_input);
    cudaFree(d_output);

    
    return 0;
}