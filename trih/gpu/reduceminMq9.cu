#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <iostream>
#include <cassert>
#include <random>
#include <cuda_fp16.h> // Add include for half precision support

namespace cg = cooperative_groups;

constexpr int SEGMENTS_PER_QUERY = 8000;
constexpr int SEGMENT_SIZE = 125;
constexpr int PADDED_SEGMENT_SIZE = 128; // 内存对齐填充
constexpr int QUERIES = 1000;
constexpr int TOTAL_SEGMENTS = QUERIES * SEGMENTS_PER_QUERY;

struct SegmentResult {
    float min_val;
    int min_idx;
};

__device__ __forceinline__ void warp_reduce(SegmentResult& val) {//warp内，每个线程执行相同的指令，最小值放入val中
    cg::coalesced_group active = cg::coalesced_threads();
    for (int i = active.size() / 2; i > 0; i /= 2) {
        SegmentResult other = active.shfl_down(val, i);
        if (other.min_val < val.min_val || 
           (other.min_val == val.min_val && other.min_idx < val.min_idx)) {
            val = other;
        }
    }
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

__device__ __forceinline__ void warp_reduce_index_only_half(int& idx, half& min_val) {
    cg::coalesced_group active = cg::coalesced_threads();
    for (int i = active.size() / 2; i > 0; i /= 2) {
        half other_val = active.shfl_down(min_val, i);
        int other_idx = active.shfl_down(idx, i);
        
        // Use native half comparison operators (__hlt and __heq)
        if (__hlt(other_val, min_val) || 
           (__heq(other_val, min_val) && other_idx < idx)) {
            min_val = other_val;
            idx = other_idx;
        }
    }
}

__global__ void segmented_argmin_kernel(
    const float* __restrict__ distances,
    const int* __restrict__ indices,
    SegmentResult* __restrict__ results)
{
    const int segment_id = blockIdx.x * blockDim.y + threadIdx.y; // blockIdx.x 个线程/段，blockDim.y段/Block
    if (segment_id >= TOTAL_SEGMENTS) return;

    const int query_id = segment_id / SEGMENTS_PER_QUERY;
    const int local_segment = segment_id % SEGMENTS_PER_QUERY;
    const int base_offset = (query_id * SEGMENTS_PER_QUERY + local_segment) * PADDED_SEGMENT_SIZE;

    constexpr int VEC_SIZE = 4;
    SegmentResult thread_min = {INFINITY, -1};

    #pragma unroll
    for (int i = 0; i < PADDED_SEGMENT_SIZE; i += blockDim.x * VEC_SIZE) {
        const int load_pos = i + threadIdx.x * VEC_SIZE;//每个线程处理 VEC_SIZE=4个，32个线程处理一个段
        if (load_pos < SEGMENT_SIZE) {
            // 向量化加载并处理有效元素
            const float4 dists = reinterpret_cast<const float4*>(distances + base_offset + load_pos)[0];
            const int4 idxs = reinterpret_cast<const int4*>(indices + base_offset + load_pos)[0];

            #pragma unroll
            for (int v = 0; v < VEC_SIZE; ++v) { //每个线程处理 4 个
                const int elem_pos = load_pos + v;
                if (elem_pos < SEGMENT_SIZE) { // 严格限制有效范围
                    const float curr_dist = (&dists.x)[v];
                    const int curr_idx = base_offset + elem_pos; // 计算全局索引
                    if (curr_dist < thread_min.min_val) {
                        thread_min = {curr_dist, curr_idx};
                    }
                }
            }
        }
    }

    // 多级归约
    warp_reduce(thread_min); //得到32个线程，也就是一个段的最小值

    __shared__ SegmentResult shared_mins[32]; //当前block内，32个段，每段1个最小值
    if (threadIdx.x == 0) {//只有线程x==0 进行最小值设置，事实上，warp内32个线程（x从0-31），都拿到了相同的thread_min（warp内，每个线程执行相同的指令）
        // shared_mins[threadIdx.y] = thread_min;//y 指定不同warp
        results[segment_id] = thread_min;
    }
}

// each warp process a segment, each thread process 4 elements
__global__ void segmented_argmin_kernel_half(
    const half* __restrict__ distances,
    int* __restrict__ results,
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
        results[segment_id] = thread_min_idx;
    }
}

void launch_segmented_argmin(
    const float* distances,
    const int* indices,
    SegmentResult* results,
    cudaStream_t stream = 0)
{
    dim3 block(32, 32);  // 1024 threads per block
    dim3 grid((TOTAL_SEGMENTS + block.y - 1) / block.y);
    
    segmented_argmin_kernel<<<grid, block, 0, stream>>>(distances, indices, results);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

__global__ void segmented_argmin_kernel_half_optimized(
    const half* __restrict__ distances,
    int* __restrict__ results,
    int segment_size,
    int segment_num,
    int seg_num_per_query
)
{
    const int segment_id = blockIdx.x * blockDim.y + threadIdx.y;
    if (segment_id >= segment_num) return;

    const int local_segment = segment_id % seg_num_per_query;
    const int base_offset = segment_id * segment_size;
    const int base_offset_in_query = local_segment * segment_size;

    const int VEC_SIZE = (segment_size + 31) / 32;
    half thread_min_val = __float2half(INFINITY);  // Store min as half directly
    int thread_min_idx = -1;

    #pragma unroll
    for (int i = 0; i < segment_size; i += blockDim.x * VEC_SIZE) {
        const int load_pos = i + threadIdx.x * VEC_SIZE;
        if (load_pos < segment_size) {
            for (int v = 0; v < VEC_SIZE; ++v) {
                const int elem_pos = load_pos + v;
                if (elem_pos < segment_size) {
                    // Direct comparison on half values without conversion
                    const half curr_dist = distances[base_offset + elem_pos];
                    const int curr_idx = base_offset_in_query + elem_pos;
                    
                    // Use native half comparison (__hlt requires compute capability ≥ 8.0)
                    if (__hlt(curr_dist, thread_min_val)) {
                        thread_min_val = curr_dist;
                        thread_min_idx = curr_idx;
                    }
                }
            }
        }
    }

    warp_reduce_index_only_half(thread_min_idx, thread_min_val);

    if (threadIdx.x == 0) {
        results[segment_id] = thread_min_idx;
    }
}

// Launch function for the optimized kernel
void launch_segmented_argmin_half_optimized(
    const half* distances,
    int* results,
    int segment_size,
    int segment_num,
    int seg_num_per_query,
    cudaStream_t stream = 0)
{
    dim3 block(32, 32);  // 1024 threads per block
    dim3 grid((segment_num + block.y - 1) / block.y);
    
    segmented_argmin_kernel_half_optimized<<<grid, block, 0, stream>>>(
        distances, results, segment_size, segment_num, seg_num_per_query);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Optimized half-precision kernel launch failed: " 
                  << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void launch_segmented_argmin_half(
    const half* distances,
    int* results,
    cudaStream_t stream = 0)
{
    dim3 block(32, 32);  // 1024 threads per block
    dim3 grid((TOTAL_SEGMENTS + block.y - 1) / block.y);
    
    segmented_argmin_kernel_half<<<grid, block, 0, stream>>>(distances, results, 125, 8000000, 8000);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Half-precision kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    assert(prop.major >= 8);

    // 分配带填充的内存
    const size_t padded_size = QUERIES * SEGMENTS_PER_QUERY * PADDED_SEGMENT_SIZE;
    float* h_dist = new float[padded_size]();
    int* h_idx = new int[padded_size]();

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0, 100.0);

    // 初始化数据（有效部分设置真实值，填充部分设为极大值）
    for (int q = 0; q < QUERIES; ++q) {
        for (int s = 0; s < SEGMENTS_PER_QUERY; ++s) {
            const int seg_base = (q * SEGMENTS_PER_QUERY + s) * PADDED_SEGMENT_SIZE;
            for (int i = 0; i < SEGMENT_SIZE; ++i) {
                h_dist[seg_base + i] = i; //yshen: dist(gen);
                h_idx[seg_base + i] = seg_base + i; // 存储物理位置
            }
            // 填充部分设为无效值
            for (int i = SEGMENT_SIZE; i < PADDED_SEGMENT_SIZE; ++i) {
                h_dist[seg_base + i] = 9999;//yshen:INFINITY;
                h_idx[seg_base + i] = -1;
            }
        }
    }

    // 设备内存分配
    float *d_dist;
    int *d_idx;
    SegmentResult *d_results;
    cudaMalloc(&d_dist, padded_size * sizeof(float));
    cudaMalloc(&d_idx, padded_size * sizeof(int));
    cudaMalloc(&d_results, TOTAL_SEGMENTS * sizeof(SegmentResult));

    // 拷贝数据
    cudaMemcpy(d_dist, h_dist, padded_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx, h_idx, padded_size * sizeof(int), cudaMemcpyHostToDevice);

    // 执行并计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    launch_segmented_argmin(d_dist, d_idx, d_results);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Execution time: " << ms << " ms" << std::endl;

    // 结果验证
    SegmentResult* h_results = new SegmentResult[TOTAL_SEGMENTS];
    cudaMemcpy(h_results, d_results, TOTAL_SEGMENTS * sizeof(SegmentResult), cudaMemcpyDeviceToHost);

    bool valid = true;
    for (int s = 0; s < TOTAL_SEGMENTS; ++s) {
        const int seg_base = s * PADDED_SEGMENT_SIZE;
        float expected_min = INFINITY;
        int expected_idx = -1;
        
        for (int i = 0; i < SEGMENT_SIZE; ++i) {
            if (h_dist[seg_base + i] < expected_min) {
                expected_min = h_dist[seg_base + i];
                expected_idx = seg_base + i;
            }
        }

        if (h_results[s].min_val != expected_min || h_results[s].min_idx != expected_idx) {
            std::cerr << "Segment " << s << " mismatch! Expected (" 
                      << expected_min << ", " << expected_idx << ") Got ("
                      << h_results[s].min_val << ", " << h_results[s].min_idx << ")\n";
            valid = false;
            break;
        }
    }

    if (valid) {
        std::cout << "All results are correct!" << std::endl;
    }

    // Test half-precision kernel
    std::cout << "Testing half-precision kernel..." << std::endl;

    // Convert float data to half
    half* h_dist_half = new half[padded_size];
    for (size_t i = 0; i < padded_size; ++i) {
        h_dist_half[i] = __float2half(h_dist[i]);
    }

    // Allocate device memory for half-precision data
    half* d_dist_half;
    int* d_results_half;
    cudaMalloc(&d_dist_half, padded_size * sizeof(half));
    cudaMalloc(&d_results_half, TOTAL_SEGMENTS * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_dist_half, h_dist_half, padded_size * sizeof(half), cudaMemcpyHostToDevice);

    // Execute and time half-precision kernel
    cudaEvent_t start_half, stop_half;
    cudaEventCreate(&start_half);
    cudaEventCreate(&stop_half);

    cudaEventRecord(start_half);
    launch_segmented_argmin_half(d_dist_half, d_results_half);
    // launch_segmented_argmin_half_optimized(d_dist_half, d_results_half, 125, 8000000, 8000);
    cudaEventRecord(stop_half);
    cudaEventSynchronize(stop_half);

    float ms_half = 0;
    cudaEventElapsedTime(&ms_half, start_half, stop_half);
    std::cout << "Half-precision execution time: " << ms_half << " ms" << std::endl;

    // Validate results
    int* h_results_half = new int[TOTAL_SEGMENTS];
    cudaMemcpy(h_results_half, d_results_half, TOTAL_SEGMENTS * sizeof(int), cudaMemcpyDeviceToHost);

    bool valid_half = true;
    for (int s = 0; s < TOTAL_SEGMENTS; ++s) {
        const int seg_base = s * 125;
        const int seg_base_per_query = (s % 8000) * 125;
        float expected_min = INFINITY;
        int expected_idx = -1;
        
        for (int i = 0; i < SEGMENT_SIZE; ++i) {
            if (h_dist[seg_base + i] < expected_min) {
                expected_min = h_dist[seg_base + i];
                expected_idx = seg_base_per_query + i;
            }
        }

        if (h_results_half[s] != expected_idx) {
            std::cerr << "Half-precision segment " << s << " mismatch! Expected index: " 
                      << expected_idx << " Got: " << h_results_half[s] << std::endl;
            valid_half = false;
            break;
        }
    }

    if (valid_half) {
        std::cout << "All half-precision results are correct!" << std::endl;
    }

    // Clean up half-precision resources
    delete[] h_dist_half;
    delete[] h_results_half;
    cudaFree(d_dist_half);
    cudaFree(d_results_half);
    cudaEventDestroy(start_half);
    cudaEventDestroy(stop_half);

    // 清理资源
    delete[] h_dist;
    delete[] h_idx;
    delete[] h_results;
    cudaFree(d_dist);
    cudaFree(d_idx);
    cudaFree(d_results);
    return 0;
}