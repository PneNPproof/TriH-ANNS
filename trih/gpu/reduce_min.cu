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


// __global__ void computeGroupMinimalKernel(float *dist, int n, int m, int g, int lda,
//                        std::vector<float *> minArray, std::vector<int *> indexArray)
// {
//   // Compute column and group indices
//   int num_groups_per_column = (n + g - 1) / g;
//   int col = blockIdx.x / num_groups_per_column;
//   int k = blockIdx.x % num_groups_per_column;
//   if (col >= m)
//     return;

//   // Compute group boundaries
//   int start = k * g;
//   int end = (start + g < n) ? start + g : n;
//   int num_elements = end - start;

//   int tid = threadIdx.x;

//   // Declare shared memory
//   extern __shared__ float shared[];
//   float *shared_val = shared;
//   int *shared_idx = (int *)(shared + blockDim.x);

//   // Load data into shared memory
//   if (tid < num_elements)
//   {
//     int global_idx = start + tid;
//     shared_val[tid] = dist[global_idx + col * lda];
//     shared_idx[tid] = global_idx;
//   }
//   else
//   {
//     shared_val[tid] = FLT_MAX;
//     shared_idx[tid] = -1;
//   }
//   __syncthreads();

//   // Perform reduction
//   for (int s = blockDim.x / 2; s > 0; s >>= 1)
//   {
//     if (tid < s)
//     {
//       if (shared_val[tid + s] < shared_val[tid] ||
//           (shared_val[tid + s] == shared_val[tid] && shared_idx[tid + s] < shared_idx[tid]))
//       {
//         shared_val[tid] = shared_val[tid + s];
//         shared_idx[tid] = shared_idx[tid + s];
//       }
//     }
//     __syncthreads();
//   }

//   // Write results
//   if (tid == 0)
//   {
//     minArray[col][k] = shared_val[0];
//     indexArray[col][k] = shared_idx[0];
//   }
// }


// template <typename T>
// using VectorPtr = std::vector<T*>;

// __global__ void computeGroupMinimaKernel(
//     const float* __restrict__ input,
//     int n,
//     int m,
//     int group_size,
//     float** output_min,
//     int** output_idx)
// {
//     int num_groups_per_col = (n + group_size - 1) / group_size;
//     int block_idx = blockIdx.x;
//     int col = block_idx / num_groups_per_col;
//     int group = block_idx % num_groups_per_col;

//     if (col >= m) return;

//     int start = group * group_size;
//     int end = start + group_size;
//     if (end > n) end = n;
//     int group_length = end - start;

//     int tid = threadIdx.x;

//     float local_min = FLT_MAX;
//     int local_idx = -1;
//     if (tid < group_length) {
//         int idx = start + tid;
//         local_min = input[col * n + idx];
//         local_idx = idx;
//     }

//     __shared__ float s_min[256];
//     __shared__ int s_idx[256];

//     s_min[tid] = local_min;
//     s_idx[tid] = local_idx;

//     __syncthreads();

//     for (int s = blockDim.x / 2; s > 0; s >>= 1) {
//         if (tid < s) {
//             if (s_min[tid] > s_min[tid + s]) {
//                 s_min[tid] = s_min[tid + s];
//                 s_idx[tid] = s_idx[tid + s];
//             }
//         }
//         __syncthreads();
//     }

//     if (tid == 0) {
//         output_min[col][group] = s_min[0];
//         output_idx[col][group] = s_idx[0];
//     }
// }

// void launchComputeGroupMinimaKernel(
//     const float* d_input,
//     int n,
//     int m,
//     int group_size,
//     float** d_output_min,
//     int** d_output_idx)
// {
//     int num_groups_per_col = (n + group_size - 1) / group_size;
//     int total_blocks = m * num_groups_per_col;
//     int blockSize = 256; // Block size >= group_size

//     computeGroupMinimaKernel<<<total_blocks, blockSize>>>(
//         d_input, n, m, group_size, d_output_min, d_output_idx);
// }


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

// int main() {
//     const int m = 2; // Number of columns
//     const int n = 5; // Number of rows
//     const int group_size = 3;

//     // Host input data (column-major)
//     float h_input[] = {
//         4.0f, 2.0f, 5.0f, 1.0f, 3.0f, // Column 0
//         9.0f, 7.0f, 8.0f, 6.0f, 10.0f  // Column 1
//     };

//     // Allocate device input
//     float* d_input;
//     cudaMalloc(&d_input, m * n * sizeof(float));
//     cudaMemcpy(d_input, h_input, m * n * sizeof(float), cudaMemcpyHostToDevice);

//     // Allocate output arrays for each column
//     int num_groups_per_col = (n + group_size - 1) / group_size;
//     std::vector<float*> h_output_min(m);
//     std::vector<int*> h_output_idx(m);

//     for (int i = 0; i < m; ++i) {
//         cudaMalloc(&h_output_min[i], num_groups_per_col * sizeof(float));
//         cudaMalloc(&h_output_idx[i], num_groups_per_col * sizeof(int));
//     }

//     // Allocate device arrays of pointers
//     float** d_output_min;
//     cudaMalloc(&d_output_min, m * sizeof(float*));
//     cudaMemcpy(d_output_min, h_output_min.data(), m * sizeof(float*), cudaMemcpyHostToDevice);

//     int** d_output_idx;
//     cudaMalloc(&d_output_idx, m * sizeof(int*));
//     cudaMemcpy(d_output_idx, h_output_idx.data(), m * sizeof(int*), cudaMemcpyHostToDevice);

//     // Launch kernel
//     launchComputeGroupMinimaKernel(d_input, n, m, group_size, d_output_min, d_output_idx);

//     // Copy results back to host
//     for (int col = 0; col < m; ++col) {
//         std::vector<float> h_min(num_groups_per_col);
//         cudaMemcpy(h_min.data(), h_output_min[col], num_groups_per_col * sizeof(float), cudaMemcpyDeviceToHost);

//         std::vector<int> h_idx(num_groups_per_col);
//         cudaMemcpy(h_idx.data(), h_output_idx[col], num_groups_per_col * sizeof(int), cudaMemcpyDeviceToHost);

//         std::cout << "Column " << col << ":\n";
//         std::cout << "Minima: ";
//         for (float val : h_min) std::cout << val << " ";
//         std::cout << "\nIndices: ";
//         for (int idx : h_idx) std::cout << idx << " ";
//         std::cout << std::endl;
//     }

//     // Free device memory
//     cudaFree(d_input);
//     for (int i = 0; i < m; ++i) {
//         cudaFree(h_output_min[i]);
//         cudaFree(h_output_idx[i]);
//     }
//     cudaFree(d_output_min);
//     cudaFree(d_output_idx);

//     return 0;
// }