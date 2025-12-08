#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cassert>
#include <cstdio>
#include <chrono>

struct GpuTimer {
  cudaEvent_t start, stop;
  GpuTimer() { cudaEventCreate(&start); cudaEventCreate(&stop); }
  ~GpuTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
  void Start() { cudaEventRecord(start); }
  void Stop() { cudaEventRecord(stop); }
  float ElapsedMillis() {
    cudaEventSynchronize(stop);
    float elapsed; cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
  }
};

void PrintDevices() {
  int num_devices; cudaGetDeviceCount(&num_devices);
  for (int i = 0; i < num_devices; i++) {
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, i);
    printf("Device %d: %s\n", i, prop.name);
  }
}

void RunTest(int m, int k, int n, int transa, int transb) {
  PrintDevices();

  const int A_rows = transa ? m : k;
  const int A_cols = transa ? k : m;
  const int B_rows = transb ? k : n;
  const int B_cols = transb ? n : k;
  const int C_size = n * m;

  cublasHandle_t handle; 
  cublasCreate(&handle);
  cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH); // 启用Tensor Core

  constexpr int NUM_DATASETS = 61;
  __half *host_A[NUM_DATASETS], *host_B[NUM_DATASETS];
  __half *device_A[NUM_DATASETS], *device_B[NUM_DATASETS], *device_C[NUM_DATASETS];

  for (int i = 0; i < NUM_DATASETS; ++i) {
    // 分配主机内存
    host_A[i] = new __half[A_rows * A_cols];
    host_B[i] = new __half[B_rows * B_cols];
    
    // 使用FP16初始化数据
    for (int j = 0; j < A_rows * A_cols; j++)
      host_A[i][j] = __float2half((j + i * 100) % (100 + i * 20));
    for (int j = 0; j < B_rows * B_cols; j++)
      host_B[i][j] = __float2half((j + i * 50) % (150 + i * 25));
    
    // 分配设备内存
    cudaMalloc(&device_A[i], A_rows * A_cols * sizeof(__half));
    cudaMalloc(&device_B[i], B_rows * B_cols * sizeof(__half));
    cudaMalloc(&device_C[i], C_size * sizeof(__half));
    
    // 拷贝数据到设备
    cudaMemcpy(device_A[i], host_A[i], A_rows * A_cols * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(device_B[i], host_B[i], B_rows * B_cols * sizeof(__half), cudaMemcpyHostToDevice);
  }

  __half alpha = __float2half(-2.0f);
  __half beta = __float2half(1.0f);
  cublasGemmAlgo_t algos[] = {CUBLAS_GEMM_DFALT};

  // Warm up with the first dataset
  printf("Performing warm-up run...\n");
  cublasGemmEx(
    handle,
    transb ? CUBLAS_OP_T : CUBLAS_OP_N,
    transa ? CUBLAS_OP_T : CUBLAS_OP_N,
    n, m, k,
    &alpha,
    device_B[0], CUDA_R_16F, transb ? k : n,
    device_A[0], CUDA_R_16F, transa ? m : k,
    &beta,
    device_C[0], CUDA_R_16F, n,
    CUDA_R_16F,
    algos[0]
  );
  cudaDeviceSynchronize();

  printf("m:%5d, k:%5d, n:%5d, transa:%d, transb:%d\n", m, k, n, transa, transb);
  for (int algo = 0; algo < sizeof(algos)/sizeof(algos[0]); ++algo) {
    printf("\nTesting algorithm %d (%s):\n", algo, 
           algo == 0 ? "DEFAULT" : algo == 9 ? "ALGO17" : "ALGO0-7");
    
    float total_time = 0.0f;
    bool success = true;
    
    // Create a stream for each dataset
    cudaStream_t streams[NUM_DATASETS];
    for (int round = 1; round < NUM_DATASETS; ++round) {
      cudaStreamCreate(&streams[round]);
    }
        
    // Using std::chrono for CPU-side timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Launch all operations in their own streams
    for (int round = 1; round < NUM_DATASETS; ++round) {
      // Set stream for cublas handle
      cublasSetStream(handle, streams[round]);
      
      cublasStatus_t status = cublasGemmEx(
      handle,
      transb ? CUBLAS_OP_T : CUBLAS_OP_N,
      transa ? CUBLAS_OP_T : CUBLAS_OP_N,
      n, m, k,
      &alpha,
      device_B[round], CUDA_R_16F, transb ? k : n,
      device_A[round], CUDA_R_16F, transa ? m : k,
      &beta,
      device_C[round], CUDA_R_16F, n,
      CUDA_R_16F,
      algos[algo]
      );
      
      if (status != CUBLAS_STATUS_SUCCESS) {
      printf("  Round %d failed with error %d\n", round, status);
      success = false;
      break;
      }
    }
    
    // Synchronize all streams to ensure completion
    for (int round = 1; round < NUM_DATASETS && success; ++round) {
      cudaStreamSynchronize(streams[round]);
    }
    
    // Record end time
    auto end_time = std::chrono::high_resolution_clock::now();
    total_time = std::chrono::duration<float, std::milli>(end_time - start_time).count();
    printf("  Total execution time: %6.3f ms\n", total_time);
    
    // Clean up streams
    for (int round = 1; round < NUM_DATASETS; ++round) {
      cudaStreamDestroy(streams[round]);
    }
    
    if (success) {
      float avg_time = total_time / NUM_DATASETS;
      float gflops = 2.0f * m * n * k / (avg_time * 1e6);
      printf("  Average: %6.3f ms | %.1f GFLOPS\n", avg_time, gflops);
    }
  }

  for (int i = 0; i < NUM_DATASETS; ++i) {
    cudaFree(device_A[i]);
    cudaFree(device_B[i]);
    cudaFree(device_C[i]);
    delete[] host_A[i];
    delete[] host_B[i];
  }
  
  cublasDestroy(handle);
}

int main(int argc, char *argv[]) {
  int m = 20, k = 20000, n = 200;
  int ta = 0, tb = 1;
  
  if (argc >= 6) {
    m = atoi(argv[1]);
    k = atoi(argv[2]);
    n = atoi(argv[3]);
    ta = atoi(argv[4]);
    tb = atoi(argv[5]);
  }
  
  RunTest(m, k, n, ta, tb);
  return 0;
}