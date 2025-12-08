#include <cublasLt.h>

#include <stdexcept>

inline void checkCublasStatus(cublasStatus_t status)
{
  if (status != CUBLAS_STATUS_SUCCESS)
  {
    printf("cuBLAS API failed with status %d\n", status);
    throw std::logic_error("cuBLAS API failed");
  }
}

/// Sample wrapper executing single precision gemm with cublasLtMatmul, nearly a drop-in replacement for cublasSgemm,
/// with addition of the workspace to support split-K algorithms
///
/// pointer mode is always host, to change it configure the appropriate matmul descriptor attribute
/// matmul is not using cublas handle's configuration of math mode, here tensor ops are implicitly allowed; to change
/// this configure appropriate attribute in the preference handle
void LtSgemm(
    int m,
    int n,
    int k,
    const float *A,
    const float *B,
    const float *C,
    float *D,
    void *workspace,
    size_t workspaceSize)
{
  cublasLtHandle_t ltHandle;
  checkCublasStatus(cublasLtCreate(&ltHandle));
  cublasOperation_t transa = CUBLAS_OP_T, transb = CUBLAS_OP_N;
  float alpha = -2.0f, beta = 1.0f;
  int lda = k, ldb = k, ldc = 0, ldd = m;

  cublasLtMatmulDesc_t operationDesc = NULL;
  cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL, Ddesc = NULL;
  cublasLtMatmulPreference_t preference = NULL;

  int returnedResults = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};

  // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
  // set the transforms for A and B
  checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
  // checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSC, &transc, sizeof(transc)));

  // create matrix descriptors, we are good with the details here so no need to set any extra attributes
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, k, m, lda));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, k, n, ldb));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, ldc));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_32F, m, n, ldd));

  

  // create preference handle; here we could use extra attributes to disable tensor ops or to make sure algo selected
  // will work with badly aligned A, B, C; here for simplicity we just assume A,B,C are always well aligned (e.g.
  // directly come from cudaMalloc)
  checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
  checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

  // printf("executing with algoId: %d\n", 4);

  // we just need the best available heuristic to try and run matmul. There is no guarantee this will work, e.g. if A
  // is badly aligned, you can request more (e.g. 32) algos and try to run them one by one until something works
  checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, 1, &heuristicResult, &returnedResults));

  // printf("executing with algoId: %d\n", 5);

  if (returnedResults == 0)
  {
    checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);
  }

  

  checkCublasStatus(cublasLtMatmul(ltHandle,
                                   operationDesc,
                                   &alpha,
                                   A,
                                   Adesc,
                                   B,
                                   Bdesc,
                                   &beta,
                                   C,
                                   Cdesc,
                                   D,
                                   Ddesc,
                                   &heuristicResult.algo,
                                   workspace,
                                   workspaceSize,
                                   0));

  // descriptors are no longer needed as all GPU work was already enqueued
  if (preference)
    checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
  if (Ddesc)
    checkCublasStatus(cublasLtMatrixLayoutDestroy(Ddesc));
  if (Cdesc)
    checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
  if (Bdesc)
    checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
  if (Adesc)
    checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
  if (operationDesc)
    checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
}


int main()
{
  // Using smaller dimensions for verification
  int m = 5, n = 2, k = 4;
  float *A, *B, *C, *D;
  float *h_A, *h_B, *h_C;

  // Allocate device memory
  cudaMalloc(&A, m * k * sizeof(float));
  cudaMalloc(&B, n * k * sizeof(float));
  cudaMalloc(&C, m * 1 * sizeof(float));
  cudaMalloc(&D, m * n * sizeof(float));

  // Allocate host memory and initialize with simple values
  h_A = (float*)malloc(m * k * sizeof(float));
  h_B = (float*)malloc(n * k * sizeof(float));
  h_C = (float*)malloc(m * 1 * sizeof(float));

  // Initialize matrices with simple values
  for(int i = 0; i < m * k; i++) h_A[i] = 3.0f;
  for(int i = 0; i < n * k; i++) h_B[i] = 2.0f;
  for(int i = 0; i < m * 1; i++) h_C[i] = 1.0f;

  // Copy data to device
  cudaMemcpy(A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B, h_B, n * k * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(C, h_C, m * 1 * sizeof(float), cudaMemcpyHostToDevice);

  // Allocate workspace (smaller size for this test)
  float *workspace;
  size_t workspaceSize = (size_t)1024 * 1024 * 1024 * 4;
  cudaMalloc(&workspace, workspaceSize);

  // Execute matrix multiplication
  LtSgemm(m, n, k, A, B, C, D, workspace, workspaceSize);

  // Allocate host memory for result
  float *h_D = (float*)malloc(m * n * sizeof(float));

  // Copy result back to host
  cudaMemcpy(h_D, D, m * n * sizeof(float), cudaMemcpyDeviceToHost);

  // Print the result matrix
  printf("Result matrix D:\n");
  for(int i = 0; i < m; i++) {
    for(int j = 0; j < n; j++) {
      printf("%f ", h_D[i * n + j]);
    }
    printf("\n");
  }

  // Free the temporary host memory
  free(h_D);

  // Free memory
  free(h_A);
  free(h_B);
  free(h_C);
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  cudaFree(D);
  cudaFree(workspace);

  return 0;
}






// #include <stdio.h>
// #include <stdlib.h>
// #include <cuda_runtime.h>
// #include <cublasLt.h>

// #define CHECK_CUDA(call) do { \
//     cudaError_t err = call; \
//     if (err != cudaSuccess) { \
//         fprintf(stderr, "CUDA error at %s:%d: %s\n", \
//                 __FILE__, __LINE__, cudaGetErrorString(err)); \
//         exit(EXIT_FAILURE); \
//     } \
// } while(0)

// #define CHECK_CUBLAS(call) do { \
//     cublasStatus_t status = call; \
//     if (status != CUBLAS_STATUS_SUCCESS) { \
//         fprintf(stderr, "cuBLAS error at %s:%d: %d\n", \
//                 __FILE__, __LINE__, status); \
//         exit(EXIT_FAILURE); \
//     } \
// } while(0)

// int main() {
//     // Define matrix dimensions
//     const int m = 4;  // Rows of A and D
//     const int n = 3;  // Columns of B and D
//     const int k = 2;  // Columns of A, rows of B

//     // Define scalars
//     const float alpha = 1.0f;
//     const float beta = 1.0f;

//     // Host data
//     float h_A[m * k] = {1, 2, 3, 4, 5, 6, 7, 8}; // A: m x k
//     float h_B[k * n] = {1, 2, 3, 4, 5, 6};       // B: k x n
//     float h_C[m] = {1, 2, 3, 4};                 // C: m x 1 (vector)
//     float h_D[m * n];                            // D: m x n (output)

//     // Device pointers
//     float *d_A, *d_B, *d_C, *d_D;

//     // Allocate device memory
//     CHECK_CUDA(cudaMalloc((void**)&d_A, m * k * sizeof(float)));
//     CHECK_CUDA(cudaMalloc((void**)&d_B, k * n * sizeof(float)));
//     CHECK_CUDA(cudaMalloc((void**)&d_C, m * sizeof(float)));
//     CHECK_CUDA(cudaMalloc((void**)&d_D, m * n * sizeof(float)));

//     // Copy data to device
//     CHECK_CUDA(cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice));
//     CHECK_CUDA(cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice));
//     CHECK_CUDA(cudaMemcpy(d_C, h_C, m * sizeof(float), cudaMemcpyHostToDevice));

//     // Initialize cuBLASLt handle
//     cublasLtHandle_t handle;
//     CHECK_CUBLAS(cublasLtCreate(&handle));

//     // Create matrix layout descriptors
//     cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc, Ddesc;
//     CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, m, k, m)); // A: m x k, leading dim = m
//     CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, k, n, k)); // B: k x n, leading dim = k
//     CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, 0)); // C: m x 1, packed
//     CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_32F, m, n, m)); // D: m x n, leading dim = m

//     // Create matmul descriptor
//     cublasLtMatmulDesc_t matmulDesc;
//     CHECK_CUBLAS(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

//     // // Set epilogue to include bias (broadcast C across columns)
//     // cublasLtEpilogue_t bias = CUBLASLT_EPILOGUE_BIAS;
//     // CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
//     //     matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &bias, sizeof(bias)));
    
//     // CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
//     //     matmulDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, 
//     //     &d_C, sizeof(d_C)));

//     // Perform matrix multiplication with bias broadcasting
//     // Create workspace for potential performance improvement
//     void* workspace = nullptr;
//     size_t workspaceSize = 4 * 1024 * 1024;  // 4MB workspace
//     CHECK_CUDA(cudaMalloc(&workspace, workspaceSize));

//     // Create preference descriptor for the workspace
//     cublasLtMatmulPreference_t preference = nullptr;
//     CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
//     CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
//       preference,
//       CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
//       &workspaceSize,
//       sizeof(workspaceSize)));

//     // Get the best algorithm that will use our workspace
//     cublasLtMatmulHeuristicResult_t heuristicResult = {};
//     int returnedResults = 0;
//     CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
//       handle, matmulDesc,
//       Adesc, Bdesc, Cdesc, Ddesc,
//       preference, 1,
//       &heuristicResult, &returnedResults));

//     // Perform matrix multiplication with workspace
//     CHECK_CUBLAS(cublasLtMatmul(
//       handle,
//       matmulDesc,
//       &alpha,
//       d_A, Adesc,
//       d_B, Bdesc,
//       &beta,
//       d_C, Cdesc,
//       d_D, Ddesc,
//       &heuristicResult.algo,
//       workspace,
//       workspaceSize,
//       0
//     ));

//     // Clean up workspace
//     CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));
//     CHECK_CUDA(cudaFree(workspace));

//     // Copy result back to host
//     CHECK_CUDA(cudaMemcpy(h_D, d_D, m * n * sizeof(float), cudaMemcpyDeviceToHost));

//     // Print results
//     printf("Matrix A (%d x %d):\n", m, k);
//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < k; j++) printf("%.1f ", h_A[i + j * m]);
//         printf("\n");
//     }
//     printf("\nMatrix B (%d x %d):\n", k, n);
//     for (int i = 0; i < k; i++) {
//         for (int j = 0; j < n; j++) printf("%.1f ", h_B[i + j * k]);
//         printf("\n");
//     }
//     printf("\nVector C (%d):\n", m);
//     for (int i = 0; i < m; i++) printf("%.1f\n", h_C[i]);
//     printf("\nResult D = AB + C (%d x %d):\n", m, n);
//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < n; j++) printf("%.1f ", h_D[i + j * m]);
//         printf("\n");
//     }

//     // Clean up
//     CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Adesc));
//     CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Bdesc));
//     CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Cdesc));
//     CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Ddesc));
//     CHECK_CUBLAS(cublasLtMatmulDescDestroy(matmulDesc));
//     CHECK_CUBLAS(cublasLtDestroy(handle));
//     CHECK_CUDA(cudaFree(d_A));
//     CHECK_CUDA(cudaFree(d_B));
//     CHECK_CUDA(cudaFree(d_C));
//     CHECK_CUDA(cudaFree(d_D));

//     return 0;
// }