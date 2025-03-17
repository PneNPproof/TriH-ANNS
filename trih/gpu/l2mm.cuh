#pragma once

#include <cuda_runtime.h>  
#include <stddef.h>        

void l2mm(
  int m,
  int n,
  int k,
  const float *A,
  const float *B,
  float *C,
  // void *workspace,
  // size_t workspaceSize,
  cudaStream_t stream);