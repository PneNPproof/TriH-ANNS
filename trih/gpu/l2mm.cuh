#pragma once

#include <cuda_runtime.h>  
#include <stddef.h>
#include <cuda_fp16.h>

void addVectors(half* d_A, const half* d_B, int n, cudaStream_t stream);

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

void l2mm_fp16(
  int m,
  int n,
  int k,
  const half *A,
  const half *B,
  half *C,
  // void *workspace,
  // size_t workspaceSize,
  cudaStream_t stream);