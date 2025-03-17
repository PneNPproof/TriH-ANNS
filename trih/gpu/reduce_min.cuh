#pragma once
#include <cuda_runtime.h>

void launchComputeGroupMinimaKernel(
  const float* d_input,
  int n,
  int m,
  int group_size,
  float **d_output_min,
  int **d_output_idx,
  int block_size);