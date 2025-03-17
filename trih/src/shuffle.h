#ifndef __SHUFFLE_H__
#define __SHUFFLE_H__

#include <stdint.h>

void myshuffle(float *vectors, int dim, int size, int *ids);
// void myshuffle(float *vectors, int dim, int size, int *others, int others_size);
void myshuffle2(float *vectors, int dim, int size, int *others, int others_size);
#endif