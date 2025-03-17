#ifndef __GIST_H__
#define __GIST_H__

#include "hdf5.h"

typedef struct gist_t
{
    uint32_t dim;

    float *train;
    uint32_t train_point_count;

    float *test;
    uint32_t test_point_count;

    uint32_t *neighbors;
    float *distances;
    uint32_t neighbors_per_test;
} gist;

gist load_gist(const char *filename);


#endif