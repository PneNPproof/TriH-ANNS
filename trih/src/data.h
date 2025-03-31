#ifndef __GIST_H__
#define __GIST_H__

#include "hdf5.h"

namespace trih
{

    typedef struct data_t
    {
        uint32_t dim;

        float *train;
        uint32_t train_point_count;

        float *test;
        uint32_t test_point_count;

        uint32_t *neighbors;
        uint32_t neighbors_per_test;
    } data;

    trih::data load_data(const char *filename);

}
#endif