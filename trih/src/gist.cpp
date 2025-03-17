#include <fstream>

#include "hdf5.h"
#include "gist.h"

gist load_gist(const char *filename)
{
    gist data;
    data.dim = 960;
    data.train = new float[1000000*960];
    data.train_point_count = 1000000;
    data.test = new float[1000*960];
    data.test_point_count = 1000;
    data.neighbors = new uint32_t[1000*100];
    data.distances = new float[1000*100];
    data.neighbors_per_test = 100;

    hid_t h5f = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);

    hid_t ds_train = H5Dopen(h5f, "train", H5P_DEFAULT);
    H5Dread(ds_train, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.train);
    H5Dclose(ds_train);

    hid_t ds_test = H5Dopen(h5f, "test", H5P_DEFAULT);
    H5Dread(ds_test, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.test);
    H5Dclose(ds_test);

    hid_t ds_neighbors = H5Dopen(h5f, "neighbors", H5P_DEFAULT);
    H5Dread(ds_neighbors, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.neighbors);
    H5Dclose(ds_neighbors);

    hid_t ds_distances = H5Dopen(h5f, "distances", H5P_DEFAULT);
    H5Dread(ds_distances, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.distances);
    H5Dclose(ds_distances);

    return data;
}