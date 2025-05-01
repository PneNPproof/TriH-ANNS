#include <fstream>
#include <iostream>
#include <stdio.h>
#include <sys/stat.h>
#include "hdf5.h"

#include "data.h"

namespace trih
{

data load_h5(const char *filename)
{
    data d;
    hid_t file_id, dataset_id, dataspace_id;
    hsize_t dims[2];

    file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);

    //train
    dataset_id = H5Dopen(file_id, "train", H5P_DEFAULT);
    dataspace_id = H5Dget_space(dataset_id);
    H5Sget_simple_extent_dims(dataspace_id, dims, NULL);

    d.dim = dims[1];
    d.train_point_count = dims[0];
    // d.train = new float[dims[0]*dims[1]];
    d.train = (float *)aligned_alloc(64, dims[0]*dims[1]*sizeof(float));
    H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d.train);
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);

    //test
    dataset_id = H5Dopen(file_id, "test", H5P_DEFAULT);
    dataspace_id = H5Dget_space(dataset_id);
    H5Sget_simple_extent_dims(dataspace_id, dims, NULL);

    d.test_point_count = dims[0];
    d.test = new float[dims[0]*dims[1]];
    H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d.test);
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);

    //neighbors
    dataset_id = H5Dopen(file_id, "neighbors", H5P_DEFAULT);
    // dataset_id = H5Dopen(file_id, "neighbor_indices", H5P_DEFAULT);
    dataspace_id = H5Dget_space(dataset_id);
    H5Sget_simple_extent_dims(dataspace_id, dims, NULL);

    d.neighbors_per_test = dims[1];
    d.neighbors = new uint32_t[dims[0]*dims[1]];
    H5Dread(dataset_id, H5T_NATIVE_UINT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d.neighbors);
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);


    H5Fclose(file_id);

    return d;
}


data load_vecs(const char *filename_prefix)
{
    data d;
    char filename[100];
    FILE *file;
    long file_size;

    //1. train
    sprintf(filename, "%s_base.fvecs", filename_prefix);
    file = fopen(filename, "rb");  // 以二进制模式打开

    //维度
    fread(&d.dim, sizeof(int), 1, file);

    //向量数
    fseek(file, 0, SEEK_END);
    file_size = ftell(file);
    d.train_point_count = file_size / (d.dim * sizeof(float) + sizeof(int));  // (dim+1)*4

    //读取向量
    rewind(file);  // 回到文件起始
    d.train = new float[d.dim*d.train_point_count];
    for(int i=0; i<d.train_point_count; i++){
        fseek(file, sizeof(int), SEEK_CUR);
        fread(d.train+d.dim*i, sizeof(float)*d.dim, 1, file);
    }
    fclose(file);

    //2. test
    sprintf(filename, "%s_query.fvecs", filename_prefix);
    file = fopen(filename, "rb");  // 以二进制模式打开

    //向量数
    fseek(file, 0, SEEK_END);
    file_size = ftell(file);
    d.test_point_count = file_size / (d.dim * sizeof(float) + sizeof(int));  // (dim+1)*4

    //读取向量
    rewind(file);  // 回到文件起始
    d.test = new float[d.dim*d.test_point_count];
    for(int i=0; i<d.test_point_count; i++){
        fseek(file, sizeof(int), SEEK_CUR);
        fread(d.test+d.dim*i, sizeof(float)*d.dim, 1, file);
    }
    fclose(file);

    //3. neighbors
    sprintf(filename, "%s_groundtruth.ivecs", filename_prefix);
    file = fopen(filename, "rb");  // 以二进制模式打开

    //维度
    fread(&d.neighbors_per_test, sizeof(int), 1, file);

    //向量数
    fseek(file, 0, SEEK_END);
    file_size = ftell(file);
    int vec_count = file_size / (d.neighbors_per_test * sizeof(int) + sizeof(int));  // (dim+1)*4

    //读取向量
    rewind(file);  // 回到文件起始
    d.neighbors = new uint32_t[d.neighbors_per_test*vec_count];
    for(int i=0; i<vec_count; i++){
        fseek(file, sizeof(int), SEEK_CUR);
        fread(d.neighbors+d.neighbors_per_test*i, sizeof(uint32_t)*d.neighbors_per_test, 1, file);
    }
    fclose(file);

    return d;
}

data load_data(const char *filename) {
    struct stat buffer;
    data d;
    if(stat(filename, &buffer) == 0) //文件存在， hdf5
        d = load_h5(filename);
    else
        d = load_vecs(filename);
    
    #ifdef DETAILED_LOG
    std::cout << "\tData info: " << std::endl;
    std::cout << "\tdim: " << d.dim << std::endl;
    std::cout << "\ttrain: " << d.train_point_count << std::endl;
    std::cout << "\ttest: " << d.test_point_count << std::endl;
    std::cout << "\tneigbors per test: " << d.neighbors_per_test << std::endl;
    #endif

    return d;
}

}