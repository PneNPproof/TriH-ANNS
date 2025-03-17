#pragma once

#ifndef __PCA_H__
#define __PCA_H__

#include <vector>
#include <fstream>

using namespace std;

struct EigenPair {
    float value;

    vector<float> vec;    
};


struct pca_index {
    //原始数据信息
    int dim;//数据维度，960
    int record_num;//数据向量个数
    //第一层PCA信息
    float *pca_data;//特征向量，完整的dim个特征向量, dim x dim
    int column_num;//特征向量个数，满足需求的特征向量个数，如 128
    float ratio;//满足需求的特征值累积和 比率

    float *trans_data;//原始数据在PCA投影后的数据，降维后的数据，如 降维为 128
    float *trans_data_remain;//原始数据主维度之后的数据在PCA投影后的数据

    //第二层PCA信息
    float *pca_data2; //第二层特征向量，完整的 dim - column_num 个特征向量, (dim - column_num) x (dim - column_num)
    int column_num2;//特征向量个数，满足需求的个数
    float ratio2; 

    float *trans_data2;// trans_data_remain 在 PCA2 上降维后的数据

    //shuffle trans_data 的remain数据，以及相关数据
    int *ids_shuffle; //针对原始数据下标 shuffle 后的下标
    float *pca_data2_shuffle; //用 trans_data_shuffle 生成的 pca_data2
    float *trans_data2_shuffle; //用 pca_data2_shuffle 生成的 trans_data2

    pca_index() {
        pca_data = nullptr;
        trans_data = nullptr;
        pca_data2 = nullptr;
        trans_data2 = nullptr;

        ids_shuffle = nullptr;
        pca_data2_shuffle = nullptr;
        trans_data2_shuffle = nullptr;
    }

    ~pca_index() {   

        if(pca_data != nullptr) {
            delete [] pca_data;
            pca_data = nullptr;
        }

        if(trans_data != nullptr) {
            delete [] trans_data;
            trans_data = nullptr;
        }

        if(pca_data2 != nullptr) {
            delete [] pca_data2;
            pca_data2 = nullptr;
        }

        if(trans_data2 != nullptr) {
            delete [] trans_data2;
            trans_data2 = nullptr;
        }

        if(ids_shuffle != nullptr) {
            delete [] ids_shuffle;
            ids_shuffle = nullptr;
        }

        if(pca_data2_shuffle != nullptr) {
            delete [] pca_data2_shuffle;
            pca_data2_shuffle = nullptr;
        }

        if(trans_data2_shuffle != nullptr) {
            delete [] trans_data2_shuffle;
            trans_data2_shuffle = nullptr;
        }
    }
};

inline bool compareEigenPairs(const EigenPair &a, const EigenPair &b){
    return a.value > b.value;
}

void PCA(const float* src, const int N0, const int D, float &ratio, int &d, float*& pca_data);

void save_pca_index(float *data, int record_num, int dim, //原始数据
            float &ratio, int &column_num, //PCA 参数
            float &ratio2, int column_num2, //PCA2 参数
            ofstream &ofs);

void load_pca_index(ifstream &ifs, pca_index &index);

void queryProject(const float *query, pca_index &index, float *trans_data, float *trans_data_remain, float *trans_data2, float *trans_data2_shuffle);


#endif