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

    pca_index() {
        pca_data = nullptr;
        trans_data = nullptr;
        trans_data_remain = nullptr;
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

        if(trans_data_remain != nullptr) {
            delete [] trans_data_remain;
            trans_data_remain = nullptr;
        }
    }
};

inline bool compareEigenPairs(const EigenPair &a, const EigenPair &b){
    return a.value > b.value;
}

void PCA(const float* src, const int N0, const int D, float &ratio, int &d, float*& pca_data);

void save_pca_index(float *data, int record_num, int dim, //原始数据
            float &ratio, int &column_num, //PCA 参数
            ofstream &ofs);

void load_pca_index(ifstream &ifs, pca_index &index);

void queryProject(const float *query, pca_index &index, float *trans_data, float *trans_data_remain);

void queryProjectMain(const float *query, pca_index &index, float *trans_data);
void queryProjectRest(const float *query, pca_index &index, float *trans_data_remain);

void queryProjectRest2(const float *query, pca_index &index, float *trans_data_remain);
#endif