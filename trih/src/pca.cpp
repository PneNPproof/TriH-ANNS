#include <Eigen/Dense>
#include <iostream>
#include <fstream>

#include "pca.h"
#include "shuffle.h"

#define MAX_SAMPLE 1000000

using namespace std;

//remain: 剩余的pca特征向量
//N0 原始数据个数； D 原始数据维度；ratio / d 降维特征值累积和比率或维度目标；
//pca_data pca 所有特征值，个数 D 个；
void PCA(const float* src, const int N0, const int D, float &ratio, int &d, float*& pca_data) {
    
    int N = N0;
    if(N > MAX_SAMPLE) N = MAX_SAMPLE;

    float *data = new float[N*D]{0};

    memcpy(data, src, sizeof(float)*N*D);

    float *mean = new float[D]{0};
    for(int i=0; i<N; i++) {
        for(int j=0; j<D; j++) 
            mean[j] += data[i*D+j];
    }
    for(int j=0; j<D; j++) 
        mean[j] /= N;

    Eigen::MatrixXf matrix(N, D);
    for(int i=0; i<N; i++) {
        for(int j=0; j<D; j++) {
            data[i*D+j] -= mean[j];
            matrix(i,j) = data[i*D+j];
        }
    }

    //1. 协方差
    Eigen::MatrixXf cov_matrix = (matrix.transpose() * matrix) / float(matrix.rows() -1);

    //2. 特征值分解
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver(cov_matrix);

    //3. 获取特征值
    Eigen::VectorXf eigen_values = solver.eigenvalues().real();

    //4. 获取特征向量
    Eigen::MatrixXf eigen_vectors = solver.eigenvectors().real();

    //5. 降序排列
    std::vector<EigenPair> eigen_pairs(D);
    for(int i=0; i<D; i++) {
        eigen_pairs[i].value = eigen_values[i];
        eigen_pairs[i].vec = std::vector<float>(D);
        for(int j=0; j<D; j++)
            eigen_pairs[i].vec[j] = eigen_vectors(j, i);
    }

    std::sort(eigen_pairs.begin(), eigen_pairs.end(), compareEigenPairs);


    float sum = 0;
    for(int i=0; i<D; i++) 
        sum += eigen_pairs[i].value;

    //修改 ratio 以及 d，表示达到降维要求的目标
    if(d <= 0) { //根据ratio输出

        if(ratio <= 0) {
            cout << "Wrong parameter: ratio" << endl;
            return;
        } 

        float total = 0;
        d = 0;
        for(; d<D; d++) {
            total += eigen_pairs[d].value;
            if(total >= sum*ratio) {
                ratio = total/sum;
                break;
            }
        }
        d+=1; // total d vectors
    }
    else { //根据 d 输出
        float total = 0;
        for(int i=0; i<d; i++) 
            total += eigen_pairs[i].value;
        ratio = total/sum;
    }

    //输出完整的D个PCA特征向量
    // float *comp = new float[D*d];
    // for(int i=0; i<D; i++) {
    //     for(int j=0; j<d; j++)
    //         comp[i*d+j] = eigen_pairs[j].vec[i];
    // }
    pca_data = new float[D*D];
    for(int i=0; i<D; i++) {
        for(int j=0; j<D; j++)
            pca_data[i*D+j] = eigen_pairs[j].vec[i];
    }

    delete [] data;
    delete [] mean;
}


//shuffle trans_data_remain, 进行PCA，并保存到文杰
void shuffle_pca2(float *trans_data_remain, int dim, int size, float &ratio, int &column_num, ofstream &ofs) {

    float *tdr = new float[dim*size];
    memcpy(tdr, trans_data_remain, dim*size*sizeof(float));

    int ids_shuffle[size];
    for(int i=0; i<size; i++) ids_shuffle[i] = i;

    cout << "Start PCA 2 shuffling..." << endl;
    myshuffle(trans_data_remain, dim, size, ids_shuffle); //shuffle trans_data_remain，ids中存储shuffle 后的id

    float *pca_data2_shuffle;
    PCA(trans_data_remain, size, dim, ratio, column_num, pca_data2_shuffle); //针对shuffle后的数据，得到dimxdim 的矩阵， (dim-column_num)

    ofs.write((const char *)ids_shuffle, sizeof(int)*size);//保存 shuffle 的ids下标
    ofs.write((const char *)pca_data2_shuffle, sizeof(float)*dim*dim);//保存特征向量

    cout << "Start shuffled PCA 2 projecting..." << endl; //将shuffle后的数据，进行投影，取 column_num 维度，并保存
    float *trans_data2_shuffle = new float[size*column_num]{0};
#pragma omp parallel for
    for(int r=0; r<size; r++) {//原始数据每一个向量, shape record_num x dim
        for(int c=0; c<column_num; c++) {//投影的向量的每一个元素, trans_data shape record_num x column_num, 1000000 x column_num
            for(int d=0; d<dim; d++) //trans_data_remain 是 dim-column_num 维
                trans_data2_shuffle[r*column_num+c] += trans_data_remain[r*dim+d] * pca_data2_shuffle[d*dim+c];
        }
    }

    ofs.write((const char *)trans_data2_shuffle, sizeof(float)*size*column_num);

    delete [] pca_data2_shuffle;
    delete [] trans_data2_shuffle;
}


//为一组数据求PCA，并且保存特征向量，以及投影向量
void save_pca_index(float *data, int record_num, int dim, //原始数据
            float &ratio, int &column_num, //PCA 参数
            float &ratio2, int column_num2, //PCA2 参数
            ofstream &ofs) {

    cout << "Start PCA 1..." << endl;
    float *pca_data;
    PCA(data, record_num, dim, ratio, column_num, pca_data); //得到  dim * dim 矩阵, columns可以是函数输出

    cout << "PCA 1 ratio=" << ratio << "  column_num=" << column_num << endl;


    //保存 基础信息
    ofs.write((const char *)&dim, sizeof(int)); //数据维度，960
    ofs.write((const char *)&record_num, sizeof(int)); //数据向量个数
    ofs.write((const char *)&column_num, sizeof(int)); //特征向量个数，128
    ofs.write((const char *)&ratio, sizeof(float)); //比率
    //保存 PCA 得到的 col_num 个特征向量
    ofs.write((const char *)pca_data, sizeof(float)*dim*dim);

    cout << "Start PCA 1 projecting..." << endl;
    //使用特征向量，得到原始数据的PCA降维投影
    float *trans_data = new float[record_num*column_num]{0};
#pragma omp parallel for
    for(int r=0; r<record_num; r++) {//原始数据每一个向量, shape record_num x dim, 1000000 x 960
        for(int c=0; c<column_num; c++) {//投影的向量的每一个元素, trans_data shape record_num x column_num, 1000000 x column_num
            for(int d=0; d<dim; d++)
                trans_data[r*column_num+c] += data[r*dim+d] * pca_data[d*dim+c];
        }
    }

    ofs.write((const char *)trans_data, sizeof(float)*record_num*column_num);
    delete [] trans_data;
    

    cout << "Start PCA 1 remain projecting..." << endl;
    //使用特征向量，得到原始数据的除了PCA上述投影之外的投影
    float *trans_data_remain = new float[record_num*(dim-column_num)]{0};
#pragma omp parallel for
    for(int r=0; r<record_num; r++) {//原始数据每一个向量, shape record_num x dim, 1000000 x 960
        for(int c=0; c<dim-column_num; c++) {//投影的向量的每一个元素, trans_data shape record_num x column_num, 1000000 x column_num
            for(int d=0; d<dim; d++)
                trans_data_remain[r*(dim-column_num)+c] += data[r*dim+d] * pca_data[d*dim+column_num+c];
        }
    }

    ofs.write((const char *)trans_data_remain, sizeof(float)*record_num*(dim-column_num));

    delete [] pca_data;

    cout << "Start PCA 2..." << endl;
    //针对trans_data_remain进行PCA2
    float *pca_data2;
    PCA(trans_data_remain, record_num, dim-column_num, ratio2, column_num2, pca_data2); //得到  (dim-column_num) * (dim-column_num) 矩阵, columns可以是函数输出

    ofs.write((const char *)&column_num2, sizeof(int)); //特征向量个数，128
    ofs.write((const char *)&ratio2, sizeof(float)); //比率
    ofs.write((const char *)pca_data2, sizeof(float)*(dim-column_num)*(dim-column_num));

    cout << "Start PCA 2 projecting..." << endl;
    float *trans_data2 = new float[record_num*column_num2]{0};
#pragma omp parallel for
    for(int r=0; r<record_num; r++) {//原始数据每一个向量, shape record_num x dim, 1000000 x 960
        for(int c=0; c<column_num2; c++) {//投影的向量的每一个元素, trans_data shape record_num x column_num, 1000000 x column_num
            for(int d=0; d<dim-column_num; d++) //trans_data_remain 是 dim-column_num 维
                trans_data2[r*column_num2+c] += trans_data_remain[r*(dim-column_num)+d] * pca_data2[d*(dim-column_num)+c];
        }
    }

    ofs.write((const char *)trans_data2, sizeof(float)*record_num*column_num2);

    delete [] pca_data2;
    delete [] trans_data2;

    //shuffle pca2 并 保存
    shuffle_pca2(trans_data_remain, dim-column_num, record_num, ratio2, column_num2, ofs);

    delete [] trans_data_remain;
}

void load_pca_index(ifstream &ifs, pca_index &index) {

    ifs.read((char *)&index.dim, sizeof(int)); //数据维度，960
    ifs.read((char *)&index.record_num, sizeof(int)); //数据向量个数
    ifs.read((char *)&index.column_num, sizeof(int)); //特征向量个数，128
    ifs.read((char *)&index.ratio, sizeof(float)); //比率

    index.pca_data = new float[index.dim*index.dim]{0};
    ifs.read((char *)index.pca_data, sizeof(float)*index.dim*index.dim);

    index.trans_data = new float[index.record_num*index.column_num];
    ifs.read((char *)index.trans_data, sizeof(float)*index.record_num*index.column_num);

    index.trans_data_remain = new float[index.record_num*(index.dim-index.column_num)];
    ifs.read((char *)index.trans_data_remain, sizeof(float)*index.record_num*(index.dim-index.column_num));

    //PCA 2
    ifs.read((char *)&index.column_num2, sizeof(int)); //特征向量个数，128
    ifs.read((char *)&index.ratio2, sizeof(float)); //比率

    index.pca_data2 = new float[(index.dim-index.column_num)*(index.dim-index.column_num)]{0};
    ifs.read((char *)index.pca_data2, sizeof(float)*(index.dim-index.column_num)*(index.dim-index.column_num));

    index.trans_data2 = new float[index.record_num*index.column_num2];
    ifs.read((char *)index.trans_data2, sizeof(float)*index.record_num*index.column_num2);

    index.ids_shuffle = new int[index.record_num];
    ifs.read((char *)index.ids_shuffle, sizeof(int)*index.record_num);

    index.pca_data2_shuffle = new float[(index.dim-index.column_num)*(index.dim-index.column_num)];
    ifs.read((char *)index.pca_data2_shuffle, sizeof(float)*(index.dim-index.column_num)*(index.dim-index.column_num));

    index.trans_data2_shuffle = new float[index.record_num*index.column_num2];
    ifs.read((char *)index.trans_data2_shuffle, sizeof(float)*index.record_num*index.column_num2);
}


//获取 query 在PCA上的投影
void queryProject(const float *query, pca_index &index, float *trans_data, float *trans_data_remain, float *trans_data2, float *trans_data2_shuffle) {

    memset(trans_data, 0, sizeof(float)*index.column_num); 
    memset(trans_data_remain, 0, sizeof(float)*(index.dim-index.column_num)); 
    memset(trans_data2, 0, sizeof(float)*index.column_num2); 
    memset(trans_data2_shuffle, 0, sizeof(float)*index.column_num2); 

    for(int c=0; c<index.column_num; c++) {
        for(int d=0; d<index.dim; d++) 
            trans_data[c] += query[d] * index.pca_data[d*index.dim+c];
    }

    for(int c=0; c<index.dim-index.column_num; c++) {
        for(int d=0; d<index.dim; d++) 
            trans_data_remain[c] += query[d] * index.pca_data[d*index.dim + index.column_num + c];
    }

    for(int c=0; c<index.column_num2; c++) {
        for(int d=0; d<index.dim-index.column_num; d++) 
            trans_data2[c] += trans_data_remain[d] * index.pca_data2[d*(index.dim-index.column_num)+c];
    }

    for(int c=0; c<index.column_num2; c++) {
        for(int d=0; d<index.dim-index.column_num; d++) 
            trans_data2_shuffle[c] += trans_data_remain[d] * index.pca_data2_shuffle[d*(index.dim-index.column_num)+c];
    }
}