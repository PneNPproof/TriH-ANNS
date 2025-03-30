#include <iostream>
#include <fstream>
#include <cstring>

#include "pca.h"
#include "gist.h"
#include "search.h"
#include "shuffle.h"

using namespace std;

//保存shuffle数据
void save_shuffled_gist(const char* filename, gist &gdata) {
  ofstream ofs(filename, std::ios::binary);

  ofs.write((const char*)gdata.train, sizeof(float)*gdata.dim*gdata.train_point_count);
  ofs.write((const char*)gdata.neighbors, sizeof(uint32_t)*gdata.neighbors_per_test*gdata.test_point_count);

  ofs.close();
}

//用shuffle的数据覆盖gist数据
void load_shuffled_gist(const char* filename, gist &gdata) {
  ifstream ifs(filename, std::ios::binary);

  ifs.read((char*)gdata.train, sizeof(float)*gdata.dim*gdata.train_point_count);
  ifs.read((char*)gdata.neighbors, sizeof(uint32_t)*gdata.neighbors_per_test*gdata.test_point_count);

  ifs.close();
}

//构建索引，保存为文件
//base 数据，dimension 原始数据维度，size 原始数据大小，column_num PCA 特征向量个数，ratio PCA 累积特征值比率
void build_index(float *base, int dimension, int size, int column_num, float ratio, int column_num2, float ratio2, ofstream &ofs) {
  
  std::cout << "Starting to build PCA index..." << std::endl;
  
  save_pca_index(base, size, dimension, ratio, column_num, ofs);


  std::cout << "PCA index is ready." << std::endl;
}

