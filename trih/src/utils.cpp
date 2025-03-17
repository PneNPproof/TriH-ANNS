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
  
  save_pca_index(base, size, dimension, ratio, column_num, ratio2, column_num2, ofs);


  std::cout << "PCA index is ready." << std::endl;
}


void recall_test(gist &gdata, pca_index &index, int section_num, int top_k) {

  int ids[top_k];
  int top100_ids0[100], top100_ids[100], top100_ids_sq[100], top100_ids_shuffle[100];
  uint32_t neighbors[100];
  int recalls_ids[1000] = {0};
  int recalls0[1000] = {0};
  int recalls[1000] = {0};
  int recalls_sq[1000] = {0};
  int recalls_shuffle[1000] = {0};

  int end_query = 1000;
      
  for(int q=0; q<end_query; q++) {

      search_index(gdata.test+q*960, gdata.train, index, section_num, top_k, ids, 100, top100_ids0, top100_ids, top100_ids_sq, top100_ids_shuffle);

      memcpy(neighbors, gdata.neighbors+q*100, 100*sizeof(uint32_t));
      for(int i=0; i<100; i++) {
          for(int k=0; k<top_k; k++) {
              if(neighbors[i] == (uint32_t)ids[k]) {
                  recalls_ids[q]++; break;
              }
          }
      }

      for(int i=0; i<100; i++) {
          for(int k=0; k<100; k++) {
              if(neighbors[i] == (uint32_t)top100_ids0[k]) {
                  recalls0[q]++; break;
              }
          }
      }

      for(int i=0; i<100; i++) {
          for(int k=0; k<100; k++) {
              if(neighbors[i] == (uint32_t)top100_ids[k]) {
                  recalls[q]++; break;
              }
          }
      }

      for(int i=0; i<100; i++) {
          for(int k=0; k<100; k++) {
              if(neighbors[i] == (uint32_t)top100_ids_sq[k]) {
                  recalls_sq[q]++; break;
              }
          }
      }

      for(int i=0; i<100; i++) {
          for(int k=0; k<100; k++) {
              if(neighbors[i] == (uint32_t)top100_ids_shuffle[k]) {
                  recalls_shuffle[q]++; break;
              }
          }
      }

      // if(q % 10 == 0)
          std::cout << "query " << q << "    recalls_ids=" << recalls_ids[q] << " recall0=" << recalls0[q] << "  recall=" << recalls[q] << "  recall_sq=" << recalls_sq[q] << "  recall_shuffle=" << recalls_shuffle[q];

          if(recalls_ids[q] > recalls0[q])
              std::cout << "  ******" << std::endl;
          else
              std::cout << std::endl;
  }

  int total_recall_ids = 0, total_recall0 = 0, total_recall = 0, total_recall_sq = 0, total_recall_shuffle = 0;
  for(int i=0; i<end_query; i++) {
      total_recall_ids += recalls_ids[i];
      total_recall0 += recalls0[i];
      total_recall += recalls[i];
      total_recall_sq += recalls_sq[i];
      total_recall_shuffle += recalls_shuffle[i];
  }

  std::cout << "Avg recalls_ids: " << total_recall_ids/1.0/end_query << "  Avg recall0: " << total_recall0/1.0/end_query << "  Avg recall: " << total_recall/1.0/end_query << "  Avg recall_sq: " << total_recall_sq/1.0/end_query << "  Avg recall shuffle: " << total_recall_shuffle/1.0/end_query << std::endl;
}

//增加了sq量化
void recall_test_sq(gist &gdata, pca_index &index, int section_num, int top_k) {

  int ids[top_k];
  int top100_ids0[100], top100_ids[100], top100_ids_sq[100], top100_ids_sq_all_vec[100], top100_ids_shuffle[100];
  uint32_t neighbors[100];
  int recalls_ids[1000] = {0};
  int recalls0[1000] = {0};
  int recalls[1000] = {0};
  int recalls_sq[1000] = {0};
  int recalls_sq_all_vec[1000] = {0};
  int recalls_shuffle[1000] = {0};

  int end_query = 1000;
  
  /// 量化到0-255
  int dimension = index.dim - index.column_num;
  auto base = index.trans_data_remain;

  // int dimension = index.dim;
  // auto base = gdata.train;
  int b = 256;

  // float max_all_dim = FLT_MIN;
  // float min_all_dim = FLT_MAX;
  // for(int i=0; i<index.record_num; i++) {
  //     for(int j=0; j<dimension; j++) {
  //         if(max_all_dim < base[i*dimension+j])
  //             max_all_dim = base[i*dimension+j];
  //         if(min_all_dim > base[i*dimension+j])
  //             min_all_dim = base[i*dimension+j];
  //     }
  // }

  // float scale = (max_all_dim - min_all_dim) / (b - 1);
  // float zp = min_all_dim;

  float scale = (1.5 - 0) / (b - 1);
  float zp = 0;
  
  uint8_t *quant_base = new uint8_t[index.record_num * dimension];

  std::cout << "quant point begin" << std::endl;

  for (int i = 0; i < index.record_num; i++)
  {
      quant_point(base + i * dimension, quant_base + i * dimension, scale, zp, dimension, b);
  }

  std::cout << "quant point finished" << std::endl;




  for(int q=0; q<end_query; q++) {

      // search_index(gdata.test+q*960, gdata.train, index, section_num, top_k, ids, 100, top100_ids0, top100_ids, top100_ids_sq, top100_ids_shuffle);
      search_index(gdata.test+q*960, gdata.train, index, section_num, top_k, ids, 100, top100_ids0, top100_ids, top100_ids_sq, top100_ids_shuffle, quant_base, scale, zp, b, top100_ids_sq_all_vec);

      memcpy(neighbors, gdata.neighbors+q*100, 100*sizeof(uint32_t));
      for(int i=0; i<100; i++) {
          for(int k=0; k<top_k; k++) {
              if(neighbors[i] == (uint32_t)ids[k]) {
                  recalls_ids[q]++; break;
              }
          }
      }

      for(int i=0; i<100; i++) {
          for(int k=0; k<100; k++) {
              if(neighbors[i] == (uint32_t)top100_ids0[k]) {
                  recalls0[q]++; break;
              }
          }
      }

      for(int i=0; i<100; i++) {
          for(int k=0; k<100; k++) {
              if(neighbors[i] == (uint32_t)top100_ids[k]) {
                  recalls[q]++; break;
              }
          }
      }

      for(int i=0; i<100; i++) {
          for(int k=0; k<100; k++) {
              if(neighbors[i] == (uint32_t)top100_ids_sq[k]) {
                  recalls_sq[q]++; break;
              }
          }
      }
      
      for(int i=0; i<100; i++) {
          for(int k=0; k<100; k++) {
              if(neighbors[i] == (uint32_t)top100_ids_sq_all_vec[k]) {
                  recalls_sq_all_vec[q]++; break;
              }
          }
      }

      for(int i=0; i<100; i++) {
          for(int k=0; k<100; k++) {
              if(neighbors[i] == (uint32_t)top100_ids_shuffle[k]) {
                  recalls_shuffle[q]++; break;
              }
          }
      }

      // if(q % 10 == 0)
          std::cout << "query " << q << "    recalls_ids=" << recalls_ids[q] << " recall0=" << recalls0[q] << "  recall=" << recalls[q] << "  recall_sq=" << recalls_sq[q] << "  recall_sq_all_vec=" << recalls_sq_all_vec[q] << "  recall_shuffle=" << recalls_shuffle[q];

          if(recalls_ids[q] > recalls0[q])
              std::cout << "  ******" << std::endl;
          else
              std::cout << std::endl;
  }

  int total_recall_ids = 0, total_recall0 = 0, total_recall = 0, total_recall_sq = 0, total_recall_sq_all_vec = 0, total_recall_shuffle = 0;
  for(int i=0; i<end_query; i++) {
      total_recall_ids += recalls_ids[i];
      total_recall0 += recalls0[i];
      total_recall += recalls[i];
      total_recall_sq += recalls_sq[i];
      total_recall_sq_all_vec += recalls_sq_all_vec[i];
      
      total_recall_shuffle += recalls_shuffle[i];
  }

  std::cout << "Avg recalls_ids: " << total_recall_ids/1.0/end_query << "  Avg recall0: " << total_recall0/1.0/end_query << "  Avg recall: " << total_recall/1.0/end_query << "  Avg recall_sq: " << total_recall_sq/1.0/end_query << "  Avg recall_sq_all_vec: " << total_recall_sq_all_vec/1.0/end_query << "  Avg recall shuffle: " << total_recall_shuffle/1.0/end_query << std::endl;
}
