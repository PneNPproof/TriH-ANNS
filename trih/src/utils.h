#pragma once

#include <iostream>
#include <fstream>

#include "gist.h"
#include "pca.h"

using namespace std;

void save_shuffled_gist(const char* filename, gist &gdata);
void load_shuffled_gist(const char* filename, gist &gdata);
void build_index(float *base, int dimension, int size, int column_num, float ratio, int column_num2, float ratio2, ofstream &ofs);