#include <random>
#include <cstring>
#include <iostream>

void myshuffle(float *vectors, int dim, int size, int *ids) {
    
    int ex;

    float vec_tmp[dim];
    int tmp;

    srand(time(NULL));

    for(int i=size-1; i>1; i--){

        ex = rand() % i; //[0, i-1]

        //exchange train point: i <-> ex
        memcpy(vec_tmp, vectors+i*dim, sizeof(float)*dim);
        memcpy(vectors+i*dim, vectors+ex*dim, sizeof(float)*dim);
        memcpy(vectors+ex*dim, vec_tmp, sizeof(float)*dim);

        tmp = ids[i];
        ids[i] = ids[ex];
        ids[ex] = tmp;
    }
}

// 用于neighbors有问题，修改后，再次循环可能被再次修改？
// void myshuffle(float *vectors, int dim, int size, int *others, int others_size) {
    
//     int ids[size];
//     for(int i=0; i<size; i++) ids[i] = i;

//     myshuffle(vectors, dim, size, ids);

//     for(int i=0; i<size; i++) {
//         if(ids[i] != i) {//已经发生改变，位置 i 上面的值不是 i ，而已经变换成 ids[i] 了

//             for(int k=0; k<others_size; k++) {//遍历others，如果原来的值是 i，则需要调整为 ids[i]
//                 if(others[k] == i) others[k] = ids[i]; 
//             }
//         }
//     }
// }

//用于gdata shuffle with neighbors
void myshuffle2(float *vectors, int dim, int size, int *others, int others_size) {
    
    int ex;

    float vec_tmp[dim];

    srand(time(NULL));

    for(int i=size-1; i>1; i--){

        ex = rand() % i; //[0, i-1]

        //exchange train point: i <-> ex
        memcpy(vec_tmp, vectors+i*dim, sizeof(float)*dim);
        memcpy(vectors+i*dim, vectors+ex*dim, sizeof(float)*dim);
        memcpy(vectors+ex*dim, vec_tmp, sizeof(float)*dim);

        for(int k=0; k<others_size; k++) {
            if(others[k] == i) 
                others[k] = ex;
            else if(others[k] == ex)
                others[k] = i;
        }
    }
}