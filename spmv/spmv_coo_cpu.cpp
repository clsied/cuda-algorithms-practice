#include "dense_to_sparse.h"

void spmv_coo_cpu(const float* A, const float* B, float* C, int M, int K){

    COO h_coo = dense_to_coo(A, M, K);
    
    int none_zeros = h_coo.row.size();

    for(int i = 0; i < none_zeros; ++i){
        C[h_coo.row[i]] += h_coo.val[i] * B[h_coo.col[i]];
    }
}