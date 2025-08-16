#include <iostream>
#include <cuda_runtime.h>
#include "dense_to_sparse.h"
#define blockSize 256

using namespace std;
__global__ 
void spmv_ell_kernel(const int* col_indices, const float* vals, int rows, int max_non_zero, const float* B, float* C){

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows){
        float sum = 0.0f;
        for (int i = 0; i < max_non_zero; i++) {
            int col = col_indices[row * max_non_zero + i];
            float val = vals[row * max_non_zero + i];
            
            if (col != -1) {
                sum += val * B[col];
            }
        }
        C[row] = sum;
    }

}
void spmv_ell_gpu(const float* h_A, const float* h_B, float* h_C, int M, int K){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    ELL h_ell = dense_to_ell(h_A, M, K);
    int *d_col_indices;
    float *d_vals, *d_B, *d_C;

    const int max_non_zero = h_ell.max_non_zero;
    const int rows = h_ell.rows;

    cudaMalloc(&d_col_indices, sizeof(int) * (rows * max_non_zero));
    cudaMalloc(&d_vals, sizeof(float) * (rows * max_non_zero));
    cudaMalloc(&d_B, sizeof(float) * K);
    cudaMalloc(&d_C, sizeof(float) * M);

    cudaMemcpy(d_col_indices, h_ell.col_indices.data(), sizeof(int) *  (rows * max_non_zero), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, h_ell.vals.data(), sizeof(float) *  (rows * max_non_zero), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * K, cudaMemcpyHostToDevice);
    // initialize to 0
    cudaMemset(d_C, 0, sizeof(float) * M);
    
    cudaEventRecord(start);
    spmv_ell_kernel<<< (rows + blockSize - 1) / blockSize, blockSize>>>(d_col_indices, d_vals, rows, max_non_zero, d_B, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cout << "GPU (ELL) time: " << ms << " ms\n";

    cudaMemcpy(h_C, d_C, sizeof(float) * M, cudaMemcpyDeviceToHost);

    cudaFree(d_col_indices);
    cudaFree(d_vals);
    cudaFree(d_B);
    cudaFree(d_C);

}