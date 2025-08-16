#include <iostream>
#include <cuda_runtime.h>
#include "dense_to_sparse.h"
#define blockSize 256

using namespace std;
__global__ 
void spmv_coo_kernel(const int* row, const int* col, const float* val, int non_zeros, const float* B, float* C){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < non_zeros) {
        // avoid bank conflicts
        atomicAdd(&C[row[idx]], val[idx] * B[col[idx]]);
    }
}
void spmv_coo_gpu(const float* h_A, const float* h_B, float* h_C, int M, int K){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    COO h_coo = dense_to_coo(h_A, M, K);
    int *d_row, *d_col;
    float *d_val, *d_B, *d_C;

    const int non_zeros = h_coo.val.size();

    cudaMalloc(&d_row, sizeof(int) * non_zeros);
    cudaMalloc(&d_col, sizeof(int) * non_zeros);
    cudaMalloc(&d_val, sizeof(float) * non_zeros);
    cudaMalloc(&d_B, sizeof(float) * K);
    cudaMalloc(&d_C, sizeof(float) * M);

    cudaMemcpy(d_row, h_coo.row.data(), sizeof(int) * non_zeros, cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, h_coo.col.data(), sizeof(int) * non_zeros, cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, h_coo.val.data(), sizeof(float) * non_zeros, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * M, cudaMemcpyHostToDevice);
    // initialize to 0
    cudaMemset(d_C, 0, sizeof(float) * K);
    
    cudaEventRecord(start);
    spmv_coo_kernel<<< (non_zeros + blockSize - 1) / blockSize, blockSize>>>(d_row, d_col, d_val, non_zeros, d_B, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cout << "GPU (COO) time: " << ms << " ms\n";

    cudaMemcpy(h_C, d_C, sizeof(float) * M, cudaMemcpyDeviceToHost);

    cudaFree(d_row);
    cudaFree(d_col);
    cudaFree(d_val);
    cudaFree(d_B);
    cudaFree(d_C);

}