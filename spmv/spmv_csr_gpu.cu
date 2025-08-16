#include <iostream>
#include <cuda_runtime.h>
#include "dense_to_sparse.h"
#define blockSize 256

using namespace std;
__global__ 
void spmv_csr_kernel(const int* row_ptr, const int* col_idx, const float* val, int rows, const float* B, float* C){
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows) {
        float sum = 0.0f;
        int start = row_ptr[row];
        int end   = row_ptr[row + 1];
        for (int i = start; i < end; ++i) {
            sum += val[i] * B[col_idx[i]];
        }
        C[row] = sum;
    }
}
void spmv_csr_gpu(const float* h_A, const float* h_B, float* h_C, int M, int K){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    CSR h_csr = dense_to_csr(h_A, M, K);
    int *d_row_ptr, *d_col_idx;
    float *d_val, *d_B, *d_C;

    const int non_zeros = h_csr.col_idx.size();
    const int rows = h_csr.row_ptr.size() - 1;

    cudaMalloc(&d_row_ptr, sizeof(int) * (rows + 1));
    cudaMalloc(&d_col_idx, sizeof(int) * non_zeros);
    cudaMalloc(&d_val, sizeof(float) * non_zeros);
    cudaMalloc(&d_B, sizeof(float) * K);
    cudaMalloc(&d_C, sizeof(float) * M);


    cudaMemcpy(d_row_ptr, h_csr.row_ptr.data(), sizeof(int) * (rows + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, h_csr.col_idx.data(), sizeof(int) * non_zeros, cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, h_csr.val.data(), sizeof(float) * non_zeros, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * K, cudaMemcpyHostToDevice);
    
    // initialize to 0
    cudaMemset(d_C, 0, sizeof(float) * M);
    
    cudaEventRecord(start);
    // each thread map to a row, to avoid the use of atomic operation
    spmv_csr_kernel<<< (rows + blockSize - 1) / blockSize, blockSize>>>(d_row_ptr, d_col_idx, d_val, rows, d_B, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cout << "GPU (CSR) time: " << ms << " ms\n";

    cudaMemcpy(h_C, d_C, sizeof(float) * M, cudaMemcpyDeviceToHost);

    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_val);
    cudaFree(d_B);
    cudaFree(d_C);

}