#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__
void matrix_mul_naive_kernel(const float* A, const float* B, float* C, int M, int N, int K){
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    float sum = 0.0f;
    if(row < M && col < N){
        sum = 0.0f;
        for(int k = 0; k < K; ++k){
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void matrix_mul_naive_gpu(const float* h_A, const float* h_B, float* h_C, int M, int N, int K){
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float* d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeof(float) * M * K);
    cudaMalloc(&d_B, sizeof(float) * K * N);
    cudaMalloc(&d_C, sizeof(float) * M * N);

    cudaMemcpy(d_A, h_A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * K * N, cudaMemcpyHostToDevice);


    dim3 blockDim(16, 16);
    
    // Important
    // N is col & M is row
    dim3 gridDim((N + 15) / 16, (M + 15) / 16); 

    cudaEventRecord(start);
    matrix_mul_naive_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cout << "GPU (naive) time: " << ms << " ms\n";

    cudaMemcpy(h_C, d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}