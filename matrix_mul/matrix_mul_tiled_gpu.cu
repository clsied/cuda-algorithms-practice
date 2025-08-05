#include <iostream>
#include <cuda_runtime.h>
using namespace std;
#define TILE_SIZE 16

__global__
void matrix_mul_tiled_kernel(const float* A, const float* B, float* C, int M, int N, int K){

    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = TILE_SIZE * blockIdx.y + threadIdx.y;
    int col = TILE_SIZE * blockIdx.x + threadIdx.x;
    
    float value = 0.0f;

    // looped the second dimension of tile
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < M && t * TILE_SIZE + threadIdx.x < K){
            // left -> relative position & right -> absolute position
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        }
        else{
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;

        }
        if (col < N && t * TILE_SIZE + threadIdx.y < K){
            tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        }
        else{
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i) {
            value += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
        }

        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}
void matrix_mul_tiled_gpu(const float* h_A, const float* h_B, float* h_C, int M, int N, int K){

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeof(float) * M * K);
    cudaMalloc(&d_B, sizeof(float) * K * N);
    cudaMalloc(&d_C, sizeof(float) * M * N);

    cudaMemcpy(d_A, h_A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * K * N, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    cudaEventRecord(start);
    matrix_mul_tiled_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cout << "GPU (tiled) time: " << ms << " ms\n";

    cudaMemcpy(h_C, d_C, sizeof(float) * M * K, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}