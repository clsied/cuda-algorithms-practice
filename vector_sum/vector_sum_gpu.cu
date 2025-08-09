#include <iostream>
#include <cuda_runtime.h>
using namespace std;

#define BLOCK_SIZE 256

__global__
void vector_sum_kernel(const float* A, float* blockSums, int N){
    __shared__ float partialSum[BLOCK_SIZE];
    unsigned int t = threadIdx.x;
    unsigned int start = blockIdx.x * (blockDim.x * 2);

    float temp = 0.0f;
    if (start + t < N){
        temp += A[start + t];
    }
    if (start + t + blockDim.x < N){
        temp += A[start + blockDim.x + t];
    }

    partialSum[t] = temp;
    __syncthreads();

    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if(t < stride){
            partialSum[t] += partialSum[t + stride];
        }
        __syncthreads();
    }

    if (t == 0){
        blockSums[blockIdx.x] = partialSum[0];
    }
}

float vector_sum_gpu(const float* h_A, int N){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *d_A, *d_blockSums; 
    float h_sum = 0.0f;

    const int gridSize = (N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);

    cudaMalloc(&d_A, sizeof(float) * N);
    cudaMalloc(&d_blockSums, sizeof(float) * gridSize);

    cudaMemcpy(d_A, h_A, sizeof(float) * N, cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    // use d_blockSums instead of d_totalSum to avoid race condition between blocks
    vector_sum_kernel<<< gridSize, BLOCK_SIZE >>>(d_A, d_blockSums, N);
    
    float* h_blockSums = new float[gridSize];
    cudaMemcpy(h_blockSums, d_blockSums, sizeof(float) * gridSize, cudaMemcpyDeviceToHost);
    
    for(int i = 0; i < gridSize; ++i){
        h_sum += h_blockSums[i];
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cout << "GPU time: " << ms << " ms\n";

    delete[] h_blockSums;
    cudaFree(d_A);
    cudaFree(d_blockSums);

    return h_sum;
}
