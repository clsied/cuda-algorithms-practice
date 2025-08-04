#include <iostream>
#include <cuda_runtime.h>

__global__
void vector_add_kernel(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

void vector_add_gpu(const float* h_A, const float* h_B, float* h_C, int N){
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    
    float *d_A, *d_B, *d_C; // pointer
    
    cudaMalloc(&d_A, sizeof(float) * N); // allocate memory on GPU, input agruments is address of pointer
    cudaMalloc(&d_B, sizeof(float) * N);
    cudaMalloc(&d_C, sizeof(float) * N);
    
    cudaMemcpy(d_A, h_A, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * N, cudaMemcpyHostToDevice);
    
    const int blockSize = 256;
    
    // compute the number of grid requires,
    // addition of blockSize - 1 is equivalent to ceil
    const int gridSize = (N + blockSize - 1) / blockSize; 
    
    
    // only include the time of the kernel execution
    cudaEventRecord(start);
    vector_add_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "GPU time: " << ms << " ms\n";

    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}