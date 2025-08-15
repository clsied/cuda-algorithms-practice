#include <iostream>
#include <cuda_runtime.h>
#define blockSize 256
using namespace std;

// block independent inclusive scane using kogge-stone algorithm
__global__
void inclusive_scan_kogge_kernel(const float* input, float* output, int N){

    __shared__ float temp[blockSize];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
        temp[threadIdx.x] = input[idx];
    }
    __syncthreads();

    for(int stride = 1; stride < blockSize; stride <<=1){
        float val = temp[threadIdx.x];
        __syncthreads();
        if(threadIdx.x >= stride){
            val += temp[threadIdx.x-stride];
        }
        __syncthreads();
        temp[threadIdx.x] = val;
    }

    if (idx < N){
        output[idx] = temp[threadIdx.x];
    }
}

// collect the last element of each block

__global__
void collect_last_kogge_kernel(const float* output, float* last_list, int N, int num_blocks){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x == blockDim.x - 1 && idx < N) {
        last_list[blockIdx.x] = output[idx];
    }
}

__global__
void add_last_kogge_kernel(float* output, const float* last_list, int N, int num_blocks){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = output[idx];

    if (idx < N && blockIdx.x > 0){
        for(int i = 0; i < blockIdx.x; ++i){
            temp += last_list[i];
        }
    }
    if (idx < N){
        output[idx] = temp;
    }
}

void inclusive_scan_kogge_gpu(const float* h_input, float* h_output, int N){

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *d_input, *d_output, *d_last_list;
    int num_blocks = (N + blockSize - 1) / (blockSize);

    cudaMalloc(&d_input, sizeof(float) * N);
    cudaMalloc(&d_output, sizeof(float) * N);
    cudaMalloc(&d_last_list, sizeof(float) * num_blocks);


    cudaMemcpy(d_input, h_input, sizeof(float) * N, cudaMemcpyHostToDevice);

    dim3 blockDim(blockSize, 1, 1);
    dim3 gridDim(num_blocks, 1, 1);
    
    cudaEventRecord(start);
    inclusive_scan_kogge_kernel<<<gridDim, blockDim>>>(d_input, d_output, N);
    collect_last_kogge_kernel<<<gridDim, blockDim>>>(d_output, d_last_list, N, num_blocks);
    add_last_kogge_kernel<<<gridDim, blockDim>>>(d_output, d_last_list, N, num_blocks);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);


    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cout << "GPU (kogge) time: " << ms << " ms\n";

    cudaMemcpy(h_output, d_output, sizeof(float) * N, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_last_list);
}