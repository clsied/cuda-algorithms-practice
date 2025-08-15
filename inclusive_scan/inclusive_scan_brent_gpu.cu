#include <iostream>
#include <cuda_runtime.h>
using namespace std;
#define blockSize 256

__global__
void inclusive_scan_brent_kernel(const float* input, float* output, int N){

    // 1:2 mapping of input to shared memory
    __shared__ float temp[blockSize * 2];
    int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    int j = i + 1;

    if(i < N){
        temp[threadIdx.x * 2] = input[i];
    }
    if(j < N){
        temp[threadIdx.x * 2 + 1] = input[j];
    }
    __syncthreads();
    // pre scan
    for(int stride = 1; stride < 2 * blockSize; stride <<= 1){
        int idx = (threadIdx.x+1)*stride*2 - 1;
        if(idx - stride >= 0 && idx < blockSize * 2){
            temp[idx] += temp[idx - stride];
        }
        __syncthreads();
    }
    // post scan
    for(int stride =  (2 * blockSize) / 4; stride > 0; stride >>= 1){
        int idx = (threadIdx.x + 1) * stride * 2 - 1;
        if(idx + stride < blockSize * 2){
            temp[idx + stride] += temp[idx];
        }
        __syncthreads();
    }
    
    __syncthreads();
    if(i < N){
        output[i] = temp[threadIdx.x * 2];
    }
    if(j < N){
        output[j] = temp[threadIdx.x * 2 + 1];
    }
}

__global__
void collect_last_brent_kernel(const float* output, float* last_list, int N, int num_blocks) {

    // global index for last element of the block
    int idx = (blockIdx.x + 1)* blockDim.x * 2 - 1;

    if (threadIdx.x == blockDim.x - 1 && idx < N) {
        last_list[blockIdx.x] = output[idx];
    }
}

__global__
void add_last_brent_kernel(float* output, const float* last_list, int N, int num_blocks){

    int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    int j = i + 1;

    float temp1 = output[i];
    float temp2 = output[j];
    
    if (blockIdx.x > 0){
        for(int idx = 0; idx < blockIdx.x; ++idx){
            temp1 += last_list[idx];
            temp2 += last_list[idx];
        }
    }

    if (i < N){
        output[i] = temp1;
    }
    if (j < N){
        output[j] = temp2;
    }
}

void inclusive_scan_brent_gpu(const float* h_input, float* h_output, int N){

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float *d_input, *d_output, *d_last_list;
    int num_blocks = (N + blockSize - 1) / (blockSize * 2);

    cudaMalloc(&d_input, sizeof(float) * N);
    cudaMalloc(&d_output, sizeof(float) * N);
    cudaMalloc(&d_last_list, sizeof(float) * num_blocks);


    cudaMemcpy(d_input, h_input, sizeof(float) * N, cudaMemcpyHostToDevice);

    dim3 blockDim(blockSize, 1, 1);
    dim3 gridDim(num_blocks, 1, 1);
    
    cudaEventRecord(start);
    inclusive_scan_brent_kernel<<<gridDim, blockDim>>>(d_input, d_output, N);
    collect_last_brent_kernel<<<gridDim, blockDim>>>(d_output, d_last_list, N, num_blocks);
    add_last_brent_kernel<<<gridDim, blockDim>>>(d_output, d_last_list, N, num_blocks);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);


    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cout << "GPU (brent) time: " << ms << " ms\n";

    cudaMemcpy(h_output, d_output, sizeof(float) * N, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_last_list);

}