#include <iostream>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define KERNEL_SIZE 3 // current kernel size 3

using namespace std;

// load into constant memory
__constant__ float const_kernel[KERNEL_SIZE * KERNEL_SIZE]; 

__global__
void matrix_conv_tiled_kernel(const float* input, const float* /*kernel*/, float* output, int H, int W, int KH, int KW){
    int padh = KH / 2;
    int padw = KW / 2;

    // consider the halo
    // share memory should with fixed size when compiling
    __shared__ float tile[TILE_SIZE + KERNEL_SIZE - 1][TILE_SIZE + KERNEL_SIZE - 1];
    
    // the size that actually used
    int sharedw = TILE_SIZE + KW - 1;
    int sharedh = TILE_SIZE + KH - 1;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global coordinates for shared memory loading
    int row = (blockIdx.y * TILE_SIZE + ty) - padh;
    int col = (blockIdx.x * TILE_SIZE + tx) - padw;

    // Load input patch to shared memory (with halo)
    if (ty < sharedh && tx < sharedw) {
        if (row >= 0 && row < H && col >= 0 && col < W){
            tile[ty][tx] = input[row * W + col];
        }
        else{
            tile[ty][tx] = 0.0f;
        }
    }

    __syncthreads();

    // global coordinate for output matrix
    float sum;
    int out_x = blockIdx.x * TILE_SIZE + tx;
    int out_y = blockIdx.y * TILE_SIZE + ty;

    // here tx & ty are mapped to the output matrix 1 to 1
    if (tx < TILE_SIZE && ty < TILE_SIZE && out_x < W && out_y < H) {

        sum = 0.0f;
        for (int ki = 0; ki < KH; ++ki) {
            for (int kj = 0; kj < KW; ++kj) {
                sum += tile[ty + ki][tx + kj] * const_kernel[ki * KW + kj];
            }
        }

        output[out_y * W + out_x] = sum;
    }
}

void matrix_conv_tiled_gpu(const float* h_input, const float* h_kernel, float* h_output, int H, int W, int KH, int KW){

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *d_input , *d_kernel, *d_output;
    cudaMalloc(&d_input, sizeof(float) * H * W);
    cudaMalloc(&d_kernel, sizeof(float) * KH* KW);
    cudaMalloc(&d_output, sizeof(float) * H * W);

    cudaMemcpy(d_input, h_input, sizeof(float) * H * W, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(const_kernel, h_kernel, KH * KW * sizeof(float));
    
    dim3 blockDim(TILE_SIZE + KERNEL_SIZE - 1, TILE_SIZE + KERNEL_SIZE - 1);
    dim3 gridDim((W + TILE_SIZE - 1)/TILE_SIZE, (H + TILE_SIZE - 1)/TILE_SIZE);

    cudaEventRecord(start);
    matrix_conv_tiled_kernel<<<gridDim, blockDim>>>(d_input, d_kernel, d_output, H, W, KH, KW);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cout << "GPU (tiled) time: " << ms << " ms\n";

    cudaMemcpy(h_output, d_output, sizeof(float) * H * W, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
