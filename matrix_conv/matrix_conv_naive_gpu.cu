#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__
void matrix_conv_naive_kernel(const float* input, const float* kernel, float* output, int H, int W, int KH, int KW){
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int padh = KH / 2;
    int padw = KW / 2;

    float sum = 0.0f ; int ii, jj;

    if (col < W && row < H) {
        for(int ki = 0; ki < KH; ++ki){
            for(int kj = 0; kj < KW; ++kj){
                ii = row + (ki - padh);
                jj = col + (kj - padw);

                if(ii >= 0 && ii < H && jj >= 0 && jj < W){
                    sum += input[ii * W + jj] * kernel[ki * KH + kj];
                }
            }
        }
        output[row * W + col] = sum;
    }

}


void matrix_conv_naive_gpu(const float* h_input, const float* h_kernel, float* h_output, int H, int W, int KH, int KW){
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *d_input, *d_kernel, *d_output;
    
    cudaMalloc(&d_input, sizeof(float) * H * W);
    cudaMalloc(&d_kernel, sizeof(float) * KH * KW);
    cudaMalloc(&d_output, sizeof(float) * H * W);

    cudaMemcpy(d_input, h_input, sizeof(float) * H * W, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, sizeof(float) * KH * KW, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((W + 15) / 16, (H + 15) / 16);

    cudaEventRecord(start);
    matrix_conv_naive_kernel<<<gridDim, blockDim>>>(d_input, d_kernel, d_output, H, W, KH, KW);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cout << "GPU (naive) time: " << ms << " ms\n";

    cudaMemcpy(h_output, d_output, sizeof(float) * H * W, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);


}
