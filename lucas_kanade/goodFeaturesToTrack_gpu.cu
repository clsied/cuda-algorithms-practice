#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <vector>
#include <algorithm>
#define BLOCK_SIZE 16
#define MAX_WIN_SIZE 7

__global__
void corner_response_kernel(const uchar* d_img, int W, int H, float* d_response,int blockSize, bool useHarris, float k){
    __shared__ uchar tile[BLOCK_SIZE + MAX_WIN_SIZE + 2][BLOCK_SIZE + MAX_WIN_SIZE + 2];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    
    int half = blockSize / 2;
    
    for(int shared_y = ty; shared_y < BLOCK_SIZE + blockSize + 2; shared_y += blockDim.y){
        for(int shared_x = tx; shared_x < BLOCK_SIZE + blockSize + 2; shared_x += blockDim.x){
            
            int global_x = blockIdx.x * BLOCK_SIZE + shared_x - (half + 1);
            int global_y = blockIdx.y * BLOCK_SIZE + shared_y - (half + 1);
            
            // mirror the coordinates
            if (global_x < 0) {
                global_x = -global_x;
            } 
            else if (global_x >= W) {
                global_x = 2 * W - global_x - 2;
            }
            if (global_y < 0) {
                global_y = -global_y;
            } 
            else if (global_y >= H) {
                global_y = 2 * H - global_y - 2;
            }

            tile[shared_y][shared_x] = d_img[global_y * W + global_x];
        }
    }
    __syncthreads();

    if(x < W && y < H) {
        float sumIx2= 0.f, sumIy2=0.f, sumIxIy=0.f;

        for(int dy = -half; dy <= half; ++dy){
            for(int dx = -half; dx <= half; ++dx){
                float gx = 0.5f * (tile[(ty + half + 1) + dy][(tx + half + 1) + dx + 1] - tile[(ty + half + 1) + dy][(tx + half + 1) + dx - 1]);
                float gy = 0.5f * (tile[(ty + half + 1) + dy + 1][(tx + half + 1) + dx] - tile[(ty + half + 1) + dy - 1][(tx + half + 1) + dx]);
                sumIx2 += gx * gx;
                sumIy2 += gy * gy;
                sumIxIy += gx * gy;
            }
        }

        float R = 0.f;
        if(useHarris){
            R = sumIx2 * sumIy2 - sumIxIy * sumIxIy - k * (sumIx2 + sumIy2) * (sumIx2 + sumIy2);
        } 
        else {
            float trace = sumIx2 + sumIy2;
            float temp = sqrtf((sumIx2 - sumIy2) * (sumIx2 - sumIy2) + 4.f * sumIxIy * sumIxIy);
            float lambda1 = 0.5f * (trace + temp);
            float lambda2 = 0.5f * (trace - temp);
            R = fminf(lambda1, lambda2);
        }
        d_response[y * W + x] = R;
    }
}

__global__
void vector_max_kernel(const float* A, float* blockMaxs, int N){
    __shared__ float partialMax[BLOCK_SIZE];
    unsigned int t = threadIdx.x;
    unsigned int start = blockIdx.x * (blockDim.x * 2);

    float temp = -FLT_MAX;
    if (start + t < N) temp = max(temp, A[start + t]);
    if (start + t + blockDim.x < N) temp = max(temp, A[start + blockDim.x + t]);

    partialMax[t] = temp;
    __syncthreads();

    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if(t < stride){
            partialMax[t] = max(partialMax[t], partialMax[t + stride]);
        }
        __syncthreads();
    }

    if (t == 0){
        blockMaxs[blockIdx.x] = partialMax[0];
    }
}

__global__ 
void threshold_kernel(float* d_response, int W, int H, float threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < W && y < H) {
        int idx = y * W + x;
        if (d_response[idx] < threshold) {
            d_response[idx] = 0.0f;
        }
    }
}

struct CompareResponse {
    const float* d_res_ptr;
    CompareResponse(const float* _ptr) : d_res_ptr(_ptr) {}
    __host__ __device__ 
    bool operator()(int a, int b) const {
        return d_res_ptr[a] > d_res_ptr[b];
    }
};

void goodFeaturesToTrack_gpu(cv::InputArray image, cv::OutputArray corners,
                             int maxCorners, double qualityLevel,
                             double minDistance, cv::InputArray mask = cv::noArray(),
                             int blockSize = 3, bool useHarrisDetector = false,
                             double k = 0.04) {

    cv::Mat img = image.getMat();

    int W = img.cols;
    int H = img.rows;
    int N = W * H;

    uchar *d_img;
    float *d_response;
    cudaMalloc(&d_img, sizeof(uchar) * N);
    cudaMalloc(&d_response, sizeof(float) * N);

    cudaMemcpy(d_img, img.ptr<uchar>(), sizeof(uchar) * N, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((W + BLOCK_SIZE - 1) / BLOCK_SIZE, (H + BLOCK_SIZE - 1) / BLOCK_SIZE);
    corner_response_kernel<<<gridDim, blockDim>>>(d_img, W, H, d_response, blockSize, useHarrisDetector, k);
    cudaDeviceSynchronize();

    // iterative max reduction
    float* d_buf1;
    float* d_buf2;
    cudaMalloc(&d_buf1, sizeof(float) * ((N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2)));
    cudaMalloc(&d_buf2, sizeof(float) * ((N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2)));

    float* d_in = d_response;
    float* d_out = nullptr;

    int n = N;
    while (n > 1) {
        int blocks = (n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
        if (d_in == d_response || d_in == d_buf2){
            d_out = d_buf1;
        }
        else{
            d_out = d_buf2;
        }

        vector_max_kernel<<<blocks, BLOCK_SIZE>>>(d_in, d_out, n);
        cudaDeviceSynchronize();

        d_in = d_out;
        n = blocks;
    }

    float Rmax;
    cudaMemcpy(&Rmax, d_in, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_buf1);
    cudaFree(d_buf2);

    // thresholding
    float threshold = static_cast<float>(qualityLevel * Rmax);
    threshold_kernel<<<gridDim, blockDim>>>(d_response, W, H, threshold);
    cudaDeviceSynchronize();

    // GPU sort using Thrust
    thrust::device_vector<int> d_indices(N);
    thrust::sequence(d_indices.begin(), d_indices.end());
    float* d_res_ptr = d_response;

    thrust::sort(d_indices.begin(), d_indices.end(), CompareResponse(d_res_ptr));
    cudaDeviceSynchronize();

    // copy sorted indices to CPU
    std::vector<int> h_indices(N);
    cudaMemcpy(h_indices.data(), thrust::raw_pointer_cast(d_indices.data()), N * sizeof(int), cudaMemcpyDeviceToHost);

    // CPU minDistance filtering
    struct Corner {
        float response; 
        int x, y; 
    };

    std::vector<Corner> candidates;
    for (int i = 0; i < N; ++i) {
        int idx = h_indices[i];
        float R = 0.0f;
        cudaMemcpy(&R, d_res_ptr + idx, sizeof(float), cudaMemcpyDeviceToHost);
        if (R <= 0) {
            continue;
        }
        candidates.push_back({R, idx % W, idx / W});
    }

    std::vector<Corner> finalCorners;
    for (const auto& c : candidates) {
        bool keep = true;
        for (const auto& s : finalCorners) {
            float dx = c.x - s.x;
            float dy = c.y - s.y;
            if (dx * dx + dy * dy < minDistance * minDistance) {
                keep = false;
                break;
            }
        }
        if (keep) {
            finalCorners.push_back(c);
            if (finalCorners.size() >= maxCorners) {
                break;
            }
        }
    }

    // write output
    cv::Mat cornersMat(finalCorners.size(), 1, CV_32FC2);
    for (size_t i = 0; i < finalCorners.size(); ++i) {
        cornersMat.at<cv::Point2f>(i, 0) = cv::Point2f(finalCorners[i].x, finalCorners[i].y);
    }
    cornersMat.copyTo(corners);

}
