#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#define KERNEL_SIZE 5
#define BLOCK_SIZE 16
#define MAX_WIN_SIZE 15
__constant__ float kernel[5 * 5];

__global__
void gaussian_subsample_kernel(const uchar* input, uchar* output, int currH, int currW) {
    int padh = KERNEL_SIZE / 2;
    int padw = KERNEL_SIZE / 2;

    // shared memory with halo
    __shared__ uchar tile[BLOCK_SIZE + KERNEL_SIZE - 1][BLOCK_SIZE + KERNEL_SIZE - 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global coordinates for shared memory loading
    int row = (blockIdx.y * BLOCK_SIZE + ty) - padh;
    int col = (blockIdx.x * BLOCK_SIZE + tx) - padw;

    if (row >= 0 && row < currH && col >= 0 && col < currW) {
        tile[ty][tx] = input[row * currW + col];
    } 
    else {
        if (row < 0) {
            row = 0;
        }
        if (row >= currH){
            row = currH - 1;
        }
        if (col < 0) {
            col = 0;
        }
        if (col >= currW) {
            col = currW - 1;
        }
        tile[ty][tx] = input[row * currW + col];
    }

    __syncthreads();

    // Compute Gaussian blur at this pixel
    int out_x = blockIdx.x * BLOCK_SIZE + tx;
    int out_y = blockIdx.y * BLOCK_SIZE + ty;

    if (tx < BLOCK_SIZE && ty < BLOCK_SIZE && out_x < currW && out_y < currH) {
        float sum = 0.0f;
        for (int ki = 0; ki < KERNEL_SIZE; ++ki) {
            for (int kj = 0; kj < KERNEL_SIZE; ++kj) {
                int sm_y = ty + ki;
                int sm_x = tx + kj;
                sum += static_cast<float>(tile[sm_y][sm_x]) * kernel[ki * KERNEL_SIZE + kj];
            }
        }

        // Subsample
        if (out_x % 2 == 0 && out_y % 2 == 0) {
            int new_x = out_x / 2;
            int new_y = out_y / 2;

            sum = fminf(fmaxf(sum  + 0.5f, 0.0f), 255.0f);  // clamp
            output[new_y * (currW / 2) + new_x] = static_cast<uchar>(sum);
        }
    }
}

__global__
void lucas_kanade_kernel(
    const uchar* prev_img, const uchar* next_img, const float* prev_pts, float* next_pts,
    uchar* status, int currH, int currW, int winH, int winW, int max_iter, float eps,
    float minEigThreshold, int curr_level, float* u, float* v)
{

    uchar tile[(MAX_WIN_SIZE + 2) * (MAX_WIN_SIZE + 2)];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (status[idx] == 0){
        return;
    }

    int tx = (int)prev_pts[idx * 2 + 0];
    int ty = (int)prev_pts[idx * 2 + 1];

    int scale = 1 << curr_level;
    tx /= scale;
    ty /= scale;

    const int halfH = winH / 2;
    const int halfW = winW / 2;

    // load with mirroring
    for (int i = 0; i < winH + 2; ++i) {
        for (int j = 0; j < winW + 2; ++j) {
            int gx = tx - (halfW + 1) + j;
            int gy = ty - (halfH + 1) + i;
            if (gx < 0) {gx = -gx;} else if (gx >= currW) {gx = 2 * currW - gx - 2;}
            if (gy < 0) {gy = -gy;} else if (gy >= currH) {gy = 2 * currH - gy - 2;}
            tile[i * (winW + 2) + j] = prev_img[gy * currW + gx];
        }
    }

    float sumIx2 = 0.f, sumIy2 = 0.f, sumIxIy = 0.f;
    for (int dy = -halfH; dy <= halfH; ++dy) {
        for (int dx = -halfW; dx <= halfW; ++dx) {
            float gx = 0.5f * (
                tile[(halfH + 1 + dy) * (winW + 2) + (halfW + 1 + dx + 1)] -
                tile[(halfH + 1 + dy) * (winW + 2) + (halfW + 1 + dx - 1)]
            );
            float gy = 0.5f * (
                tile[(halfH + 1 + dy + 1) * (winW + 2) + (halfW + 1 + dx)] -
                tile[(halfH + 1 + dy - 1) * (winW + 2) + (halfW + 1 + dx)]
            );
            sumIx2 += gx * gx;
            sumIy2 += gy * gy;
            sumIxIy += gx * gy;
        }
    }

    {
        float trace = sumIx2 + sumIy2;
        float tmp   = sqrtf((sumIx2 - sumIy2)*(sumIx2 - sumIy2) + 4.f*sumIxIy*sumIxIy);
        float lambda1 = 0.5f * (trace + tmp);
        float lambda2 = 0.5f * (trace - tmp);
        float minlambda = fminf(lambda1, lambda2);
        if (minlambda < minEigThreshold) { status[idx] = 0; return; }
    }

    float det = sumIx2 * sumIy2 - sumIxIy * sumIxIy;
    if (det < 1e-6f) { status[idx] = 0; return; }
    float inv_det = 1.0f / det;

    float curr_u = u[idx] / scale;
    float curr_v = v[idx] / scale;

    for (int iter = 0; iter < max_iter; ++iter) {
        float sumIxIt = 0.f, sumIyIt = 0.f;

        for (int dy = -halfH; dy <= halfH; ++dy) {
            for (int dx = -halfW; dx <= halfW; ++dx) {
                float gx = 0.5f * (
                    tile[(halfH + 1 + dy) * (winW + 2) + (halfW + 1 + dx + 1)] -
                    tile[(halfH + 1 + dy) * (winW + 2) + (halfW + 1 + dx - 1)]
                );
                float gy = 0.5f * (
                    tile[(halfH + 1 + dy + 1) * (winW + 2) + (halfW + 1 + dx)] -
                    tile[(halfH + 1 + dy - 1) * (winW + 2) + (halfW + 1 + dx)]
                );

                int px = max(0, min(tx + dx, currW - 1));
                int py = max(0, min(ty + dy, currH - 1));
                float I1p = prev_img[py * currW + px];

                // neigherest neighbor
                int qx = (int)floorf(tx + dx + curr_u + 0.5f);
                int qy = (int)floorf(ty + dy + curr_v + 0.5f);
                qx = max(0, min(qx, currW - 1));
                qy = max(0, min(qy, currH - 1));
                float I2p = next_img[qy * currW + qx];

                float It = I2p - I1p;
                sumIxIt += gx * It;
                sumIyIt += gy * It;
            }
        }

        float du = (-sumIy2 * sumIxIt + sumIxIy * sumIyIt) * inv_det;
        float dv = ( sumIxIy * sumIxIt - sumIx2 * sumIyIt) * inv_det;

        curr_u += du;
        curr_v += dv;

        if (fabsf(du) < eps && fabsf(dv) < eps) {
            break;
        }
    }

    next_pts[idx * 2 + 0] = (tx + curr_u) * scale;
    next_pts[idx * 2 + 1] = (ty + curr_v) * scale;
    u[idx] = curr_u * scale;
    v[idx] = curr_v * scale;
    status[idx] = 1;
}

void calcOpticalFlowPyrLK_gpu(cv::InputArray prevImg, cv::InputArray nextImg, cv::InputArray prevPts,
    cv::InputOutputArray nextPts, cv::OutputArray status, cv::OutputArray err, cv::Size winSize = cv::Size(21, 21), int maxLevel = 3, 
    cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
    int flags = 0, double minEigThreshold = 1e-4)
    {

        cv::Mat prev_img = prevImg.getMat();
        cv::Mat next_img = nextImg.getMat();

        int W = prev_img.cols;
        int H = prev_img.rows;
        int N = W * H;

        // 1. Image pryamid
        float h_kernel[5*5] = {
            1/256.f,  4/256.f,  6/256.f,  4/256.f, 1/256.f,
            4/256.f, 16/256.f, 24/256.f, 16/256.f, 4/256.f,
            6/256.f, 24/256.f, 36/256.f, 24/256.f, 6/256.f,
            4/256.f, 16/256.f, 24/256.f, 16/256.f, 4/256.f,
            1/256.f,  4/256.f,  6/256.f,  4/256.f, 1/256.f
        };

        cudaMemcpyToSymbol(kernel, h_kernel, sizeof(float) * 25);

        std::vector<uchar*> d_prev_img(maxLevel);
        std::vector<uchar*> d_next_img(maxLevel);

        cudaMalloc(&d_prev_img[0], sizeof(uchar) * N);
        cudaMalloc(&d_next_img[0], sizeof(uchar) * N);
        
        cudaMemcpy(d_prev_img[0], prev_img.ptr<uchar>(), sizeof(uchar) * N, cudaMemcpyHostToDevice);
        cudaMemcpy(d_next_img[0], next_img.ptr<uchar>(), sizeof(uchar) * N, cudaMemcpyHostToDevice);

        for(int i = 1; i < maxLevel; ++i){

            int prevH = H >> (i-1);
            int prevW = W >> (i-1);
            int curH = H >> i;
            int curW = W >> i;
            int curN = curH * curW;

            cudaMalloc(&d_prev_img[i], sizeof(uchar) * curN);
            cudaMalloc(&d_next_img[i], sizeof(uchar) * curN);

            dim3 gridDim((prevW + KERNEL_SIZE +BLOCK_SIZE - 1)/BLOCK_SIZE, (prevH + BLOCK_SIZE - 1)/BLOCK_SIZE);
            dim3 blockDim(KERNEL_SIZE + BLOCK_SIZE - 1, KERNEL_SIZE + BLOCK_SIZE - 1);
            
            gaussian_subsample_kernel<<<gridDim, blockDim>>>(d_prev_img[i-1], d_prev_img[i], prevH, prevW);
            cudaDeviceSynchronize();

            gaussian_subsample_kernel<<<gridDim, blockDim>>>(d_next_img[i-1], d_next_img[i], prevH, prevW);
            cudaDeviceSynchronize();

        }

        // 2. Lucas Kanade Optical Flow
        cv::Mat prevPtsMat = prevPts.getMat();
        std::vector<cv::Point2f> prevPtsVec ;
        cv::Mat reshaped = prevPtsMat.reshape(2, prevPtsMat.total());
        for (int i = 0; i < reshaped.rows; i++) {
            cv::Vec2f v = reshaped.at<cv::Vec2f>(i, 0);
            prevPtsVec.push_back(cv::Point2f(v[0], v[1]));
        }

        int N_pts = prevPtsVec.size();
        float *d_prevPts, *d_nextPts;
        float *u, *v;
        uchar* d_status;

        status.create(N_pts, 1, CV_8U);
        cv::Mat statusMat = status.getMat();

        cudaMalloc(&d_prevPts, sizeof(float) * N_pts * 2);
        cudaMalloc(&d_nextPts, sizeof(float) * N_pts * 2);
        cudaMalloc(&u, sizeof(float) * N_pts);
        cudaMalloc(&v, sizeof(float) * N_pts);
        cudaMalloc(&d_status, sizeof(uchar) * N_pts);

        cudaMemcpy(d_prevPts, prevPtsMat.ptr<float>(), sizeof(float) * N_pts * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_nextPts, prevPtsMat.ptr<float>(), sizeof(float) * N_pts * 2, cudaMemcpyHostToDevice);

        cudaMemset(u, 0, sizeof(float) * N_pts);
        cudaMemset(v, 0, sizeof(float) * N_pts);
        cudaMemset(d_status, 1, N_pts * sizeof(uchar));
        
        int blockSize = 256;
        int gridSize = (N_pts + blockSize - 1) / blockSize;
        
        int winH = winSize.height;
        int winW = winSize.width;

        int max_iter = criteria.maxCount;
        float eps = criteria.epsilon;


        for(int i = maxLevel - 1; i >= 0; --i){
            lucas_kanade_kernel<<< (N_pts + 255) / 256, 256>>>(d_prev_img[i], d_next_img[i], d_prevPts, d_nextPts, d_status, H >> i, W >> i, winH, winW, max_iter, eps, minEigThreshold, i, u, v);
            cudaDeviceSynchronize();
        }

        cudaMemcpy(status.getMat().ptr<uchar>(), d_status, N_pts*sizeof(uchar), cudaMemcpyDeviceToHost);
        
        std::vector<float> h_nextPts(N_pts * 2);
        cudaMemcpy(h_nextPts.data(), d_nextPts, N_pts * 2 * sizeof(float), cudaMemcpyDeviceToHost);

        cv::Mat nextPtsMat(N_pts, 1, CV_32FC2);
        for (int i = 0; i < N_pts; i++) {
            nextPtsMat.at<cv::Point2f>(i, 0) = cv::Point2f(h_nextPts[i * 2], h_nextPts[i * 2 + 1]);
        }

        nextPtsMat.copyTo(nextPts);

        for (int i = 0; i < maxLevel; i++) {
            cudaFree(d_prev_img[i]);
            cudaFree(d_next_img[i]);
        }

        cudaFree(d_prevPts);
        cudaFree(d_nextPts);
        cudaFree(u);
        cudaFree(v);
        cudaFree(d_status);
}