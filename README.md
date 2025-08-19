# CUDA Algorithms Practice

A collection of CUDA implementations covering fundamental parallel primitives, sparse matrix formats, and application-level algorithms (e.g., Lucas-Kanade Optical Flow).  

This repository serves as my learning record and experiments in GPU programming.

### Basic Primitives
- [x] Vector Addition
- [x] Vector Reduction (sum)
- [x] Inclusive Scan
  - [x] Kogge-Stone
  - [x] Brent-Kung

### Matrix Operations
- [x] Matrix Multiplication
  - [x] Naive
  - [x] Tiled (shared memory)
- [x] 2D Convolution
  - [x] Naive
  - [x] Tiled (shared memory)

### Sparse Matrix-Vector Multiplication (SpMV)
- [x] COO format
- [x] CSR format
- [x] ELL format

### Applications
- [x] Lucas-Kanade Optical Flow (in progress)
  - [x] goodFeaturesToTrack
    - [x] Harris Corner (shared memory)
    - [x] Vector Reduction (max)
    - [x] Thresholding
    - [x] Non-Maximum Suppression (on cpu)
  - [x] calcOpticalFlowPyrLK
    - [x] Image Pyramids (With Gaussian Blurring)
    - [x] Lucas-Kanade (with Iterative refinement)

## Benchmarks

All experiments were tested on **Intel Core i5-12400 (CPU)** and **RTX 3060 Ti (GPU)**.  
Reported numbers are average **kernel execution times** (ms), excluding host-device memory transfers.

GPU speedup is computed against the single-thread CPU baseline.

---

### Vector Addition (1M elements)

| Version     | Runtime (ms) | Speedup (×) |
|-------------|--------------|-------------|
| CPU         | 0.3027         | 1× |
| GPU | 0.1216         | 2.49× |
---

### Reduction Sum (1M elements)

| Version     | Runtime (ms) | Speedup (×) |
|-------------|--------------|-------------|
| CPU         | 0.5085         | 1× |
| GPU (tiled) | 0.1682 | 3.02× |

---

### Inclusive Scan (1M elements)

| Version             | Runtime (ms) | Speedup (×) |
|---------------------|--------------|-------------|
| CPU                 | 1.2482         | 1× |
| GPU (Kogge-Stone)   | 0.5408         | 2.31× |
| GPU (Brent-Kung)    | 0.4738         | 2.63× |

---

### Matrix Multiplication (512×512)

| Version     | Runtime (ms) | Speedup (×) |
|-------------|--------------|-------------|
| CPU         | 72.9396        | 1× |
| GPU (naive) | 0.5195         | 140.40× |
| GPU (tiled) | 0.3359          | 217.15× |

---

### 2D Convolution (512×512, 3×3)

| Version     | Runtime (ms) | Speedup (×) |
|-------------|--------------|-------------|
| CPU         | 3.1210        | 1× |
| GPU (naive) | 0.4004         | 7.79× |
| GPU (tiled) | 0.1075          | 29.03× |

---

### SpMV (512 x 512, 512 x 1)

| Version     | Runtime (ms) | Speedup (×) |
|-------------|--------------|-------------|
| CPU         | 0.7538         | 1× |
| GPU (COO)   | 0.3615          | 2.09× |
| GPU (CSR)   | 0.2765          | 2.73× |
| GPU (ELL)   | 0.2847          | 2.65× |

---
### Lucas-Kanade Optical Flow (MPI Sintel Dataset)

| Version     | Runtime (ms) | Speedup (×) |
|-------------|--------------|-------------|
| CPU (goodFeaturesToTrack)  | 5.6301          | 1× |
| GPU (goodFeaturesToTrack)  | 2.6394          | 2.13× |
| CPU (calcOpticalFlowPyrLK)  | 1.3990          | 1× |
| GPU (calcOpticalFlowPyrLK)   | 0.5485          | 2.55× |
## Environment / Dependencies

- OS: Windows 11
- GPU: NVIDIA GeForce RTX 3060 Ti (8GB)
- CUDA Toolkit: 12.5
- Compiler: MSVC (via Visual Studio 2022), C++17 standard
- OpenCV: 4.12 (for image processing and Optical Flow experiments)

Optional:
- CMake (for larger multi-file projects, not required for single-file `.cu`)

## Dataset / Citation

Optical Flow experiments use the [**MPI Sintel Dataset**](http://sintel.is.tue.mpg.de/downloads)

- Butler, D. J., Wulff, J., Stanley, G. B., & Black, M. J. (2012). 
*A naturalistic open source movie for optical flow evaluation*. 
European Conference on Computer Vision (ECCV), 611–625.