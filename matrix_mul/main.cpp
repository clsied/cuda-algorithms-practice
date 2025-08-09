#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <cmath>
using namespace std;

// M x N & N x K matrix multiplication
void matrix_mul_cpu(const float* A, const float* B, float* C, int M, int N, int K);
void matrix_mul_naive_gpu(const float* A, const float* B, float* C, int M, int N, int K);
void matrix_mul_tiled_gpu(const float* A, const float* B, float* C, int M, int N, int K);


bool is_close(float a, float b, float tol = 1e-4f) {
    return fabs(a - b) < tol;
}

int main() {
    const int M = 512, K = 512, N = 512;

    vector<float> A(M * K), B(K * N), C_cpu(M * N), C_gpu_naive(M * N), C_gpu_tiled(M * N);

    for (int i = 0; i < A.size(); ++i) A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < B.size(); ++i) B[i] = static_cast<float>(rand()) / RAND_MAX;

    auto t1 = chrono::high_resolution_clock::now();
    matrix_mul_cpu(A.data(), B.data(), C_cpu.data(), M, N, K);
    auto t2 = chrono::high_resolution_clock::now();
    cout << "CPU time: " << chrono::duration<float, milli>(t2 - t1).count() << " ms\n";

    matrix_mul_naive_gpu(A.data(), B.data(), C_gpu_naive.data(), M, N, K);
    matrix_mul_tiled_gpu(A.data(), B.data(), C_gpu_tiled.data(), M, N, K);

    float errors = 0;
    for (int i = 0; i < M * N; ++i) {
        if (!is_close(C_cpu[i], C_gpu_naive[i])) {
            if (errors < 10)
                cout << "Mismatch at " << i << ": CPU=" << C_cpu[i] << " GPU (naive)=" << C_gpu_naive[i] << "\n";
            errors++;
        }
    }
    cout << "Total mismatches (naive): " << errors << "\n";

    errors = 0;
    for (int i = 0; i < M * N; ++i) {
        if (!is_close(C_cpu[i], C_gpu_tiled[i])) {
            if (errors < 10)
                cout << "Mismatch at " << i << ": CPU=" << C_cpu[i] << " GPU (tiled)=" << C_gpu_tiled[i] << "\n";
            errors++;
        }
    }
    cout << "Total mismatches (tiled): " << errors << "\n";
    return 0;
}