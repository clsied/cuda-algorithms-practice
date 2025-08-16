#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <random>
#include "dense_to_sparse.h"
using namespace std;

// M x N & N x 1 matrix multiplication
void spmv_coo_cpu(const float* A, const float* B, float* C, int M, int K);
void spmv_coo_gpu(const float* h_A, const float* h_B, float* h_C, int M, int K);
void spmv_csr_gpu(const float* h_A, const float* h_B, float* h_C, int M, int K);
void spmv_ell_gpu(const float* h_A, const float* h_B, float* h_C, int M, int K);

bool is_close(float a, float b, float tol = 1e-4f) {
    return fabs(a - b) < tol;
}


int main() {
    const int M = 512, K = 512, N = 1;

    vector<float> A(M * K), B(K);
    vector<float> C_cpu(M, 0.0f), C_gpu_coo(M, 0.0f), C_gpu_csr(M, 0.0f), C_gpu_ell(M, 0.0f);

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> prob(0.0f, 1.0f);
    uniform_real_distribution<float> val(1.0f, 10.0f);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            if (prob(gen) > 0.9) {
                A[i * K + j] = val(gen);
            }
        }
    }

    for (int i = 0; i < B.size(); ++i) B[i] = static_cast<float>(rand()) / RAND_MAX;

    auto t1 = chrono::high_resolution_clock::now();
    spmv_coo_cpu(A.data(), B.data(), C_cpu.data(), M, K);
    auto t2 = chrono::high_resolution_clock::now();
    cout << "CPU time: " << chrono::duration<float, milli>(t2 - t1).count() << " ms\n";

    spmv_coo_gpu(A.data(), B.data(), C_gpu_coo.data(), M, K);
    spmv_csr_gpu(A.data(), B.data(), C_gpu_csr.data(), M, K);
    spmv_ell_gpu(A.data(), B.data(), C_gpu_ell.data(), M, K);


    float errors = 0;
    for (int i = 0; i < M * N; ++i) {
        if (!is_close(C_cpu[i], C_gpu_coo[i])) {
            if (errors < 10)
                cout << "Mismatch at " << i << ": CPU=" << C_cpu[i] << " GPU (COO)=" << C_gpu_coo[i] << "\n";
            errors++;
        }
    }
    cout << "Total mismatches (COO): " << errors << "\n";

    errors = 0;
    for (int i = 0; i < M * N; ++i) {
        if (!is_close(C_cpu[i], C_gpu_csr[i])) {
            if (errors < 10)
                cout << "Mismatch at " << i << ": CPU=" << C_cpu[i] << " GPU (CSR)=" << C_gpu_csr[i] << "\n";
            errors++;
        }
    }
    cout << "Total mismatches (CSR): " << errors << "\n";

        errors = 0;
    for (int i = 0; i < M * N; ++i) {
        if (!is_close(C_cpu[i], C_gpu_ell[i])) {
            if (errors < 10)
                cout << "Mismatch at " << i << ": CPU=" << C_cpu[i] << " GPU (ELL)=" << C_gpu_ell[i] << "\n";
            errors++;
        }
    }
    cout << "Total mismatches (ELL): " << errors << "\n";
    return 0;
}