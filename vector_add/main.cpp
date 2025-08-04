#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <cmath>

void vector_add_cpu(const float* A, const float* B, float* C, int N);
void vector_add_gpu(const float* A, const float* B, float* C, int N);

bool is_close(float a, float b, float tol = 1e-5f) {
    return std::fabs(a - b) < tol;
}

int main() {
    const int N = 1 << 20; // 1M
    std::vector<float> A(N), B(N), C_cpu(N), C_gpu(N);

    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // CPU version
    auto t1 = std::chrono::high_resolution_clock::now();
    vector_add_cpu(A.data(), B.data(), C_cpu.data(), N);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "CPU time: "
              << std::chrono::duration<float, std::milli>(t2 - t1).count()
              << " ms\n";

    // GPU version
    vector_add_gpu(A.data(), B.data(), C_gpu.data(), N);

    // correctness check
    int error = 0;
    for (int i = 0; i < N; ++i) {
        if (!is_close(C_cpu[i], C_gpu[i])) {
            error++;
            if (error < 10)
                std::cout << "Mismatch at " << i << ": CPU=" << C_cpu[i]
                          << " GPU=" << C_gpu[i] << "\n";
        }
    }
    std::cout << "Total mismatches: " << error << "\n";
    return 0;
}
