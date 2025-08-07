#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <cmath>
using namespace std;

void matrix_conv_cpu(const float* input, const float* kernel, float* output, int H, int W, int KH, int KW);
void matrix_conv_naive_gpu(const float* input, const float* kernel, float* output, int H, int W, int KH, int KW);
void matrix_conv_tiled_gpu(const float* input, const float* kernel, float* output, int H, int W, int KH, int KW);

bool is_close(float a, float b, float tol = 1e-3f) {
    return fabs(a - b) < tol;
}

int main() {
    const int H = 512, W = 512;
    const int KH = 3, KW = 3;

    vector<float> input(H * W), kernel(KH * KW), output_cpu(H * W), output_gpu_naive(H * W), output_gpu_tiled(H * W);

    for (int i = 0; i < input.size(); ++i)
        input[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < kernel.size(); ++i)
        kernel[i] = static_cast<float>(rand()) / RAND_MAX;

    auto t1 = chrono::high_resolution_clock::now();
    matrix_conv_cpu(input.data(), kernel.data(), output_cpu.data(), H, W, KH, KW);
    auto t2 = chrono::high_resolution_clock::now();
    cout << "CPU time: " << chrono::duration<float, milli>(t2 - t1).count() << " ms\n";

    matrix_conv_naive_gpu(input.data(), kernel.data(), output_gpu_naive.data(), H, W, KH, KW);
    matrix_conv_tiled_gpu(input.data(), kernel.data(), output_gpu_tiled.data(), H, W, KH, KW);

    int errors = 0;
    for (int i = 0; i < H * W; ++i) {
        if (!is_close(output_cpu[i], output_gpu_naive[i])) {
            if (errors < 10)
                cout << "Mismatch at " << i << ": CPU=" << output_cpu[i] << "GPU (naive)=" << output_gpu_naive[i] << "\n";
            errors++;
        }
    }
    cout << "Total mismatches (naive): " << errors << "\n";

    errors = 0;
    for (int i = 0; i < H * W; ++i) {
        if (!is_close(output_cpu[i], output_gpu_tiled[i])) {
            if (errors < 10)
                cout << "Mismatch at " << i << ": CPU=" << output_cpu[i] << "GPU (tiled)=" << output_gpu_tiled[i] << "\n";
            errors++;
        }
    }
    cout << "Total mismatches (tiled): " << errors << "\n";
    return 0;
}
