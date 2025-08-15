#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
using namespace std;

void inclusive_scan_cpu(const float* input, float* output, int N);
void inclusive_scan_kogge_gpu(const float* input, float* output, int N);
void inclusive_scan_brent_gpu(const float* input, float* output, int N);

bool is_close(float a, float b, float tol = 1e-5f) {
    return fabs(a - b) < tol;
}


int main() {
    int N = 1 << 20; // 1M elements
    vector<float> data(N);
    vector<float> output_cpu(N);
    vector<float> kogge_gpu(N);
    vector<float> brent_gpu(N);


    for (int i = 0; i < N; ++i) {
        // monotonic increasing sequence to prevent different
        data[i] = static_cast<float>(1.);
        // data[i] = static_cast<float>(i);
    }

    // CPU version
    auto t1 = chrono::high_resolution_clock::now();
    inclusive_scan_cpu(data.data(), output_cpu.data(), N);
    auto t2 = chrono::high_resolution_clock::now();
    cout << "CPU time: "
              << chrono::duration<float, milli>(t2 - t1).count()
              << " ms\n";

    // GPU version
    inclusive_scan_kogge_gpu(data.data(), kogge_gpu.data(), N);
    inclusive_scan_brent_gpu(data.data(), brent_gpu.data(), N);

    int error = 0;
    for (int i = 0; i < N; ++i) {
        if (!is_close(output_cpu[i], kogge_gpu[i])) {
            error++;
            if (error < 10)
                cout << "Mismatch at " << i << ": CPU=" << output_cpu[i]
                          << " GPU=" << kogge_gpu[i] << "\n";
        }
    }
    cout << "Total mismatches (kogge): " << error << "\n";
    
    error = 0;
    for (int i = 0; i < N; ++i) {
        if (!is_close(output_cpu[i], brent_gpu[i])) {
            error++;
            if (error < 10)
                cout << "Mismatch at " << i << ": CPU=" << output_cpu[i]
                          << " GPU=" << brent_gpu[i] << "\n";
        }
    }
    cout << "Total mismatches (brent): " << error << "\n";
    return 0;
}
