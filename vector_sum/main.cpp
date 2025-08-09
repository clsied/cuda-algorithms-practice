#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
using namespace std;

float vector_sum_cpu(const float* A, int N);
float vector_sum_gpu(const float* A, int N);

bool is_close(float a, float b, float tol = 1e-5f) {
    return fabs(a - b) < tol;
}

int main() {
    int N = 1 << 20; // 1M elements
    vector<float> data(N);

    for (int i = 0; i < N; ++i) {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // CPU version
    auto t1 = chrono::high_resolution_clock::now();
    float sum_cpu = vector_sum_cpu(data.data(), N);
    auto t2 = chrono::high_resolution_clock::now();
    cout << "CPU time: "
              << chrono::duration<float, milli>(t2 - t1).count()
              << " ms\n";

    // GPU version
    float sum_gpu = vector_sum_gpu(data.data(), N);

    float error = 0;
    cout << "Total mismatches : " << abs(sum_cpu - sum_gpu) << "\n";
    return 0;
}
