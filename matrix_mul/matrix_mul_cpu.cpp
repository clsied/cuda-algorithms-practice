void matrix_mul_cpu(const float* A, const float* B, float* C, int M, int N, int K){

    float sum = 0.0f;
    
    for(int i = 0; i < M ; ++i){
        for(int j = 0; j < N; ++j){
            sum = 0.0f;
            for(int k = 0; k < K; ++k){
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}