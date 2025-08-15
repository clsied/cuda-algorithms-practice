void inclusive_scan_cpu(const float* input, float* output, int N){
    
    float partial_sum = 0.0f;
    
    for(int i = 0; i < N; i++){
        partial_sum += input[i];
        output[i] = partial_sum;
    }
}