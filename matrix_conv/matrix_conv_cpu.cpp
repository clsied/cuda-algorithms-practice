void matrix_conv_cpu(const float* input, const float* kernel, float* output, int H, int W, int KH, int KW){
    
    // flooring
    int padh = KH / 2;
    int padw = KW / 2;
    
    float sum; int ii, jj;

    for(int i = 0; i < H; ++i){
        for(int j = 0; j < W; ++j){

                sum = 0.0f;
                for (int ki = 0; ki < KH; ++ki) {
                    for (int kj = 0; kj < KW; ++kj) {
                        // i & j as center
                        int ii = i + (ki - padh);
                        int jj = j + (kj - padw);
                        if (ii >= 0 && ii < H && jj >= 0 && jj < W) {
                            sum += input[ii * W + jj] * kernel[ki * KW + kj];
                        }
                }
                output[i * W + j] = sum;
            }
        }
    }
}