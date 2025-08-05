#!/bin/bash

OUTPUT="analysis.exe"
SRC="main.cpp matrix_mul_cpu.cpp matrix_mul_naive_gpu.cu matrix_mul_tiled_gpu.cu"
FLAGS="-O2 -std=c++17"

nvcc $FLAGS -o $OUTPUT $SRC