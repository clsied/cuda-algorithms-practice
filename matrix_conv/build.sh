#!/bin/bash

OUTPUT="analysis.exe"
SRC="main.cpp matrix_conv_cpu.cpp matrix_conv_naive_gpu.cu matrix_conv_tiled_gpu.cu"
FLAGS="-O2 -std=c++17"

nvcc $FLAGS -o $OUTPUT $SRC