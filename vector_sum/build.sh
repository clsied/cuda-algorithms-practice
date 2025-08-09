#!/bin/bash

OUTPUT="analysis.exe"
SRC="main.cpp vector_sum_cpu.cpp vector_sum_gpu.cu"
FLAGS="-O2 -std=c++17"

nvcc $FLAGS -o $OUTPUT $SRC