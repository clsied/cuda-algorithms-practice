#!/bin/bash

OUTPUT="analysis.exe"
SRC="main.cpp vector_add_cpu.cpp vector_add_gpu.cu"
FLAGS="-O2 -std=c++17"

nvcc $FLAGS -o $OUTPUT $SRC