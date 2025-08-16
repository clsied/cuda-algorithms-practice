#!/bin/bash

OUTPUT="analysis.exe"
SRC="main.cpp spmv_coo_cpu.cpp spmv_coo_gpu.cu spmv_csr_gpu.cu spmv_ell_gpu.cu"
FLAGS="-O2 -std=c++17"

nvcc $FLAGS -o $OUTPUT $SRC