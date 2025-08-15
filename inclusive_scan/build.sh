#!/bin/bash

OUTPUT="analysis.exe"
SRC="main.cpp inclusive_scan_cpu.cpp inclusive_scan_kogge_gpu.cu inclusive_scan_brent_gpu.cu"
FLAGS="-O2 -std=c++17"

nvcc $FLAGS -o $OUTPUT $SRC