nvcc -O2 -std=c++17 -I"C:\Program Files\opencv\build\include" ^
    -L"C:\Program Files\opencv\build\x64\vc16\lib" ^
    -lopencv_world4120 ^
    -o analysis main.cpp goodFeaturesToTrack_gpu.cu calcOpticalFlowPyrLK_gpu.cu