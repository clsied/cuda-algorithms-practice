#include <vector>
#include <random>
#include <iostream>
#include <algorithm>
#include <cassert>
using namespace std;

struct COO {
    vector<int> row;
    vector<int> col;
    vector<float> val;
    int rows = 0;
    int cols = 0;
};

struct CSR {
    vector<int> row_ptr;
    vector<int> col_idx;
    vector<float> val;
    int rows = 0;
    int cols = 0;
};

struct ELL {
    vector<int> col_indices;    // column indices
    vector<float> vals; // values (linearized matrix)
    int max_non_zero = 0;    // max non-zero per row
    int rows = 0;
    int cols = 0;
};


// dense -> COO
inline COO dense_to_coo(const float* dense, int H, int W) {
    COO coo;
    coo.rows = H;
    coo.cols = W;
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            float val = dense[i * W + j];
            if (val != 0.0f) {
                coo.row.push_back(i);
                coo.col.push_back(j);
                coo.val.push_back(val);
            }
        }
    }
    return coo;
}

// dense -> CSR
inline CSR dense_to_csr(const float* dense, int H, int W) {

    CSR csr;
    csr.rows = H;
    csr.cols = W;

    // intialize CSR row pointers
    csr.row_ptr.push_back(0);
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            float val = dense[i * W + j];
            if (val != 0.0f) {
                csr.col_idx.push_back(j);
                csr.val.push_back(val);
            }
        }
        csr.row_ptr.push_back(static_cast<int>(csr.col_idx.size()));
    }

    return csr;
}

// dense -> ELL
inline ELL dense_to_ell(const float* dense, int H, int W) {

    ELL ell;
    int max_non_zero = 0;
    ell.rows = H;
    ell.cols = W;

    for (int i = 0; i < H; ++i) {
        int non_zeros = 0;
        for (int j = 0; j < W; ++j) {
            if (dense[i * W + j] != 0.0f) {
                ++non_zeros;
            }
        }
        max_non_zero = max(max_non_zero, non_zeros);
    }
    ell.max_non_zero = max_non_zero;

    // intialize ELL values
    ell.col_indices.resize(H * max_non_zero, -1);
    ell.vals.resize(H * max_non_zero, 0.0f);

    for (int i = 0; i < H; ++i) {
        int count = 0;
        for (int j = 0; j < W; ++j) {
            float val = dense[i * W + j];
            if (val != 0.0f) {
                ell.col_indices[i * max_non_zero + count] = j;
                ell.vals[i * max_non_zero + count] = val;
                ++count;
            }
        }
    }

    return ell;
}