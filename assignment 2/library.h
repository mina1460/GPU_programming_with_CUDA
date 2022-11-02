#ifndef library_H
#define library_H

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <assert.h>
#include <cuda_runtime.h>

#include "GPU_utils.h"
#include "time_helper.h"
#include "matrix.h"

#define A_ROWS 5
#define A_COLS 5
#define B_ROWS A_COLS
#define B_COLS 6
#define BLOCK_SIZE 2




template<typename T>
void multiply(const matrix<T>& A, const matrix<T>& B, matrix<T>& out_C){
    // multiplies A by B and stores it into C
    const int a_rows = A.get_rows();
    const int a_cols = A.get_columns();
    const int b_rows = B.get_rows();
    const int b_cols = B.get_columns();
    const int c_rows = out_C.get_rows();
    const int c_cols = out_C.get_columns();

    if(a_cols != b_rows || c_rows != a_rows || c_cols != b_cols)
        throw std::invalid_argument("Invalid matrix dimensions, cannot multiply\n");
    
    for(int i=0; i<a_rows; i++){
        for(int j=0; j<b_cols; j++){
            for(int k=0; k<b_rows; k++){
                out_C.data[i*c_cols + j] += A.data[i*a_cols + k] * B.data[k*b_cols + j];
            }
        }
    }
}

template<typename T>
bool compare_with_tolerance(const matrix<T>& A, const matrix<T>& B, float tolerance){
    // compares two matrices and returns true if they are within tolerance
    const int a_rows = A.get_rows();
    const int a_cols = A.get_columns();
    const int b_rows = B.get_rows();
    const int b_cols = B.get_columns();

    if(a_rows != b_rows || a_cols != b_cols)
        throw std::invalid_argument("Invalid matrix dimensions, cannot compare\n");
    
    for(int i=0; i<a_rows; i++){
        for(int j=0; j<a_cols; j++){
            if(abs(A.data[i*a_cols + j] - B.data[i*a_cols + j]) > tolerance)
                {
                    std::cout << "index: [" << i << "][" << j << "]  B: ["<< i << "]["  << j << "] " << "\n";
                    std::cout << "A: " << A.data[i*a_cols + j] << " B: " << B.data[i*a_cols + j] << "\n";
                    std::cout <<"Difference: "<< abs(A.data[i*a_cols + j] - B.data[i*a_cols + j]) << "\n";
                    return false; 
                }
        }
    }
    return true;
}

#endif