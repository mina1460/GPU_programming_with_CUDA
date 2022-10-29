#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <fstream>

#include "time_helper.h"
#include "matrix.h"
#include "GPU_task1.h"

#define A_ROWS 1000
#define A_COLS 900
#define B_ROWS A_COLS
#define B_COLS 1200



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

bool compare_with_tolerance(const matrix<float>& A, const matrix<float>& B, float tolerance){
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
                    std::cout << "index: [" << i << "][" << j << "]  B: ["i << "]["  << j << "] " << "\n";
                    std::cout << "A: " << A.data[i*a_cols + j] << " B: " << B.data[i*a_cols + j] << "\n";
                    return false; 
                }
        }
    }
    return true;
}

int main(int argc, char** argv){
    
    std::cout << "Starting..." << std::endl;
    std::cout << "Dimensions of A: " << A_ROWS << "x" << A_COLS << std::endl;
    std::cout << "Dimensions of B: " << B_ROWS << "x" << B_COLS << std::endl;
   
    assert(A_COLS == B_ROWS && "matrix A columns must be equal to matrix B rows");
    
    // allocate memory for matrices
    matrix<float> A(A_ROWS, A_COLS, "A");
    matrix<float> B(B_ROWS, B_COLS, "B");
    matrix<float> C_GPU(A_ROWS, B_COLS, "C_GPU");
    matrix<float> C_CPU(A_ROWS, B_COLS, "C_CPU");
    
    // generate matrices
    A.auto_generate(250.0);
    B.auto_generate(400.0);

    // CPU matrix multiplication 
    std::cout << "Starting CPU matrix multiplication..." << std::endl;
    std::chrono::high_resolution_clock::time_point start = get_time();
    multiply(A, B, C_CPU);
    std::chrono::high_resolution_clock::time_point end = get_time();
    time_unit time_unit = nanoseconds;
    long long time_in_chosen_unit = get_time_diff(start, end, time_unit);
    
    std::cout << "CPU matrix multiplication took " << time_in_chosen_unit << " " + unit_name(time_unit) << std::endl;
    // get filename as a timestamp 
    std::string filename = "./results_" + get_timestamp() + ".txt";
    //write matrix to file
    A.write_to_file(filename);
    B.write_to_file(filename);
    C_CPU.write_to_file(filename);

    // report GFLOPS
    float gflops = get_GFLOPS(A, B, time_in_chosen_unit, time_unit);
    std::cout << "CPU GFLOPS: " << gflops << std::endl;


    // GPU matrix multiplication
    std::cout << "Starting GPU matrix multiplication..." << std::endl;
    
    start = get_time();
    int block_size = 32;
    long long time_in_kernel = GPU_matrix_multiplication(A, B, C_GPU, block_size);
    end = get_time();
    
    time_in_chosen_unit = get_time_diff(start, end, time_unit);
    
    std::cout << "GPU matrix multiplication took " << time_in_chosen_unit << " " + unit_name(time_unit) << std::endl;
    
    
    // write matrix to file
    C_GPU.write_to_file(filename);
    
    
    // report GFLOPS
    gflops = get_GFLOPS(A, B, time_in_kernel, nanoseconds);
    std::cout << "GPU GFLOPS of kernel only: " << gflops << std::endl;
    
    gflops = get_GFLOPS(A, B, time_in_chosen_unit, time_unit);
    std::cout << "GPU GFLOPS with copy: " << gflops << std::endl;

    // compare matrices
    std::cout << "Comparing matrices..." << std::endl;
    if(compare_with_tolerance(C_CPU, C_GPU, 0.0001))
        std::cout << "Matrices are equal" << std::endl;
    else
        std::cout << "Matrices are not equal" << std::endl;

    return 0;
}