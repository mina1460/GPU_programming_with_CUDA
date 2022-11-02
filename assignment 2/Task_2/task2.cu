#include "../library.h"
#include "GPU_task2.h"


int main(int argc, char** argv){
    
    std::cout << "Starting..." << std::endl;
    std::cout << "Dimensions of A: " << A_ROWS << "x" << A_COLS << std::endl;
    std::cout << "Dimensions of B: " << B_ROWS << "x" << B_COLS << std::endl;
   
    assert(A_COLS == B_ROWS && "matrix A columns must be equal to matrix B rows");
    
    // allocate memory for matrices
    matrix<double> A(A_ROWS, A_COLS, "A");
    matrix<double> B(B_ROWS, B_COLS, "B");
    matrix<double> C_GPU(A_ROWS, B_COLS, "C_GPU");
    matrix<double> C_CPU(A_ROWS, B_COLS, "C_CPU");
    
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
    int block_size = BLOCK_SIZE;
    long long time_in_kernel = GPU_matrix_multiplication( &A, &B, &C_GPU, block_size);
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
    if(compare_with_tolerance(C_CPU, C_GPU, 0.00001))
        std::cout << "Matrices are equal" << std::endl;
    else
        std::cout << "Matrices are not equal" << std::endl;

    return 0;
}