#ifndef GPU_TASK1_H
#define GPU_TASK1_H

#include "matrix.h"
#include <cuda_runtime.h>


#define cudaCheckError() {                                                                  \
 cudaError_t e=cudaGetLastError();                                                          \
 if(e!=cudaSuccess) {                                                                       \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));            \
   exit(0);                                                                                 \
 }                                                                                          \
}

int get_max_threads_per_block(){
    int max_threads_per_block;
    cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, 0);
    return max_threads_per_block;
}

int get_max_threads_per_multiprocessor(){
    int max_threads_per_multiprocessor;
    cudaDeviceGetAttribute(&max_threads_per_multiprocessor, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
    return max_threads_per_multiprocessor;
}


_global__ void MatrixMulKernel(float* mat_a, float* mat_b, float* mat_c, int A_rows, int A_cols, int B_rows, int B_cols, int C_rows, int C_cols){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0;
    if (row < A_rows && col < B_cols){
        for (int i = 0; i < A_cols; i++){
            sum += mat_a[row * A_cols + i] * mat_b[i * B_cols + col];
        }
        mat_c[row * C_cols + col] = sum;
    }

}


template <typename T>
long long GPU_matrix_multiplication(matrix<T> &A, matrix<T> &B, matrix<T> &C, int block_size){
    
    float *d_mat_a, *d_mat_b, *d_mat_c;
    int size_a = A.get_rows() * A.get_columns() * sizeof(T);
    int size_b = B.get_rows() * B.get_columns() * sizeof(T);
    int size_c = C.get_rows() * C.get_columns() * sizeof(T);

    cudaMalloc((void **)&d_mat_a, size_a);
    cudaCheckError();

    cudaMalloc((void **)&d_mat_b, size_b);
    cudaCheckError();

    cudaMalloc((void **)&d_mat_c, size_c);
    cudaCheckError();

    cudaMemcpy(d_mat_a, A.data, size_a, cudaMemcpyHostToDevice);
    cudaCheckError();

    cudaMemcpy(d_mat_b, B.data, size_b, cudaMemcpyHostToDevice);
    cudaCheckError();

    cudaMemcpy(d_mat_c, C.data, size_c, cudaMemcpyHostToDevice);
    cudaCheckError();

    dim3 blockSizes(block_size, block_size);
    int grid_x = std::ceil(static_cast<double>(B.get_columns()) / static_cast<double>(blockSizes.x));
    int grid_y = std::ceil(static_cast<double>(A.get_rows()) / static_cast<double>(blockSizes.y));

    std::cout << "Total number of blocks: " << grid_x * grid_y << std::endl;
    std::cout << "Total number of threads: " << grid_x * grid_y * blockSizes.x * blockSizes.y << std::endl;

    dim3 gridSizes(grid_x, grid_y);

    std::chrono::high_resolution_clock::time_point start = get_time();

    MatrixMulKernel<<<gridSizes, blockSizes>>>(d_mat_a, d_mat_b, d_mat_c, A.get_columns(), A.get_rows());


    // cuda device synchronize
    cudaDeviceSynchronize();
    cudaCheckError();

    std::chrono::high_resolution_clock::time_point end = get_time();
    long long kernel_duration = get_time_diff(start, end, nanoseconds);
    

    cudaMemcpy(C.data, d_mat_c, size_c, cudaMemcpyDeviceToHost);
    cudaCheckError();

    cudaFree(d_mat_a);
    cudaCheckError();

    cudaFree(d_mat_b);
    cudaCheckError();

    cudaFree(d_mat_c);
    cudaCheckError();


    return kernel_duration;
    
}
#endif