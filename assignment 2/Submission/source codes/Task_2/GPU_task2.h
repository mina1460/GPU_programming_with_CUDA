#ifndef GPU_TASK3_H
#define GPU_TASK3_H

#include "../library.h"

__global__
void MatrixMulKernel(double* mat_a, double* mat_b, double* mat_c, int block_size, int A_width, int A_height, int B_width, int B_height, int C_width, int C_height){

    //Using Tiling to improve the performance
    int tile_size = block_size; 
    __shared__ double ds_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double ds_B[BLOCK_SIZE][BLOCK_SIZE];
    
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * tile_size + ty;
    int Col = bx * tile_size + tx;
    double result = 0;

    for(int ph = 0; ph < (A_width+tile_size - 1)/tile_size; ph++)
    {
        if(Row < A_height && ph*tile_size + tx < A_width)
            ds_A[ty][tx] = mat_a[Row * A_width + ph * tile_size + tx];
        else 
            ds_A[ty][tx] = 0.0;
        if(Col < B_width && ph*tile_size + ty < B_height)
            ds_B[ty][tx] = mat_b[(ph * tile_size + ty) * B_width + Col];
        else
            ds_B[ty][tx] = 0.0;
            
        __syncthreads();

        for(int k = 0; k < tile_size; k++)
        {
            // if(Row < A_height && Col < B_width && ph*tile_size + k < A_width && ph*tile_size + k < B_height)
                result += ds_A[ty][k] * ds_B[k][tx];
        }
        __syncthreads();
    }
    if(Row < C_height && Col < C_width)
        mat_c[Row * C_width + Col] = result;
}


template <typename T> 
long long GPU_matrix_multiplication(matrix<T> *A, matrix<T> *B, matrix<T> *C, int block_size){
    
    T *d_mat_a, *d_mat_b, *d_mat_c;
    int size_a = A->get_rows() * A->get_columns() * sizeof(T);
    int size_b = B->get_rows() * B->get_columns() * sizeof(T);
    int size_c = C->get_rows() * C->get_columns() * sizeof(T);

    cudaMalloc((void **)&d_mat_a, size_a);
    cudaCheckError();

    cudaMalloc((void **)&d_mat_b, size_b);
    cudaCheckError();

    cudaMalloc((void **)&d_mat_c, size_c);
    cudaCheckError();

    cudaMemcpy(d_mat_a, A->data, size_a, cudaMemcpyHostToDevice);
    cudaCheckError();

    cudaMemcpy(d_mat_b, B->data, size_b, cudaMemcpyHostToDevice);
    cudaCheckError();

    cudaMemcpy(d_mat_c, C->data, size_c, cudaMemcpyHostToDevice);
    cudaCheckError();

    dim3 blockSizes(block_size, block_size);
    int grid_x = std::ceil(static_cast<double>(B->get_columns()) / static_cast<double>(blockSizes.x));
    int grid_y = std::ceil(static_cast<double>(A->get_rows()) / static_cast<double>(blockSizes.y));

    std::cout << "Total number of blocks: " << grid_x * grid_y << std::endl;
    std::cout << "Total number of threads: " << grid_x * grid_y * blockSizes.x * blockSizes.y << std::endl;

    dim3 gridSizes(grid_x, grid_y);

    std::chrono::high_resolution_clock::time_point start = get_time();

    MatrixMulKernel<<<gridSizes, blockSizes>>>(d_mat_a, d_mat_b, d_mat_c, block_size, A->get_columns(), A->get_rows(), B->get_columns(), B->get_rows(), C->get_columns(), C->get_rows());


    // cuda device synchronize
    cudaDeviceSynchronize();
    cudaCheckError();

    std::chrono::high_resolution_clock::time_point end = get_time();
    long long kernel_duration = get_time_diff(start, end, nanoseconds);
    

    cudaMemcpy(C->data, d_mat_c, size_c, cudaMemcpyDeviceToHost);
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