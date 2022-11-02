#ifndef GPU_TASK2_H
#define GPU_TASK2_H

#include "../library.h"
#define bx blockIdx.x
#define by blockIdx.y
#define tx threadIdx.x
#define ty threadIdx.y
#define tile_size BLOCK_SIZE

__global__
void MatrixMulKernel(double* mat_a, double* mat_b, double* mat_c, int A_width, int A_height, int B_width, int B_height, int C_width, int C_height){

    //Using Tiling to improve the performance
    __shared__ double ds_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double ds_B[BLOCK_SIZE][BLOCK_SIZE];
    

    int Row = by * tile_size + ty;
    int Col = granularity * (bx * tile_size + tx);
    const int result_sz = granularity; 
    double results[result_sz] = {};
    for(int g = 0; g<granularity; g++){
        results[g] = 0;
    }

    for(int ph = 0; ph < (A_width+tile_size - 1)/tile_size; ph++)
    {
        if(Row < A_height && ph*tile_size + tx < A_width)
            ds_A[ty][tx] = mat_a[Row * A_width + ph * tile_size + tx];
        else 
            ds_A[ty][tx] = 0.0;

        // for(int g = 0 ; g< granularity ; g++)
        // {
        //     if(ph*tile_size + ty < A_width && Col + g < B_height)
        //         ds_B[ty][(Col+g)%BLOCK_SIZE] = mat_b[(ph * tile_size + ty) * B_width + Col + g];
          
        // }
            
        __syncthreads();

        for(int g = 0; g<granularity; g++)
            for(int k = 0; k < tile_size; k++)
            {
                if(Row < A_height && Col + g < B_width && ph*tile_size + k < A_width && ph*tile_size + k < B_height)
                    results[g] += ds_A[ty][k] * mat_b[( (ph) * tile_size + k) * B_width + Col + g];
                
            }
        __syncthreads();
    }
    for(int g = 0; g<granularity; g++)
        if(Row < C_height && Col + g < C_width)
            mat_c[Row * C_width + Col + g ] = results[g];
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
    int grid_x = std::ceil(static_cast<double>(B->get_columns()) / static_cast<double>(blockSizes.x * granularity));
    int grid_y = std::ceil(static_cast<double>(A->get_rows()) / static_cast<double>(blockSizes.y))+1;

    std::cout << "Total number of blocks: " << grid_x * grid_y << std::endl;
    std::cout << "Total number of threads: " << grid_x * grid_y * blockSizes.x * blockSizes.y << std::endl;

    dim3 gridSizes(grid_x, grid_y);

    std::chrono::high_resolution_clock::time_point start = get_time();

    MatrixMulKernel<<<gridSizes, blockSizes>>>(d_mat_a, d_mat_b, d_mat_c, A->get_columns(), A->get_rows(), B->get_columns(), B->get_rows(), C->get_columns(), C->get_rows());


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