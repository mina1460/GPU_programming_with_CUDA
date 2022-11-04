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
    __shared__ double ds_B[BLOCK_SIZE][BLOCK_SIZE * granularity];
    
    int Row = (by * tile_size) + ty;
    int Col = granularity * (bx * tile_size) + tx;

    double results[granularity] = {};


    for(int start_tile = 0; start_tile < A_width; start_tile+= BLOCK_SIZE)
    {
        if(Row < A_height && start_tile + tx < A_width)
            ds_A[ty][tx] = mat_a[Row * A_width + start_tile + tx];
        else 
            ds_A[ty][tx] = 0.0;

        // Samething applies to the column 
        #pragma unroll
        for(int g = 0 ; g < granularity * BLOCK_SIZE ; g+=BLOCK_SIZE)
        {
            if(Col + g < B_width && start_tile + ty < B_height)
                ds_B[ty][tx + g] = mat_b[(start_tile + ty) * B_width + (Col + g)];
            else 
                ds_B[ty][tx + g] = 0.0; 
        }
        __syncthreads(); 
        #pragma unroll
        for(int k = 0; k < tile_size; k++)
            {  
                for(int g_y = 0 ; g_y < granularity; g_y++)
                    results[g_y] += ds_A[ty][k] * ds_B[k][tx + g_y * BLOCK_SIZE]; 
                
            }
        __syncthreads();
    }

    //Updating 
    #pragma unroll
    for(int g_y = 0 ; g_y < granularity; g_y++)
    {
        if(Row < C_height && Col + g_y * BLOCK_SIZE < C_width)
        {
            mat_c[Row * C_width + Col + g_y * BLOCK_SIZE] = results[g_y];
        }
    }

    
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
    int grid_y = std::ceil(static_cast<double>(A->get_rows()) / static_cast<double>(blockSizes.y));

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