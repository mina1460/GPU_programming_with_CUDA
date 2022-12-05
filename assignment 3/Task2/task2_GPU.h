#ifndef GPU_TASK2_GPU_H
#define GPU_TASK2_GPU_H
#include "GPU_utils.h"
#define needed_threads 512
#define sqrt_needed_threads 16 // sqrt(512/2)
#define i blockIdx.x * blockDim.x + threadIdx.x
#define j blockIdx.y * blockDim.y + threadIdx.y
#define sectionIdx blockIdx.y * blockDim.x + blockIdx.x
template<typename T>
__global__ void prefixSumScanKernel(T* input_matrix, T* total_sum_sections, T* result_matrix, int img_width, int img_height)
{
    __shared__ T cache[needed_threads];

    // img_width is the number of columns
    // img_height is the number of rows
    int idx = j * img_width + i;

    if(idx < (img_width * img_height))
    {
        if(i < img_width && j < img_height)
        {
            cache[threadIdx.x] = input_matrix[idx];
        }
        else
        {
            cache[threadIdx.x] = 0;
        }

        for (int stride = 1 ; stride < blockDim.x ; stride*=2)
        {
            if(threadIdx.x >= stride)
            {
                __syncthreads(); 
                float temp = cache[threadIdx.x] + cache[threadIdx.x - stride];
                __syncthreads();
                cache[threadIdx.x]=temp; 
            }
        }
            result_matrix[idx] = cache[threadIdx.x];


        // add the results of the end of each section in total_sum_sections 
        if(threadIdx.x == blockDim.x - 1)
        {
            total_sum_sections[sectionIdx] = cache[threadIdx.x];
        }

    }

}

template<typename T>
__global__ addBoundaryValueKernel(T* row_prefix_result, T* total_sum_sections, int img_width, int img_height)
{
    int idx = j * img_width + i;
    if(idx < (img_width * img_height))
    {
        if(sectionIdx >=1)
        {
            row_prefix_result[idx] += total_sum_sections[sectionIdx - 1];
        }
          
    }
}

template<typename T>
__global__ transposeKernel(T* input_matrix, T* result_matrix, int img_width, int img_height)
{
    int idx = j * img_width + i;
    if(idx < (img_width * img_height))
    {
        result_matrix[idx] = input_matrix[j * img_width + i];
    }
}

template<typename T>
__global__ efficientTransposeKernel(T* input_matrix, T* result_matrix, int img_width, int img_height)
{
    __shared__ T cache[sqrt_needed_threads][sqrt_needed_threads];

    int idx = j * img_width + i;
    
    if(i < img_width && j< img_height)
    {
        cache[threadIdx.y][threadIdx.x] = input_matrix[idx];
    }
    else
    {
        cache[threadIdx.y][threadIdx.x] = 0;
    }
    
    __syncthreads();

    int t_i = blockIdx.y * blockDim.y + threadIdx.x;
    int t_j = blockIdx.x * blockDim.x + threadIdx.y;

    if(tj < img_width && ti < img_height)
    {
        result_matrix[tj * img_height + ti] = cache[threadIdx.x][threadIdx.y];
    }

}




template <typename T> 
long long GPU_summed_area_table(T* input_matrix, T* result_image, int img_width, int img_height)
{
    T* original_matrix;    
    T* result_matrix; 
    T* result_matrix_2;
    T*total_sum_sections; 

    int matrixSize = img_width * img_height * sizeof(T);
    int total_sum_sections_size = needed_threads * img_height * sizeof(T);

    cudaMalloc((void **)&original_matrix, matrixSize);
    cudaCheckError();

    cudaMalloc((void **)&result_matrix, matrixSize);
    cudaCheckError();

    cudaMalloc((void **)&result_matrix_2, matrixSize);
    cudaCheckError();

    cudaMalloc((void **)&total_sum_sections, total_sum_sections_size);
    cudaCheckError();

    cudaMemcpy(original_matrix, input_matrix, matrixSize, cudaMemcpyHostToDevice);
    cudaCheckError();
    
    // The logic is as follows: (1)Scan
    //                          (2) get the accumulated values from each section in an array 
    //                          (3) Scan again to get the desired result
    //                          (4) transpose 
    //                          (5) do from 1 to 3 again
    //                          (6) transpose again
    //                          (7) get the desired result
    
    
    // 1st scan
    int threads = (needed_threads-1)/2;
    dim3 dimBlock(threads, 1);
    dim3 dimGrid(needed_threads, img_height);

    std::chrono::high_resolution_clock::time_point start = get_time();

    prefixSumScanKernel<<<dimGrid, dimBlock>>>(original_matrix, total_sum_sections, result_matrix, img_width, img_height);

    // 2nd scan
    addBoundaryValueKernel<<<dimGrid, dimBlock>>>(result_matrix, total_sum_sections, img_width, img_height);

    // transpose 
    dim3 dimBlock_t(sqrt_needed_threads, sqrt_needed_threads);
    dim3 dimGrid_t((img_width + sqrt_needed_threads - 1)/sqrt_needed_threads, (img_height + sqrt_needed_threads - 1)/sqrt_needed_threads);
    efficientTransposeKernel<<dimGrid_t, dimBlock_t>>>(result_matrix, result_matrix, img_width, img_height);

    // Do the exact samething again 
    cudaMemset(total_sum_sections, 0, total_sum_sections_size);
    prefixSumScanKernel<<<dimGrid, dimBlock>>>(result_matrix, total_sum_sections, result_matrix_2, img_height, img_width);

    addBoundaryValueKernel<<<dimGrid, dimBlock>>>(result_matrix_2, total_sum_sections, img_height, img_width);

    efficientTransposeKernel<<dimGrid_t, dimBlock_t>>>(result_matrix_2, result_matrix_2, img_height, img_width);

    // cuda device synchronize
    cudaDeviceSynchronize();
    cudaCheckError();

    std::chrono::high_resolution_clock::time_point end = get_time();

    std::chrono::high_resolution_clock::time_point end = get_time();
    long long kernel_duration = get_time_diff(start, end, nanoseconds);

    
    cudaMemcpy(result_image, result_matrix_2, matrixSize, cudaMemcpyDeviceToHost);
    cudaCheckError();

    cudaFree(original_matrix);
    cudaCheckError();

    cudaFree(result_matrix);
    cudaCheckError();

    cudaFree(result_matrix_2);
    cudaCheckError();

    cudaFree(total_sum_sections);
    cudaCheckError();

    return kernel_duration;

} 















#endif