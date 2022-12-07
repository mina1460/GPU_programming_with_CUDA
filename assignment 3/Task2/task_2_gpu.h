#ifndef GPU_TASK2_GPU_H
#define GPU_TASK2_GPU_H
#include "GPU_utils.h"
#define sqrt_needed_threads 16 // sqrt(512/2)
#define i blockIdx.x * blockDim.x + threadIdx.x
#define j blockIdx.y * blockDim.y + threadIdx.y
#define sectionIdx blockIdx.y * blockDim.x + blockIdx.x

#define needed_threads 256
#define block_width needed_threads


__global__ void scanRows(int32_t * input_matrix, int32_t * result_matrix, int img_width, int img_height)
{
        /*
            (1) Each thread has some sequential Work to be done
            (2) Then, using the scan algorithm, each thread will add its work to the previous thread's work
            (3) The result will be stored in the result_matrix
        */

        int number_of_sections = ceil(float(block_width)/blockDim.y);
        int row_number         = blockIdx.y;

        int start_idx = number_of_sections * threadIdx.y;
        int end_idx   = min(start_idx + number_of_sections - 1, img_width - 1);

        __shared__ int32_t shared_memory[block_width];

        if(start_idx < img_width)
        {
            // First phase: Sequential - normal prefix sum in this section

            result_matrix[row_number * img_width + start_idx] = input_matrix[row_number * img_width + start_idx];
            for(int idx = start_idx+1 ; idx<=end_idx ; idx++)
            {
                if(idx <img_width)
                    result_matrix[row_number * img_width + idx] = result_matrix[row_number * img_width + idx - 1] + input_matrix[row_number * img_width + idx];
            } 
        }
        __syncthreads();

        // Second phase: Parallel - 
        //                          (1) Add the last element of the previous section to the shared memory
        //                          (2) Do prefix sum to this shared memory

        if(end_idx < img_width)
        {
            shared_memory[thread_idx.y] = result_matrix[row_number * img_width + end_idx];
        }
        else 
        {
            shared_memory[thread_idx.y] = 0;
        }

        // Do the prefix using Kogge-Stone algorithm

        for(int stride = 1 ; stride < blockDim.y ; stride *= 2)
        {
            if(threadIdx.y >= stride)
            {
                __syncthreads();
                int temp = shared_memory[threadIdx.y - stride] + shared_memory[threadIdx.y];
                __syncthreads();
                shared_memory[threadIdx.y] = temp;
            }
        }
        

}


