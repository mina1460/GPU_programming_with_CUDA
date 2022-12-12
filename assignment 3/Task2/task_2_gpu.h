#include "GPU_utils.h"
// #define sqrt_needed_threads 16 // sqrt(512/2)


#define needed_threads 256
#define block_width needed_threads


__global__ void scanRowsKernel(long long * input_matrix, long long * result_matrix, int img_width, int img_height)
{
        /*
            (1) Each thread has some sequential Work to be done
            (2) Then, using the scan algorithm, each thread will add its work to the previous thread's work
            (3) The result will be stored in the result_matrix
        */
        int section_length = ceil(float(img_width)/blockDim.x);
        int row_number         = blockIdx.x;
        int start_idx          = section_length * threadIdx.x;
        int end_idx             = min(start_idx + section_length - 1, img_width - 1);

        __shared__ long long shared_memory[block_width];

        if(start_idx < img_width)
        {
            // First phase: Sequential - normal prefix sum in this section
            result_matrix[row_number * img_width + start_idx] = input_matrix[row_number * img_width + start_idx];
            for(int idx = start_idx+1 ; idx<=end_idx ; idx++)
            {
                if(idx <img_width)
                    result_matrix[row_number * img_width + idx] = result_matrix[row_number * img_width + idx - 1] + input_matrix[row_number * img_width + idx];
            } 
            __syncthreads();

            // Second phase: Parallel - 
            //                          (1) Add the last element of the previous section to the shared memory
            //                          (2) Do prefix sum to this shared memory

            if(end_idx < img_width)
                shared_memory[threadIdx.x] = result_matrix[row_number * img_width + end_idx];
            else 
                shared_memory[threadIdx.x] = 0;
            for(int stride = 1 ; stride < blockDim.x ; stride *= 2)
            {
                if(threadIdx.x >= stride)
                {
                    // __syncthreads();
                    int temp = shared_memory[threadIdx.x - stride] + shared_memory[threadIdx.x];
                    // __syncthreads();
                    shared_memory[threadIdx.x] = temp;
                }
            }


            __syncthreads();
            // printf("shared_memory[threadIdx.x] = %d", shared_memory[threadIdx.x]);

            // Third phase: Each sharedMemory value is added to the next section of the result matrix
            long long temp = 0; 
            if(threadIdx.x > 0)
                temp = shared_memory[threadIdx.x - 1];

            
            for(int idx = start_idx ; idx <= end_idx ; idx++)
            {
                if(idx < img_width)
                    result_matrix[row_number * img_width + idx] += temp;
            }


        }
}

__global__ void scanColumnsKernel(long long * result_matrix, int img_width, int img_height)
{
        /*
            (1) Each thread has some sequential Work to be done
            (2) Then, using the scan algorithm, each thread will add its work to the previous thread's work
            (3) The result will be stored in the result_matrix
        */

        int section_length = ceil(float(block_width)/blockDim.x);
        int col_number         = blockIdx.x;

        int start_idx = section_length * threadIdx.x;
        int end_idx   = min(start_idx + section_length - 1, img_height - 1);

        __shared__ long long shared_memory[block_width];

        if(start_idx < img_height)
        {
            // First phase: Sequential - normal prefix sum in this section

            for(int idx = start_idx+1 ; idx<=end_idx ; idx++)
            {
                if(idx <img_width)
                    result_matrix[idx * img_width + start_idx] += result_matrix[idx * img_width + start_idx - 1];
            }
            __syncthreads();

            // Second phase: Parallel - 
            //                          (1) Add the last element of the previous section to the shared memory
            //                          (2) Do prefix sum to this shared memory

            if(end_idx < img_height)
            {
                shared_memory[threadIdx.x] = result_matrix[end_idx * img_width + col_number];
            }
            else 
            {
                shared_memory[threadIdx.x] = 0;
            }

            // Do the prefix using Kogge-Stone algorithm

            for(int stride = 1 ; stride < blockDim.x ; stride *= 2)
            {
                __syncthreads();
                if(threadIdx.x >= stride)
                {
                    // __syncthreads();
                    int temp = shared_memory[threadIdx.x - stride] + shared_memory[threadIdx.x];
                    __syncthreads();
                    shared_memory[threadIdx.x] = temp;
                }
            }

            __syncthreads();

            // Third phase: Each sharedMemory value is added to the next section of the result matrix
            long long temp = 0; 
            if(threadIdx.x > 0)
                temp = shared_memory[threadIdx.x - 1];

            
            for(int idx = start_idx ; idx <= end_idx ; idx++)
            {
                if(idx < img_height && col_number < img_width)
                    result_matrix[idx * img_width + col_number] += temp;
            }
        }

    // printf("Thread %d, %d finished", threadIdx.x, blockIdx.x);
    return;
}



long long GPU_summed_area_table_not_Generalized(long long* input_matrix, long long* result_image, int img_width, int img_height)
{
    long long* original_matrix;    
    long long* result_matrix; 

    int matrixSize = img_width * img_height * sizeof(long long);

    cudaMalloc((void **)&original_matrix, matrixSize);
    cudaCheckError();

    cudaMalloc((void **)&result_matrix, matrixSize);
    cudaCheckError();


    cudaMemcpy(original_matrix, input_matrix, matrixSize, cudaMemcpyHostToDevice);
    cudaCheckError();
    
    // As an assumption, we will use 1 block for each row. 
    //We can extend further by using multiple blocks for each row, but we will need less Grid Size.

    dim3 dimBlock(block_width);
    dim3 dimGrid(img_height);
    dim3 dimGrid2(img_width);

    std::chrono::high_resolution_clock::time_point start = get_time();
    scanRowsKernel<<<dimGrid, dimBlock>>>(original_matrix, result_matrix, img_width, img_height);
    scanColumnsKernel<<<dimGrid2, dimBlock>>>(result_matrix, img_width, img_height);
    cudaDeviceSynchronize();
    cudaCheckError();
    std::chrono::high_resolution_clock::time_point end = get_time();
    long long kernel_duration = get_time_diff(start, end, nanoseconds);

    
    cudaMemcpy(result_image, result_matrix, matrixSize, cudaMemcpyDeviceToHost);
    cudaCheckError();

    cudaFree(original_matrix);
    cudaCheckError();

    cudaFree(result_matrix);
    cudaCheckError();

    return kernel_duration;

} 
