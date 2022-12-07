#include "GPU_utils.h"
#define sqrt_needed_threads 16 // sqrt(512/2)


#define needed_threads 2048
// #define block_width needed_threads


// __global__ void scanRowsKernel(int32_t * input_matrix, int32_t * result_matrix, int32_t* add_matrix, int img_width, int img_height, int n_blocks_row)
// {
//     int block_width = ceil(float(img_width)/n_blocks_row);
//     int start_idx   = blockIdx.x * block_width;
//     int end_idx     = min(start_idx + block_width -1, img_width - 1);
//     int row_number  = blockIdx.x / n_blocks_row;
//     // Phase1: Each block will do the sequential part of the scan for a row
//     if(start_idx < img_width)
//     {
//         result_matrix[row_number * img_width + start_idx] = input_matrix[row_number * img_width + start_idx];
//         for (int i = start_idx+1; i <= end_idx; i++)
//         {
//             int idx = row_number * img_width + i;
//             if (idx < img_width)
//                 result_matrix[idx] = result_matrix[idx - 1] + input_matrix[idx];
//         }
//         __syncthreads();
//         // Last Element of each block will be the sum of all elements in the block
//         //Phase 2: We need to have a 2D shared Memory, where each row will have a block_width number of elements

//         if(end_idx < img_width)
//                 shared_memory[threadIdx.x] = result_matrix[row_number * img_width + end_idx];
//             else 
//                 shared_memory[threadIdx.x] = 0;

//             for(int stride = 1 ; stride < blockDim.x ; stride *= 2)
//             {
//                 __syncthreads();
//                 if(threadIdx.x >= stride)
//                 {
//                     int temp = shared_memory[threadIdx.x - stride] + shared_memory[threadIdx.x];
//                     __syncthreads();
//                     shared_memory[threadIdx.x] = temp;
//                 }
//             }

//             // printf("shared_memory[threadIdx.x] = %d", shared_memory[threadIdx.x]);

//             __syncthreads();

//             // Third phase: Each sharedMemory value is added to the next section of the result matrix
//             int32_t temp = 0; 
//             if(threadIdx.x > 0)
//                 temp = shared_memory[threadIdx.x - 1];

            
//             for(int idx = start_idx ; idx <= end_idx ; idx++)
//             {
//                 if(idx < img_width)
//                     result_matrix[row_number * img_width + idx] += temp;
//             }





//     }

// }

__global__ void prefixSumScanKernel(int32_t* input_matrix, int32_t* add_matrix, int32_t* result_matrix, int img_width, int img_height, int n_blocks_row, bool isadded)
{
    __shared__ int32_t cache[needed_threads];

    int i           = blockIdx.x * blockDim.x + threadIdx.x;
    int j           = blockIdx.y * blockDim.y + threadIdx.y;
    int sectionIdx  =  blockIdx.x + blockIdx.y *n_blocks_row; 

    // img_width is the number of columns
    // img_height is the number of rows
    int idx = j * img_width + i;

    if(idx < (img_width * img_height))
    {
        if(i < img_width && j < img_height)
            cache[threadIdx.x] = input_matrix[idx];
        else
            cache[threadIdx.x] = 0;

        if(idx < (img_width * img_height))
        {
            if(i < img_width && j < img_height)
                cache[threadIdx.x] = input_matrix[idx];
            else
                cache[threadIdx.x] = 0;

            printf("cache[%d] = %d , input_matrix[%d]=%d, i=%d, j=%d\n",threadIdx.x, cache[threadIdx.x], idx, input_matrix[idx], i, j);
            int temp = 0;
            for (int stride = 1 ; stride < blockDim.x ; stride*=2)
            {
                if(threadIdx.x >= stride)
                {
                    __syncthreads(); 
                    temp = cache[threadIdx.x] + cache[threadIdx.x - stride];
                    __syncthreads();
                    cache[threadIdx.x]=temp; 
                }
            }
            //print the result 
            result_matrix[idx] = cache[threadIdx.x];
            printf("cache[%d] = %d, result_matrix[%d]=%d\n",threadIdx.x, cache[threadIdx.x],idx, result_matrix[idx]);

            // add the results of the end of each section in add_matrix 
            if(threadIdx.x == blockDim.x - 1 && isadded){
                add_matrix[sectionIdx] = cache[threadIdx.x];
            }
        }
    }
}



__global__ void addSectionsKernel(int32_t* result_matrix, int32_t* add_matrix, int img_width, int img_height, int n_blocks_row)
{
    int i           = blockIdx.x * blockDim.x + threadIdx.x;
    int j           = blockIdx.y * blockDim.y + threadIdx.y;
    int sectionIdx  = blockIdx.x + blockIdx.y *n_blocks_row - 1; 

    int idx = j * img_width + i;

    if(idx < (img_width * img_height))
    {
        if(i < img_width && j < img_height )
        {   if(sectionIdx > 0)
                result_matrix[idx] += add_matrix[sectionIdx];
                //print
                // printf("result_matrix[%d] = %d, add_matrix[%d] = %d; \n", idx, result_matrix[idx], sectionIdx, add_matrix[sectionIdx]);
        }
    }
}




long long GPU_summed_area_table(int32_t* input_matrix, int32_t* result_image, int img_width, int img_height)
{
    int32_t* original_matrix;    
    int32_t* result_matrix; 
    int32_t* add_matrix;
    int32_t* prefix_add_matrix;
    

    int matrixSize = img_width * img_height * sizeof(int32_t);
    
    const int maxNumberOfBlocks = 65535;
    const int maxNumberOfThreads = 5;

    int n_blocks_row = ceil(float(img_width)/maxNumberOfThreads);
    //each block will be a section of a row
    int block_width = ceil(float(img_width)/n_blocks_row);

    int n_blocks_col = ceil(float(img_height)/maxNumberOfBlocks);
    int block_height = ceil(float(img_height)/n_blocks_col);


    cudaMalloc((void **)&original_matrix, matrixSize);
    cudaCheckError();

    cudaMalloc((void **)&result_matrix, matrixSize);
    cudaCheckError();
        // img_height * n_blocks_row * sizeof(int32_t)
    cudaMalloc((void **)&add_matrix, img_height * n_blocks_row * sizeof(int32_t)); // n_blocks_row is the number of sections in a row for each row
    cudaCheckError();
    
    cudaMalloc((void **)&prefix_add_matrix, matrixSize); // n_blocks_row is the number of sections in a row for each row
    cudaCheckError();

    cudaMemcpy(original_matrix, input_matrix, matrixSize, cudaMemcpyHostToDevice);
    cudaCheckError();
    
    // Genralizing the block width: Not necessary to consider a block for each row
    // Each block will be conisdered as a section of a row

    
    int row_threads = min(maxNumberOfThreads, block_width);
    // The question has mentioned that the matrix will be square (num of rows = num of columns)
    dim3 dimBlock(row_threads);
    dim3 dimGrid(n_blocks_row, img_height); // number of rows and number of blocks needed for a specific row
    

    dim3 dimBlock2(n_blocks_row);
    dim3 dimGrid2(img_height, n_blocks_row);

    std::chrono::high_resolution_clock::time_point start = get_time();

        
    
    prefixSumScanKernel<<<dimGrid, dimBlock>>>(original_matrix,add_matrix, result_matrix, img_width, img_height, n_blocks_row, true);
    cudaDeviceSynchronize();
    cudaCheckError();
    printf("STOP\n\n");
    // Do prefix sum to the add matrix 
    prefixSumScanKernel<<<dimGrid2, dimBlock2>>>(add_matrix, NULL, prefix_add_matrix, img_height, n_blocks_row, n_blocks_row, false);
    cudaDeviceSynchronize();
    cudaCheckError();
    // Propgate the sum of prefix_add_matrix in the result matrix
    addSectionsKernel<<<dimGrid, dimBlock>>>(result_matrix, prefix_add_matrix, img_width, img_height, n_blocks_row);
    // addSectionsKernel<<>>(result_matrix, add_matrix, img_width, img_height, n_blocks_row);
   

   
    //print the n_blocks_row
    dim3 dimBlock3(img_height, n_blocks_row);
    dim3 dimGrid3(1, 1);


    

     
    std::chrono::high_resolution_clock::time_point end = get_time();
    long long kernel_duration = get_time_diff(start, end, nanoseconds);
    
   

    cudaMemcpy(result_image, prefix_add_matrix, img_height * n_blocks_row * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaCheckError();
    
    cudaFree(original_matrix);
    cudaCheckError();

    cudaFree(result_matrix);
    cudaCheckError();

    return kernel_duration;

} 
