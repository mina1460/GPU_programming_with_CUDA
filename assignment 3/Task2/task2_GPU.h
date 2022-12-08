#include "GPU_utils.h"
#define sqrt_needed_threads 16 // sqrt(512/2)


#define needed_threads 2048
// #define block_width needed_threads


__global__ void prefixRowSumScanKernel(int32_t* input_matrix, int32_t* add_matrix, int32_t* result_matrix, int img_width, int img_height, int n_blocks_row, bool isadded)
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
            //printing blockidx.x, blockdim.x, threadidx.x

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
            if(i < img_width && j < img_height)
                result_matrix[idx] = cache[threadIdx.x];

            // add the results of the end of each section in add_matrix 
            if(threadIdx.x == blockDim.x - 1 && isadded){
                add_matrix[sectionIdx] = cache[threadIdx.x];
            }
    }
}

__global__ void addRowSectionsKernel(int32_t* result_matrix, int32_t* add_matrix, int img_width, int img_height, int n_blocks_row)
{
    int i           = blockIdx.x * blockDim.x + threadIdx.x;
    int j           = blockIdx.y * blockDim.y + threadIdx.y;
    int sectionIdx  = blockIdx.x + blockIdx.y *n_blocks_row - 1; 
    
    if((sectionIdx+1) %n_blocks_row == 0 && sectionIdx > 0)
        add_matrix[sectionIdx] = 0;

    int idx = j * img_width + i;

    if(idx < (img_width * img_height) && idx%img_width != 0)
    {
        if(i < img_width && j < img_height )
        {   if(sectionIdx >= 0)
                {
                    result_matrix[idx] += add_matrix[sectionIdx];
                }
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

        
    
    prefixRowSumScanKernel<<<dimGrid, dimBlock>>>(original_matrix,add_matrix, result_matrix, img_width, img_height, n_blocks_row, true);
    cudaDeviceSynchronize();
    cudaCheckError();
    // Do prefix sum to the add matrix 
    prefixRowSumScanKernel<<<dimGrid2, dimBlock2>>>(add_matrix, NULL, prefix_add_matrix, img_height, n_blocks_row, n_blocks_row, false);
    cudaDeviceSynchronize();
    cudaCheckError();
    // Propgate the sum of prefix_add_matrix in the result matrix
    addRowSectionsKernel<<<dimGrid, dimBlock>>>(result_matrix, prefix_add_matrix, img_width, img_height, n_blocks_row);
    cudaDeviceSynchronize();
    cudaCheckError();

    // Same thing applies to the columns



    

     
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
