#include "GPU_utils.h"
#define needed_threads 2048
#define TILE_WIDTH 10

__global__ void prefixSumScanKernel(long long* input_matrix, long long* add_matrix, long long* result_matrix, int img_width, int img_height, int n_blocks_row, bool isadded)
{
    __shared__ long long cache[needed_threads];

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

__global__ void addRowSectionsKernel(long long* result_matrix, long long* add_matrix, int img_width, int img_height, int n_blocks_row)
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

__global__ void efficientTransposeKernel(long long* input_matrix, long long* result_matrix, int img_width, int img_height)
{
    __shared__ long long cache[TILE_WIDTH][TILE_WIDTH + 1];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int width = gridDim.x * TILE_WIDTH;
    if(col < img_width && row < img_height){
        for(int j = 0; j < TILE_WIDTH ; j+=blockDim.y)
        {
            if(col < img_width && (row + j) < img_height)
                cache[(threadIdx.y + j)][threadIdx.x] = input_matrix[(row + j)*width + col];
        }

        __syncthreads();

   
        col = blockIdx.y * TILE_WIDTH + threadIdx.x;
        row = blockIdx.x * TILE_WIDTH + threadIdx.y;

        for(int j = 0; j < TILE_WIDTH ; j+=blockDim.y)
        {   if((col + j) < img_height && row < img_width)
                result_matrix[(row + j)*width + col] = cache[threadIdx.x][(threadIdx.y + j)];
        }
    }
}


long long GPU_summed_area_table(long long* input_matrix, long long* result_image, int img_width, int img_height)
{
    long long* original_matrix;    
    long long* result_matrix; 
    long long* add_matrix;
    long long* prefix_add_matrix;
    long long* transpose_matrix;

    //for col
    long long* add_matrix_2;
    long long* prefix_add_matrix_2;

    int matrixSize = img_width * img_height * sizeof(long long);
    const int maxNumberOfBlocks = 65535;
    const int maxNumberOfThreads = 6;

    int n_blocks_row = ceil(float(img_width)/maxNumberOfThreads);
    //each block will be a section of a row
    int block_width = ceil(float(img_width)/n_blocks_row);

    int n_blocks_col = ceil(float(img_height)/maxNumberOfThreads);
    int block_height = ceil(float(img_height)/n_blocks_col);


    cudaMalloc((void **)&original_matrix, matrixSize);
    cudaCheckError();

    cudaMalloc((void **)&transpose_matrix, matrixSize);
    cudaCheckError();

    cudaMalloc((void **)&result_matrix, matrixSize);
    cudaCheckError();
        // img_height * n_blocks_row * sizeof(long long)
    cudaMalloc((void **)&add_matrix, img_height * n_blocks_row * sizeof(long long)); // n_blocks_row is the number of sections in a row for each row
    cudaCheckError();

    cudaMalloc((void **)&add_matrix_2, img_width * n_blocks_col * sizeof(long long)); // n_blocks_row is the number of sections in a row for each row
    cudaCheckError();

    cudaMalloc((void **)&prefix_add_matrix, img_height * n_blocks_row * sizeof(long long)); // n_blocks_row is the number of sections in a row for each row
    cudaCheckError();

    cudaMalloc((void **)&prefix_add_matrix, img_width * n_blocks_col * sizeof(long long)); 
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
    dim3 dimBlock3(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid3(ceil(float(img_width)/TILE_WIDTH), ceil(float(img_height)/TILE_WIDTH));


    // with columns 

    dim3 dimGrid4(n_blocks_col, img_width);
    dim3 dimBlock5(n_blocks_col);
    dim3 dimGrid5(img_width, n_blocks_col);

    std::chrono::high_resolution_clock::time_point start = get_time();

        
    
    prefixSumScanKernel<<<dimGrid, dimBlock>>>(original_matrix,add_matrix, result_matrix, img_width, img_height, n_blocks_row, true);
    cudaDeviceSynchronize();
    cudaCheckError();
    // Do prefix sum to the add matrix 
    prefixSumScanKernel<<<dimGrid2, dimBlock2>>>(add_matrix, NULL, prefix_add_matrix, img_height, n_blocks_row, n_blocks_row, false);
    cudaDeviceSynchronize();
    cudaCheckError();
    // Propgate the sum of prefix_add_matrix in the result matrix
    if(n_blocks_row > 1)
    {
        addRowSectionsKernel<<<dimGrid, dimBlock>>>(result_matrix, prefix_add_matrix, img_width, img_height, n_blocks_row);
        cudaDeviceSynchronize();
        cudaCheckError();
    }
    //Do transpose
    efficientTransposeKernel<<<dimGrid3, dimBlock3>>>(result_matrix,transpose_matrix, img_width, img_height);
    cudaDeviceSynchronize();
    cudaCheckError();

    // Same thing applies to the columns
    prefixSumScanKernel<<<dimGrid4, dimBlock>>>(transpose_matrix,add_matrix, result_matrix, img_height, img_width, n_blocks_col, true);
    cudaDeviceSynchronize();
    cudaCheckError();
    
    prefixSumScanKernel<<<dimGrid5, dimBlock5>>>(add_matrix, NULL, prefix_add_matrix, img_width, n_blocks_col, n_blocks_col, false);
    cudaDeviceSynchronize();
    cudaCheckError();
    
    if(n_blocks_col > 1){
        addRowSectionsKernel<<<dimGrid4, dimBlock>>>(result_matrix, prefix_add_matrix, img_height, img_width, n_blocks_col);
        cudaDeviceSynchronize();
        cudaCheckError();
    }
    //transpose again
    efficientTransposeKernel<<<dimGrid3, dimBlock3>>>(result_matrix,transpose_matrix, img_height, img_width);
     
    std::chrono::high_resolution_clock::time_point end = get_time();
    long long kernel_duration = get_time_diff(start, end, nanoseconds);
    
   
//img_height * n_blocks_row * sizeof(long long)
    cudaMemcpy(result_image, result_matrix,  matrixSize, cudaMemcpyDeviceToHost);
    cudaCheckError();
    
    cudaFree(original_matrix);
    cudaCheckError();

    cudaFree(result_matrix);
    cudaCheckError();

    return kernel_duration;

} 
