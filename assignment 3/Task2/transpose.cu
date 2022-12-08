#include "GPU_utils.h"
#include <iostream>
#include <string>
#define needed_threads 2048
#define TILE_WIDTH 5

__global__ void efficientTransposeKernel( long long *input_matrix, long long *result_matrix, int img_width, int img_height)
{
	__shared__ long long cache[TILE_WIDTH][TILE_WIDTH+1];
	
	unsigned int xIndex = blockIdx.x * TILE_WIDTH + threadIdx.x;
	unsigned int yIndex = blockIdx.y * TILE_WIDTH + threadIdx.y;
	if((xIndex < img_width) && (yIndex < img_height))
	{
		unsigned int index_in = yIndex * img_width + xIndex;
		cache[threadIdx.y][threadIdx.x] = input_matrix[index_in];
	}

	__syncthreads();

	xIndex = blockIdx.y * TILE_WIDTH + threadIdx.x;
	yIndex = blockIdx.x * TILE_WIDTH + threadIdx.y;
	if((xIndex < img_height) && (yIndex < img_width))
	{
		unsigned int index_out = yIndex * img_width + xIndex;
		result_matrix[index_out] = cache[threadIdx.x][threadIdx.y];
	}
}

int main()
{
	int width = 10 , height = 5;
	long long* input_matrix = (long long*)malloc(width*height*sizeof(long long));
	long long* result_matrix = (long long*)malloc(width*height*sizeof(long long));

	for(int i = 0; i < width*height; i++)
	{
		input_matrix[i] = i;
	}

	//print the input 
	for(int i = 0; i < width*height; i++)
	{
		printf("%lld ", input_matrix[i]);
		if((i+1)%width == 0)
			printf("\n");
	}

	long long *d_input_matrix, *d_result_matrix;
	cudaMalloc((void**)&d_input_matrix, width*height*sizeof(long long));
	cudaMalloc((void**)&d_result_matrix, width*height*sizeof(long long));
	
	cudaMemcpy(d_input_matrix, input_matrix, width*height*sizeof(long long), cudaMemcpyHostToDevice);

	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
	dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

	efficientTransposeKernel<<<dimGrid, dimBlock>>>(d_input_matrix, d_result_matrix, width, height);

	cudaMemcpy(result_matrix, d_result_matrix, width*height*sizeof(long long), cudaMemcpyDeviceToHost);

	for(int i = 0; i < width*height; i++)
	{
		printf("%lld ", result_matrix[i]);
		if((i+1)%width == 0)
			printf("\n");
	}

}