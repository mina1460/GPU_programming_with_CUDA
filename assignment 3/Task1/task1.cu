#include <iostream>
#include <string>
#include "time_helper.h"
#include <cuda_runtime.h>
#define cimg_display 0
#define cimg_use_jpeg
#include "CImg.h"
using namespace cimg_library;

using namespace std;


#define MAX_MASK_SIZE 9
#define MAX_MASK_HEIGHT 3
#define MAX_MASK_WIDTH 3

__constant__ float d_mask[MAX_MASK_SIZE];

#define TILE_SIZE 3
#define BLOCK_SIZE ( TILE_SIZE + MAX_MASK_WIDTH - 1 )

#define cudaCheckError() {                                                                  \
 cudaError_t e=cudaGetLastError();                                                          \
 if(e!=cudaSuccess) {                                                                       \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));            \
   exit(0);                                                                                 \
 }}    

const float blur_kernel[3][3] = {
    {0.0625, 0.125, 0.0625},
    {0.125, 0.25, 0.125},
    {0.0625, 0.125, 0.0625}
};

const float emboss_kernel[3][3] = {
    {-2, -1, 0},
    {-1, 1, 1},
    {0, 1, 2}
};

const float outline_kernel[3][3] = {
    {-1, -1, -1},
    {-1, 8, -1},
    {-1, -1, -1}
};

const float sharpen_kernel[3][3] = {
    {0, -1, 0},
    {-1, 5, -1},
    {0, -1, 0}
};

const float left_sobel_kernel[3][3] = {
    {1, 0, -1},
    {2, 0, -2},
    {1, 0, -1}
};

const float right_sobel_kernel[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

const float top_sobel_kernel[3][3] = {
    {1, 2, 1},
    {0, 0, 0},
    {-1, -2, -1}
};

const float bottom_sobel_kernel[3][3] = {
    {-1, -2, -1},
    {0, 0, 0},
    {1, 2, 1}
};

// array of pointers to the kernels
const float *kernels[8] = {
    blur_kernel[0],
    emboss_kernel[0],
    outline_kernel[0],
    sharpen_kernel[0],
    left_sobel_kernel[0],
    right_sobel_kernel[0],
    top_sobel_kernel[0],
    bottom_sobel_kernel[0],
};
const string kernel_names[8] = {
    "blur",
    "emboss",
    "outline",
    "sharpen",
    "left_sobel",
    "right_sobel",
    "top_sobel",
    "bottom_sobel",
};

__global__ void convolution_2D_tiled_kernel(float *d_input, float *d_output, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int mask_width = MAX_MASK_WIDTH;
    __shared__ float N_ds[TILE_SIZE][TILE_SIZE];
    if (row < height && col < width) {
        N_ds[threadIdx.y][threadIdx.x] = d_input[row * width + col];
    } 
    __syncthreads();

    int tile_row_start = blockIdx.y * blockDim.y ;
    int tile_row_end = tile_row_start +  blockDim.y;
    int tile_col_start = blockIdx.x * blockDim.x ;
    int tile_col_end = tile_col_start+ blockDim.x;

    if(threadIdx.x == 3 && threadIdx.y == 2 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("tile_row_start: %d, tile_row_end: %d, tile_col_start: %d, tile_col_end: %d\n", tile_row_start, tile_row_end, tile_col_start, tile_col_end);
        //printing blockdim 
        printf("blockDim.x: %d, blockDim.y: %d\n", blockDim.x, blockDim.y);
    }

    float pvalue = 0;
    int row_start = row - (mask_width / 2);
    int col_start = col - (mask_width / 2);

    for(int i = 0; i < mask_width; i++) {
        for(int j = 0; j < mask_width; j++) {
            int cur_row = row_start + i;
            int cur_col = col_start + j;
            if (cur_row < 0){
                cur_row = 0;
            }  
            else if(cur_row >= height){
                cur_row = height - 1;
            }
            
            if (cur_col < 0){
                cur_col = 0;
            }  
            else if(cur_col >= width){
                cur_col = width - 1;
            }

            int x = threadIdx.x + (cur_col - tile_col_start);
            int y = threadIdx.y + (cur_row - tile_row_start); 

            
            if(cur_row >= tile_row_start && cur_row < tile_row_end 
            && cur_col >= tile_col_start && cur_col < tile_col_end 
            && x < TILE_SIZE && y < TILE_SIZE && x>=0 && y>=0) {
                if(threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
                printf("cur_row: %d, cur_col: %d, x: %d, y: %d \n", cur_row, cur_col, x, y);
                //printing row and column start
                printf("row_start: %d, col_start: %d\n", row_start, col_start);
                //print i and j 
                printf("i: %d, j: %d\n", i, j);
            }
                pvalue += N_ds[y][x] * d_mask[i * mask_width + j];
            } 
            else if(cur_row < height && cur_col < width) {
                // uses general caching 
                pvalue += d_input[cur_row * width + cur_col] * d_mask[i * mask_width + j];
            }
        }


        if(row * width + col < width*height)
                d_output[row * width + col] = pvalue;
    }
}

__global__ void convolution_output_tiling_2D_kernel(float *d_input, float *d_output, int width, int height) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row_o = blockIdx.y * TILE_SIZE + ty; 
    int col_o = blockIdx.x * TILE_SIZE + tx;
    
    int row_i = row_o - MAX_MASK_WIDTH/2; 
    int col_i = col_o - MAX_MASK_WIDTH/2;
    // this is the same block width = tile size + mask width - 1
    __shared__ float N_ds[TILE_SIZE+MAX_MASK_WIDTH-1][TILE_SIZE+MAX_MASK_HEIGHT-1];
    // grid size is width / tile size
    if(row_i < 0) row_i = 0;
    if(row_i >= height) row_i = height - 1;
    if(col_i < 0) col_i = 0;
    if(col_i >= width) col_i = width - 1;

    N_ds[ty][tx] = d_input[row_i*width+col_i];
    __syncthreads();

    float output = 0.0f;
    if(ty < TILE_SIZE && tx < TILE_SIZE){
        for(int i = 0; i < MAX_MASK_WIDTH; i++) { 
            for(int j = 0; j < MAX_MASK_WIDTH; j++) {
                output += d_mask[i*MAX_MASK_WIDTH+j] * N_ds[i+ty][j+tx]; 
            }
        }
    
        if(row_o < height && col_o < width){ 
            d_output[row_o*width + col_o] = output;
        } 
    }
    return;
}
    
    
void GPU_apply_convolution_kernel(float* h_image, int img_width, int img_height, const float *h_mask, float* output, float *output_2) {

    float *d_image;
    float *d_kernel;
    float *d_result;
    float *d_result_2;
    
    cudaMalloc((void **)&d_image, img_width * img_height* sizeof(float));
    cudaCheckError();
    
    // copy filter to symbol memory
    cudaMalloc((void **)&d_result, img_width * img_height * sizeof(float));
    cudaCheckError();

    cudaMalloc((void **)&d_result_2, img_width * img_height * sizeof(float));
    cudaCheckError();

    // copy image to device
    cudaMemcpy(d_image, h_image, img_width * img_height * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError();

    // and copy filter to static memory
    cudaMemcpyToSymbol(d_mask, h_mask, 9 * sizeof(float));
    cudaCheckError();

    
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(ceil((float)img_width / TILE_SIZE), ceil((float)img_height / TILE_SIZE));
    convolution_output_tiling_2D_kernel<<<dimGrid, dimBlock>>>(d_image, d_result, img_width,img_height);
    cudaDeviceSynchronize();
    cudaCheckError();

    dim3 dimBlock_generalCache(TILE_SIZE, TILE_SIZE);
    convolution_2D_tiled_kernel<<<dimGrid, dimBlock_generalCache>>>(d_image, d_result_2, img_width,img_height);
    cudaDeviceSynchronize();
    cudaCheckError();
    
    // copy result back to host
    
    cudaMemcpy(output, d_result, img_width * img_height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError();

    cudaMemcpy(output_2, d_result_2, img_width * img_height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError();
    
    cudaFree(d_image); 
    cudaCheckError();
    
    cudaFree(d_result);
    cudaCheckError();
    
    cudaFree(d_result_2);
    cudaCheckError();
    return;
}


void apply_convolution_kernel(float* input_data, float* output_data, const float* kernel, int kernel_width, int kernel_height, int img_width, int img_height)
{
    int output_index, input_r, input_c, input_index, kernel_index;
    float p_value;
    for(int pixel_r=0; pixel_r < img_height; pixel_r++){
        for(int pixel_c=0; pixel_c < img_width; pixel_c++){
            output_index = pixel_r * img_width + pixel_c;
            p_value = 0;
            for(int kernel_r=0; kernel_r < kernel_height; kernel_r++){
                for(int kernel_c=0; kernel_c < kernel_width; kernel_c++){
                
                    input_r = pixel_r + kernel_r - kernel_height/2;
                    input_c = pixel_c + kernel_c - kernel_width/2;
                    
                    if(input_r < 0) input_r = 0;
                    else if(input_r >= img_height) input_r = img_height - 1;
    
                    if(input_c < 0) input_c = 0;   
                    else if(input_c >= img_width) input_c = img_width - 1;
                    
                    input_index = input_r * img_width + input_c;
                    kernel_index = kernel_r * kernel_width + kernel_c;
                    
                    p_value += input_data[input_index] * kernel[kernel_index];
                }
            }
            output_data[output_index] = p_value;
        }
    }
}

bool compare_with_tolerance(float* a, float* b, int size, float tolerance){
    for(int i=0; i < size; i++){
        if(abs(a[i] - b[i]) > tolerance){
            cout << "a[" << i << "] = " << a[i] << " b[" << i << "] = " << b[i] << endl;
            return false;
        }
    }
    return true;
}


int main(int argc, char *argv[])
{
    if(argc != 2)
    {
        cout << "Usage: " << argv[0] << " <img_file_path>" << endl;
        return -1;
    }

    string img_path = argv[1];

    // multiline c++ string 
    const char *  choose_conv_kernel_msg = 
    R"(
        enter a number to choose a convolution kernel:
        1. Blur 
        2. Emboss
        3. Outline
        4. Sharpen
        5. Left Sobel
        6. Right Sobel
        7. Top Sobel
        8. Bottom Sobel
    )";

    cout << choose_conv_kernel_msg << endl;
    int kernel_choice(1);
    cin >> kernel_choice;


    if (kernel_choice < 1 || kernel_choice > 8)
    {
        cout << "invalid kernel choice" << endl;
        return -2;
    }
    kernel_choice -= 1;
    // get the kernel pointer
    const float *kernel = kernels[kernel_choice];

    // open the image with cimg library
    CImg<float> img(img_path.c_str());

    // get the image dimensions
    int width = img.width();
    int height = img.height();
    int depth = img.depth();

    cout << "image dimensions: " << width << "x" << height << "x" << depth << endl;

    string img_basename = img_path.substr(img_path.find_last_of("/\\") + 1);
    cout << "image basename: " << img_basename << endl;
    
    // create a new image to store the result
    float* orig_values = img.data();
    float* result_values = new float[width * height * depth];
    
    cout << "Applying kernel " << kernel_names[kernel_choice] << endl;
    
    for(int i=0; i < width * height * depth; i++){
        result_values[i] = 0;
    }
    
    // apply the convolution kernel
    apply_convolution_kernel(orig_values, result_values, kernel, 3, 3, width, height);
    
    // create a new image to store the result
    CImg<float> result_img(result_values, width, height, 1, depth);
    string result_img_path = img_basename + "_" + kernel_names[kernel_choice] + ".jpeg";
    result_img.save(result_img_path.c_str());
    cout << "result image saved to " << result_img_path << endl;    
    
    float* gpu_result_values = new float[width * height * depth];
    float* gpu_result_values_2 = new float[width * height * depth];
    GPU_apply_convolution_kernel(img.data(),img.width(), img.height(), kernel, gpu_result_values,gpu_result_values_2);

    CImg<float> gpu_result_img(gpu_result_values, width, height, 1, depth);
    
    CImg<float> gpu_result_img_2(gpu_result_values_2, width, height, 1, depth);

    string gpu_result_img_path = img_basename + "_" + kernel_names[kernel_choice] + "_gpu.jpeg";
    string gpu_result_img_path_2 = img_basename + "_" + kernel_names[kernel_choice] + "_gpu_2.jpeg";
    gpu_result_img.save(gpu_result_img_path.c_str());
    cout << "gpu result image saved to " << gpu_result_img_path << endl;
    gpu_result_img_2.save(gpu_result_img_path_2.c_str());
    cout << "gpu result_2 image saved to " << gpu_result_img_path << endl;


    bool res = compare_with_tolerance(result_values, gpu_result_values, width * height * depth, 1e-3);
    if(res){
        cout << "CPU is the same as GPU results\n\n";
    }
    else{
        cout << "Failed!!! CPU is different from GPU\n\n";
    }

    bool res2 = compare_with_tolerance(result_values, gpu_result_values_2, width * height * depth, 1e-3);
    if(res2){
        cout << "CPU is the same as GPU results\n";
    }
    else{
        cout << "Failed!!! CPU is different from GPU\n";
    }
    delete[] result_values;
    return 0;
}
