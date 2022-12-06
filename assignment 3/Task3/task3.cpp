#include <iostream>
#include <string>
#include "time_helper.h"
#define cimg_display 1
#define cimg_use_jpeg
#include "CImg.h"
#include <vector>

using namespace cimg_library;
using namespace std;
#define BLOCK_SIZE 32

__global__ void privatized_interleaved_histogram(unsigned char *d_input, int32_t *d_histogram, int width, int height, int numBins) {

    extern __shared__ int smem[];
    
    int bin_size = 256 / numBins;

    // pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // grid dimensions
    int n_of_threads_in_x = blockDim.x * gridDim.x;
    int n_of_threads_in_y = blockDim.y * gridDim.y;

    // linear thread index within 2D block
    int tid = threadIdx.x + threadIdx.y * blockDim.x; 

    if(tid < numBins) 
        smem[i] = 0;
    __syncthreads();
    
    for (int col = x; col < width; col += n_of_threads_in_x)
    { 
        for (int row = y; row < height; row += n_of_threads_in_y) { 
            unsigned int bin_val = d_input[col + row * width] / bin_size;
            if(bin_val >= numBins){
                bin_val = numBins - 1;
            }
            atomicAdd(&smem[bin_val], 1);
        }
    }
    __syncthreads();

    if(tid < numBins)
        atomicAdd(&d_histogram[i], smem[i]);
}

void GPU_compute_histogram(char* input_values, int32_t* h_histogram, int img_width, int img_height, int num_of_hist_bins){
    unsigned char* d_input_values;
    int32_t* d_histogram;
    
    cuda_malloc((void**)&d_input_values, img_width*img_height*sizeof(unsigned char));
    cuda_memcpy(d_input_values, input_values, img_width*img_height*sizeof(unsigned char), cudaMemcpyHostToDevice);


    cuda_malloc((void**)&d_histogram, num_of_hist_bins*sizeof(int32_t));
    cuda_memset(d_histogram, 0, num_of_hist_bins*sizeof(int32_t));

    // launch kernel with number of rows = img_height, number of columns = 1
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(ceil(img_width/(float)BLOCK_SIZE), ceil(img_height/(float)BLOCK_SIZE));
    privatized_interleaved_histogram<<<grid, block, num_of_hist_bins*sizeof(int32_t)>>>(d_input_values, d_histogram, img_width, img_height, num_of_hist_bins);

    int32_t* h_histogram = (int32_t*)malloc(num_of_hist_bins*sizeof(int32_t));
    cuda_memcpy(h_histogram, d_histogram, num_of_hist_bins*sizeof(int32_t), cudaMemcpyDeviceToHost);

    cuda_free(d_input_values);
    cuda_free(d_histogram);
    
}



void compute_histogram(unsigned char* input_values, int32_t* histogram, int img_width, int img_height, int num_of_hist_bins){
    unsigned char pixel_value = 0;
    int bin = 0;
    for(int i=0; i<img_height*img_width; i++){
        pixel_value = input_values[i];
        
        bin = pixel_value / (256/num_of_hist_bins);
        if(bin >= num_of_hist_bins){
            // assign to last bin as they are the result of integer division 
            // for example (256 / 10 = 25.6) so the 0.6 will accumulate in the last bin
            bin = num_of_hist_bins - 1;
        }
        histogram[bin] ++;
    }
}

void test_compute_histogram(){
    unsigned char input_test_data[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 ,12, 13, 14, 15};
    int32_t histogram[4] = {0};
    compute_histogram(input_test_data, histogram, 4, 4, 4);
    for(int i=0; i<4; i++){
        if (histogram[i] != 4){
            cout << "test_compute_histogram failed" << endl;
            return;
        }
        
    }
    cout << "1: test_compute_histogram passed!!" << endl;
    unsigned char input_test_data2[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ,1, 1, 1, 1, 1};
    int32_t histogram2[4] = {0};
    compute_histogram(input_test_data2, histogram2, 4, 4, 4);
    if(histogram2[0] != 16 && histogram2[1] != 0 && histogram2[2] != 0 && histogram2[3] != 0){
        cout << "test_compute_histogram failed" << endl;
        return;
    }
    cout << "2: test_compute_histogram passed!!" << endl;
    cout << endl;
}

bool compare(int32_t* cpu_results, int32_t* gpu_results, int size){
    for(int i=0; i<size; i++){
        if(cpu_results[i] != gpu_results[i]){
            cout << "compare failed at index: " << i << endl;
            cout <<"Cpu: " << cpu_results[i] << " Gpu: " << gpu_results[i] << endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]){
    if(argc != 2){
        cout << "Usage: " << argv[0] << " <img_file_path>" << endl;
        return -1;
    }
    test_compute_histogram();
    
    string img_file_path = argv[1];
    CImg<unsigned char> img(img_file_path.c_str());
    int img_width = img.width();
    int img_height = img.height();
    int img_channels = img.spectrum();
    int img_size = img_width * img_height * img_channels;
    unsigned char* img_values = img.data();
    cout << "Image width: " << img_width << endl;
    cout << "Image height: " << img_height << endl;
    cout << "Image channels: " << img_channels << endl;
    cout << "Image size: " << img_size << endl;
    
    cout << endl << "Please enter the number of histogram bins to use: ";
    int num_bins;
    cin >> num_bins;
    int32_t* histogram = new int32_t[num_bins];
    compute_histogram(img_values, histogram, img_width, img_height, num_bins);

    for(int i=0; i<num_bins; i++){
        cout << "Bin " << i << ": " << histogram[i] << endl;
    }
    int32_t* gpu_histogram;
    GPU_compute_histogram(img_values, gpu_histogram, img_width, img_height, num_bins);

    if(compare(histogram, gpu_histogram, num_bins)){
        cout << "GPU_compute_histogram passed!!" << endl;
    }
    else{
        cout << "GPU_compute_histogram failed!!" << endl;
    }

    return 0;
}

