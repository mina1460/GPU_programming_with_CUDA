#include <iostream>
#include <string>
#include "time_helper.h"
#define cimg_display 0
#define cimg_use_jpeg
#include "CImg.h"
#include <vector>
#include <cuda_runtime.h>
using namespace cimg_library;
using namespace std;
#define BLOCK_SIZE 32

#define cudaCheckError() {                                                                  \
 cudaError_t e=cudaGetLastError();                                                          \
 if(e!=cudaSuccess) {                                                                       \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));            \
   exit(0);                                                                                 \
 }}    

 

__global__ void privatized_interleaved_histogram(int32_t *d_input, int32_t *d_histogram, int width, int height, int numBins) {
   
    extern __shared__ int32_t smem[];
    
    int bin_size = 256 / numBins;
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
   
    // // pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // // grid dimensions
    int n_of_threads_in_x = blockDim.x * gridDim.x;
    int n_of_threads_in_y = blockDim.y * gridDim.y;

    int32_t bin_val = 0;
  
    int local_thread = ty * blockDim.x + tx;
    // initialize private histogram

    if (local_thread < numBins)
    {
        smem[local_thread] = 0;
    }
    __syncthreads();
    for (int col = x; col < width; col += n_of_threads_in_x)
    { 
        for (int row = y; row < height; row += n_of_threads_in_y) { 
            
                bin_val = d_input[x + y * width] / bin_size;
                if(bin_val >= numBins){
                    bin_val = numBins - 1;
                }
                atomicAdd(&smem[bin_val], 1);
        
        }
    }

    __syncthreads();

    if(local_thread < numBins){
        atomicAdd(&d_histogram[local_thread], smem[local_thread]);
    }
}

long long GPU_compute_histogram(int32_t* input_values, int32_t* h_histogram, int img_width, int img_height, int num_of_hist_bins){
    int32_t* d_input_values;
    int32_t* d_histogram;
    
    cudaMalloc((void**)&d_input_values, img_width*img_height*sizeof(int32_t));
    cudaCheckError();
    cudaMemcpy(d_input_values, input_values, img_width*img_height*sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaCheckError();

    const int value = 0;
    cudaMalloc((void**)&d_histogram, num_of_hist_bins*sizeof(int32_t));
    cudaCheckError();
    cudaMemset(d_histogram, value, num_of_hist_bins*sizeof(int32_t));
    cudaCheckError();


    // launch kernel with number of rows = img_height, number of columns = 1
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(ceil(img_width/(float)BLOCK_SIZE), ceil(img_height/(float)BLOCK_SIZE));
    
    
    //timer 
    auto start = std::chrono::high_resolution_clock::now();
    privatized_interleaved_histogram<<<grid, block, num_of_hist_bins * sizeof(int32_t)>>>(d_input_values, d_histogram, img_width, img_height, num_of_hist_bins);
    cudaCheckError();
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    long long duration = get_time_diff(start, end, nanoseconds);


    cudaMemcpy(h_histogram, d_histogram, num_of_hist_bins*sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaCheckError(); 

    cudaFree(d_input_values);
    cudaCheckError();

    cudaFree(d_histogram);
    cudaCheckError() ;

    return duration;
    
}



void compute_histogram(int32_t* input_values, int32_t* histogram, int img_width, int img_height, int num_of_hist_bins, int max){
    int32_t pixel_value = 0;
    int bin = 0;
    int bin_size = max/num_of_hist_bins;

    for(int i=0; i<img_height*img_width; i++){
        pixel_value = input_values[i];
        
        bin = pixel_value / bin_size;
        if(bin >= num_of_hist_bins){
            // assign to last bin as they are the result of integer division 
            // for example (256 / 10 = 25.6) so the 0.6 will accumulate in the last bin
            bin = num_of_hist_bins - 1;
        }
        histogram[bin] ++;
    }
}

void test_compute_histogram(){
    int32_t input_test_data[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 ,12, 13, 14, 15};
    int32_t histogram[4] = {0};
    compute_histogram(input_test_data, histogram, 4, 4, 4, 16);
    for(int i=0; i<4; i++){
        if (histogram[i] != 4){
            cout << "test_compute_histogram failed" << endl;
            return;
        }
        
    }
    cout << "1: test_compute_histogram passed!!" << endl;
    int32_t input_test_data2[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ,1, 1, 1, 1, 1};
    int32_t histogram2[4] = {0};
    compute_histogram(input_test_data2, histogram2, 4, 4, 4, 16);
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
float get_GFLOPS(int img_width, int img_height, long long time, time_unit t_unit){

    // std::cout << "\nFLOPS: " << flops << "\n";
    float flops = 2 * img_width * img_height;

    float factor = 0;
    switch(t_unit){
        case nanoseconds:
            factor = 1e-9;
            break;
        case microseconds:
            factor = 1e-6;
            break;
        case milliseconds:
            factor = 1e-3;
            break;
        case seconds:
            factor = 1;
            break;
        default:
            throw std::invalid_argument("Invalid time unit\n");
    }
    // std::cout << "\nTime in seconds: " << time * factor << "\n";
    return flops / (factor * time);
}



int main(int argc, char* argv[]){
    if(argc != 2){
        cout << "Usage: " << argv[0] << " <img_file_path>" << endl;
        return -1;
    }
    test_compute_histogram();
    
    string img_file_path = argv[1];
    CImg<int32_t> img(img_file_path.c_str());
    int img_width = img.width();
    int img_height = img.height();
    int img_channels = img.spectrum();
    int img_size = img_width * img_height * img_channels;
    img_width = 6000, img_height = 4000;
    int32_t* img_values = new int32_t[img_width*img_height];
    for(int i=0; i<img_size; i++){
        img_values[i] = rand() % 256;
    }
    // int32_t* img_values = img.data();
    cout << "Image width: " << img_width << endl;
    cout << "Image height: " << img_height << endl;
    cout << "Image channels: " << img_channels << endl;
    cout << "Image size: " << img_size << endl;
    
    cout << endl << "Please enter the number of histogram bins to use: ";
    int num_bins;
    cin >> num_bins;
    int32_t* histogram = new int32_t[num_bins];
    memset(histogram, 0, num_bins*sizeof(int32_t)); 
    //timer 
    cout <<"CPU Results\n";
    auto start = chrono::high_resolution_clock::now();
    compute_histogram(img_values, histogram, img_width, img_height, num_bins, 256);
    auto end = chrono::high_resolution_clock::now();
    long long duration =  get_time_diff(start, end, nanoseconds);
    cout << "CPU compute histogram time: " << duration << " ns" << endl;
    cout << "CPU compute histogram GFLOPS: " << get_GFLOPS(img_width, img_height, duration, nanoseconds) << endl;


    cout <<"GPU Results\n";

    int32_t* gpu_histogram = new int32_t[num_bins];
    memset(gpu_histogram, 0, num_bins*sizeof(int32_t));
    auto start2 = chrono::high_resolution_clock::now();
    long long gpu_duration = GPU_compute_histogram(img_values, gpu_histogram, img_width, img_height, num_bins);
    auto end2 = chrono::high_resolution_clock::now();
    cout << "GPU compute histogram time without Transfer: " << gpu_duration << " ns" << endl;
    cout << "GPU compute histogram time with Transfer: " << get_time_diff(start2, end2, nanoseconds) << " ns" << endl;
    cout << "GPU compute histogram GFLOPS without Transfer: " << get_GFLOPS(img_width, img_height, gpu_duration, nanoseconds) << endl;
    cout << "GPU compute histogram GFLOPS with Transfer: " << get_GFLOPS(img_width, img_height, get_time_diff(start2, end2, nanoseconds), nanoseconds) << endl;

    // print gpu histogram
    // for(int i=0; i<num_bins; i++){
    //     cout << "Bin " << i << ": " << histogram[i]<< " " << gpu_histogram[i] << endl;
    // }

    if(compare(histogram, gpu_histogram, num_bins)){
        cout << "GPU_compute_histogram passed!!" << endl;
    }
    else{
        cout << "GPU_compute_histogram failed!!" << endl;
    }

    return 0;
}




