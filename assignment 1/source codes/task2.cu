#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MIN(a,b) ((a) < (b) ? (a) : (b))

#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

__global__ void multiply_kernel(int* d_img, int x, int y, int* d_out) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int index = row * x + col;
    if(col < x && row < y) {
       d_out[index] = MIN(d_img[index] * 3, 255);
    }
}


void GPU_multiply_image(int* h_input_img, int* h_output_img, int x, int y){
    int* d_input_img;
    int* d_output_img;
    int size = x*y*sizeof(int);

    cudaMalloc((void**)&d_input_img, size);
    cudaCheckError();
    
    cudaMalloc((void**)&d_output_img, size);
    cudaCheckError();
    
    cudaMemcpy(d_input_img, h_input_img, size, cudaMemcpyHostToDevice);
    cudaCheckError();

    dim3 dimGrid(ceil(x/16.0), ceil(y/16.0), 1);
    dim3 dimBlock(16, 16, 1);

    multiply_kernel<<<dimGrid, dimBlock>>>(d_input_img, x, y, d_output_img);
    cudaCheckError();
    
    cudaMemcpy(h_output_img, d_output_img, size, cudaMemcpyDeviceToHost);
    cudaCheckError();

    cudaFree(d_input_img);
    cudaCheckError();

    cudaFree(d_output_img);
    cudaCheckError();
}

void CPU_multiply_image(int* h_input_img, int* h_output_img, int x, int y){
    for(int i = 0; i < x*y; i++){
        h_output_img[i] = MIN(h_input_img[i] * 3, 255);
    }
}

bool compareImages(int* img1, int* img2, int x, int y){
    for(int i = 0; i < x*y; i++){
        if(img1[i] != img2[i]){
            printf("Images are not equal at index %d value 1: %i and value 2: %i \n", i, img1[i], img2[i]);
            return false;
        }
    }
    return true;
}

void print_vector(const int *v, int n) {
    for (int i=0; i<n; i++) {
        printf("%i ", v[i]);
    }
    printf("\n");
}

int main(){

    int* img = (int*)malloc(1000*800*sizeof(int));
    int* gpu_out = (int*)malloc(1000*800*sizeof(int));
    int* cpu_out = (int*)malloc(1000*800*sizeof(int));

    for(int i=0; i<1000*800; i++){
        img[i] = rand()%256;
    }
    
    printf("generated Image: \n");
    print_vector(img, 1000*800);

    GPU_multiply_image(img, gpu_out, 1000, 800);
    printf("The GPU output is: \n");
    print_vector(gpu_out, 1000*800);

    CPU_multiply_image(img, cpu_out, 1000, 800);
    printf("The CPU output is: \n");
    print_vector(cpu_out, 1000*800);

    if (compareImages(gpu_out, cpu_out, 1000, 800)){
        printf("Images are equal\n");
    } else {
        printf("Images are not equal\n");
    }


    return 0;
}