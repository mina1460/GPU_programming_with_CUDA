#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

void CPU_add_vectors(float *vec1, float *vec2, float *out, int n) {
    for (int i=0; i<n; i++) {
        out[i] = vec1[i] + vec2[i];
    }
}

__global__ void addKernel(float *d_vec1, float *d_vec2, float *d_out, int n){
    
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    i = i * 4;
    for(int index=0; index<4; index++)
    {
        if(i+index < n)
        {
            d_out[i+index] = d_vec1[i+index] + d_vec2[i+index];
        }
    }
}

void GPU_add_vectors(float *vec1, float *vec2, float *out, int n, int threadsPerBlock){

    int size = n * sizeof(float);
    float *d_vec1, *d_vec2, *d_out;

    cudaMalloc((void **) &d_vec1, size);
    cudaCheckError();

    cudaMalloc((void **) &d_vec2, size);
    cudaCheckError();

    cudaMalloc((void **) &d_out, size);     
    cudaCheckError();

    cudaMemcpy(d_vec1, vec1, size, cudaMemcpyHostToDevice);
    cudaCheckError();

    cudaMemcpy(d_vec2, vec2, size, cudaMemcpyHostToDevice);
    cudaCheckError();

    int number_of_blocks = ceil(n / 4.0 / (float)threadsPerBlock);
    printf("Number of blocks: %d\n", number_of_blocks);

    dim3 dimGrid(ceil(number_of_blocks), 1, 1);
    dim3 dimBlock(threadsPerBlock, 1, 1);
    
    addKernel<<<dimGrid, dimBlock>>>(d_vec1, d_vec2, d_out, n);
    cudaCheckError();

    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);
    cudaCheckError();
    
    cudaFree(d_vec1);   cudaCheckError();
    
    cudaFree(d_vec2);   cudaCheckError();
    
    cudaFree(d_out);    cudaCheckError();
}

bool compare_vectors(const float *v1, const float *v2, int n, bool idenfity_block_and_thread = true) {
    bool flag = true;
    for (int i=0; i<n; i++) {
        if (v1[i] != v2[i]) {
            printf("v1[%d] = %f, v2[%d] = %f\n", i, v1[i], i, v2[i]);
            flag = false;
            if (idenfity_block_and_thread) {
                printf("Block: %d and Thread: %d\n", i/1024/4, i%(1024));
            }
            
            break;
        }
    }
    return flag;
}

void print_vector(const float *v, int n) {
    for (int i=0; i<n; i++) {
        printf("%.2f ", v[i]);
    }
    printf("\n");
}

int main(){
    
    srand(time(NULL));
    int size = rand() % 100000 + 1;
    printf("Size of the vector: %d\n", size);
    float* vec1 = (float *) calloc(size, sizeof(float));
    float* vec2 = (float *) calloc(size, sizeof(float));
    float* cpu_out = (float *) calloc(size, sizeof(float));
    float* gpu_out = (float *) calloc(size, sizeof(float));
    
    float upper_limit = 1000.0;

    for(int i=0; i<size; i++){
        vec1[i] = (float)rand()/(RAND_MAX/upper_limit);
        vec2[i] = (float)rand()/(RAND_MAX/upper_limit);
    }
    int nDevices;

    cudaGetDeviceCount(&nDevices);
    int maxThreadsPerBlock = INT_MAX;
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        maxThreadsPerBlock = min(prop.maxThreadsPerBlock, maxThreadsPerBlock);
        printf("  max number of theads: %i\n", maxThreadsPerBlock);
    }

    GPU_add_vectors(vec1, vec2, gpu_out, size, maxThreadsPerBlock);
    CPU_add_vectors(vec1, vec2, cpu_out, size);
    

    if (compare_vectors(cpu_out, gpu_out, size)){
        printf("The vectors are equal\n");
    }
    else{
        printf("The vectors are not equal\n");
    }

    printf("Input vector 1:\n");
    print_vector(vec1, size);
    printf("Input vector 2:\n");
    print_vector(vec2, size);
    printf("GPU output vector:\n");
    print_vector(gpu_out, size);

    return 0;
}