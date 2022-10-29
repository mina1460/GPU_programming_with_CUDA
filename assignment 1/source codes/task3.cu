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


void print_mat(float* mat, int width, int height){
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            printf("%.2f ", mat[i*width + j]);
        }
        printf("\n");
}
}

__global__ void mat_add(float* d_A, float* d_B, float* d_C, int height){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(col>=height){
       return;
    }
    for(int i=0; i<height; i++){
        d_C[i*height + col] += d_A[i*height + col] + d_B[i*height + col];
    }
}

int get_max_threads_per_block(int device_number){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_number);
    return prop.maxThreadsPerBlock;
}

void GPU_add_matrices(const float* h_matA, const float* h_matB, float* h_matC, const int dimA, const int dimB){
    /*
    this function is designed to add two square matrices of the same size
    */
    if(dimA != dimB){
        printf("Error: Matrices are not the same size\n\n");
        return;
    }
    int size = dimA * dimA * sizeof(float);
    float* d_matA;
    float* d_matB;
    float* d_matC;
    
    cudaMalloc((void**)&d_matA, size);
    cudaCheckError();
    cudaMalloc((void**)&d_matB, size);
    cudaCheckError();
    cudaMalloc((void**)&d_matC, size);   
    cudaCheckError();

    cudaMemcpy(d_matA, h_matA, size, cudaMemcpyHostToDevice);
    cudaCheckError();

    cudaMemcpy(d_matB, h_matB, size, cudaMemcpyHostToDevice);
    cudaCheckError();

    int maxThreadsPerBlock = get_max_threads_per_block(0);
    int numBlocks = ceil(dimA / (float)maxThreadsPerBlock);

    mat_add<<<numBlocks, maxThreadsPerBlock>>>(d_matA, d_matB, d_matC, dimA);
    cudaCheckError();

    cudaMemcpy(h_matC, d_matC, size, cudaMemcpyDeviceToHost);
    cudaCheckError();

    cudaFree(d_matA);    cudaCheckError();

    cudaFree(d_matB);    cudaCheckError();

    cudaFree(d_matC);    cudaCheckError();

}

void CPU_add_mat(const float* h_matA, const float* h_matB, float* h_matC, const int x){
    for(int i = 0; i < x*x; i++){
        h_matC[i] = h_matA[i] + h_matB[i];
    }
}

bool compareMatrices(const float* matA, const float* matB, int x){
    for(int i = 0; i < x*x; i++){
        if(matA[i] != matB[i]){
            printf("Matrices are not equal at index %d value 1: %.2f and value 2: %.2f \n", i, matA[i], matB[i]);
            return false;
        }
    }
    return true;
}
int main(){

    srand(time(NULL));

    int dimA = rand()%30000 + 1;
    int dimB = dimA;

    printf("dimension matrix A: %d x %d \n", dimA, dimA);
    printf("dimension matrix B: %d x %d \n", dimB, dimB);
    float* h_matA = (float*)malloc(dimA*dimA*sizeof(float));
    float* h_matB = (float*)malloc(dimB*dimB*sizeof(float));
    float* h_mat_cpu = (float*)calloc(dimA*dimA, sizeof(float));
    float* h_mat_gpu = (float*)calloc(dimA*dimA, sizeof(float));

    for (int i = 0; i < dimA * dimA; i++)
    {
        h_matA[i] = rand();
        h_matB[i] = rand();
    }

    GPU_add_matrices(h_matA, h_matB, h_mat_gpu, dimA, dimB);
    CPU_add_mat(h_matA, h_matB, h_mat_cpu, dimA);
    printf("Matrix A: \n");
    print_mat(h_matA, dimA, dimA);

    printf("Matrix B: \n");
    print_mat(h_matB, dimB, dimB);

    printf("Result matrix: \n");
    print_mat(h_mat_gpu, dimB, dimB);


    if (compareMatrices(h_mat_cpu, h_mat_gpu, dimA)){
        printf("Matrices are equal\n");
    }
    else{
        printf("Matrices are not equal\n");
    }

    return 0;
}