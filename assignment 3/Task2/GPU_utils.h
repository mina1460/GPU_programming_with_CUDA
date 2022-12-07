#ifndef GPU_utils_H
#define GPU_utils_H


#define cudaCheckError() {                                                                  \
 cudaError_t e=cudaGetLastError();                                                          \
 if(e!=cudaSuccess) {                                                                       \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));            \
   exit(0);                                                                                 \
 }                                                                                          \
}

int get_max_threads_per_block(){
    int max_threads_per_block;
    cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, 0);
    return max_threads_per_block;
}

int get_max_threads_per_multiprocessor(){
    int max_threads_per_multiprocessor;
    cudaDeviceGetAttribute(&max_threads_per_multiprocessor, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
    return max_threads_per_multiprocessor;
}


#endif