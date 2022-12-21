# GPU_programming_with_CUDA
This repo contains all our answers for famous problems that can be solved in parallel and make use of massively parallel architectures of General Purpose Graphics Processing Units (GPGPUS) using nVidia CUDA.
We implemented everything from scratch. For example, the famous MVMul (Matrix-Vector multiplicatoin) and MMMul (Matrix-Matrix Multiplication) can be achieved by easily calling an NVidia CUDA function with little to no preparation; however, we decided to go the hard way of learning all the little bits of the CUDA by doing everything from Scratch. 

We provide all the sequential implementation for proving correctness of the parallel algorithm, timer helper functions, GFLOPS calculator, and a modular Matrix class.

The repo has 3 folders, with each folder containing multiple mini-projects:
## assignment 1:
#### 1. Simple vector addition
#### 2. Image value-scaler with 2D Blocks
#### 3. Matrix addition 

## assignment 2: 
#### 1. Matrix multiplication without tiling or shared memory
#### 2. Matrix multiplication with tiling and shared memory 
#### 3. Same as 2 but with thread-work granularity 

## assignment 3: 
#### 2D image convolution on greyscale images
- We support gaussian Blur, emboss, sharpen, sobel (Top, Bottom, left, right), and Outline filters. 
- we use the constant memory for the masks, and general caching (L2 cache) for halo cells. For ghost cells, we replicate the nearest edge values.
- The code uses the famous Cimg library for reading jpeg, jpg, and PNG images. You are advised to use ImageMagik instead as it is more generic and works easier with MacBooks.
- Note: We also provided an implementation for output-tiling 2D convolution where the input tile contains all the values a convolution kernel needs, and the output tile is smaller. For example, if tile size is 8x8 and the filter is 3x3. The output tile will be a 6x6 tile.

#### Summed-area Table calculator with Prefix sum. 
- This is by the far the most challenging problem. We used Brent-kung and Kogge-stone implementations for a work-efficient parallel implementation. 
- We provided an O(1) Query answering function afterwards. 

#### A 2D histogram calculator with custom bins. 
- We provide an efficent memory-coalesced kernel that uses tiling and privatization for fewer global memory traffic and faster atomics. 
- This works with greyscale images and any custom number of bins you want.
