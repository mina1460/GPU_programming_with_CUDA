#include <iostream>
#include <string>
#include "time_helper.h"

#define cimg_display 1
#define cimg_use_jpeg
#include "CImg.h"
using namespace cimg_library;

using namespace std;

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


void apply_convolution_kernel(float* input_data, float* output_data, const float* kernel, int kernel_width, int kernel_height, int img_width, int img_height){
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
    //cin >> kernel_choice;

    if (kernel_choice < 1 || kernel_choice > 8)
    {
        cout << "invalid kernel choice" << endl;
        return -2;
    }

    // get the kernel pointer
    const float *kernel = kernels[kernel_choice - 1];

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
    
    for(int kernel_choice=7; kernel_choice >=0 ; kernel_choice--){
        cout << "Applying kernel " << kernel_names[kernel_choice] << endl;
        
        for(int i=0; i < width * height * depth; i++){
            result_values[i] = 0;
        }
        const float *kernel = kernels[kernel_choice];
        // apply the convolution kernel
        apply_convolution_kernel(orig_values, result_values, kernel, 3, 3, width, height);
        
        // create a new image to store the result
        CImg<float> result_img(result_values, width, height, 1, depth);
        string result_img_path = img_basename + "_" + kernel_names[kernel_choice] + ".jpeg";
        result_img.save(result_img_path.c_str());
        cout << "result image saved to " << result_img_path << endl;

    }
    
    
    
    delete[] result_values;
    return 0;
}
