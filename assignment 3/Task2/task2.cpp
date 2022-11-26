#include <iostream>
#include <string>
#include "time_helper.h"
#define cimg_display 1
#define cimg_use_jpeg
#include "CImg.h"
using namespace cimg_library;
using namespace std;


void compute_summed_area_table(int32_t* input_values, int32_t* output_values, int img_width, int img_height){
    int top_left = 0;
    int left = 0; 
    int top = 0;
    for(int i=0; i<img_height; i++){
        for(int j=0; j<img_width; j++){
            top_left = left = top = 0;
            if(i> 0 && j > 0){
                top_left = output_values[(i-1)*img_width + j-1];
                left =  output_values[(i)*img_width + j-1];
                top= output_values[(i-1)*img_width + j];
            }
            else if(i> 0 ){
                top= output_values[(i-1)*img_width + j];
            }
            else if(j> 0 ){
                left =  output_values[(i)*img_width + j-1];
            }
            output_values[i*img_width + j] =   top + left + input_values[i*img_width + j] - top_left; 
        }
    }
}

void test_compute_summed_area_table(){
    int32_t input_values[16] = {5,2,5,2, 3,6,3,6, 5,2,5,2, 3,6,3,6};
    int32_t output_values[16] = {0};
    int32_t correct_values[16] = {5,7,12,14, 8,16,24,32, 13, 23, 36, 46, 16, 32, 48, 64};
    compute_summed_area_table(input_values, output_values, 4, 4);
    for(int i=0; i<4; i++){
        for(int j=0; j<4; j++){
            cout << input_values[i*4 + j] << " ";
        }
        cout << endl;
    }
    cout << endl <<"Results: " << endl << endl;
    for(int i=0; i<4; i++){
        for(int j=0; j<4; j++){
            cout << output_values[i*4 + j] << " ";
        }
        cout << endl;
    }
    for(int i=0; i<16; i++){
        if(output_values[i] != correct_values[i]){
            cout << "Error at index " << i << endl;
            return;
        }
    }
    cout << endl << "Test passed" << endl;
}

int main(int argc, char* argv[]){
    if(argc != 2)
    {
        cout << "Usage: " << argv[0] << " <img_file_path>" << endl;
        return -1;
    }

    string img_path = argv[1];
    // open the image with cimg library
    CImg<int32_t> img(img_path.c_str());

    // get the image dimensions
    int width = img.width();
    int height = img.height();
    int depth = img.depth();

    cout << "image dimensions: " << width << "x" << height << "x" << depth << endl;

    string img_basename = img_path.substr(img_path.find_last_of("/\\") + 1);
    cout << "image basename: " << img_basename << endl;
    
    // create a new image to store the result
    int* orig_values = img.data();
    int* result_values = (int*) calloc(width * height * depth, sizeof(int));

    compute_summed_area_table(orig_values, result_values, width, height);

    CImg<int32_t> result_img(result_values, width, height, depth, 1, true);
    // for (int i = 0; i < height; i++) {
    //     for (int j = 0; j < width; j++) {
    //         cout << result_values[i*width + j] << " ";
    //     }
    //     cout << endl;
    // }
    test_compute_summed_area_table();
    result_img.save("summed_area_table_result.jpeg");

    delete[] result_values;

    return 0;
}

