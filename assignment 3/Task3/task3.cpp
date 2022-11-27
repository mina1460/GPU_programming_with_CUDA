#include <iostream>
#include <string>
#include "time_helper.h"
#define cimg_display 1
#define cimg_use_jpeg
#include "CImg.h"
#include <vector>

using namespace cimg_library;
using namespace std;

void compute_histogram(int32_t* input_values, int32_t* histogram, int img_width, int img_height, int num_of_hist_bins){
    int pixel_value = 0;
    int bin = 0;
    for(int i=0; i<img_height*img_width; i++){
        pixel_value = input_values[i];
        // pixel_value = max(0, pixel_value);
        // pixel_value = min(255, pixel_value);
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
    int32_t input_test_data[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 ,12, 13, 14, 15};
    int32_t histogram[4] = {0};
    compute_histogram(input_test_data, histogram, 4, 4, 4);
    for(int i=0; i<4; i++){
        if (histogram[i] != 4){
            cout << "test_compute_histogram failed" << endl;
            return;
        }
        
    }
    cout << "1: test_compute_histogram passed!!" << endl;
    int32_t input_test_data2[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ,1, 1, 1, 1, 1};
    int32_t histogram2[4] = {0};
    compute_histogram(input_test_data2, histogram2, 4, 4, 4);
    if(histogram2[0] != 16 && histogram2[1] != 0 && histogram2[2] != 0 && histogram2[3] != 0){
        cout << "test_compute_histogram failed" << endl;
        return;
    }
    cout << "2: test_compute_histogram passed!!" << endl;
    cout << endl;
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
    int32_t* img_values = img.data();
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


    return 0;
}

