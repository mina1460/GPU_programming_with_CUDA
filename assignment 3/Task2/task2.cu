#include <iostream>
#include <string>
#include "time_helper.h"
#define cimg_display 0
#define cimg_use_jpeg
#include "CImg.h"
#include <cuda_runtime.h>
#include <vector>
#include "task2_GPU.h"
#include "task_2_gpu.h"

using namespace cimg_library;
using namespace std;


class Point {
    public:
        int x;
        int y;
        Point(int x, int y) : x(x), y(y) {}
};

class query{
    public:
        Point p1;
        Point p2;
        Point p3;
        Point p4;
        query(Point p1, Point p2, Point p3, Point p4) : p1(p1), p2(p2), p3(p3), p4(p4) {}

};


void compute_summed_area_table(long long* input_values, long long* output_values, int img_width, int img_height){
    
    output_values[0] = input_values[0];

    for(int i=1; i<img_width; i++)
        output_values[i] = output_values[i-1] + input_values[i];

    for(int j=1; j<img_height; j++)
        output_values[j*img_width] = output_values[(j-1)*img_width]+ input_values[j*img_width];

    for(int i = 1 ; i < img_height; i++)
    {
        for(int j = 1; j < img_width; j++)
            output_values[i*img_width + j] = input_values[i*img_width + j] + output_values[(i-1)*img_width + j] + output_values[i*img_width + j-1] - output_values[(i-1)*img_width + j - 1];
    }

}

void test_compute_summed_area_table(){
    long long input_values[16] = {5,2,5,2, 3,6,3,6, 5,2,5,2, 3,6,3,6};
    long long output_values[16] = {0};
    long long correct_values[16] = {5,7,12,14, 8,16,24,32, 13, 23, 36, 46, 16, 32, 48, 64};
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
int intensity_sum(long long* summed_area_table, query q, int width, int height){
    
        if(q.p1.x < 0 || q.p1.y < 0 || q.p2.x < 0 || q.p2.y < 0 || q.p3.x < 0 || q.p3.y < 0 || q.p4.x < 0 || q.p4.y < 0){
            cout << "Error: negative index" << endl;
            return -1;
        }

        if(q.p1.x >= width || q.p1.y >= height || q.p2.x >= width || q.p2.y >= height || q.p3.x >= width || q.p3.y >= height || q.p4.x >= width || q.p4.y >= height){
            cout << "Error: index out of bounds" << endl;
            return -1;
        }

        long long A = summed_area_table[q.p1.x * width + q.p1.y];
        long long B = summed_area_table[q.p2.x * width + q.p2.y];
        long long C = summed_area_table[q.p3.x * width + q.p3.y];
        long long D = summed_area_table[q.p4.x * width + q.p4.y];
        long long sum = A + D - B - C;
        return sum;
}

int test_intensity_sum(){
    long long correct_values[16] = {5,7,12,14, 8,16,24,32, 13, 23, 36, 46, 16, 32, 48, 64};
    query q(Point(1,1), Point(1,3), Point(3,1), Point(3,3));
    long long result = intensity_sum(correct_values, q, 4, 4);
    if(result != 16){
        cout << "Error, expected 16, got " << result << endl;
        return -1;
    }
    else{
        cout << "Test passed!!" << endl;
        return 0;
    }
}

// compare the CPU and GPU results
bool compare_with_tolerance(long long* cpu_results, long long* gpu_results, int img_width, int img_height, double tolerance){
    
    int num_elements = img_width * img_height;

    for(int i=0; i<num_elements; i++){
        if(abs (cpu_results[i]-gpu_results[i]) > tolerance){
            cout << "Error at index " << i << endl;
            printf("Error at Index %d: CPU: %d, GPU: %d\n", i, cpu_results[i], gpu_results[i]);
            return false;
        }
    }
    cout << "Test passed" << endl;
    return true;
}

int main(int argc, char* argv[]){
    if(argc != 2)
    {
        cout << "Usage: " << argv[0] << " <img_file_path>" << endl;
        return -1;
    }
    
    test_compute_summed_area_table();
    test_intensity_sum();
    string img_path = argv[1];

    CImg<long long> img(img_path.c_str());

    // get the image dimensions
    int width = img.width();
    int height = img.height();
    int depth = img.depth();

    cout << "image dimensions: " << width << "x" << height << "x" << depth << endl;

    string img_basename = img_path.substr(img_path.find_last_of("/\\") + 1);
    cout << "image basename: " << img_basename << endl;
    
    // create a new image to store the result

    // int32_t* orig_values = img.data();
    width = 10; 
    height = 10;
    long long *orig_values = (long long*) calloc(width * height * depth, sizeof(long long));
    
    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
            orig_values[i*width + j] = i+j;
        }
    }
    cout <<"Original: " << endl;
    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
            cout << orig_values[i*width + j] << " ";
        }
        cout << endl;
    }
    
    long long* CPU_result_values = (long long*) calloc(width * height * depth, sizeof(long long));
    
    compute_summed_area_table(orig_values, CPU_result_values, width, height);
    cout <<"CPU result: " << endl;
    for(int i=0; i<height; i++){
        for(int j=0; j< width; j++){
            cout << CPU_result_values[i*width + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
    // Add the GPU one here 
    long long* GPU_result_values = (long long*) calloc(width * height * depth, sizeof(long long));
    long long kernel_duration = GPU_summed_area_table(orig_values, GPU_result_values, width, height);
    cout << "Kernel duration: " << kernel_duration << " ms" << endl;

    cout <<"GPU result: " << endl;
    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
            cout << GPU_result_values[i*width + j] << " ";
        }
        cout << endl;
    }
    // compare the CPU and GPU results
    bool result = compare_with_tolerance(CPU_result_values, GPU_result_values, width, height, 0.001);
    if(result)
        cout << "The GPU Test Passed" << endl;
    else
    {
        cout << "Test failed" << endl;
        exit(1);
    }
    
    // cout << "Enter the number of queries to execute : " << endl;
    // int number_of_queries = 0; 
    // cin >> number_of_queries;
    
    // vector<query> queries_vec;

    // for(int i=0; i<number_of_queries; i++){
    //     cout << "Enter query " << i <<" :" << endl;
    //     cout << "Enter A(x1,y1) " << endl;
    //     int x1, y1;
    //     cin >> x1 >> y1;
    //     cout << "Enter B(x2,y2) " << endl;
    //     int x2, y2;
    //     cin >> x2 >> y2;
    //     cout << "Enter C(x3,y3) " << endl;
    //     int x3, y3;
    //     cin >> x3 >> y3;
    //     cout << "Enter D(x4,y4) " << endl;
    //     int x4, y4;
    //     cin >> x4 >> y4;
    //     Point p1(x1, y1);
    //     Point p2(x2, y2);
    //     Point p3(x3, y3);
    //     Point p4(x4, y4);
    //     queries_vec.push_back(query(p1, p2, p3, p4));
    // }

    // for(auto q: queries_vec){
    //     int32_t result = intensity_sum(GPU_result_values, q, width, height);
    //     cout << "Result: " << result << endl;
    // }
    
    
    
    

    delete[] GPU_result_values;
    delete[] CPU_result_values;

    return 0;
}

