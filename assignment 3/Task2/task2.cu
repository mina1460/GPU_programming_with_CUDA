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
float get_GPU_GFLOPS(int img_width, int img_height, long long time, time_unit t_unit);
float get_CPU_GFLOPS(int img_width, int img_height, long long time, time_unit t_unit);
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
long long intensity_sum(long long* summed_area_table, query q, int width, int height){
    
        if(q.p1.x < 0 || q.p1.y < 0 || q.p2.x < 0 || q.p2.y < 0 || q.p3.x < 0 || q.p3.y < 0 || q.p4.x < 0 || q.p4.y < 0){
            cout << "Error: negative index" << endl;
            return -1;
        }

        if(q.p1.y >= width || q.p1.x >= height || q.p2.y >= width || q.p2.x >= height || q.p3.y >= width || q.p3.x >= height || q.p4.y >= width || q.p4.x >= height){
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
    
    long long num_elements = img_width * img_height;
    printf("Comparing %d elements\n", num_elements);
    for(long long i=0; i<num_elements; i++){
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

    long long* orig_values = img.data();
  


    long long* CPU_result_values = (long long*) calloc(width * height * depth, sizeof(long long));
    // calculate the time for the cpu 
    auto start = chrono::high_resolution_clock::now();
    compute_summed_area_table(orig_values, CPU_result_values, width, height);
    auto end = chrono::high_resolution_clock::now();
    auto duration = get_time_diff(start, end, nanoseconds);
    
    float cpu_Flops = get_CPU_GFLOPS(width, height, duration, nanoseconds);
    cout << "CPU GFLOPS: " << cpu_Flops << endl;
    
    cout << "CPU duration: " << duration << " ns" << endl;

    cout << "The Genralized Kernel of GPU\n";
    long long* GPU_result_values = (long long*) calloc(width * height * depth, sizeof(long long));
    auto start2 = chrono::high_resolution_clock::now();
    long long kernel_duration = GPU_summed_area_table(orig_values, GPU_result_values, width, height);
    auto end2 = chrono::high_resolution_clock::now();
    long long duration2 =  get_time_diff(start2, end2, nanoseconds);
    cout << "GPU duration without Data Transfer: " << kernel_duration << " ns" << endl;
    cout << "GPU duration with Data Transfer: " << duration2 << " ns" << endl;
    // number of flops for the GPU 
    float GPU_Flops = get_GPU_GFLOPS(width, height, kernel_duration, nanoseconds);
    cout << "GPU GFLOPS Without Transfer: " << GPU_Flops << endl;
    float GPU_Flops2 = get_GPU_GFLOPS(width, height, duration2, nanoseconds);
    cout << "GPU GFLOPS with Transfer: " << GPU_Flops2 << endl;

    // compare the CPU and GPU results
    bool result = compare_with_tolerance(CPU_result_values, GPU_result_values, width, height, 0.001);
    if(result)
        cout << "The GPU Test Passed" << endl;
    else
    {
        cout << "Test failed" << endl;
        exit(1);
    }
    
//     cout <<"The Non-Gneralized Kernel of GPU\n";
//     // width = 48; height = 48; depth = 1;
//     // create an origValues array
//     long long* orig_values2 = (long long*) calloc(width * height * depth, sizeof(long long));
//     //dump values 
//     for(int i = 0; i < width * height * depth; i++){
//         orig_values2[i] = i%250;
//     }
//     // cout the orig_values
//     for(int i = 0 ; i < height; i++)
//     {
//         for(int j = 0; j < width; j++)
//         {
//             cout << orig_values2[i * width + j] << " ";
//         }
//         cout << endl;
//     }
//     cout << endl;
//     // get the CPU result from this array
//     long long* CPU_result_values2 = (long long*) calloc(width * height * depth, sizeof(long long));
    
//     // get the cpu result
//     compute_summed_area_table(orig_values2, CPU_result_values2, width, height);
//     // cout the CPU result
//    for(int i = 0 ; i < height; i++)
//     {
//         for(int j = 0; j < width; j++)
//         {
//             cout << CPU_result_values2[i * width + j] << " ";
//         }
//         cout << endl;
//     }

//     long long* GPU_result_values2 = (long long*) calloc(width * height * depth, sizeof(long long));
//     auto start3 = chrono::high_resolution_clock::now();
//     long long kernel_duration2 = GPU_summed_area_table_not_Generalized(orig_values2, GPU_result_values2, width, height);

//     // cout the GPU result
//     for(int i = 0 ; i < height; i++)
//     {
//         for(int j = 0; j < width; j++)
//         {
//             cout << GPU_result_values2[i * width + j] << " ";
//         }
//         cout << endl;
//     }
//     //compare the CPU and GPU results
//     bool result2 = compare_with_tolerance(CPU_result_values2, GPU_result_values2, width, height, 0.001);
//     if(result2)
//         cout << "The GPU of the second kernel Test Passed" << endl;
//     else
//     {
//         cout << "Test failed" << endl;
//         exit(1);
//     }
//     auto end3 = chrono::high_resolution_clock::now();
//     long long duration3 =  get_time_diff(start3, end3, nanoseconds);
//     cout << "GPU duration without Data Transfer: " << kernel_duration2 << " ns" << endl;
//     cout << "GPU duration with Data Transfer: " << duration3 << " ns" << endl;
//     // number of flops for the GPU
//     // float GPU_Flops3 = get_GPU_GFLOPS(width, height, kernel_duration2, nanoseconds);
//     // cout << "GPU GFLOPS Without Transfer: " << GPU_Flops3 << endl;
//     // float GPU_Flops4 = get_GPU_GFLOPS(width, height, duration3, nanoseconds);
//     // cout << "GPU GFLOPS with Transfer: " << GPU_Flops4 << endl;


    cout << "Enter the number of queries to execute : " << endl;
    int number_of_queries = 0; 
    cin >> number_of_queries;
    
    vector<query> queries_vec;

    for(int i=0; i<number_of_queries; i++){
        cout << "Enter query " << i <<" :" << endl;
        cout << "Enter A(x1,y1) " << endl;
        int x1, y1;
        cin >> x1 >> y1;
        cout << "Enter B(x2,y2) " << endl;
        int x2, y2;
        cin >> x2 >> y2;
        cout << "Enter C(x3,y3) " << endl;
        int x3, y3;
        cin >> x3 >> y3;
        cout << "Enter D(x4,y4) " << endl;
        int x4, y4;
        cin >> x4 >> y4;
        Point p1(x1, y1);
        Point p2(x2, y2);
        Point p3(x3, y3);
        Point p4(x4, y4);
        vector<Point>v = {p1, p2, p3, p4};
        sort(v.begin(), v.end(), [](Point a, Point b){
            if(a.x == b.x)
                return a.y < b.y;
            return a.x < b.x;
        }); 
        if(v[0].x != v[1].x || v[0].y != v[2].y  || v[1].y != v[3].y || v[2].x != v[3].x)
        {
            cout << "Invalid query" << endl;
            continue; 
        }

        queries_vec.push_back(query(v[0], v[1], v[2], v[3]));
    }

    for(auto q: queries_vec){
        int32_t result = intensity_sum(GPU_result_values, q, width, height);
        cout << "Result: " << result << endl;
    }
    

    delete[] GPU_result_values;
    delete[] CPU_result_values;

    return 0;
}

float get_GFLOPS(int img_width, int img_height, long long time, time_unit t_unit, float flops){

    // std::cout << "\nFLOPS: " << flops << "\n";

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

float get_GPU_GFLOPS_Not_Generalized(int img_width, int img_height, long long time, time_unit t_unit)
{
    // kernel is 2D PrefixSum
    // sequentialVersion = ceil(img_width/1024) * img_height; 
    // parallelVersion = ceil(img_width/1024) * img_height * log2(ceil(img_width/1024));
    // Adding_to_next_section =  ceil(img_width/1024) * img_height * ceil(img_width/1024);
    // kernel1 = sequentialVersion + parallelVersion + Adding_to_next_section;
    // Same thing for column version
    // sequentialVersion2 = ceil(img_height/1024) * img_width;
    // parallelVersion2 = ceil(img_height/1024) * img_width * log2(ceil(img_height/1024));
    // Adding_to_next_section2 =  ceil(img_height/1024) * img_width * ceil(img_height/1024);
    // kernel2 = sequentialVersion2 + parallelVersion2 + Adding_to_next_section2;

    float sequentialVersion = ceil(img_width/1024.0) * img_height;
    float parallelVersion = ceil(img_width/1024.0) * img_height * log2(ceil(img_width/1024.0));
    float Adding_to_next_section =  ceil(img_width/1024.0) * img_height * ceil(img_width/1024.0);
    float kernel1 = sequentialVersion + parallelVersion + Adding_to_next_section;
    float sequentialVersion2 = ceil(img_height/1024.0) * img_width;
    float parallelVersion2 = ceil(img_height/1024.0) * img_width * log2(ceil(img_height/1024.0));
    float Adding_to_next_section2 =  ceil(img_height/1024.0) * img_width * ceil(img_height/1024.0);
    float kernel2 = sequentialVersion2 + parallelVersion2 + Adding_to_next_section2;
    float flops = kernel1 + kernel2;
    return get_GFLOPS(img_width, img_height, time, t_unit, flops);
}


float get_GPU_GFLOPS(int img_width, int img_height, long long time, time_unit t_unit)
{
        // returns the GFLOPS of a 2D PrefixSum 
    // Kernel1 = img_height*img_width*log2(img_width)
    // Block_width_size = ceil(img_width/1024)
    // Kernel2 =  Block_width_size*img_height*log2(Block_width_size)
    // Kernel3 =  img_height * img_width
    // Kernel4 =  img_height * img_width * 2  // Transpose 
    // Kernel5 = img_width * img_height * log2(img_height) // PrefixSum
    // Block_height_size = ceil(img_height/1024)
    // Kernel6 = Block_height_size * img_width * log2(Block_height_size) // PrefixSum
    // Kernel7 = img_width * img_height // Transpose
    // GFLOPS = Sum 7 kernels / time
    float kernel1 = img_height * img_width * log2(img_width);
    float kernel2 = ceil(img_width/1024.0) * img_height * log2(ceil(img_width/1024.0));
    float kernel3 = img_height * img_width;
    float kernel4 = img_height * img_width * 2;
    float kernel5 = img_width * img_height * log2(img_height);
    float kernel6 = ceil(img_height/1024.0) * img_width * log2(ceil(img_height/1024.0));
    float kernel7 = img_width * img_height;
    float flops = kernel1 + kernel2 + kernel3 + kernel4 + kernel5 + kernel6 + kernel7;

    return get_GFLOPS(img_width, img_height, time, t_unit, flops);

}


float get_CPU_GFLOPS(int img_width, int img_height, long long time, time_unit t_unit)
{
    //Returns the GFLOPS of a 2D PrefixSum
    int numReadAccess   = 3; 
    int numWriteAccess  = 1; 
    int operations      = 4; 
    float flops = img_width * img_height * (operations + numReadAccess + numWriteAccess);
    return get_GFLOPS(img_width, img_height, time, t_unit, flops); 
}


