#include <iostream>
#include <string>
#include "time_helper.h"
#define cimg_display 1
#define cimg_use_jpeg
#include "CImg.h"
using namespace cimg_library;
using namespace std;


struct Point {
    int x;
    int y;
    Point(int x, int y) : x(x), y(y) {}
};

struct query{
    Point p1;
    Point p2;
    Point p3;
    Point p4;
    query(Point p1, Point p2, Point p3, Point p4) : p1(p1), p2(p2), p3(p3), p4(p4) {}
};


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
    
    test_compute_summed_area_table();

    string img_path = argv[1];

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

    
    cout << "Enter the number of queries to execute : " << endl;
    int number_of_queries = 0; 
    cin >> number_of_queries;
    
    vector<query> queries_vector(number_of_queries);

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
        queries_vec[i] = query(Point(x1, y1), Point(x2, y2), Point(x3, y3), Point(x4, y4));
    }

    for(auto q: queries_vec){
        int32_t A = result_values[q.p1.x * width + q.p1.y];
        int32_t B = result_values[q.p2.x * width + q.p2.y];
        int32_t C = result_values[q.p3.x * width + q.p3.y];
        int32_t D = result_values[q.p4.x * width + q.p4.y];
        int32_t sum = A + D - B - C;
        cout << "A(x1,y1) = " << A.x << " " << A.y << endl;
        cout << "B(x2,y2) = " << B.x << " " << B.y << endl;
        cout << "C(x3,y3) = " << C.x << " " << C.y << endl;
        cout << "D(x4,y4) = " << D.x << " " << D.y << endl;
        cout << "Sum of the query is " << sum << endl;
    }
    
    
    // result_img.save("summed_area_table_result.jpeg");

    delete[] result_values;

    return 0;
}

