#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include "time_helper.h"

#ifndef MATRIX_H
#define MATRIX_H

template <typename T>
class matrix{
    private:
        int rows;
        int columns; 
        std::string name;
    public:
        T* data; 
        matrix(int rows, int columns, std::string name){
            this->rows = rows;
            this->columns = columns;
            this->name = name;
            data = (T*) calloc(rows * columns, sizeof(T));
        }
        void auto_generate(T upper_limit){
            // generates random matrix of any size to the max value specified in max_rand 
            srand(time(0));
            for(long i = 0; i < rows * columns; i++){
                data[i] = (T) rand()/(RAND_MAX/upper_limit);
            }
        }
        void display(){
            std::cout << "Displaying Matrix: " << name << "\n";
            for(int i=0; i<rows; i++){
                for(int j=0; j<columns; j++){
                    std::cout << data[i*columns + j] << " ";
                }
                std::cout << "\n";
            }   
        }
        void write_to_file(std::string filename){
            std::ofstream file;
            // append to file if it exists
            file.open(filename, std::ios::out | std::ios::app);
            file << "Matrix: " << name << "\n";
            for(int i=0; i<rows; i++){
                for(int j=0; j<columns; j++){
                    file << data[i*columns + j] << " ";
                }
                file << "\n";
            }
            file << "\n";
            file.close();
        }
        int get_rows() const{
            return rows; 
        }
        int get_columns() const{
            return columns; 
        }
};


template<typename T>
float get_GFLOPS(matrix<T>& A, matrix<T>& B, long long time, time_unit t_unit){
    // returns the GFLOPS of a matrix multiplication
    // (N * P) * (P * M)
    // GFLOPS = N × M × (2P−1)
    float N = A.get_rows();
    float M = B.get_columns();
    float P = A.get_columns();
    float flops = N * M * (2 * P - 1);
    std::cout << "\nFLOPS: " << flops << "\n";

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
    std::cout << "\nTime in seconds: " << time * factor << "\n";
    return flops / (factor * time);
    
}

#endif