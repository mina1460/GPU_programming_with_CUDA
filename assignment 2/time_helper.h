#include <chrono>
#include <iostream>
#include <string>

#ifndef TIME_HELPER_H
#define TIME_HELPER_H

std::chrono::high_resolution_clock::time_point get_time() {
    return std::chrono::high_resolution_clock::now();
}
enum time_unit { nanoseconds, microseconds, milliseconds, seconds };
std::string unit_name(time_unit t){
    switch(t){
        case nanoseconds:
            return "nanoseconds";
        case microseconds:
            return "microseconds";
        case milliseconds:
            return "milliseconds";
        case seconds:
            return "seconds";
        default:
            throw std::invalid_argument("Invalid time unit\n");
    }
}
long long get_time_diff(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end, time_unit unit) {
    switch (unit) {
        case nanoseconds:
            return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        case microseconds:
            return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        case milliseconds:
            return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        case seconds:
            return std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
        default:
            throw std::invalid_argument("Invalid time unit\n");
    }
}

std::string get_timestamp() {
    //get day_hour_minute
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    time_t tt = std::chrono::system_clock::to_time_t(now);
    std::string day_hour_minute = std::ctime(&tt);
    day_hour_minute.erase(day_hour_minute.end() - 1, day_hour_minute.end());
    return day_hour_minute;
}

#endif