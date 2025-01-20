#pragma once
#include <cuda_fp16.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// Helper function to convert __half to float
inline void convert_for_print(__half value) {
  std::cout << std::setw(3) << std::fixed << std::setprecision(4)
            << __half2float(value) << " ";
}

// Identity function for types that don't need conversion
template <typename T> inline void convert_for_print(T value) {
  std::cout << std::setw(3) << std::fixed << std::setprecision(4)
            << __half2float(value) << " ";
}

template <typename T>
void printVector(const std::vector<T>& vec, const std::string& label) {
  std::cout << label << ": ";
  for (const auto& elem : vec) {
    convert_for_print(elem);
  }
  std::cout << "\n";
}
