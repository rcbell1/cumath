#include <cuda_fp16.h>
#include <vector>

#include "../src/cumsum.cuh"
#include "../src/utils.cuh"

template <typename T> void testOnes() {}

int main() {
  int threadsPerBlock = 2;
  int numel = 5;

  // test double type
  std::vector<double> h_input_f64(numel, 1);
  std::vector<double> h_output_f64(numel);

  double* d_input_f64 = nullptr;
  cudaMalloc(&d_input_f64, numel * sizeof(double));
  cudaMemset(d_input_f64, 0, numel * sizeof(double));
  cudaMemcpy(d_input_f64, h_input_f64.data(), numel * sizeof(double),
             cudaMemcpyHostToDevice);

  double* d_output_f64 = nullptr;
  cudaMalloc(&d_output_f64, numel * sizeof(double));

  cumsum(d_input_f64, d_output_f64, numel, threadsPerBlock);

  cudaMemcpy(h_output_f64.data(), d_output_f64, numel * sizeof(double),
             cudaMemcpyDeviceToHost);

  printVector(h_input_f64, "input_f64");
  printVector(h_output_f64, "output_f64");
  std::cout << "\n";

  // test float type
  std::vector<float> h_input_f32(numel, 1);
  std::vector<float> h_output_f32(numel);

  float* d_input_f32 = nullptr;
  cudaMalloc(&d_input_f32, numel * sizeof(float));
  cudaMemset(d_input_f32, 0, numel * sizeof(float));
  cudaMemcpy(d_input_f32, h_input_f32.data(), numel * sizeof(float),
             cudaMemcpyHostToDevice);

  float* d_output_f32 = nullptr;
  cudaMalloc(&d_output_f32, numel * sizeof(float));

  cumsum(d_input_f32, d_output_f32, numel, threadsPerBlock);

  cudaMemcpy(h_output_f32.data(), d_output_f32, numel * sizeof(float),
             cudaMemcpyDeviceToHost);

  printVector(h_input_f32, "input_f32");
  printVector(h_output_f32, "output_f32");
  std::cout << "\n";

  // test int32 type
  std::vector<int32_t> h_input_i32(numel, 1);
  std::vector<int32_t> h_output_i32(numel);

  int32_t* d_input_i32 = nullptr;
  cudaMalloc(&d_input_i32, numel * sizeof(int32_t));
  cudaMemset(d_input_i32, 0, numel * sizeof(int32_t));
  cudaMemcpy(d_input_i32, h_input_i32.data(), numel * sizeof(int32_t),
             cudaMemcpyHostToDevice);

  int32_t* d_output_i32 = nullptr;
  cudaMalloc(&d_output_i32, numel * sizeof(int32_t));

  cumsum(d_input_i32, d_output_i32, numel, threadsPerBlock);

  cudaMemcpy(h_output_i32.data(), d_output_i32, numel * sizeof(int32_t),
             cudaMemcpyDeviceToHost);

  printVector(h_input_i32, "input_i32");
  printVector(h_output_i32, "output_i32");
  std::cout << "\n";

  // test __half type
  std::vector<__half> h_input_f16(numel, 1);
  std::vector<__half> h_output_f16(numel);

  __half* d_input_f16 = nullptr;
  cudaMalloc(&d_input_f16, numel * sizeof(__half));
  cudaMemset(d_input_f16, 0, numel * sizeof(__half));
  cudaMemcpy(d_input_f16, h_input_f16.data(), numel * sizeof(__half),
             cudaMemcpyHostToDevice);

  __half* d_output_f16 = nullptr;
  cudaMalloc(&d_output_f16, numel * sizeof(__half));

  cumsum(d_input_f16, d_output_f16, numel, threadsPerBlock);

  cudaMemcpy(h_output_f16.data(), d_output_f16, numel * sizeof(__half),
             cudaMemcpyDeviceToHost);

  printVector(h_input_f16, "input_f16");
  printVector(h_output_f16, "output_f16");
}
