#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void vector_add(const int* a, const int* b, int* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

int main() {
  const int N = 1e8; // Size of the sequences
  const int SIZE = N * sizeof(int);

  // Host vectors
  std::vector<int> h_a(N, 1); // Sequence 1 (all 1s)
  std::vector<int> h_b(N, 2); // Sequence 2 (all 2s)
  std::vector<int> h_c(N);

  // Device vectors
  int *d_a, *d_b, *d_c;
  cudaMalloc((void**)&d_a, SIZE);
  cudaMalloc((void**)&d_b, SIZE);
  cudaMalloc((void**)&d_c, SIZE);

  // Copy from host to device
  cudaMemcpy(d_a, h_a.data(), SIZE, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), SIZE, cudaMemcpyHostToDevice);

  // Launch kernel
  const int THREADS = 256;
  const int BLOCKS = (N + THREADS - 1) / THREADS;
  vector_add<<<BLOCKS, THREADS>>>(d_a, d_b, d_c, N);

  // Copy result back to host
  cudaMemcpy(h_c.data(), d_c, SIZE, cudaMemcpyDeviceToHost);

  // Output results
  for (int i = 0; i < 10; i++) {
    std::cout << h_c[i] << " ";
  }
  std::cout << std::endl;

  // Free memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
