#include <iostream>

// Compute the convolution of two sequences.
int conv(int* x, int* h, int* y, int N, int M) {
  int i, j;
  for (i = 0; i < N + M - 1; i++) {
    y[i] = 0;
    for (j = 0; j < N; j++) {
      if (i - j >= 0 && i - j < M) {
        y[i] += x[j] * h[i - j];
      }
    }
  }
  return 0;
}

int main() { std::cout << "Hello World!\n"; }
