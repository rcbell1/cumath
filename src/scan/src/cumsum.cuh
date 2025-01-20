#pragma once
#include <cuda_runtime.h>
#define NUMEL_PER_BLOCK 2
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_MEM_BANKS)

template <typename TYPE>
void __global__ addBlockSums(TYPE* block_sums, TYPE* outp, int numel,
                             int rstride) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < numel && blockIdx.x > 0) {
    outp[gid] += block_sums[blockIdx.x];
  }
}

template <typename TYPE> __device__ TYPE* shared_memory_proxy() {
  // See the discussion here for reasoning behind this function
  // https://stackoverflow.com/questions/27570552/templated-cuda-kernel-with-dynamic-shared-memory
  extern __shared__ unsigned char memory[];
  return reinterpret_cast<TYPE*>(memory);
}

template <typename TYPE>
void __global__ blockScan(const TYPE* d_inp, TYPE* d_outp, TYPE* d_block_sums,
                          int numel_chunk) {
  // A good discussion of the algorithm implemented here can be found at
  // https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
  __shared__ TYPE sdata[NUMEL_PER_BLOCK];
  // auto sdata = shared_memory_proxy<TYPE>(); // see function comment

  int numel_block = 2 * blockDim.x;
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int gid = bid * numel_block + tid;
  int gid1 = bid * numel_block + 2 * tid;
  int gid2 = bid * numel_block + 2 * tid + 1;

  // Copy global input to local shared block memory
  sdata[tid] = (TYPE)0;
  sdata[tid + blockDim.x] = (TYPE)0;
  __syncthreads();

  if (gid < numel_chunk) {
    sdata[tid] = d_inp[gid];
  }
  if (gid + blockDim.x < numel_chunk) {
    sdata[tid + blockDim.x] = d_inp[gid + blockDim.x];
  }
  __syncthreads();

  // Up-sweep (Reduce) phase
  for (int stride = 1; stride <= blockDim.x; stride *= 2) {
    int index = (tid + 1) * stride * 2 - 1;
    if (index < numel_block) {
      sdata[index] += sdata[index - stride];
    }
    __syncthreads();
  }

  // Store block sum and clear the last element if multiple blocks
  if (tid == 0) {
    if (gridDim.x > 1) {
      d_block_sums[bid] = sdata[numel_block - 1];
    }
    sdata[numel_block - 1] = 0;
  }
  __syncthreads();

  // Down-sweep phase
  for (int stride = blockDim.x; stride > 0; stride /= 2) {
    int index = (tid + 1) * stride * 2 - 1;
    if (index < numel_block) {
      TYPE temp = sdata[index];
      sdata[index] += sdata[index - stride];
      sdata[index - stride] = temp;
    }
    __syncthreads();
  }

  // Copy to global output
  if (gid < numel_chunk) {
    d_outp[gid] = sdata[tid];
  }
  if (gid + blockDim.x < numel_chunk) {
    d_outp[gid + blockDim.x] = sdata[tid + blockDim.x];
  }
}

template <typename TYPE>
void recursiveScan(const TYPE* d_inp, TYPE* d_outp, int numel, int blockDim,
                   int recurDepth = 0) {

  int numelBlock = 2 * blockDim;
  int gridDim = (numel + numelBlock - 1) / numelBlock;

  // Allocate memory for block sums, if multiple blocks are needed
  TYPE* d_blockSums = nullptr;
  if (gridDim > 1) {
    cudaMalloc(&d_blockSums, gridDim * sizeof(TYPE));
    cudaMemset(d_blockSums, 0, gridDim * sizeof(TYPE));
  }

  // Run block level scans
  blockScan<TYPE><<<gridDim, blockDim>>>(d_inp, d_outp, d_blockSums, numel);

  if (gridDim > 1) {
    ++recurDepth;

    // Run recursive_scan on the block sums, if more than one block exists
    recursiveScan(d_blockSums, d_blockSums, gridDim, blockDim, recurDepth);

    addBlockSums<TYPE>
        <<<gridDim, numelBlock>>>(d_blockSums, d_outp, numel, recurDepth);

    cudaFree(d_blockSums);
  }
}

template <typename TYPE>
__global__ void shiftAndAddLast(const TYPE* d_inp, TYPE* d_exScan, int numel) {
  // Shift left all elements left
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numel - 1;
       i += blockDim.x * gridDim.x) {
    d_exScan[i] = d_exScan[i + 1];
  }

  // Add last input element to last scan element
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    d_exScan[numel - 1] += d_inp[numel - 1];
  }
}

template <typename TYPE>
void exclusiveToInclusiveScan(const TYPE* d_inp, TYPE* d_exScan, int numel,
                              int blockDim) {

  int gridDim = (numel + blockDim - 1) / blockDim;
  shiftAndAddLast<TYPE><<<gridDim, blockDim>>>(d_inp, d_exScan, numel);
}

template <typename TYPE>
void cumsum(TYPE* d_inp, TYPE* d_outp, int numel, int threadsPerBlock) {

  // Implements exclusive scan recursively to enable arbitrary input lengths
  recursiveScan<TYPE>(d_inp, d_outp, numel, threadsPerBlock);

  // Convert exclusive scan to inclusive scan, what MATLABs cumsum produces
  exclusiveToInclusiveScan<TYPE>(d_inp, d_outp, numel, threadsPerBlock);
}
