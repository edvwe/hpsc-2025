#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

const int N = 50;
const int RANGE = 5;

// Kernel 1: Initialize bucket
__global__ void init_bucket(int *bucket) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < RANGE) {
    bucket[i] = 0;
  }
}

// Kernel 2: Count occurrences
__global__ void count_instances(int *key, int *bucket, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    atomicAdd(&bucket[key[i]], 1);
  }
}

// Kernel 3: Write sorted output
__global__ void write_sorted(int *bucket, int *sorted) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < RANGE) {
    int offset = 0;
    // Sequential scan inside thread block (RANGE is small)
    for (int j = 0; j < i; j++) {
      offset += bucket[j];
    }
    for (int k = 0; k < bucket[i]; k++) {
      sorted[offset + k] = i;
    }
  }
}

int main() {
  int *key;
  int *bucket;
  int *sorted;

  cudaMallocManaged(&key, N * sizeof(int));
  cudaMallocManaged(&bucket, RANGE * sizeof(int));
  cudaMallocManaged(&sorted, N * sizeof(int));

  // Fill key with random values
  printf("Original: ");
  for (int i = 0; i < N; i++) {
    key[i] = rand() % RANGE;
    printf("%d ", key[i]);
  }
  printf("\n");

  // Kernel 1: Init bucket
  init_bucket<<<1, RANGE>>>(bucket);
  cudaDeviceSynchronize();

  // Kernel 2: Count values
  count_instances<<<(N + 31) / 32, 32>>>(key, bucket, N);
  cudaDeviceSynchronize();

  // Kernel 3: Fill sorted array based on bucket counts
  write_sorted<<<1, RANGE>>>(bucket, sorted);
  cudaDeviceSynchronize();

  // Output sorted array
  printf("Sorted:   ");
  for (int i = 0; i < N; i++) {
    printf("%d ", sorted[i]);
  }
  printf("\n");

  cudaFree(key);
  cudaFree(bucket);
  cudaFree(sorted);
  return 0;
}
