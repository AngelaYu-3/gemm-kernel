#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                             const float *B, float beta, float *C) {
  // compute position in C that this thread is responsible for
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    // C = α*(A@B)+β*C
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}

int main() {
  const int M = 4096, N = 4096, K = 4096;
  const float alpha = 1.0f, beta = 0.0f;

  size_t sizeA = M * K * sizeof(float);
  size_t sizeB = K * N * sizeof(float);
  size_t sizeC = M * N * sizeof(float);

  float *hA = (float *)malloc(sizeA);
  float *hB = (float *)malloc(sizeB);
  float *hC = (float *)malloc(sizeC);

  for (int i = 0; i < M * K; ++i) hA[i] = 1.0f;
  for (int i = 0; i < K * N; ++i) hB[i] = 1.0f;
  for (int i = 0; i < M * N; ++i) hC[i] = 0.0f;

  float *dA, *dB, *dC;
  cudaMalloc(&dA, sizeA);
  cudaMalloc(&dB, sizeB);
  cudaMalloc(&dC, sizeC);

  cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice);
  cudaMemcpy(dC, hC, sizeC, cudaMemcpyHostToDevice);

  // create as many blocks as necessary to map all of C
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
  // 32 * 32 = 1024 thread per block
  dim3 blockDim(32, 32, 1);

  // warm-up run, not timed (avoids counting one-time GPU/driver init cost)
  sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  const int num_runs = 10;
  cudaEventRecord(start);
  for (int i = 0; i < num_runs; ++i) {
    sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  float avg_ms = ms / num_runs;

  double gflops = (2.0 * M * N * K) / (avg_ms / 1000.0) / 1e9;

  cudaMemcpy(hC, dC, sizeC, cudaMemcpyDeviceToHost);

  // every entry should be K (1*1 summed K times), sanity check one value
  printf("C[0] = %f (expected %f)\n", hC[0], (float)K);
  printf("Avg kernel time: %.3f ms\n", avg_ms);
  printf("Achieved: %.2f GFLOPS\n", gflops);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  free(hA);
  free(hB);
  free(hC);

  return 0;
}
