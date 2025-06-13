#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#define SHMEM_SIZE 99264
#define SHMEM_INT (SHMEM_SIZE / 4)
#define BLOCK_SIZE 1024
#define GRID_SIZE 2000

// each block handles SHMEM_INT data
__global__ void max_shmem_kernel(float* out, int N) {

  int sid = threadIdx.x;

  extern __shared__ float shmem[]; 

  // within shmem, each thread calculate sid, sid + blockDim.x, sid + 2 * blockDim.x ...
  #pragma unroll
  for(int id = sid; id < SHMEM_INT; id += BLOCK_SIZE) {
    shmem[id] = (id + blockIdx.x * SHMEM_INT) * 2.0f;
  }

  __syncthreads();

  // store global
  for(int id = sid; id < SHMEM_INT; id += BLOCK_SIZE) {
    out[id + blockIdx.x * SHMEM_INT] = shmem[id];
  }
}

void cpu_version(std::vector<float>& ref, int N) {
  for (int i = 0; i < N; ++i) {
    ref[i] = i * 2.0f;
  }
}

int main() {

  const int N = SHMEM_INT * GRID_SIZE;

  std::vector<float> host_ref(N);
  std::vector<float> host_out(N);
  
  float* d_out;
  cudaMalloc(&d_out, N * sizeof(float));

  // Allow large shared memory size
  cudaFuncSetAttribute(
    max_shmem_kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    SHMEM_SIZE 
  );

  max_shmem_kernel<<<GRID_SIZE, BLOCK_SIZE, SHMEM_SIZE>>>(d_out, N);
  cudaDeviceSynchronize();

  cudaMemcpy(host_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

  // CPU computation
  cpu_version(host_ref, N);

  // Check correctness
  bool correct = true;
  for (int i = 0; i < N; ++i) {
    if (std::abs(host_ref[i] - host_out[i]) > 1e-5) {
      std::cerr << "Mismatch at " << i << ": CPU = " << host_ref[i]
                << ", GPU = " << host_out[i] << '\n';
      correct = false;
      break;
    }
  }

  if (correct) {
    std::cout << "✅ Output matches CPU reference for N = " << N << '\n';
  } else {
    std::cout << "❌ Output mismatch\n";
  }

  cudaFree(d_out);
  return 0;

} 
