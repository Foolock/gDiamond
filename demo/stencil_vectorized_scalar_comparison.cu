#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <cstdlib>
#include <vector>
#include <chrono>

#define PAD 4

#define FLOAT4(ptr) (reinterpret_cast<float4*>(&(ptr))[0])
#define CUDACHECK(err) if (err != cudaSuccess) { \
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << "\n"; exit(EXIT_FAILURE); }

// 3D indexing
__device__ int flatten(int i, int j, int k, int Nx, int Ny) {
    return i + j * Nx + k * Nx * Ny;
}

// Scalar 3D stencil
__global__ void stencil3D_scalar(float* __restrict__ in, float* __restrict__ out, int Nx, int Ny, int Nz) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int N = Nx * Ny * Nz;
    if (tid < N) {
        int i = tid % Nx;
        int j = (tid % (Nx * Ny)) / Nx;
        int k = tid / (Nx * Ny);

        if (i >= 1 && i < Nx - 1 && j >= 1 && j < Ny - 1 && k >= 1 && k < Nz - 1) {
            int idx = flatten(i, j, k, Nx, Ny);
            out[idx] = in[flatten(i - 1, j, k, Nx, Ny)] +
                       in[flatten(i + 1, j, k, Nx, Ny)] +
                       in[flatten(i, j - 1, k, Nx, Ny)] +
                       in[flatten(i, j + 1, k, Nx, Ny)] +
                       in[flatten(i, j, k - 1, Nx, Ny)] +
                       in[flatten(i, j, k + 1, Nx, Ny)] +
                       in[idx];
        }
    }
}

// Vectorized 3D stencil (x-dimension vectorization)
__global__ void stencil3D_vectorized(float* __restrict__ in, float* __restrict__ out, int Nx, int Ny, int Nz) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int scalar_idx = tid * 4;
  int total = Nx * Ny * Nz;

  if (scalar_idx + 3 >= total) return;

  int i = scalar_idx % Nx;
  int j = (scalar_idx % (Nx * Ny)) / Nx;
  int k = scalar_idx / (Nx * Ny);

  if(j < 1 || j >= Ny - 1 || k < 1 || k >= Nz - 1)
      return;

  int stride_y = Nx;
  int stride_z = Nx * Ny;

  float4 center = FLOAT4(in[scalar_idx]);
  // float4 left   = FLOAT4(in[scalar_idx - 1]);
  // float4 right  = FLOAT4(in[scalar_idx + 1]);
  float left = in[scalar_idx - 1];
  float right = in[scalar_idx + 4];
  float4 up     = FLOAT4(in[scalar_idx - stride_y]);
  float4 down   = FLOAT4(in[scalar_idx + stride_y]);
  float4 front  = FLOAT4(in[scalar_idx - stride_z]);
  float4 back   = FLOAT4(in[scalar_idx + stride_z]);

  float4 result;
  result.x = (i == 0)? 0 : (left + center.y + up.x + down.x + front.x + back.x + center.x);
  result.y = center.x + center.z + up.y + down.y + front.y + back.y + center.y;
  result.z = center.y + center.w + up.z + down.z + front.z + back.z + center.z;
  result.w = (i + 3 == Nx - 1)? 0 : (center.z + right + up.w + down.w + front.w + back.w + center.w);

  // FLOAT4(out[scalar_idx]) = result;    
  out[scalar_idx] = result.x;
  out[scalar_idx + 1] = result.y;
  out[scalar_idx + 2] = result.z;
  out[scalar_idx + 3] = result.w;
}

int main() {
  const int Nx = 128, Ny = 128, Nz = 128;
  const int N = Nx * Ny * Nz;

  std::vector<float> h_in(N);

  std::random_device rd;  // seed
  std::mt19937 gen(rd()); // random number engine
  std::uniform_real_distribution<float> dist(0.0f, 1.0f); // range [0.0, 1.0)

  for (auto& val : h_in) {
      val = dist(gen);
  }
  std::vector<float> h_out_scalar(N, 0);
  std::vector<float> h_out_vector(N, 0);

  float *d_in_scalar, *d_in_vector;
  float *d_out_scalar, *d_out_vector;

  CUDACHECK(cudaMalloc(&d_in_scalar, sizeof(float) * N));
  CUDACHECK(cudaMalloc(&d_in_vector, sizeof(float) * N));
  CUDACHECK(cudaMalloc(&d_out_scalar, sizeof(float) * N));
  CUDACHECK(cudaMalloc(&d_out_vector, sizeof(float) * N));

  CUDACHECK(cudaMemcpyAsync(d_in_scalar, h_in.data(), sizeof(float) * N, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(d_in_vector, h_in.data(), sizeof(float) * N, cudaMemcpyHostToDevice));

  size_t block_size = 512; 
  size_t grid_size_scalar = (N + block_size - 1) / block_size;
  size_t grid_size_vector = (N / 4 + block_size - 1) / block_size;

  // scalar version
  auto start = std::chrono::steady_clock::now();
  stencil3D_scalar<<<grid_size_scalar, block_size>>>(d_in_scalar, d_out_scalar, Nx, Ny, Nz);
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  std::cout << "scalar runtime: " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << "s\n"; 

  // vector version
  start = std::chrono::steady_clock::now();
  stencil3D_vectorized<<<grid_size_vector, block_size>>>(d_in_vector, d_out_vector, Nx, Ny, Nz);
  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  std::cout << "vector runtime: " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << "s\n"; 

  CUDACHECK(cudaMemcpyAsync(h_out_scalar.data(), d_out_scalar, sizeof(float) * N, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpyAsync(h_out_vector.data(), d_out_vector, sizeof(float) * N, cudaMemcpyDeviceToHost));

  cudaDeviceSynchronize();

  bool correct = true;

  for(size_t i=0; i<Nx*Ny*Nz; i++) {
    if(fabs(h_out_scalar[i] - h_out_vector[i]) >= 1e-8
    ) {
      int x = i % Nx;
      int y = (i % (Nx * Ny)) / Nx;
      int z = i / (Nx * Ny);
      std::cout << "wrong at x = " << x << ", y = " << y << ", z = " << z << "\n";
      std::cout << "h_out_scalar[i] = " << h_out_scalar[i] << ", h_out_vector[i] = " << h_out_vector[i] << "\n";
      correct = false;
    }
  }

  if(!correct) {
    std::cerr << "results are wrong!\n";
  }

  // Clean up
  CUDACHECK(cudaFree(d_in_scalar));
  CUDACHECK(cudaFree(d_in_vector));
  CUDACHECK(cudaFree(d_out_scalar));
  CUDACHECK(cudaFree(d_out_vector));

  return 0;
}
















