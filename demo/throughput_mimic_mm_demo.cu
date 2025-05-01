#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>
#include <algorithm>

// mix mapping
#define BLT 4

// one-to-one mapping in X dimension
#define NTX 16

// one-to-many mapping in Y dimension
#define NTY 4

// one-to-many mapping in Z dimension
#define NTZ 4

// tile size
#define BLX 16 
#define BLY 10 
#define BLZ 10 

/*

  mimic memory access pattern in mix-mapping, one-to-one on X, one-to-many on Y, Z 

  each tile is of size BLX * BLY * BLZ

  no shared memory use, check throughput on update E

  Ex[idx] = Cax[idx] * Ex[idx] + Cbx[idx] *
            ((Hz[idx] + Hz[idx - Nx]) + (Hy[idx] + Hy[idx - Nx * Ny]) - Jx[idx] * dx);
  Ey[idx] = Cay[idx] * Ey[idx] + Cby[idx] *
            ((Hx[idx] + Hx[idx - Nx * Ny]) + (Hz[idx] + Hz[idx - 1]) - Jy[idx] * dx);
  Ez[idx] = Caz[idx] * Ez[idx] + Cbz[idx] *
            ((Hy[idx] + Hy[idx - 1]) + (Hx[idx] + Hx[idx - Nx]) - Jz[idx] * dx);

*/

// handle errors in CUDA call
#define CUDACHECK(call)                                                        \
{                                                                          \
   const cudaError_t error = call;                                         \
   if (error != cudaSuccess)                                               \
   {                                                                       \
      printf("Error: %s:%d, ", __FILE__, __LINE__);                        \
      printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));   \
      exit(1);                                                             \
   }                                                                       \
} (void)0  // Ensures a semicolon is required after the macro call.


void sequential(std::vector<float>& Ex, std::vector<float>& Ey, std::vector<float>& Ez,
                const std::vector<float>& Hx, const std::vector<float>& Hy, const std::vector<float>& Hz,
                const std::vector<float>& Cax, const std::vector<float>& Cay, const std::vector<float>& Caz,
                const std::vector<float>& Cbx, const std::vector<float>& Cby, const std::vector<float>& Cbz,
                const std::vector<float>& Jx, const std::vector<float>& Jy, const std::vector<float>& Jz,
                float dx,
                int Nx, int Ny, int Nz,
                int timesteps) {

  auto start = std::chrono::high_resolution_clock::now();

  for(int t = 0; t < timesteps; t++) {
    for(int z = 1; z <= Nz - 2; z++) {
      for(int y = 1; y <= Ny - 2; y++) {
        for(int x = 1; x <= Nx - 2; x++) {
          int idx = x + y * Nx + z * Nx * Ny;
          Ex[idx] = Cax[idx] * Ex[idx] + Cbx[idx] *
                    ((Hz[idx] + Hz[idx - Nx]) + (Hy[idx] + Hy[idx - Nx * Ny]) - Jx[idx] * dx);
          Ey[idx] = Cay[idx] * Ey[idx] + Cby[idx] *
                    ((Hx[idx] + Hx[idx - Nx * Ny]) + (Hz[idx] + Hz[idx - 1]) - Jy[idx] * dx);
          Ez[idx] = Caz[idx] * Ez[idx] + Cbz[idx] *
                    ((Hy[idx] + Hy[idx - 1]) + (Hx[idx] + Hx[idx - Nx]) - Jz[idx] * dx);
        }
      }
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "seq runtime: " << std::chrono::duration<double>(end-start).count() << " s\n"; 
}

__global__ void update_naive(float* Ex, float* Ey, float* Ez,
                             float* Hx, float* Hy, float* Hz,
                             float* Cax, float* Cay, float* Caz,
                             float* Cbx, float* Cby, float* Cbz,
                             float* Jx, float* Jy, float* Jz,
                             float dx,
                             int Nx, int Ny, int Nz) {

  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  unsigned int global_x = tid % Nx;
  unsigned int global_y = (tid / Nx) % Ny;  
  unsigned int global_z = tid / (Nx * Ny);

  if (global_x >= 1 && global_x <= Nx - 2 && global_y >= 1 && global_y <= Ny - 2 && global_z >= 1 && global_z <= Nz - 2)
  {
    int idx = global_x + global_y * Nx + global_z * (Nx * Ny);

    Ex[idx] = Cax[idx] * Ex[idx] + Cbx[idx] *
              ((Hz[idx] + Hz[idx - Nx]) + (Hy[idx] + Hy[idx - Nx * Ny]) - Jx[idx] * dx);
    Ey[idx] = Cay[idx] * Ey[idx] + Cby[idx] *
              ((Hx[idx] + Hx[idx - Nx * Ny]) + (Hz[idx] + Hz[idx - 1]) - Jy[idx] * dx);
    Ez[idx] = Caz[idx] * Ez[idx] + Cbz[idx] *
              ((Hy[idx] + Hy[idx - 1]) + (Hx[idx] + Hx[idx - Nx]) - Jz[idx] * dx);
  }
}

void gpu_naive(std::vector<float>& Ex, std::vector<float>& Ey, std::vector<float>& Ez,
               const std::vector<float>& Hx, const std::vector<float>& Hy, const std::vector<float>& Hz,
               const std::vector<float>& Cax, const std::vector<float>& Cay, const std::vector<float>& Caz,
               const std::vector<float>& Cbx, const std::vector<float>& Cby, const std::vector<float>& Cbz,
               const std::vector<float>& Jx, const std::vector<float>& Jy, const std::vector<float>& Jz,
               float dx,
               int Nx, int Ny, int Nz,
               int timesteps
) {

  float *Ex_d, *Ey_d, *Ez_d;
  float *Hx_d, *Hy_d, *Hz_d;
  float *Cax_d, *Cay_d, *Caz_d;
  float *Cbx_d, *Cby_d, *Cbz_d;
  float *Jx_d, *Jy_d, *Jz_d;

  int length = Nx * Ny * Nz;

  CUDACHECK(cudaMalloc(&Ex_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Ey_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Ez_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Hx_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Hy_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Hz_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Cax_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Cay_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Caz_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Cbx_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Cby_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Cbz_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Jx_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Jy_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Jz_d, sizeof(float) * length));

  CUDACHECK(cudaMemcpyAsync(Ex_d, Ex.data(), sizeof(float) * length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Ey_d, Ey.data(), sizeof(float) * length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Ez_d, Ez.data(), sizeof(float) * length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Hx_d, Hx.data(), sizeof(float) * length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Hy_d, Hy.data(), sizeof(float) * length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Hz_d, Hz.data(), sizeof(float) * length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Jx_d, Jx.data(), sizeof(float) * length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Jy_d, Jy.data(), sizeof(float) * length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Jz_d, Jz.data(), sizeof(float) * length, cudaMemcpyHostToDevice));

  CUDACHECK(cudaMemcpyAsync(Cax_d, Cax.data(), sizeof(float) * length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Cay_d, Cay.data(), sizeof(float) * length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Caz_d, Caz.data(), sizeof(float) * length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Cbx_d, Cbx.data(), sizeof(float) * length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Cby_d, Cby.data(), sizeof(float) * length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Cbz_d, Cbz.data(), sizeof(float) * length, cudaMemcpyHostToDevice));

  size_t block_size = NTX * NTY * NTZ;
  size_t grid_size = (length + block_size - 1)/block_size;

  auto start = std::chrono::high_resolution_clock::now();

  for(int t=0; t<timesteps; t++) {
    update_naive<<<grid_size, block_size>>>(Ex_d, Ey_d, Ez_d,
                                      Hx_d, Hy_d, Hz_d,
                                      Cax_d, Cay_d, Caz_d,
                                      Cbx_d, Cby_d, Cbz_d,
                                      Jx_d, Jy_d, Jz_d,
                                      dx,
                                      Nx, Ny, Nz
                                     );
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "gpu naive runtime (exclude memcpy): " << std::chrono::duration<double>(end-start).count() << " s\n"; 

  cudaMemcpy(Ex.data(), Ex_d, sizeof(float) * length, cudaMemcpyDeviceToHost);
  cudaMemcpy(Ey.data(), Ey_d, sizeof(float) * length, cudaMemcpyDeviceToHost);
  cudaMemcpy(Ez.data(), Ez_d, sizeof(float) * length, cudaMemcpyDeviceToHost);

  CUDACHECK(cudaFree(Ex_d));
  CUDACHECK(cudaFree(Ey_d));
  CUDACHECK(cudaFree(Ez_d));
  CUDACHECK(cudaFree(Hx_d));
  CUDACHECK(cudaFree(Hy_d));
  CUDACHECK(cudaFree(Hz_d));
  CUDACHECK(cudaFree(Cax_d));
  CUDACHECK(cudaFree(Cay_d));
  CUDACHECK(cudaFree(Caz_d));
  CUDACHECK(cudaFree(Cbx_d));
  CUDACHECK(cudaFree(Cby_d));
  CUDACHECK(cudaFree(Cbz_d));
  CUDACHECK(cudaFree(Jx_d));
  CUDACHECK(cudaFree(Jy_d));
  CUDACHECK(cudaFree(Jz_d));

} 

__global__ void update_tiling(float* Ex, float* Ey, float* Ez,
                              float* Hx, float* Hy, float* Hz,
                              float* Cax, float* Cay, float* Caz,
                              float* Cbx, float* Cby, float* Cbz,
                              float* Jx, float* Jy, float* Jz,
                              float dx,
                              int Nx, int Ny, int Nz,
                              int* xx_heads, int* yy_heads, int* zz_heads,
                              int Tx, int Ty, int Tz) {

  const unsigned int block_id = blockIdx.x;
  const unsigned int thread_id = threadIdx.x;

  const unsigned int xx = block_id % Tx;
  const unsigned int yy = (block_id / Tx) % Ty;
  const unsigned int zz = block_id / (Tx * Ty);

  const int local_x = thread_id % NTX;
  const int local_y = (thread_id / NTX) % NTY;
  const int local_z = thread_id / (NTX * NTY);

  int global_x, global_y, global_z;

  global_x = xx_heads[xx] + local_x; // one-to-one mapping in X  

  // one-to-many mapping in Y and Z
  const int ytile_head = yy_heads[yy] + local_y;
  const int ytile_tail = yy_heads[yy] + BLY - 1;
  const int ztile_head = zz_heads[zz] + local_z;
  const int ztile_tail = zz_heads[zz] + BLZ - 1;

  for(global_y = ytile_head; global_y <= ytile_tail; global_y += NTY) {
    for(global_z = ztile_head; global_z <= ztile_tail; global_z += NTZ) {
      if (global_x >= 1 && global_x <= Nx - 2 && global_y >= 1 && global_y <= Ny - 2 && global_z >= 1 && global_z <= Nz - 2)
      {
        int idx = global_x + global_y * Nx + global_z * (Nx * Ny);

        Ex[idx] = Cax[idx] * Ex[idx] + Cbx[idx] *
                  ((Hz[idx] + Hz[idx - Nx]) + (Hy[idx] + Hy[idx - Nx * Ny]) - Jx[idx] * dx);
        Ey[idx] = Cay[idx] * Ey[idx] + Cby[idx] *
                  ((Hx[idx] + Hx[idx - Nx * Ny]) + (Hz[idx] + Hz[idx - 1]) - Jy[idx] * dx);
        Ez[idx] = Caz[idx] * Ez[idx] + Cbz[idx] *
                  ((Hy[idx] + Hy[idx - 1]) + (Hx[idx] + Hx[idx - Nx]) - Jz[idx] * dx);
      }
    }
  }

}

void gpu_tiling(std::vector<float>& Ex, std::vector<float>& Ey, std::vector<float>& Ez,
                const std::vector<float>& Hx, const std::vector<float>& Hy, const std::vector<float>& Hz,
                const std::vector<float>& Cax, const std::vector<float>& Cay, const std::vector<float>& Caz,
                const std::vector<float>& Cbx, const std::vector<float>& Cby, const std::vector<float>& Cbz,
                const std::vector<float>& Jx, const std::vector<float>& Jy, const std::vector<float>& Jz,
                float dx,
                int Nx, int Ny, int Nz,
                int Tx, int Ty, int Tz,
                int timesteps
) {

  // tiling parameters
  std::vector<int> xx_heads(Tx, 0);
  std::vector<int> yy_heads(Ty, 0);
  std::vector<int> zz_heads(Tz, 0);

  for(int index = 0; index < Tx; index++) {
    xx_heads[index] = index * BLX; 
  }
  for(int index = 0; index < Ty; index++) {
    yy_heads[index] = index * BLY; 
  }
  for(int index = 0; index < Tz; index++) {
    zz_heads[index] = index * BLZ; 
  }

  float *Ex_d, *Ey_d, *Ez_d;
  float *Hx_d, *Hy_d, *Hz_d;
  float *Cax_d, *Cay_d, *Caz_d;
  float *Cbx_d, *Cby_d, *Cbz_d;
  float *Jx_d, *Jy_d, *Jz_d;

  int *xx_heads_d, *yy_heads_d, *zz_heads_d;

  int length = Nx * Ny * Nz;

  CUDACHECK(cudaMalloc(&Ex_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Ey_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Ez_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Hx_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Hy_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Hz_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Cax_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Cay_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Caz_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Cbx_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Cby_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Cbz_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Jx_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Jy_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Jz_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&xx_heads_d, sizeof(int) * Tx));
  CUDACHECK(cudaMalloc(&yy_heads_d, sizeof(int) * Ty));
  CUDACHECK(cudaMalloc(&zz_heads_d, sizeof(int) * Tz));

  CUDACHECK(cudaMemcpyAsync(Ex_d, Ex.data(), sizeof(float) * length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Ey_d, Ey.data(), sizeof(float) * length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Ez_d, Ez.data(), sizeof(float) * length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Hx_d, Hx.data(), sizeof(float) * length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Hy_d, Hy.data(), sizeof(float) * length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Hz_d, Hz.data(), sizeof(float) * length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Jx_d, Jx.data(), sizeof(float) * length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Jy_d, Jy.data(), sizeof(float) * length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Jz_d, Jz.data(), sizeof(float) * length, cudaMemcpyHostToDevice));

  CUDACHECK(cudaMemcpyAsync(Cax_d, Cax.data(), sizeof(float) * length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Cay_d, Cay.data(), sizeof(float) * length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Caz_d, Caz.data(), sizeof(float) * length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Cbx_d, Cbx.data(), sizeof(float) * length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Cby_d, Cby.data(), sizeof(float) * length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Cbz_d, Cbz.data(), sizeof(float) * length, cudaMemcpyHostToDevice));

  CUDACHECK(cudaMemcpyAsync(xx_heads_d, xx_heads.data(), sizeof(int) * Tx, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(yy_heads_d, yy_heads.data(), sizeof(int) * Ty, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(zz_heads_d, zz_heads.data(), sizeof(int) * Tz, cudaMemcpyHostToDevice));

  size_t block_size = NTX * NTY * NTZ;
  size_t grid_size = Tx * Ty * Tz;

  auto start = std::chrono::high_resolution_clock::now();

  for(int t=0; t<timesteps; t++) {
    update_tiling<<<grid_size, block_size>>>(Ex_d, Ey_d, Ez_d,
                                             Hx_d, Hy_d, Hz_d,
                                             Cax_d, Cay_d, Caz_d,
                                             Cbx_d, Cby_d, Cbz_d,
                                             Jx_d, Jy_d, Jz_d,
                                             dx,
                                             Nx, Ny, Nz,
                                             xx_heads_d, yy_heads_d, zz_heads_d,
                                             Tx, Ty, Tz
                                             );
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "gpu tiling runtime (exclude memcpy): " << std::chrono::duration<double>(end-start).count() << " s\n"; 

  cudaMemcpy(Ex.data(), Ex_d, sizeof(float) * length, cudaMemcpyDeviceToHost);
  cudaMemcpy(Ey.data(), Ey_d, sizeof(float) * length, cudaMemcpyDeviceToHost);
  cudaMemcpy(Ez.data(), Ez_d, sizeof(float) * length, cudaMemcpyDeviceToHost);

  CUDACHECK(cudaFree(Ex_d));
  CUDACHECK(cudaFree(Ey_d));
  CUDACHECK(cudaFree(Ez_d));
  CUDACHECK(cudaFree(Hx_d));
  CUDACHECK(cudaFree(Hy_d));
  CUDACHECK(cudaFree(Hz_d));
  CUDACHECK(cudaFree(Cax_d));
  CUDACHECK(cudaFree(Cay_d));
  CUDACHECK(cudaFree(Caz_d));
  CUDACHECK(cudaFree(Cbx_d));
  CUDACHECK(cudaFree(Cby_d));
  CUDACHECK(cudaFree(Cbz_d));
  CUDACHECK(cudaFree(Jx_d));
  CUDACHECK(cudaFree(Jy_d));
  CUDACHECK(cudaFree(Jz_d));
  CUDACHECK(cudaFree(xx_heads_d));
  CUDACHECK(cudaFree(yy_heads_d));
  CUDACHECK(cudaFree(zz_heads_d));

}

// unroll for loop at compile time but with a runtime if check statement
__global__ void update_tiling_unroll_with_rt_if(float* Ex, float* Ey, float* Ez,
                                                float* Hx, float* Hy, float* Hz,
                                                float* Cax, float* Cay, float* Caz,
                                                float* Cbx, float* Cby, float* Cbz,
                                                float* Jx, float* Jy, float* Jz,
                                                float dx,
                                                int Nx, int Ny, int Nz,
                                                int* xx_heads, int* yy_heads, int* zz_heads,
                                                int Tx, int Ty, int Tz) {

  const unsigned int block_id = blockIdx.x;
  const unsigned int thread_id = threadIdx.x;

  const unsigned int xx = block_id % Tx;
  const unsigned int yy = (block_id / Tx) % Ty;
  const unsigned int zz = block_id / (Tx * Ty);

  const int local_x = thread_id % NTX;
  const int local_y = (thread_id / NTX) % NTY;
  const int local_z = thread_id / (NTX * NTY);

  int global_x, global_y, global_z;

  global_x = xx_heads[xx] + local_x; // one-to-one mapping in X  

  // one-to-many mapping in Y and Z
  const int ytile_head = yy_heads[yy] + local_y;
  const int ytile_tail = yy_heads[yy] + BLY - 1;
  const int ztile_head = zz_heads[zz] + local_z;
  const int ztile_tail = zz_heads[zz] + BLZ - 1;

  constexpr int num_y_iters = (BLY + NTY - 1) / NTY;
  constexpr int num_z_iters = (BLZ + NTZ - 1) / NTZ;

  #pragma unroll
  for(int y_iter = 0; y_iter < num_y_iters; y_iter++) {
    global_y = ytile_head + y_iter * NTY;
    if(global_y <= ytile_tail) {

      #pragma unroll
      for(int z_iter = 0; z_iter < num_z_iters; z_iter++) {
        global_z = ztile_head + z_iter * NTZ;
        if(global_z <= ztile_tail) {
          if (global_x >= 1 && global_x <= Nx - 2 && global_y >= 1 && global_y <= Ny - 2 && global_z >= 1 && global_z <= Nz - 2)
          {
            int idx = global_x + global_y * Nx + global_z * (Nx * Ny);

            Ex[idx] = Cax[idx] * Ex[idx] + Cbx[idx] *
                      ((Hz[idx] + Hz[idx - Nx]) + (Hy[idx] + Hy[idx - Nx * Ny]) - Jx[idx] * dx);
            Ey[idx] = Cay[idx] * Ey[idx] + Cby[idx] *
                      ((Hx[idx] + Hx[idx - Nx * Ny]) + (Hz[idx] + Hz[idx - 1]) - Jy[idx] * dx);
            Ez[idx] = Caz[idx] * Ez[idx] + Cbz[idx] *
                      ((Hy[idx] + Hy[idx - 1]) + (Hx[idx] + Hx[idx - Nx]) - Jz[idx] * dx);
          }
        }
      }
    }
  }
}

void gpu_tiling_unroll_with_rt_if(std::vector<float>& Ex, std::vector<float>& Ey, std::vector<float>& Ez,
                                  const std::vector<float>& Hx, const std::vector<float>& Hy, const std::vector<float>& Hz,
                                  const std::vector<float>& Cax, const std::vector<float>& Cay, const std::vector<float>& Caz,
                                  const std::vector<float>& Cbx, const std::vector<float>& Cby, const std::vector<float>& Cbz,
                                  const std::vector<float>& Jx, const std::vector<float>& Jy, const std::vector<float>& Jz,
                                  float dx,
                                  int Nx, int Ny, int Nz,
                                  int Tx, int Ty, int Tz,
                                  int timesteps
) {

  // tiling parameters
  std::vector<int> xx_heads(Tx, 0);
  std::vector<int> yy_heads(Ty, 0);
  std::vector<int> zz_heads(Tz, 0);

  for(int index = 0; index < Tx; index++) {
    xx_heads[index] = index * BLX; 
  }
  for(int index = 0; index < Ty; index++) {
    yy_heads[index] = index * BLY; 
  }
  for(int index = 0; index < Tz; index++) {
    zz_heads[index] = index * BLZ; 
  }

  float *Ex_d, *Ey_d, *Ez_d;
  float *Hx_d, *Hy_d, *Hz_d;
  float *Cax_d, *Cay_d, *Caz_d;
  float *Cbx_d, *Cby_d, *Cbz_d;
  float *Jx_d, *Jy_d, *Jz_d;

  int *xx_heads_d, *yy_heads_d, *zz_heads_d;

  int length = Nx * Ny * Nz;

  CUDACHECK(cudaMalloc(&Ex_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Ey_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Ez_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Hx_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Hy_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Hz_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Cax_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Cay_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Caz_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Cbx_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Cby_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Cbz_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Jx_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Jy_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&Jz_d, sizeof(float) * length));
  CUDACHECK(cudaMalloc(&xx_heads_d, sizeof(int) * Tx));
  CUDACHECK(cudaMalloc(&yy_heads_d, sizeof(int) * Ty));
  CUDACHECK(cudaMalloc(&zz_heads_d, sizeof(int) * Tz));

  CUDACHECK(cudaMemcpyAsync(Ex_d, Ex.data(), sizeof(float) * length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Ey_d, Ey.data(), sizeof(float) * length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Ez_d, Ez.data(), sizeof(float) * length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Hx_d, Hx.data(), sizeof(float) * length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Hy_d, Hy.data(), sizeof(float) * length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Hz_d, Hz.data(), sizeof(float) * length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Jx_d, Jx.data(), sizeof(float) * length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Jy_d, Jy.data(), sizeof(float) * length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Jz_d, Jz.data(), sizeof(float) * length, cudaMemcpyHostToDevice));

  CUDACHECK(cudaMemcpyAsync(Cax_d, Cax.data(), sizeof(float) * length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Cay_d, Cay.data(), sizeof(float) * length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Caz_d, Caz.data(), sizeof(float) * length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Cbx_d, Cbx.data(), sizeof(float) * length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Cby_d, Cby.data(), sizeof(float) * length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Cbz_d, Cbz.data(), sizeof(float) * length, cudaMemcpyHostToDevice));

  CUDACHECK(cudaMemcpyAsync(xx_heads_d, xx_heads.data(), sizeof(int) * Tx, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(yy_heads_d, yy_heads.data(), sizeof(int) * Ty, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(zz_heads_d, zz_heads.data(), sizeof(int) * Tz, cudaMemcpyHostToDevice));

  size_t block_size = NTX * NTY * NTZ;
  size_t grid_size = Tx * Ty * Tz;

  auto start = std::chrono::high_resolution_clock::now();

  for(int t=0; t<timesteps; t++) {
    update_tiling_unroll_with_rt_if<<<grid_size, block_size>>>(Ex_d, Ey_d, Ez_d,
                                                               Hx_d, Hy_d, Hz_d,
                                                               Cax_d, Cay_d, Caz_d,
                                                               Cbx_d, Cby_d, Cbz_d,
                                                               Jx_d, Jy_d, Jz_d,
                                                               dx,
                                                               Nx, Ny, Nz,
                                                               xx_heads_d, yy_heads_d, zz_heads_d,
                                                               Tx, Ty, Tz
                                                               );

  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "gpu tiling unroll with runtime if check runtime (exclude memcpy): " << std::chrono::duration<double>(end-start).count() << " s\n"; 

  cudaMemcpy(Ex.data(), Ex_d, sizeof(float) * length, cudaMemcpyDeviceToHost);
  cudaMemcpy(Ey.data(), Ey_d, sizeof(float) * length, cudaMemcpyDeviceToHost);
  cudaMemcpy(Ez.data(), Ez_d, sizeof(float) * length, cudaMemcpyDeviceToHost);

  CUDACHECK(cudaFree(Ex_d));
  CUDACHECK(cudaFree(Ey_d));
  CUDACHECK(cudaFree(Ez_d));
  CUDACHECK(cudaFree(Hx_d));
  CUDACHECK(cudaFree(Hy_d));
  CUDACHECK(cudaFree(Hz_d));
  CUDACHECK(cudaFree(Cax_d));
  CUDACHECK(cudaFree(Cay_d));
  CUDACHECK(cudaFree(Caz_d));
  CUDACHECK(cudaFree(Cbx_d));
  CUDACHECK(cudaFree(Cby_d));
  CUDACHECK(cudaFree(Cbz_d));
  CUDACHECK(cudaFree(Jx_d));
  CUDACHECK(cudaFree(Jy_d));
  CUDACHECK(cudaFree(Jz_d));
  CUDACHECK(cudaFree(xx_heads_d));
  CUDACHECK(cudaFree(yy_heads_d));
  CUDACHECK(cudaFree(zz_heads_d));

}



int main(int argc, char* argv[]) {

  std::cerr << "BLX = " << BLX << ", BLY = " << BLY << ", BLZ = " << BLZ << "\n"; 
  std::cerr << "Nx = Tx * BLX\n";
  if(argc != 5) {
    std::cerr << "usage: ./a.out Tx Ty Tz timesteps\n";
    std::exit(EXIT_FAILURE);
  }

  int Tx = std::atoi(argv[1]);
  int Ty = std::atoi(argv[2]);
  int Tz = std::atoi(argv[3]);
  int timesteps = std::atoi(argv[4]);

  const float dx = 1.0;

  int Nx = Tx * BLX;
  int Ny = Ty * BLY;
  int Nz = Tz * BLZ; 

  std::cout << "Nx = " << Nx << ", Ny = " << Ny << ", Nz = " << Nz << "\n";

  int array_size = Nx * Ny * Nz;
  std::vector<float> Cax(array_size, 1.0);
  std::vector<float> Cay(array_size, 1.0);
  std::vector<float> Caz(array_size, 1.0);
  std::vector<float> Cbx(array_size, 1.0);
  std::vector<float> Cby(array_size, 1.0);
  std::vector<float> Cbz(array_size, 1.0);

  std::vector<float> Jx(array_size, 1.0);
  std::vector<float> Jy(array_size, 1.0);
  std::vector<float> Jz(array_size, 1.0);
  
  std::vector<float> Hx(array_size, 1.0);
  std::vector<float> Hy(array_size, 1.0);
  std::vector<float> Hz(array_size, 1.0);

  std::vector<float> Ex_cpu_single(array_size, 1.0);
  std::vector<float> Ey_cpu_single(array_size, 1.0);
  std::vector<float> Ez_cpu_single(array_size, 1.0);
  std::vector<float> Ex_gpu_naive(array_size, 1.0);
  std::vector<float> Ey_gpu_naive(array_size, 1.0);
  std::vector<float> Ez_gpu_naive(array_size, 1.0);
  std::vector<float> Ex_gpu_tiling(array_size, 1.0);
  std::vector<float> Ey_gpu_tiling(array_size, 1.0);
  std::vector<float> Ez_gpu_tiling(array_size, 1.0);
  std::vector<float> Ex_gpu_tiling_unroll_with_rt_if(array_size, 1.0);
  std::vector<float> Ey_gpu_tiling_unroll_with_rt_if(array_size, 1.0);
  std::vector<float> Ez_gpu_tiling_unroll_with_rt_if(array_size, 1.0);

  sequential(Ex_cpu_single, Ey_cpu_single, Ez_cpu_single,
             Hx, Hy, Hz,
             Cax, Cay, Caz,
             Cbx, Cby, Cbz,
             Jx, Jy, Jz,
             dx,
             Nx, Ny, Nz,
             timesteps);

  gpu_naive(Ex_gpu_naive, Ey_gpu_naive, Ez_gpu_naive,
            Hx, Hy, Hz,
            Cax, Cay, Caz,
            Cbx, Cby, Cbz,
            Jx, Jy, Jz,
            dx,
            Nx, Ny, Nz,
            timesteps);

  gpu_tiling(Ex_gpu_tiling, Ey_gpu_tiling, Ez_gpu_tiling,
             Hx, Hy, Hz,
             Cax, Cay, Caz,
             Cbx, Cby, Cbz,
             Jx, Jy, Jz,
             dx,
             Nx, Ny, Nz,
             Tx, Ty, Tz,
             timesteps);

  gpu_tiling_unroll_with_rt_if(Ex_gpu_tiling_unroll_with_rt_if, 
             Ey_gpu_tiling_unroll_with_rt_if, 
             Ez_gpu_tiling_unroll_with_rt_if,
             Hx, Hy, Hz,
             Cax, Cay, Caz,
             Cbx, Cby, Cbz,
             Jx, Jy, Jz,
             dx,
             Nx, Ny, Nz,
             Tx, Ty, Tz,
             timesteps);

  for(size_t i=0; i<Nx*Ny*Nz; i++) {
    if(fabs(Ex_cpu_single[i] - Ex_gpu_naive[i]) > 1e-5 ||
       fabs(Ey_cpu_single[i] - Ey_gpu_naive[i]) > 1e-5 ||
       fabs(Ez_cpu_single[i] - Ez_gpu_naive[i]) > 1e-5) {
      std::cerr << "results incorrect!\n";
      std::exit(EXIT_FAILURE);
    }
  }

  for(size_t i=0; i<Nx*Ny*Nz; i++) {
    if(fabs(Ex_cpu_single[i] - Ex_gpu_tiling[i]) > 1e-5 ||
       fabs(Ey_cpu_single[i] - Ey_gpu_tiling[i]) > 1e-5 ||
       fabs(Ez_cpu_single[i] - Ez_gpu_tiling[i]) > 1e-5) {
      std::cerr << "results incorrect!\n";
      std::exit(EXIT_FAILURE);
    }
  }

  for(size_t i=0; i<Nx*Ny*Nz; i++) {
    if(fabs(Ex_cpu_single[i] - Ex_gpu_tiling_unroll_with_rt_if[i]) > 1e-5 ||
       fabs(Ey_cpu_single[i] - Ey_gpu_tiling_unroll_with_rt_if[i]) > 1e-5 ||
       fabs(Ez_cpu_single[i] - Ez_gpu_tiling_unroll_with_rt_if[i]) > 1e-5) {
      std::cerr << "results incorrect!\n";
      std::exit(EXIT_FAILURE);
    }
  }

  std::cerr << "results matched.\n";

  return 0;
}





















































