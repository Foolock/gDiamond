#ifndef GDIAMOND_GPU_CUH
#define GDIAMOND_GPU_CUH

#include "gdiamond.hpp"
#include "kernels.cuh"
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

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

namespace gdiamond {

void gDiamond::update_FDTD_gpu(size_t num_timesteps) {

  // E, H, J, M on device 
  float *Ex, *Ey, *Ez, *Hx, *Hy, *Hz, *Jx, *Jy, *Jz, *Mx, *My, *Mz;

  // Ca, Cb, Da, Db on device
  float *Cax, *Cay, *Caz, *Cbx, *Cby, *Cbz;
  float *Dax, *Day, *Daz, *Dbx, *Dby, *Dbz;

  CUDACHECK(cudaMalloc(&Ex, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Ey, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Ez, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Hx, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Hy, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Hz, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Jx, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Jy, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Jz, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Mx, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&My, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Mz, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Cax, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Cbx, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Cay, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Cby, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Caz, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Cbz, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Dax, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Dbx, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Day, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Dby, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Daz, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Dbz, sizeof(float) * _Nx * _Ny * _Nz)); 

  // initialize J, M, Ca, Cb, Da, Db as 1
  CUDACHECK(cudaMemset(Jx, 1.0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Jy, 1.0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Jz, 1.0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Mx, 1.0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(My, 1.0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Mz, 1.0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Cax, 1.0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Cbx, 1.0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Cay, 1.0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Cby, 1.0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Caz, 1.0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Cbz, 1.0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Dax, 1.0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Dbx, 1.0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Day, 1.0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Dby, 1.0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Daz, 1.0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Dbz, 1.0, sizeof(float) * _Nx * _Ny * _Nz));
  
  auto start = std::chrono::high_resolution_clock::now();

  // copy initial E, H to device
  CUDACHECK(cudaMemcpy(Ex, _Ex.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Ey, _Ey.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Ez, _Ez.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Hx, _Hx.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Hy, _Hy.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Hz, _Hz.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));

  // set block and grid
  dim3 blockSize(BLOCK_SIZE, 1);
  dim3 gridSize((_Nx + blockSize.x - 1) / blockSize.x, _Ny);

  for(size_t t=0; t<num_timesteps; t++) {
    
    // update E
    updateE<<<gridSize, blockSize, 0>>>(Ex, Ey, Ez,
          Hx, Hy, Hz, Cax, Cbx, Cay, Cby, Caz, Cbz,
          Jx, Jy, Jz, _dx, _Nx, _Ny, _Nz);

    // update H
    updateH<<<gridSize, blockSize, 0>>>(Ex, Ey, Ez,
          Hx, Hy, Hz, Dax, Dbx, Day, Dby, Daz, Dbz,
          Mx, My, Mz, _dx, _Nx, _Ny, _Nz);
  }
  cudaDeviceSynchronize();

  // copy E, H back to host 
  CUDACHECK(cudaMemcpy(_Ex_gpu.data(), Ex, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Ey_gpu.data(), Ey, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Ez_gpu.data(), Ez, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Hx_gpu.data(), Hx, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Hy_gpu.data(), Hy, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Hz_gpu.data(), Hz, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));

  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "gpu runtime: " << std::chrono::duration<double>(end-start).count() << "s\n"; 

  CUDACHECK(cudaFree(Ex));
  CUDACHECK(cudaFree(Ey));
  CUDACHECK(cudaFree(Ez));
  CUDACHECK(cudaFree(Hx));
  CUDACHECK(cudaFree(Hy));
  CUDACHECK(cudaFree(Hz));
  CUDACHECK(cudaFree(Jx));
  CUDACHECK(cudaFree(Jy));
  CUDACHECK(cudaFree(Jz));
  CUDACHECK(cudaFree(Mx));
  CUDACHECK(cudaFree(My));
  CUDACHECK(cudaFree(Mz));
  CUDACHECK(cudaFree(Cax));
  CUDACHECK(cudaFree(Cbx));
  CUDACHECK(cudaFree(Cay));
  CUDACHECK(cudaFree(Cby));
  CUDACHECK(cudaFree(Caz));
  CUDACHECK(cudaFree(Cbz));
  CUDACHECK(cudaFree(Dax));
  CUDACHECK(cudaFree(Dbx));
  CUDACHECK(cudaFree(Day));
  CUDACHECK(cudaFree(Dby));
  CUDACHECK(cudaFree(Daz));
  CUDACHECK(cudaFree(Dbz));
}

} // end of namespace gdiamond

#endif































