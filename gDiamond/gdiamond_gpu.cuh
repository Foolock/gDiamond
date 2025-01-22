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

void gDiamond::update_FDTD_gpu_2D(size_t num_timesteps) {

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

  // initialize E, H as 0 
  CUDACHECK(cudaMemset(Ex, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Ey, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Ez, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Hx, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Hy, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Hz, 0, sizeof(float) * _Nx * _Ny * _Nz));

  // initialize J, M, Ca, Cb, Da, Db as 0 
  CUDACHECK(cudaMemset(Jx, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Jy, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Jz, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Mx, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(My, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Mz, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Cax, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Cbx, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Cay, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Cby, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Caz, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Cbz, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Dax, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Dbx, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Day, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Dby, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Daz, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Dbz, 0, sizeof(float) * _Nx * _Ny * _Nz));
  
  auto start = std::chrono::high_resolution_clock::now();

  // copy Ca, Cb, Da, Db
  CUDACHECK(cudaMemcpy(Cax, _Cax.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Cay, _Cay.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Caz, _Caz.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Cbx, _Cbx.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Cby, _Cby.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Cbz, _Cbz.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Dax, _Dax.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Day, _Day.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Daz, _Daz.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Dbx, _Dbx.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Dby, _Dby.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Dbz, _Dbz.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));

  // set block and grid
  dim3 blockSize_2D(BLOCK_SIZE, 1);
  dim3 gridSize_2D((_Nx + blockSize_2D.x - 1) / blockSize_2D.x, _Ny);

  // size_t plot_interval = num_timesteps / 10;

  for(size_t t=0; t<num_timesteps; t++) {

    // Current source
    float Mz_value = M_source_amp * std::sin(SOURCE_OMEGA * t * dt);

    CUDACHECK(cudaMemcpy(Mz + _source_idx, &Mz_value, sizeof(float), cudaMemcpyHostToDevice));
    
    // update E
    updateE_2Dmap<<<gridSize_2D, blockSize_2D, 0>>>(Ex, Ey, Ez,
          Hx, Hy, Hz, Cax, Cbx, Cay, Cby, Caz, Cbz,
          Jx, Jy, Jz, _dx, _Nx, _Ny, _Nz);

    // update H
    updateH_2Dmap<<<gridSize_2D, blockSize_2D, 0>>>(Ex, Ey, Ez,
          Hx, Hy, Hz, Dax, Dbx, Day, Dby, Daz, Dbz,
          Mx, My, Mz, _dx, _Nx, _Ny, _Nz);

    /*
    // Record the field using a monitor, once in a while
    if (t % plot_interval == 0)
    {
      printf("Iter: %ld / %ld \n", t, num_timesteps);

      float *H_time_monitor_xy;
      H_time_monitor_xy = (float *)malloc(_Nx * _Ny * sizeof(float));
      memset(H_time_monitor_xy, 0, _Nx * _Ny * sizeof(float));

      // ------------ plotting time domain
      // File name initialization
      char field_filename[50];
      size_t slice_pitch = _Nx * sizeof(float); // The size in bytes of the 2D slice row
      size_t k = _Nz / 2;  // Assuming you want the middle slice
      for (size_t j = 0; j < _Ny; ++j)
      {
        float* device_ptr = Hz + j * _Nx + k * _Nx * _Ny; // Pointer to the start of the row in the desired slice
        float* host_ptr = H_time_monitor_xy + j * _Nx;  // Pointer to the host memory
        cudaMemcpy(host_ptr, device_ptr, slice_pitch, cudaMemcpyDeviceToHost);
      }

      snprintf(field_filename, sizeof(field_filename), "figures/Hz_%04ld.png", t);
      save_field_png(H_time_monitor_xy, field_filename, _Nx, _Ny, 1.0 / sqrt(mu0 / eps0));

      free(H_time_monitor_xy);
    }
    */
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
  std::cout << "gpu runtime (2-D mapping): " << std::chrono::duration<double>(end-start).count() << "s\n"; 

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

void gDiamond::update_FDTD_gpu_3D_warp_underutilization(size_t num_timesteps) {

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

  // initialize E, H as 0 
  CUDACHECK(cudaMemset(Ex, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Ey, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Ez, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Hx, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Hy, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Hz, 0, sizeof(float) * _Nx * _Ny * _Nz));

  // initialize J, M, Ca, Cb, Da, Db as 0 
  CUDACHECK(cudaMemset(Jx, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Jy, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Jz, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Mx, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(My, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Mz, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Cax, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Cbx, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Cay, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Cby, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Caz, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Cbz, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Dax, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Dbx, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Day, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Dby, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Daz, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Dbz, 0, sizeof(float) * _Nx * _Ny * _Nz));
  
  auto start = std::chrono::high_resolution_clock::now();

  // copy Ca, Cb, Da, Db
  CUDACHECK(cudaMemcpy(Cax, _Cax.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Cay, _Cay.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Caz, _Caz.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Cbx, _Cbx.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Cby, _Cby.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Cbz, _Cbz.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Dax, _Dax.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Day, _Day.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Daz, _Daz.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Dbx, _Dbx.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Dby, _Dby.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Dbz, _Dbz.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));

  // set block and grid
  dim3 blockSize_3D(BLOCK_SIZE, 1, 1);
  dim3 gridSize_3D((_Nx + blockSize_3D.x - 1) / blockSize_3D.x, _Ny, _Nz);

  for(size_t t=0; t<num_timesteps; t++) {

    // Current source
    float Mz_value = M_source_amp * std::sin(SOURCE_OMEGA * t * dt);

    CUDACHECK(cudaMemcpy(Mz + _source_idx, &Mz_value, sizeof(float), cudaMemcpyHostToDevice));
    
    // update E
    updateE_3Dmap<<<gridSize_3D, blockSize_3D, 0>>>(Ex, Ey, Ez,
          Hx, Hy, Hz, Cax, Cbx, Cay, Cby, Caz, Cbz,
          Jx, Jy, Jz, _dx, _Nx, _Ny, _Nz);

    // update H
    updateH_3Dmap<<<gridSize_3D, blockSize_3D, 0>>>(Ex, Ey, Ez,
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
  std::cout << "gpu runtime (3-D mapping): " << std::chrono::duration<double>(end-start).count() << "s\n"; 

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

void gDiamond::update_FDTD_gpu_3D_warp_underutilization_fix(size_t num_timesteps) {

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

  // initialize E, H as 0 
  CUDACHECK(cudaMemset(Ex, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Ey, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Ez, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Hx, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Hy, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Hz, 0, sizeof(float) * _Nx * _Ny * _Nz));

  // initialize J, M, Ca, Cb, Da, Db as 0 
  CUDACHECK(cudaMemset(Jx, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Jy, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Jz, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Mx, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(My, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Mz, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Cax, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Cbx, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Cay, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Cby, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Caz, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Cbz, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Dax, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Dbx, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Day, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Dby, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Daz, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Dbz, 0, sizeof(float) * _Nx * _Ny * _Nz));
  
  auto start = std::chrono::high_resolution_clock::now();

  // copy Ca, Cb, Da, Db
  CUDACHECK(cudaMemcpy(Cax, _Cax.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Cay, _Cay.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Caz, _Caz.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Cbx, _Cbx.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Cby, _Cby.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Cbz, _Cbz.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Dax, _Dax.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Day, _Day.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Daz, _Daz.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Dbx, _Dbx.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Dby, _Dby.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(Dbz, _Dbz.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));

  // set block and grid
  size_t grid_size = (_Nx*_Ny*_Nz + BLOCK_SIZE - 1) / BLOCK_SIZE;

  for(size_t t=0; t<num_timesteps; t++) {

    // Current source
    float Mz_value = M_source_amp * std::sin(SOURCE_OMEGA * t * dt);

    CUDACHECK(cudaMemcpy(Mz + _source_idx, &Mz_value, sizeof(float), cudaMemcpyHostToDevice));
    
    // update E
    updateE_3Dmap_fix<<<grid_size, BLOCK_SIZE, 0>>>(Ex, Ey, Ez,
          Hx, Hy, Hz, Cax, Cbx, Cay, Cby, Caz, Cbz,
          Jx, Jy, Jz, _dx, _Nx, _Ny, _Nz);

    // update H
    updateH_3Dmap_fix<<<grid_size, BLOCK_SIZE, 0>>>(Ex, Ey, Ez,
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
  std::cout << "gpu runtime (3-D mapping): " << std::chrono::duration<double>(end-start).count() << "s\n"; 

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































