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

void gDiamond::update_FDTD_gpu_figures(size_t num_timesteps) { // only use for result checking

  if (std::filesystem::create_directory("gpu_figures")) {
      std::cerr << "gpu_figures created successfully. " << std::endl;
  } else {
      std::cerr << "failed to create gpu_figures or it already exists." << std::endl;
      std::exit(EXIT_FAILURE);
  }

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
  
  // // specify source
  // for(size_t t=0; t<num_timesteps; t++) {
  //   // Current source
  //   float Mz_value = M_source_amp * std::sin(SOURCE_OMEGA * t * dt);
  //   CUDACHECK(cudaMemcpy(Mz + _source_idx, &Mz_value, sizeof(float), cudaMemcpyHostToDevice));
  // }
 
  std::chrono::duration<double> gpu_runtime;
  
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

  auto end = std::chrono::high_resolution_clock::now();

  gpu_runtime = end - start;

  // set block and grid
  size_t grid_size = (_Nx*_Ny*_Nz + BLOCK_SIZE - 1) / BLOCK_SIZE;

  for(size_t t=0; t<num_timesteps; t++) {

    auto start1 = std::chrono::high_resolution_clock::now();

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

    auto end1 = std::chrono::high_resolution_clock::now();

    gpu_runtime += end1 - start1;

    // Record the field using a monitor, once in a while
    if (t % (num_timesteps/10) == 0)
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

      snprintf(field_filename, sizeof(field_filename), "gpu_figures/Hz_naive_gpu_%04ld.png", t);
      save_field_png(H_time_monitor_xy, field_filename, _Nx, _Ny, 1.0 / sqrt(mu0 / eps0));

      free(H_time_monitor_xy);
    }
  }
  cudaDeviceSynchronize();

  start = std::chrono::high_resolution_clock::now();
  // copy E, H back to host 
  CUDACHECK(cudaMemcpy(_Ex_gpu.data(), Ex, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Ey_gpu.data(), Ey, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Ez_gpu.data(), Ez, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Hx_gpu.data(), Hx, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Hy_gpu.data(), Hy, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Hz_gpu.data(), Hz, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));

  end = std::chrono::high_resolution_clock::now();

  gpu_runtime += end - start;

  std::cout << "naive gpu runtime (excluding figures output): " << gpu_runtime.count() << "s\n"; 
  std::cout << "naive gpu performance (excluding figures output): " << (_Nx * _Ny * _Nz / 1.0e6 * num_timesteps) / gpu_runtime.count() << "Mcells/s\n";

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

void gDiamond::update_FDTD_gpu_check_result(size_t num_timesteps) { // only use for result checking

  /*
  if (std::filesystem::create_directory("gpu_figures")) {
      std::cerr << "gpu_figures created successfully. " << std::endl;
  } else {
      std::cerr << "failed to create gpu_figures or it already exists." << std::endl;
      std::exit(EXIT_FAILURE);
  }
  */

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
  
  // specify source
  for(size_t t=0; t<num_timesteps; t++) {
    // Current source
    float Mz_value = M_source_amp * std::sin(SOURCE_OMEGA * t * dt);
    CUDACHECK(cudaMemcpy(Mz + _source_idx, &Mz_value, sizeof(float), cudaMemcpyHostToDevice));
  }
 
  std::chrono::duration<double> gpu_runtime;
  
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

  auto end = std::chrono::high_resolution_clock::now();

  gpu_runtime = end - start;

  // set block and grid
  size_t grid_size = (_Nx*_Ny*_Nz + BLOCK_SIZE - 1) / BLOCK_SIZE;

  for(size_t t=0; t<num_timesteps; t++) {

    auto start1 = std::chrono::high_resolution_clock::now();

    // // Current source
    // float Mz_value = M_source_amp * std::sin(SOURCE_OMEGA * t * dt);

    // CUDACHECK(cudaMemcpy(Mz + _source_idx, &Mz_value, sizeof(float), cudaMemcpyHostToDevice));
    
    // update E
    updateE_3Dmap_fix<<<grid_size, BLOCK_SIZE, 0>>>(Ex, Ey, Ez,
          Hx, Hy, Hz, Cax, Cbx, Cay, Cby, Caz, Cbz,
          Jx, Jy, Jz, _dx, _Nx, _Ny, _Nz);

    // update H
    updateH_3Dmap_fix<<<grid_size, BLOCK_SIZE, 0>>>(Ex, Ey, Ez,
          Hx, Hy, Hz, Dax, Dbx, Day, Dby, Daz, Dbz,
          Mx, My, Mz, _dx, _Nx, _Ny, _Nz);

    auto end1 = std::chrono::high_resolution_clock::now();

    gpu_runtime += end1 - start1;

    /*
    // Record the field using a monitor, once in a while
    if (t % (num_timesteps/10) == 0)
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

      snprintf(field_filename, sizeof(field_filename), "gpu_figures/Hz_naive_gpu_%04ld.png", t);
      save_field_png(H_time_monitor_xy, field_filename, _Nx, _Ny, 1.0 / sqrt(mu0 / eps0));

      free(H_time_monitor_xy);
    }
    */
  }
  cudaDeviceSynchronize();

  start = std::chrono::high_resolution_clock::now();
  // copy E, H back to host 
  CUDACHECK(cudaMemcpy(_Ex_gpu.data(), Ex, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Ey_gpu.data(), Ey, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Ez_gpu.data(), Ez, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Hx_gpu.data(), Hx, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Hy_gpu.data(), Hy, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Hz_gpu.data(), Hz, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));

  end = std::chrono::high_resolution_clock::now();

  gpu_runtime += end - start;

  std::cout << "naive gpu runtime (excluding figures output): " << gpu_runtime.count() << "s\n"; 
  std::cout << "naive gpu performance (excluding figures output): " << (_Nx * _Ny * _Nz / 1.0e6 * num_timesteps) / gpu_runtime.count() << "Mcells/s\n";

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
    if (t % 10 == 0)
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

  // transfer source
  for(size_t t=0; t<num_timesteps; t++) {
    float Mz_value = M_source_amp * std::sin(SOURCE_OMEGA * t * dt);
    CUDACHECK(cudaMemcpy(Mz + _source_idx, &Mz_value, sizeof(float), cudaMemcpyHostToDevice));
  }
  
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

    // // Current source
    // float Mz_value = M_source_amp * std::sin(SOURCE_OMEGA * t * dt);

    // CUDACHECK(cudaMemcpy(Mz + _source_idx, &Mz_value, sizeof(float), cudaMemcpyHostToDevice));
    
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
  CUDACHECK(cudaMemcpy(_Ex_gpu_bl.data(), Ex, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Ey_gpu_bl.data(), Ey, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Ez_gpu_bl.data(), Ez, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Hx_gpu_bl.data(), Hx, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Hy_gpu_bl.data(), Hy, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Hz_gpu_bl.data(), Hz, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));

  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "gpu runtime (3-D mapping): " << std::chrono::duration<double>(end-start).count() << "s\n"; 
  std::cout << "gpu performance: " << (_Nx * _Ny * _Nz / 1.0e6 * num_timesteps) / std::chrono::duration<double>(end-start).count() << "Mcells/s\n";

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

void gDiamond::update_FDTD_gpu_simulation_1_D_shmem(size_t num_timesteps) { // CPU single thread 1-D simulation of GPU workflow 

  size_t max_phases = 8;
  std::vector<int> mountain_heads_X;
  std::vector<int> mountain_tails_X;
  std::vector<int> valley_heads_X;
  std::vector<int> valley_tails_X;
  _setup_diamond_tiling_gpu(BLX_GPU, BLY_GPU, BLZ_GPU, BLT_GPU, max_phases);

  for(auto range : _Eranges_phases_X[0][0]) { 
    mountain_heads_X.push_back(range.first);
    mountain_tails_X.push_back(range.second);
  }
  for(auto range : _Hranges_phases_X[1][BLT_GPU-1]) { 
    valley_heads_X.push_back(range.first);
    valley_tails_X.push_back(range.second);
  }
  
  size_t num_mountains_X = mountain_heads_X.size();
  size_t num_valleys_X = valley_heads_X.size();

  // write 1 dimension just to check
  std::vector<float> E_simu(_Nx, 1);
  std::vector<float> H_simu(_Nx, 1);
  std::vector<float> E_seq(_Nx, 1);
  std::vector<float> H_seq(_Nx, 1);

  // seq version
  for(size_t t=0; t<num_timesteps; t++) {

    // update E
    for(size_t x=1; x<_Nx-1; x++) {
      E_seq[x] = H_seq[x-1] + H_seq[x] * 2; 
    }

    // std::cout << "t = " << t << ", E_seq =";
    // for(size_t x=0; x<_Nx; x++) {
    //   std::cout << E_seq[x] << " ";
    // }
    // std::cout << "\n";

    // update H 
    for(size_t x=1; x<_Nx-1; x++) {
      H_seq[x] = E_seq[x+1] + E_seq[x] * 2; 
    }
  }

  // tiling version
  int mountain_or_valley;
  // 1, mountain, 0, valley
  int Nx = _Nx;
  for(size_t tt=0; tt<num_timesteps/BLT_GPU; tt++) {

    // phase 1. moutains 
    mountain_or_valley = 1;
    for(size_t xx=0; xx<num_mountains_X; xx++) { // xx is block id 

      float H_shmem[BLX_GPU + 1];
      float E_shmem[BLX_GPU + 1];
      for(int local_x=0; local_x<BLX_GPU; local_x++) {
        int global_x = mountain_heads_X[xx] + local_x;
        int shared_x = local_x + 1;
        if(global_x >= 0 && global_x < Nx) {
          H_shmem[shared_x] = H_simu[global_x];

          // load HALO
          if(local_x == 0 && global_x > 0) {
            H_shmem[shared_x - 1] = H_simu[global_x - 1];
          }
        }
      }
      for(int local_x=0; local_x<BLX_GPU; local_x++) {
        int global_x = mountain_heads_X[xx] + local_x;
        int shared_x = local_x;
        if(global_x >= 0 && global_x < Nx) {
          E_shmem[shared_x] = E_simu[global_x];

          // load HALO
          if(local_x == BLX_GPU-1 && global_x < Nx-1) {
            E_shmem[shared_x + 1] = E_simu[global_x + 1];
          }
        }
      }

      for(size_t t=0; t<BLT_GPU; t++) {

        int calculate_E = 1; // calculate this E tile or not
        int calculate_H = 1; // calculate this H tile or not

        std::vector<int> indices = _get_head_tail(BLX_GPU, BLT_GPU,
                                                  mountain_heads_X, mountain_tails_X,
                                                  xx, t,
                                                  mountain_or_valley,
                                                  Nx,
                                                  &calculate_E, &calculate_H);

        // update E
        if(calculate_E) {
          for(int local_x=0; local_x<BLX_GPU; local_x++) { // local_x is thread id
            int global_x = indices[0] + local_x; 
            int Eoffset = indices[0] - mountain_heads_X[xx];
            int Hoffset = indices[2] - mountain_heads_X[xx];
            int Eshared_x = local_x + Eoffset; 
            int Hshared_x = local_x + 1 + Hoffset;   
            if(global_x>=1 && global_x<=Nx-2 && global_x<= indices[1]) {
              E_shmem[Eshared_x] = H_shmem[Hshared_x-1] + H_shmem[Hshared_x] * 2; 
            }
          }
        }

        // std::cout << "t = " << t << ", xx = " << xx << ", E_shmem = ";
        // for(auto e : E_shmem) {
        //   std::cout << e << " ";
        // }
        // std::cout << "\n";

        // update H
        if(calculate_H) {
          for(int local_x=0; local_x<BLX_GPU; local_x++) { // local_x is thread id
            int global_x = indices[2] + local_x; 
            int Eoffset = indices[0] - mountain_heads_X[xx];
            int Hoffset = indices[2] - mountain_heads_X[xx];
            int Eshared_x = local_x + Eoffset; 
            int Hshared_x = local_x + 1 + Hoffset;   
            if(global_x>=1 && global_x<=Nx-2 && global_x<= indices[3]) {
              H_shmem[Hshared_x] = E_shmem[Eshared_x+1] + E_shmem[Eshared_x] * 2; 
            }
          }
        }
      }

      // load back result
      for(int local_x=0; local_x<BLX_GPU; local_x++) {
        int global_x = mountain_heads_X[xx] + local_x;
        int Eshared_x = local_x; 
        int Hshared_x = local_x + 1;   
        if(global_x>=1 && global_x<=Nx-2) {
          H_simu[global_x] = H_shmem[Hshared_x];
          E_simu[global_x] = E_shmem[Eshared_x];
        }
      }
    }

    std::cout << "E_simu = ";
    for(auto e : E_simu) {
      std::cout << e << " ";
    }
    std::cout << "\n";

    // phase 2. valleys
    mountain_or_valley = 0;
    for(size_t xx=0; xx<num_valleys_X; xx++) { // xx is block id 

      float H_shmem[BLX_GPU + 1];
      float E_shmem[BLX_GPU + 1];
      for(int local_x=0; local_x<BLX_GPU; local_x++) {
        int global_x = valley_heads_X[xx] + local_x;
        int shared_x = local_x + 1;
        if(global_x >= 0 && global_x < Nx) {
          H_shmem[shared_x] = H_simu[global_x];

          // load HALO
          if(local_x == 0 && global_x > 0) {
            H_shmem[shared_x - 1] = H_simu[global_x - 1];
          }
        }
      }
      for(int local_x=0; local_x<BLX_GPU; local_x++) {
        int global_x = valley_heads_X[xx] + local_x;
        int shared_x = local_x;
        if(global_x >= 0 && global_x < Nx) {
          E_shmem[shared_x] = E_simu[global_x];

          // load HALO
          if(local_x == BLX_GPU-1 && global_x < Nx-1) {
            E_shmem[shared_x + 1] = E_simu[global_x + 1];
          }
        }
      }

      for(size_t t=0; t<BLT_GPU; t++) {

        int calculate_E = 1; // calculate this E tile or not
        int calculate_H = 1; // calculate this H tile or not

        std::vector<int> indices = _get_head_tail(BLX_GPU, BLT_GPU,
                                                  valley_heads_X, valley_tails_X,
                                                  xx, t,
                                                  mountain_or_valley,
                                                  Nx,
                                                  &calculate_E, &calculate_H);

        // update E
        if(calculate_E) {
          for(int local_x=0; local_x<BLX_GPU; local_x++) { // local_x is thread id
            int global_x = indices[0] + local_x; 
            int Eoffset = indices[0] - valley_heads_X[xx];
            int Hoffset = indices[2] - valley_heads_X[xx];
            int Eshared_x = local_x + Eoffset; 
            int Hshared_x = (xx == 0)? local_x + 1 + Hoffset : local_x + 1 + Hoffset + 1;   
            // if(t == 0 && xx == 1 && Eshared_x == 4) {
            //   std::cout << "E_shmem[Eshared_x] = " << E_shmem[Eshared_x] << "\n"; 
            //   std::cout << "H_shmem[Hshared_x-1] = " << H_shmem[Hshared_x-1] << "\n";
            //   std::cout << "H_shmem[Hshared_x] = " << H_shmem[Hshared_x] << "\n";
            //   std::cout << "Hshared_x = " << Hshared_x << "\n";
            // }
            if(global_x>=1 && global_x<=Nx-2 && global_x<= indices[1]) {
              E_shmem[Eshared_x] = H_shmem[Hshared_x-1] + H_shmem[Hshared_x] * 2; 
            }
          }
        }

        std::cout << "t = " << t << ", xx = " << xx << ", E_shmem = ";
        for(auto e : E_shmem) {
          std::cout << e << " ";
        }
        std::cout << "\n";

        // update H
        if(calculate_H) {
          for(int local_x=0; local_x<BLX_GPU; local_x++) { // local_x is thread id
            int global_x = indices[2] + local_x; 
            int Eoffset = indices[0] - valley_heads_X[xx];
            int Hoffset = indices[2] - valley_heads_X[xx];
            int Eshared_x = (xx == 0)? local_x + Eoffset : local_x + Eoffset - 1; 
            int Hshared_x = local_x + 1 + Hoffset;   
            if(global_x>=1 && global_x<=Nx-2 && global_x<= indices[3]) {
              H_shmem[Hshared_x] = E_shmem[Eshared_x+1] + E_shmem[Eshared_x] * 2; 
            }
          }
        }

      }

      // load back result
      for(int local_x=0; local_x<BLX_GPU; local_x++) {
        int global_x = valley_heads_X[xx] + local_x;
        int Eshared_x = local_x; 
        int Hshared_x = local_x + 1;   
        if(global_x>=1 && global_x<=Nx-2) {
          H_simu[global_x] = H_shmem[Hshared_x];
          E_simu[global_x] = E_shmem[Eshared_x];
        }
      }
    } 
  }

  std::cout << "E_seq = ";
  for(size_t x=0; x<_Nx; x++) {
    std::cout << E_seq[x] << " ";
  }
  std::cout << "\n";

  std::cout << "E_simu = ";
  for(size_t x=0; x<_Nx; x++) {
    std::cout << E_simu[x] << " ";
  }
  std::cout << "\n";

  for(size_t x=0; x<_Nx; x++) {
    if(E_seq[x] != E_simu[x] || H_seq[x] != H_simu[x]) {
      std::cerr << "1-D demo results mismatch.\n";
      std::exit(EXIT_FAILURE);
    }
  }

}

void gDiamond::update_FDTD_gpu_simulation_1_D(size_t num_timesteps) { // CPU single thread 1-D simulation of GPU workflow 

  size_t max_phases = 8;
  std::vector<int> mountain_heads_X;
  std::vector<int> mountain_tails_X;
  std::vector<int> valley_heads_X;
  std::vector<int> valley_tails_X;
  _setup_diamond_tiling_gpu(BLX_GPU, BLY_GPU, BLZ_GPU, BLT_GPU, max_phases);

  for(auto range : _Eranges_phases_X[0][0]) { 
    mountain_heads_X.push_back(range.first);
    mountain_tails_X.push_back(range.second);
  }
  for(auto range : _Hranges_phases_X[1][BLT_GPU-1]) { 
    valley_heads_X.push_back(range.first);
    valley_tails_X.push_back(range.second);
  }
  
  size_t num_mountains_X = mountain_heads_X.size();
  size_t num_valleys_X = valley_heads_X.size();

  // write 1 dimension just to check
  std::vector<float> E_simu(_Nx, 1);
  std::vector<float> H_simu(_Nx, 1);
  std::vector<float> E_seq(_Nx, 1);
  std::vector<float> H_seq(_Nx, 1);
  size_t total_timesteps = 4;

  // seq version
  for(size_t t=0; t<total_timesteps; t++) {

    // update E
    for(size_t x=1; x<_Nx-1; x++) {
      E_seq[x] = H_seq[x-1] + H_seq[x] * 2; 
    }

    std::cout << "t = " << t << ", E_seq =";
    for(size_t x=0; x<_Nx; x++) {
      std::cout << E_seq[x] << " ";
    }
    std::cout << "\n";

    // update H 
    for(size_t x=1; x<_Nx-1; x++) {
      H_seq[x] = E_seq[x+1] + E_seq[x] * 2; 
    }
  }

  // tiling version
  int mountain_or_valley;
  // 1, mountain, 0, valley
  int Nx = _Nx;
  for(size_t tt=0; tt<total_timesteps/BLT_GPU; tt++) {

    // phase 1. moutains 
    mountain_or_valley = 1;
    for(size_t xx=0; xx<num_mountains_X; xx++) {

      for(size_t t=0; t<BLT_GPU; t++) {

        int calculate_E = 1; // calculate this E tile or not
        int calculate_H = 1; // calculate this H tile or not

        std::vector<int> indices = _get_head_tail(BLX_GPU, BLT_GPU,
                                                  mountain_heads_X, mountain_tails_X,
                                                  xx, t,
                                                  mountain_or_valley,
                                                  Nx,
                                                  &calculate_E, &calculate_H);

        // update E
        for(int x=indices[0]; x<=indices[1]; x++) {
          if(x>=1 && x<=Nx-2 && calculate_E) {
            E_simu[x] = H_simu[x-1] + H_simu[x] * 2; 
          }
        }

        // update H
        for(int x=indices[2]; x<=indices[3]; x++) {
          if(x>=1 && x<=Nx-2 && calculate_H) {
            H_simu[x] = E_simu[x+1] + E_simu[x] * 2; 
          }
        }
      }
    }

    std::cout << "E_simu = ";
    for(auto e : E_simu) {
      std::cout << e << " ";
    }
    std::cout << "\n";

    // phase 2. valleys
    mountain_or_valley = 0;
    for(size_t xx=0; xx<num_valleys_X; xx++) {

      for(size_t t=0; t<BLT_GPU; t++) {

        int calculate_E = 1; // calculate this E tile or not
        int calculate_H = 1; // calculate this H tile or not
        
        std::vector<int> indices = _get_head_tail(BLX_GPU, BLT_GPU,
                                                  valley_heads_X, valley_tails_X,
                                                  xx, t,
                                                  mountain_or_valley,
                                                  Nx,
                                                  &calculate_E, &calculate_H);

        // update E
        for(int x=indices[0]; x<=indices[1]; x++) {
          if(x>=1 && x<=Nx-2 && calculate_E) {
            E_simu[x] = H_simu[x-1] + H_simu[x] * 2; 
          }
        }

        // update H
        for(int x=indices[2]; x<=indices[3]; x++) {
          if(x>=1 && x<=Nx-2 && calculate_H) {
            H_simu[x] = E_simu[x+1] + E_simu[x] * 2; 
          }
        }
      }

    } 
  }

  std::cout << "E_seq = ";
  for(size_t x=0; x<_Nx; x++) {
    std::cout << E_seq[x] << " ";
  }
  std::cout << "\n";

  std::cout << "E_simu = ";
  for(size_t x=0; x<_Nx; x++) {
    std::cout << E_simu[x] << " ";
  }
  std::cout << "\n";

  for(size_t x=0; x<_Nx; x++) {
    if(E_seq[x] != E_simu[x] || H_seq[x] != H_simu[x]) {
      std::cerr << "1-D demo results mismatch.\n";
      std::exit(EXIT_FAILURE);
    }
  }
}

void gDiamond::update_FDTD_gpu_simulation(size_t num_timesteps) { // simulation of gpu threads

  std::cout << "running update_FDTD_gpu_simulation\n";

  // clear source Mz for experiments
  _Mz.clear();

  // transfer source
  for(size_t t=0; t<num_timesteps; t++) {
    float Mz_value = M_source_amp * std::sin(SOURCE_OMEGA * t * dt);
    _Mz[_source_idx] = Mz_value;
  }

  size_t max_phases = 8;
  std::vector<int> mountain_heads_X;
  std::vector<int> mountain_tails_X;
  std::vector<int> mountain_heads_Y;
  std::vector<int> mountain_tails_Y;
  std::vector<int> mountain_heads_Z;
  std::vector<int> mountain_tails_Z;
  std::vector<int> valley_heads_X;
  std::vector<int> valley_tails_X;
  std::vector<int> valley_heads_Y;
  std::vector<int> valley_tails_Y;
  std::vector<int> valley_heads_Z;
  std::vector<int> valley_tails_Z;
  _setup_diamond_tiling_gpu(BLX_GPU, BLY_GPU, BLZ_GPU, BLT_GPU, max_phases);

  for(auto range : _Eranges_phases_X[0][0]) { 
    mountain_heads_X.push_back(range.first);
    mountain_tails_X.push_back(range.second);
  }
  for(auto range : _Eranges_phases_Y[0][0]) { 
    mountain_heads_Y.push_back(range.first);
    mountain_tails_Y.push_back(range.second);
  }
  for(auto range : _Eranges_phases_Z[0][0]) { 
    mountain_heads_Z.push_back(range.first);
    mountain_tails_Z.push_back(range.second);
  }
  for(auto range : _Hranges_phases_X[1][BLT_GPU-1]) { 
    valley_heads_X.push_back(range.first);
    valley_tails_X.push_back(range.second);
  }
  for(auto range : _Hranges_phases_Y[2][BLT_GPU-1]) { 
    valley_heads_Y.push_back(range.first);
    valley_tails_Y.push_back(range.second);
  }
  for(auto range : _Hranges_phases_Z[3][BLT_GPU-1]) { 
    valley_heads_Z.push_back(range.first);
    valley_tails_Z.push_back(range.second);
  }

  size_t num_mountains_X = mountain_heads_X.size();
  size_t num_mountains_Y = mountain_heads_Y.size();
  size_t num_mountains_Z = mountain_heads_Z.size();
  size_t num_valleys_X = valley_heads_X.size();
  size_t num_valleys_Y = valley_heads_Y.size();
  size_t num_valleys_Z = valley_heads_Z.size();

  size_t block_size = BLX_GPU * BLY_GPU * BLZ_GPU;
  size_t grid_size;

  for(size_t t=0; t<num_timesteps/BLT_GPU; t++) {
    
    // phase 1: (m, m, m)
    grid_size = num_mountains_X * num_mountains_Y * num_mountains_Z;
    _updateEH_phase_seq(_Ex_simu, _Ey_simu, _Ez_simu,
                        _Hx_simu, _Hy_simu, _Hz_simu,
                        _Cax, _Cbx,
                        _Cay, _Cby,
                        _Caz, _Cbz,
                        _Dax, _Dbx,
                        _Day, _Dby,
                        _Daz, _Dbz,
                        _Jx, _Jy, _Jz,
                        _Mx, _My, _Mz,
                        _dx, 
                        _Nx, _Ny, _Nz,
                        num_mountains_X, num_mountains_Y, num_mountains_Z, 
                        mountain_heads_X, mountain_heads_Y, mountain_heads_Z, 
                        mountain_tails_X, mountain_tails_Y, mountain_tails_Z, 
                        1, 1, 1,
                        block_size,
                        grid_size);

    // phase 2: (v, m, m)
    grid_size = num_valleys_X * num_mountains_Y * num_mountains_Z;
    _updateEH_phase_seq(_Ex_simu, _Ey_simu, _Ez_simu,
                        _Hx_simu, _Hy_simu, _Hz_simu,
                        _Cax, _Cbx,
                        _Cay, _Cby,
                        _Caz, _Cbz,
                        _Dax, _Dbx,
                        _Day, _Dby,
                        _Daz, _Dbz,
                        _Jx, _Jy, _Jz,
                        _Mx, _My, _Mz,
                        _dx, 
                        _Nx, _Ny, _Nz,
                        num_valleys_X, num_mountains_Y, num_mountains_Z, 
                        valley_heads_X, mountain_heads_Y, mountain_heads_Z, 
                        valley_tails_X, mountain_tails_Y, mountain_tails_Z, 
                        0, 1, 1,
                        block_size,
                        grid_size);

    // phase 3: (m, v, m)
    grid_size = num_mountains_X * num_valleys_Y * num_mountains_Z;
    _updateEH_phase_seq(_Ex_simu, _Ey_simu, _Ez_simu,
                        _Hx_simu, _Hy_simu, _Hz_simu,
                        _Cax, _Cbx,
                        _Cay, _Cby,
                        _Caz, _Cbz,
                        _Dax, _Dbx,
                        _Day, _Dby,
                        _Daz, _Dbz,
                        _Jx, _Jy, _Jz,
                        _Mx, _My, _Mz,
                        _dx, 
                        _Nx, _Ny, _Nz,
                        num_mountains_X, num_valleys_Y, num_mountains_Z, 
                        mountain_heads_X, valley_heads_Y, mountain_heads_Z, 
                        mountain_tails_X, valley_tails_Y, mountain_tails_Z, 
                        1, 0, 1,
                        block_size,
                        grid_size);

    // phase 4: (m, m, v)
    grid_size = num_mountains_X * num_mountains_Y * num_valleys_Z;
    _updateEH_phase_seq(_Ex_simu, _Ey_simu, _Ez_simu,
                        _Hx_simu, _Hy_simu, _Hz_simu,
                        _Cax, _Cbx,
                        _Cay, _Cby,
                        _Caz, _Cbz,
                        _Dax, _Dbx,
                        _Day, _Dby,
                        _Daz, _Dbz,
                        _Jx, _Jy, _Jz,
                        _Mx, _My, _Mz,
                        _dx, 
                        _Nx, _Ny, _Nz,
                        num_mountains_X, num_mountains_Y, num_valleys_Z, 
                        mountain_heads_X, mountain_heads_Y, valley_heads_Z, 
                        mountain_tails_X, mountain_tails_Y, valley_tails_Z, 
                        1, 1, 0,
                        block_size,
                        grid_size);

    // phase 5: (v, v, m)
    grid_size = num_valleys_X * num_valleys_Y * num_mountains_Z;
    _updateEH_phase_seq(_Ex_simu, _Ey_simu, _Ez_simu,
                        _Hx_simu, _Hy_simu, _Hz_simu,
                        _Cax, _Cbx,
                        _Cay, _Cby,
                        _Caz, _Cbz,
                        _Dax, _Dbx,
                        _Day, _Dby,
                        _Daz, _Dbz,
                        _Jx, _Jy, _Jz,
                        _Mx, _My, _Mz,
                        _dx, 
                        _Nx, _Ny, _Nz,
                        num_valleys_X, num_valleys_Y, num_mountains_Z, 
                        valley_heads_X, valley_heads_Y, mountain_heads_Z, 
                        valley_tails_X, valley_tails_Y, mountain_tails_Z, 
                        0, 0, 1,
                        block_size,
                        grid_size);

    // phase 6: (v, m, v)
    grid_size = num_valleys_X * num_mountains_Y * num_valleys_Z;
    _updateEH_phase_seq(_Ex_simu, _Ey_simu, _Ez_simu,
                        _Hx_simu, _Hy_simu, _Hz_simu,
                        _Cax, _Cbx,
                        _Cay, _Cby,
                        _Caz, _Cbz,
                        _Dax, _Dbx,
                        _Day, _Dby,
                        _Daz, _Dbz,
                        _Jx, _Jy, _Jz,
                        _Mx, _My, _Mz,
                        _dx, 
                        _Nx, _Ny, _Nz,
                        num_valleys_X, num_mountains_Y, num_valleys_Z, 
                        valley_heads_X, mountain_heads_Y, valley_heads_Z, 
                        valley_tails_X, mountain_tails_Y, valley_tails_Z, 
                        0, 1, 0,
                        block_size,
                        grid_size);

    // phase 7: (m, v, v)
    grid_size = num_mountains_X * num_valleys_Y * num_valleys_Z;
    _updateEH_phase_seq(_Ex_simu, _Ey_simu, _Ez_simu,
                        _Hx_simu, _Hy_simu, _Hz_simu,
                        _Cax, _Cbx,
                        _Cay, _Cby,
                        _Caz, _Cbz,
                        _Dax, _Dbx,
                        _Day, _Dby,
                        _Daz, _Dbz,
                        _Jx, _Jy, _Jz,
                        _Mx, _My, _Mz,
                        _dx, 
                        _Nx, _Ny, _Nz,
                        num_mountains_X, num_valleys_Y, num_valleys_Z, 
                        mountain_heads_X, valley_heads_Y, valley_heads_Z, 
                        mountain_tails_X, valley_tails_Y, valley_tails_Z, 
                        1, 0, 0,
                        block_size,
                        grid_size);

    // phase 8: (v, v, v)
    grid_size = num_valleys_X * num_valleys_Y * num_valleys_Z;
    _updateEH_phase_seq(_Ex_simu, _Ey_simu, _Ez_simu,
                        _Hx_simu, _Hy_simu, _Hz_simu,
                        _Cax, _Cbx,
                        _Cay, _Cby,
                        _Caz, _Cbz,
                        _Dax, _Dbx,
                        _Day, _Dby,
                        _Daz, _Dbz,
                        _Jx, _Jy, _Jz,
                        _Mx, _My, _Mz,
                        _dx, 
                        _Nx, _Ny, _Nz,
                        num_valleys_X, num_valleys_Y, num_valleys_Z, 
                        valley_heads_X, valley_heads_Y, valley_heads_Z, 
                        valley_tails_X, valley_tails_Y, valley_tails_Z, 
                        0, 0, 0,
                        block_size,
                        grid_size);
  }

} 



void gDiamond::update_FDTD_gpu_simulation_shmem_EH(size_t num_timesteps) { // simulation of gpu threads

  std::cout << "running update_FDTD_gpu_simulation_shmem_EH\n"; 

  // clear source Mz for experiments
  _Mz.clear();

  // transfer source
  for(size_t t=0; t<num_timesteps; t++) {
    float Mz_value = M_source_amp * std::sin(SOURCE_OMEGA * t * dt);
    _Mz[_source_idx] = Mz_value;
  }

  size_t max_phases = 8;
  std::vector<int> mountain_heads_X;
  std::vector<int> mountain_tails_X;
  std::vector<int> mountain_heads_Y;
  std::vector<int> mountain_tails_Y;
  std::vector<int> mountain_heads_Z;
  std::vector<int> mountain_tails_Z;
  std::vector<int> valley_heads_X;
  std::vector<int> valley_tails_X;
  std::vector<int> valley_heads_Y;
  std::vector<int> valley_tails_Y;
  std::vector<int> valley_heads_Z;
  std::vector<int> valley_tails_Z;
  _setup_diamond_tiling_gpu(BLX_GPU, BLY_GPU, BLZ_GPU, BLT_GPU, max_phases);

  for(auto range : _Eranges_phases_X[0][0]) { 
    mountain_heads_X.push_back(range.first);
    mountain_tails_X.push_back(range.second);
  }
  for(auto range : _Eranges_phases_Y[0][0]) { 
    mountain_heads_Y.push_back(range.first);
    mountain_tails_Y.push_back(range.second);
  }
  for(auto range : _Eranges_phases_Z[0][0]) { 
    mountain_heads_Z.push_back(range.first);
    mountain_tails_Z.push_back(range.second);
  }
  for(auto range : _Hranges_phases_X[1][BLT_GPU-1]) { 
    valley_heads_X.push_back(range.first);
    valley_tails_X.push_back(range.second);
  }
  for(auto range : _Hranges_phases_Y[2][BLT_GPU-1]) { 
    valley_heads_Y.push_back(range.first);
    valley_tails_Y.push_back(range.second);
  }
  for(auto range : _Hranges_phases_Z[3][BLT_GPU-1]) { 
    valley_heads_Z.push_back(range.first);
    valley_tails_Z.push_back(range.second);
  }

  size_t num_mountains_X = mountain_heads_X.size();
  size_t num_mountains_Y = mountain_heads_Y.size();
  size_t num_mountains_Z = mountain_heads_Z.size();
  size_t num_valleys_X = valley_heads_X.size();
  size_t num_valleys_Y = valley_heads_Y.size();
  size_t num_valleys_Z = valley_heads_Z.size();

  size_t block_size = BLX_GPU * BLY_GPU * BLZ_GPU;
  size_t grid_size;

  size_t total_cal = 0;
  for(size_t t=0; t<num_timesteps/BLT_GPU; t++) {
    
    // phase 1: (m, m, m)
    grid_size = num_mountains_X * num_mountains_Y * num_mountains_Z;
    _updateEH_phase_seq_shmem_EH(_Ex_simu_sh, _Ey_simu_sh, _Ez_simu_sh,
                                 _Hx_simu_sh, _Hy_simu_sh, _Hz_simu_sh,
                                 _Cax, _Cbx,
                                 _Cay, _Cby,
                                 _Caz, _Cbz,
                                 _Dax, _Dbx,
                                 _Day, _Dby,
                                 _Daz, _Dbz,
                                 _Jx, _Jy, _Jz,
                                 _Mx, _My, _Mz,
                                 _dx, 
                                 _Nx, _Ny, _Nz,
                                 num_mountains_X, num_mountains_Y, num_mountains_Z, 
                                 mountain_heads_X, mountain_heads_Y, mountain_heads_Z, 
                                 mountain_tails_X, mountain_tails_Y, mountain_tails_Z, 
                                 1, 1, 1,
                                 total_cal,
                                 t,
                                 block_size,
                                 grid_size);

    // phase 2: (v, m, m)
    grid_size = num_valleys_X * num_mountains_Y * num_mountains_Z;
    _updateEH_phase_seq_shmem_EH(_Ex_simu_sh, _Ey_simu_sh, _Ez_simu_sh,
                                 _Hx_simu_sh, _Hy_simu_sh, _Hz_simu_sh,
                                 _Cax, _Cbx,
                                 _Cay, _Cby,
                                 _Caz, _Cbz,
                                 _Dax, _Dbx,
                                 _Day, _Dby,
                                 _Daz, _Dbz,
                                 _Jx, _Jy, _Jz,
                                 _Mx, _My, _Mz,
                                 _dx, 
                                 _Nx, _Ny, _Nz,
                                 num_valleys_X, num_mountains_Y, num_mountains_Z, 
                                 valley_heads_X, mountain_heads_Y, mountain_heads_Z, 
                                 valley_tails_X, mountain_tails_Y, mountain_tails_Z, 
                                 0, 1, 1,
                                 total_cal,
                                 t,
                                 block_size,
                                 grid_size);

    // phase 3: (m, v, m)
    grid_size = num_mountains_X * num_valleys_Y * num_mountains_Z;
    _updateEH_phase_seq_shmem_EH(_Ex_simu_sh, _Ey_simu_sh, _Ez_simu_sh,
                                 _Hx_simu_sh, _Hy_simu_sh, _Hz_simu_sh,
                                 _Cax, _Cbx,
                                 _Cay, _Cby,
                                 _Caz, _Cbz,
                                 _Dax, _Dbx,
                                 _Day, _Dby,
                                 _Daz, _Dbz,
                                 _Jx, _Jy, _Jz,
                                 _Mx, _My, _Mz,
                                 _dx, 
                                 _Nx, _Ny, _Nz,
                                 num_mountains_X, num_valleys_Y, num_mountains_Z, 
                                 mountain_heads_X, valley_heads_Y, mountain_heads_Z, 
                                 mountain_tails_X, valley_tails_Y, mountain_tails_Z, 
                                 1, 0, 1,
                                 total_cal,
                                 t,
                                 block_size,
                                 grid_size);

    // phase 4: (m, m, v)
    grid_size = num_mountains_X * num_mountains_Y * num_valleys_Z;
    _updateEH_phase_seq_shmem_EH(_Ex_simu_sh, _Ey_simu_sh, _Ez_simu_sh,
                                 _Hx_simu_sh, _Hy_simu_sh, _Hz_simu_sh,
                                 _Cax, _Cbx,
                                 _Cay, _Cby,
                                 _Caz, _Cbz,
                                 _Dax, _Dbx,
                                 _Day, _Dby,
                                 _Daz, _Dbz,
                                 _Jx, _Jy, _Jz,
                                 _Mx, _My, _Mz,
                                 _dx, 
                                 _Nx, _Ny, _Nz,
                                 num_mountains_X, num_mountains_Y, num_valleys_Z, 
                                 mountain_heads_X, mountain_heads_Y, valley_heads_Z, 
                                 mountain_tails_X, mountain_tails_Y, valley_tails_Z, 
                                 1, 1, 0,
                                 total_cal,
                                 t,
                                 block_size,
                                 grid_size);

    // phase 5: (v, v, m)
    grid_size = num_valleys_X * num_valleys_Y * num_mountains_Z;
    _updateEH_phase_seq_shmem_EH(_Ex_simu_sh, _Ey_simu_sh, _Ez_simu_sh,
                                 _Hx_simu_sh, _Hy_simu_sh, _Hz_simu_sh,
                                 _Cax, _Cbx,
                                 _Cay, _Cby,
                                 _Caz, _Cbz,
                                 _Dax, _Dbx,
                                 _Day, _Dby,
                                 _Daz, _Dbz,
                                 _Jx, _Jy, _Jz,
                                 _Mx, _My, _Mz,
                                 _dx, 
                                 _Nx, _Ny, _Nz,
                                 num_valleys_X, num_valleys_Y, num_mountains_Z, 
                                 valley_heads_X, valley_heads_Y, mountain_heads_Z, 
                                 valley_tails_X, valley_tails_Y, mountain_tails_Z, 
                                 0, 0, 1,
                                 total_cal,
                                 t,
                                 block_size,
                                 grid_size);

    // phase 6: (v, m, v)
    grid_size = num_valleys_X * num_mountains_Y * num_valleys_Z;
    _updateEH_phase_seq_shmem_EH(_Ex_simu_sh, _Ey_simu_sh, _Ez_simu_sh,
                                 _Hx_simu_sh, _Hy_simu_sh, _Hz_simu_sh,
                                 _Cax, _Cbx,
                                 _Cay, _Cby,
                                 _Caz, _Cbz,
                                 _Dax, _Dbx,
                                 _Day, _Dby,
                                 _Daz, _Dbz,
                                 _Jx, _Jy, _Jz,
                                 _Mx, _My, _Mz,
                                 _dx, 
                                 _Nx, _Ny, _Nz,
                                 num_valleys_X, num_mountains_Y, num_valleys_Z, 
                                 valley_heads_X, mountain_heads_Y, valley_heads_Z, 
                                 valley_tails_X, mountain_tails_Y, valley_tails_Z, 
                                 0, 1, 0,
                                 total_cal,
                                 t,
                                 block_size,
                                 grid_size);

    // phase 7: (m, v, v)
    grid_size = num_mountains_X * num_valleys_Y * num_valleys_Z;
    _updateEH_phase_seq_shmem_EH(_Ex_simu_sh, _Ey_simu_sh, _Ez_simu_sh,
                                 _Hx_simu_sh, _Hy_simu_sh, _Hz_simu_sh,
                                 _Cax, _Cbx,
                                 _Cay, _Cby,
                                 _Caz, _Cbz,
                                 _Dax, _Dbx,
                                 _Day, _Dby,
                                 _Daz, _Dbz,
                                 _Jx, _Jy, _Jz,
                                 _Mx, _My, _Mz,
                                 _dx, 
                                 _Nx, _Ny, _Nz,
                                 num_mountains_X, num_valleys_Y, num_valleys_Z, 
                                 mountain_heads_X, valley_heads_Y, valley_heads_Z, 
                                 mountain_tails_X, valley_tails_Y, valley_tails_Z, 
                                 1, 0, 0,
                                 total_cal,
                                 t,
                                 block_size,
                                 grid_size);

    // phase 8: (v, v, v)
    grid_size = num_valleys_X * num_valleys_Y * num_valleys_Z;
    _updateEH_phase_seq_shmem_EH(_Ex_simu_sh, _Ey_simu_sh, _Ez_simu_sh,
                                 _Hx_simu_sh, _Hy_simu_sh, _Hz_simu_sh,
                                 _Cax, _Cbx,
                                 _Cay, _Cby,
                                 _Caz, _Cbz,
                                 _Dax, _Dbx,
                                 _Day, _Dby,
                                 _Daz, _Dbz,
                                 _Jx, _Jy, _Jz,
                                 _Mx, _My, _Mz,
                                 _dx, 
                                 _Nx, _Ny, _Nz,
                                 num_valleys_X, num_valleys_Y, num_valleys_Z, 
                                 valley_heads_X, valley_heads_Y, valley_heads_Z, 
                                 valley_tails_X, valley_tails_Y, valley_tails_Z, 
                                 0, 0, 0,
                                 total_cal,
                                 t,
                                 block_size,
                                 grid_size);
  }

  std::cout << "gpu simulation (shmem EH) total calculations: " << total_cal << "\n"; 

} 


std::vector<int> gDiamond::_get_head_tail(size_t BLX, size_t BLT,
                                          std::vector<int> xx_heads, std::vector<int> xx_tails,
                                          size_t xx, size_t t,
                                          int mountain_or_valley, // 1 = mountain, 0 = valley
                                          int Nx,
                                          int *calculate_E, int *calculate_H) 
{

  std::vector<int> results(4);

  results[0] = xx_heads[xx]; 
  results[1] = xx_tails[xx];
  results[2] = xx_heads[xx]; 
  results[3] = xx_tails[xx];

  if(mountain_or_valley == 0 && xx == 0 && t == 0) { // skip the first valley in t = 0, since it does not exist 
    *calculate_E = 0;
    *calculate_H = 0;
  }

  // adjust results[0] according to t
  if(mountain_or_valley != 0 || xx != 0) { // the 1st valley, head should not change
    results[0] = (mountain_or_valley == 1)? results[0] + t : results[0] + (BLT - t - 1) + 1;
  }
  if(results[0] > Nx - 1) { // it means this mountain/valley does not exist in t
    *calculate_E = 0;
  }

  // adjust results[1] according to t
  if(mountain_or_valley == 1) { // handle the mountains 
    int temp = results[0] + (BLX - 2 * t) - 1;
    results[1] = (temp > Nx - 1)? Nx - 1 : temp;
  }
  else if(mountain_or_valley == 0 && xx == 0) { // the 1st valley, tail should be handled differently 
    results[1] = results[0] + t - 1;
  }
  else { // handle the valleys
    int temp = results[0] + (BLX - 2 * (BLT - t - 1)) - 2;
    results[1] = (temp > Nx - 1)? Nx - 1 : temp;
  }

  if(mountain_or_valley != 0 || xx != 0) { // the 1st valley, head should not change
    results[2] = (mountain_or_valley == 1)? results[2] + t : results[2] + (BLT - t - 1);
  }
  
  if(results[2] > Nx - 1) { // it means this mountain does not exist in t
    *calculate_H = 0;
  }
  if(mountain_or_valley == 1) { // handle the mountains 
    int temp = results[2] + (BLX - 2 * t) - 2;
    results[3] = (temp > Nx - 1)? Nx - 1 : temp;
  }
  else if(mountain_or_valley == 0 && xx == 0) { // the 1st valley, tail should be handled differently 
    results[3] = results[2] + t - 1;
  }
  else { // handle the valleys
    int temp = results[2] + (BLX - 2 * (BLT - t - 1)) - 1;
    results[3] = (temp > Nx - 1)? Nx - 1 : temp;
  }

  return results;
}

void gDiamond::_updateEH_phase_seq(std::vector<float>& Ex, std::vector<float>& Ey, std::vector<float>& Ez,
                         std::vector<float>& Hx, std::vector<float>& Hy, std::vector<float>& Hz,
                         std::vector<float>& Cax, std::vector<float>& Cbx,
                         std::vector<float>& Cay, std::vector<float>& Cby,
                         std::vector<float>& Caz, std::vector<float>& Cbz,
                         std::vector<float>& Dax, std::vector<float>& Dbx,
                         std::vector<float>& Day, std::vector<float>& Dby,
                         std::vector<float>& Daz, std::vector<float>& Dbz,
                         std::vector<float>& Jx, std::vector<float>& Jy, std::vector<float>& Jz,
                         std::vector<float>& Mx, std::vector<float>& My, std::vector<float>& Mz,
                         float dx, 
                         int Nx, int Ny, int Nz,
                         int xx_num, int yy_num, int zz_num, // number of tiles in each dimensions
                         std::vector<int> xx_heads, 
                         std::vector<int> yy_heads, 
                         std::vector<int> zz_heads,
                         std::vector<int> xx_tails, 
                         std::vector<int> yy_tails, 
                         std::vector<int> zz_tails,
                         int m_or_v_X, int m_or_v_Y, int m_or_v_Z,
                         size_t block_size,
                         size_t grid_size) 
{
  for(size_t block_id=0; block_id<grid_size; block_id++) {
    int xx = block_id % xx_num;
    int yy = (block_id % (xx_num * yy_num)) / xx_num;
    int zz = block_id / (xx_num * yy_num);
  
    for(size_t t=0; t<BLT_GPU; t++) {
      
      int calculate_Ex = 1; // calculate this E tile or not
      int calculate_Hx = 1; // calculate this H tile or not
      int calculate_Ey = 1; 
      int calculate_Hy = 1; 
      int calculate_Ez = 1; 
      int calculate_Hz = 1; 
  
      // {Ehead, Etail, Hhead, Htail}
      std::vector<int> indices_X = _get_head_tail(BLX_GPU, BLT_GPU,
                                                  xx_heads, xx_tails,
                                                  xx, t,
                                                  m_or_v_X,
                                                  Nx,
                                                  &calculate_Ex, &calculate_Hx);

      /*
      if(yy == 0 && zz == 0) {
        std::cout << "X dimension, xx = " << xx << ", t = " << t << "\n";
        std::cout << "calculate_Ex = " << calculate_Ex << ", calculate_Hx = " << calculate_Hx << ", ";
        std::cout << "Ehead = " << indices_X[0] << ", " 
                  << "Etail = " << indices_X[1] << ", "
                  << "Hhead = " << indices_X[2] << ", "
                  << "Htail = " << indices_X[3] << "\n";
                                      
      }
      */

      std::vector<int> indices_Y = _get_head_tail(BLY_GPU, BLT_GPU,
                                                  yy_heads, yy_tails,
                                                  yy, t,
                                                  m_or_v_Y,
                                                  Ny,
                                                  &calculate_Ey, &calculate_Hy);

      std::vector<int> indices_Z = _get_head_tail(BLZ_GPU, BLT_GPU,
                                                  zz_heads, zz_tails,
                                                  zz, t,
                                                  m_or_v_Z,
                                                  Nz,
                                                  &calculate_Ez, &calculate_Hz);

      /*

      // update E
      if(calculate_Ex & calculate_Ey & calculate_Ez) {
        for(int x=indices_X[0]; x<=indices_X[1]; x++) {
          for(int y=indices_Y[0]; y<=indices_Y[1]; y++) {
            for(int z=indices_Z[0]; z<=indices_Z[1]; z++) {
              if(x >= 1 && x <= Nx - 2 && y >= 1 && y <= Ny - 2 && z >= 1 && z <= Nz - 2) {
                int g_idx = x + y * Nx + z * Nx * Ny; // global idx

                Ex[g_idx] = Cax[g_idx] * Ex[g_idx] + Cbx[g_idx] *
                          ((Hz[g_idx] - Hz[g_idx - Nx]) - (Hy[g_idx] - Hy[g_idx - Nx * Ny]) - Jx[g_idx] * dx);
                Ey[g_idx] = Cay[g_idx] * Ey[g_idx] + Cby[g_idx] *
                          ((Hx[g_idx] - Hx[g_idx - Nx * Ny]) - (Hz[g_idx] - Hz[g_idx - 1]) - Jy[g_idx] * dx);
                Ez[g_idx] = Caz[g_idx] * Ez[g_idx] + Cbz[g_idx] *
                          ((Hy[g_idx] - Hy[g_idx - 1]) - (Hx[g_idx] - Hx[g_idx - Nx]) - Jz[g_idx] * dx);
              }
            }
          }
        }
      }

      // update H
      if(calculate_Hx & calculate_Hy & calculate_Hz) {
        for(int x=indices_X[2]; x<=indices_X[3]; x++) {
          for(int y=indices_Y[2]; y<=indices_Y[3]; y++) {
            for(int z=indices_Z[2]; z<=indices_Z[3]; z++) {
              if(x >= 1 && x <= Nx - 2 && y >= 1 && y <= Ny - 2 && z >= 1 && z <= Nz - 2) {
                int g_idx = x + y * Nx + z * Nx * Ny; // global idx

                Hx[g_idx] = Dax[g_idx] * Hx[g_idx] + Dbx[g_idx] *
                          ((Ey[g_idx + Nx * Ny] - Ey[g_idx]) - (Ez[g_idx + Nx] - Ez[g_idx]) - Mx[g_idx] * dx);
                Hy[g_idx] = Day[g_idx] * Hy[g_idx] + Dby[g_idx] *
                          ((Ez[g_idx + 1] - Ez[g_idx]) - (Ex[g_idx + Nx * Ny] - Ex[g_idx]) - My[g_idx] * dx);
                Hz[g_idx] = Daz[g_idx] * Hz[g_idx] + Dbz[g_idx] *
                          ((Ex[g_idx + Nx] - Ex[g_idx]) - (Ey[g_idx + 1] - Ey[g_idx]) - Mz[g_idx] * dx);
              }
            }
          }
        }
      }
      */

      // update E
      if(calculate_Ex & calculate_Ey & calculate_Ez) {
        for(size_t thread_id=0; thread_id<block_size; thread_id++) {
          int local_x = thread_id % BLX_GPU;                     // X coordinate within the tile
          int local_y = (thread_id / BLX_GPU) % BLY_GPU;     // Y coordinate within the tile
          int local_z = thread_id / (BLX_GPU * BLY_GPU);     // Z coordinate within the tile

          // Ehead is offset
          int global_x = indices_X[0] + local_x; // Global X coordinate
          int global_y = indices_Y[0] + local_y; // Global Y coordinate
          int global_z = indices_Z[0] + local_z; // Global Z coordinate

          if(global_x >= 1 && global_x <= Nx-2 && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2 &&
            global_x <= indices_X[1] &&
            global_y <= indices_Y[1] &&
            global_z <= indices_Z[1]) {
            int g_idx = global_x + global_y * Nx + global_z * Nx * Ny; // global idx

            Ex[g_idx] = Cax[g_idx] * Ex[g_idx] + Cbx[g_idx] *
                      ((Hz[g_idx] - Hz[g_idx - Nx]) - (Hy[g_idx] - Hy[g_idx - Nx * Ny]) - Jx[g_idx] * dx);
            Ey[g_idx] = Cay[g_idx] * Ey[g_idx] + Cby[g_idx] *
                      ((Hx[g_idx] - Hx[g_idx - Nx * Ny]) - (Hz[g_idx] - Hz[g_idx - 1]) - Jy[g_idx] * dx);
            Ez[g_idx] = Caz[g_idx] * Ez[g_idx] + Cbz[g_idx] *
                      ((Hy[g_idx] - Hy[g_idx - 1]) - (Hx[g_idx] - Hx[g_idx - Nx]) - Jz[g_idx] * dx);
          }
        }
      }

      // update H 
      if(calculate_Hx & calculate_Hy & calculate_Hz) {
        for(size_t thread_id=0; thread_id<block_size; thread_id++) {
          int local_x = thread_id % BLX_GPU;                     // X coordinate within the tile
          int local_y = (thread_id / BLX_GPU) % BLY_GPU;     // Y coordinate within the tile
          int local_z = thread_id / (BLX_GPU * BLY_GPU);     // Z coordinate within the tile

          // Hhead is offset
          int global_x = indices_X[2] + local_x; // Global X coordinate
          int global_y = indices_Y[2] + local_y; // Global Y coordinate
          int global_z = indices_Z[2] + local_z; // Global Z coordinate

          if(global_x >= 1 && global_x <= Nx-2 && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2 &&
            global_x <= indices_X[3] &&
            global_y <= indices_Y[3] &&
            global_z <= indices_Z[3]) {
            int g_idx = global_x + global_y * Nx + global_z * Nx * Ny; // global idx

            Hx[g_idx] = Dax[g_idx] * Hx[g_idx] + Dbx[g_idx] *
                      ((Ey[g_idx + Nx * Ny] - Ey[g_idx]) - (Ez[g_idx + Nx] - Ez[g_idx]) - Mx[g_idx] * dx);
            Hy[g_idx] = Day[g_idx] * Hy[g_idx] + Dby[g_idx] *
                      ((Ez[g_idx + 1] - Ez[g_idx]) - (Ex[g_idx + Nx * Ny] - Ex[g_idx]) - My[g_idx] * dx);
            Hz[g_idx] = Daz[g_idx] * Hz[g_idx] + Dbz[g_idx] *
                      ((Ex[g_idx + Nx] - Ex[g_idx]) - (Ey[g_idx + 1] - Ey[g_idx]) - Mz[g_idx] * dx);
          }
        }
      }

    }
  } 

}



void gDiamond::_updateEH_phase_seq_shmem_EH(std::vector<float>& Ex, std::vector<float>& Ey, std::vector<float>& Ez,
                                            std::vector<float>& Hx, std::vector<float>& Hy, std::vector<float>& Hz,
                                            std::vector<float>& Cax, std::vector<float>& Cbx,
                                            std::vector<float>& Cay, std::vector<float>& Cby,
                                            std::vector<float>& Caz, std::vector<float>& Cbz,
                                            std::vector<float>& Dax, std::vector<float>& Dbx,
                                            std::vector<float>& Day, std::vector<float>& Dby,
                                            std::vector<float>& Daz, std::vector<float>& Dbz,
                                            std::vector<float>& Jx, std::vector<float>& Jy, std::vector<float>& Jz,
                                            std::vector<float>& Mx, std::vector<float>& My, std::vector<float>& Mz,
                                            float dx, 
                                            int Nx, int Ny, int Nz,
                                            int xx_num, int yy_num, int zz_num, // number of tiles in each dimensions
                                            std::vector<int> xx_heads, 
                                            std::vector<int> yy_heads, 
                                            std::vector<int> zz_heads,
                                            std::vector<int> xx_tails, 
                                            std::vector<int> yy_tails, 
                                            std::vector<int> zz_tails,
                                            int m_or_v_X, int m_or_v_Y, int m_or_v_Z,
                                            size_t &total_cal,
                                            size_t current_time,
                                            size_t block_size,
                                            size_t grid_size) 
{
  for(size_t block_id=0; block_id<grid_size; block_id++) {
    int xx = block_id % xx_num;
    int yy = (block_id % (xx_num * yy_num)) / xx_num;
    int zz = block_id / (xx_num * yy_num);

    // load data in shared memory
    float Ex_shmem[BLX_EH * BLY_EH * BLZ_EH];
    float Ey_shmem[BLX_EH * BLY_EH * BLZ_EH];
    float Ez_shmem[BLX_EH * BLY_EH * BLZ_EH];
    float Hx_shmem[BLX_EH * BLY_EH * BLZ_EH];
    float Hy_shmem[BLX_EH * BLY_EH * BLZ_EH];
    float Hz_shmem[BLX_EH * BLY_EH * BLZ_EH];

    for(size_t thread_id=0; thread_id<block_size; thread_id++) {
      int local_x = thread_id % BLX_GPU;                     // X coordinate within the tile
      int local_y = (thread_id / BLX_GPU) % BLY_GPU;     // Y coordinate within the tile
      int local_z = thread_id / (BLX_GPU * BLY_GPU);     // Z coordinate within the tile

      int global_x = xx_heads[xx] + local_x; // Global X coordinate
      int global_y = yy_heads[yy] + local_y; // Global Y coordinate
      int global_z = zz_heads[zz] + local_z; // Global Z coordinate
      int global_idx = global_x + global_y * Nx + global_z * Nx * Ny;

      // load H, stencil pattern x-1, y-1, z-1
      int shared_H_x = local_x + 1;
      int shared_H_y = local_y + 1;
      int shared_H_z = local_z + 1;
      int shared_H_idx = shared_H_x + shared_H_y * BLX_EH + shared_H_z * BLX_EH * BLY_EH;

      // if(global_x >= 0 && global_x < Nx && global_y >= 0 && global_y < Ny && global_z >= 0 && global_z < Nz &&
      //    global_x <= xx_tails[xx] &&
      //    global_y <= yy_tails[yy] &&
      //    global_z <= zz_tails[zz]) {
      if(global_x >= 0 && global_x < Nx && global_y >= 0 && global_y < Ny && global_z >= 0 && global_z < Nz) {

        // load core
        Hx_shmem[shared_H_idx] = Hx[global_idx];
        Hy_shmem[shared_H_idx] = Hy[global_idx];
        Hz_shmem[shared_H_idx] = Hz[global_idx];

        // load HALO region
        if(local_x == 0 && global_x > 0) {
          Hz_shmem[shared_H_x - 1 + shared_H_y * BLX_EH + shared_H_z * BLX_EH * BLY_EH] = Hz[global_x - 1 + global_y * Nx + global_z * Nx * Ny];
          Hy_shmem[shared_H_x - 1 + shared_H_y * BLX_EH + shared_H_z * BLX_EH * BLY_EH] = Hy[global_x - 1 + global_y * Nx + global_z * Nx * Ny];
          
        }
        if(local_y == 0 && global_y > 0) {
          Hx_shmem[shared_H_x + (shared_H_y - 1) * BLX_EH + shared_H_z * BLX_EH * BLY_EH] = Hx[global_x + (global_y - 1) * Nx + global_z * Nx * Ny];
          Hz_shmem[shared_H_x + (shared_H_y - 1) * BLX_EH + shared_H_z * BLX_EH * BLY_EH] = Hz[global_x + (global_y - 1) * Nx + global_z * Nx * Ny];
        }
        if(local_z == 0 && global_z > 0) {
          Hx_shmem[shared_H_x + shared_H_y * BLX_EH + (shared_H_z - 1) * BLX_EH * BLY_EH] = Hx[global_x + global_y * Nx + (global_z - 1) * Nx * Ny];
          Hy_shmem[shared_H_x + shared_H_y * BLX_EH + (shared_H_z - 1) * BLX_EH * BLY_EH] = Hy[global_x + global_y * Nx + (global_z - 1) * Nx * Ny];
        }
      }

      // load E, stencil pattern x+1, y+1, z+1
      // the padding does not affect origins of local idx and shared_E_idx
      // local idx and shared_E_idx still have the same origin
      int shared_E_x = local_x;
      int shared_E_y = local_y;
      int shared_E_z = local_z;

      int shared_E_idx = shared_E_x + shared_E_y * BLX_EH + shared_E_z * BLX_EH * BLY_EH;

      // if(global_x >= 0 && global_x < Nx && global_y >= 0 && global_y < Ny && global_z >= 0 && global_z < Nz &&
      //    global_x <= xx_tails[xx] &&
      //    global_y <= yy_tails[yy] &&
      //    global_z <= zz_tails[zz]) {
      if(global_x >= 0 && global_x < Nx && global_y >= 0 && global_y < Ny && global_z >= 0 && global_z < Nz) {

        Ex_shmem[shared_E_idx] = Ex[global_idx];
        Ey_shmem[shared_E_idx] = Ey[global_idx];
        Ez_shmem[shared_E_idx] = Ez[global_idx];

        // load HALO region
        if(local_x == BLX_GPU - 1 && global_x < Nx - 1) {
          Ez_shmem[shared_E_x + 1 + shared_E_y * BLX_EH + shared_E_z * BLX_EH * BLY_EH] = Ez[global_x + 1 + global_y * Nx + global_z * Nx * Ny];
          Ey_shmem[shared_E_x + 1 + shared_E_y * BLX_EH + shared_E_z * BLX_EH * BLY_EH] = Ey[global_x + 1 + global_y * Nx + global_z * Nx * Ny];
        }
        if(local_y == BLY_GPU - 1 && global_y < Ny - 1) {
          Ex_shmem[shared_E_x + (shared_E_y + 1) * BLX_EH + shared_E_z * BLX_EH * BLY_EH] = Ex[global_x + (global_y + 1) * Nx + global_z * Nx * Ny];
          Ez_shmem[shared_E_x + (shared_E_y + 1) * BLX_EH + shared_E_z * BLX_EH * BLY_EH] = Ez[global_x + (global_y + 1) * Nx + global_z * Nx * Ny];
        }
        if(local_z == BLZ_GPU - 1 && global_z < Nz - 1) {
          Ex_shmem[shared_E_x + shared_E_y * BLX_EH + (shared_E_z + 1) * BLX_EH * BLY_EH] = Ex[global_x + global_y * Nx + (global_z + 1) * Nx * Ny];
          Ey_shmem[shared_E_x + shared_E_y * BLX_EH + (shared_E_z + 1) * BLX_EH * BLY_EH] = Ey[global_x + global_y * Nx + (global_z + 1) * Nx * Ny];
        }

      }
    }


    for(size_t t=0; t<BLT_GPU; t++) {
      
      int calculate_Ex = 1; // calculate this E tile or not
      int calculate_Hx = 1; // calculate this H tile or not
      int calculate_Ey = 1; 
      int calculate_Hy = 1; 
      int calculate_Ez = 1; 
      int calculate_Hz = 1; 
  
      // {Ehead, Etail, Hhead, Htail}
      std::vector<int> indices_X = _get_head_tail(BLX_GPU, BLT_GPU,
                                                  xx_heads, xx_tails,
                                                  xx, t,
                                                  m_or_v_X,
                                                  Nx,
                                                  &calculate_Ex, &calculate_Hx);

      std::vector<int> indices_Y = _get_head_tail(BLY_GPU, BLT_GPU,
                                                  yy_heads, yy_tails,
                                                  yy, t,
                                                  m_or_v_Y,
                                                  Ny,
                                                  &calculate_Ey, &calculate_Hy);

      std::vector<int> indices_Z = _get_head_tail(BLZ_GPU, BLT_GPU,
                                                  zz_heads, zz_tails,
                                                  zz, t,
                                                  m_or_v_Z,
                                                  Nz,
                                                  &calculate_Ez, &calculate_Hz);

      // update E
      if(calculate_Ex & calculate_Ey & calculate_Ez) {
        for(size_t thread_id=0; thread_id<block_size; thread_id++) {
          int local_x = thread_id % BLX_GPU;                     // X coordinate within the tile
          int local_y = (thread_id / BLX_GPU) % BLY_GPU;     // Y coordinate within the tile
          int local_z = thread_id / (BLX_GPU * BLY_GPU);     // Z coordinate within the tile

          // Ehead is offset
          int global_x = indices_X[0] + local_x; // Global X coordinate
          int global_y = indices_Y[0] + local_y; // Global Y coordinate
          int global_z = indices_Z[0] + local_z; // Global Z coordinate

          if(global_x >= 1 && global_x <= Nx-2 && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2 &&
            global_x <= indices_X[1] &&
            global_y <= indices_Y[1] &&
            global_z <= indices_Z[1]) {

            // need to recalculate shared indices
            int Eoffset_X = indices_X[0] - xx_heads[xx];
            int Eoffset_Y = indices_Y[0] - yy_heads[yy];
            int Eoffset_Z = indices_Z[0] - zz_heads[zz];

            int Hoffset_X = indices_X[2] - xx_heads[xx];
            int Hoffset_Y = indices_Y[2] - yy_heads[yy];
            int Hoffset_Z = indices_Z[2] - zz_heads[zz];

            int shared_H_x = (m_or_v_X == 0 && xx != 0)? local_x + 1 + Hoffset_X + 1 : local_x + 1 + Hoffset_X;
            int shared_H_y = (m_or_v_Y == 0 && yy != 0)? local_y + 1 + Hoffset_Y + 1 : local_y + 1 + Hoffset_Y;
            int shared_H_z = (m_or_v_Z == 0 && zz != 0)? local_z + 1 + Hoffset_Z + 1 : local_z + 1 + Hoffset_Z;

            int shared_E_x = local_x + Eoffset_X;
            int shared_E_y = local_y + Eoffset_Y;
            int shared_E_z = local_z + Eoffset_Z;

            int s_H_idx = shared_H_x + shared_H_y * BLX_EH + shared_H_z * BLX_EH * BLY_EH; // shared memory idx for H
            int s_E_idx = shared_E_x + shared_E_y * BLX_EH + shared_E_z * BLX_EH * BLY_EH; // shared memory idx for E

            int g_idx = global_x + global_y * Nx + global_z * Nx * Ny; // global idx

            // if(m_or_v_X == 0 && xx == 1 && yy == 0 && zz == 0 && global_y == 3 && global_z == 3) {
            //   std::cout << "check update E\n";
            // }

            total_cal++;

            // check shared memory load before calculation
            // if(Hx_shmem[s_H_idx] != Hx[g_idx]) {
            //   std::cerr << "updateE, ";
            //   std::cerr << "Hx_shmem[s_H_idx] != Hx[g_idx]\n";
            //   std::exit(EXIT_FAILURE);
            // }
            // if(Hx_shmem[s_H_idx - BLX_EH * BLY_EH] != Hx[g_idx - Nx * Ny]) {
            //   std::cerr << "updateE, ";
            //   std::cerr << "Hx_shmem[s_H_idx] != Hx[g_idx - Nx * Ny]\n";
            //   std::exit(EXIT_FAILURE);
            // }
            // if(Hx_shmem[s_H_idx - BLX_EH] != Hx[g_idx - Nx]) {
            //   std::cerr << "updateE, ";
            //   std::cerr << "Hx_shmem[s_H_idx - BLX_EH] != Hx[g_idx - Nx]\n";
            //   std::exit(EXIT_FAILURE);
            // }
            // if(Hy_shmem[s_H_idx] != Hy[g_idx]) {
            //   std::cerr << "updateE, ";
            //   std::cerr << "Hy_shmem[s_H_idx] != Hy[g_idx]\n";
            //   std::exit(EXIT_FAILURE);
            // }
            // if(Hy_shmem[s_H_idx - 1] != Hy[g_idx - 1]) {
            //   std::cerr << "updateE, ";
            //   std::cerr << "Hy_shmem[s_H_idx - 1] != Hy[g_idx - 1]\n";
            //   std::exit(EXIT_FAILURE);
            // }
            // if(Hy_shmem[s_H_idx - BLX_EH * BLY_EH] != Hy[g_idx - Nx * Ny]) {
            //   std::cerr << "updateE, ";
            //   std::cerr << "Hy_shmem[s_H_idx - BLX_EH * BLY_EH] != Hy[g_idx - Nx * Ny]\n";
            //   std::exit(EXIT_FAILURE);
            // }
            // if(Hz_shmem[s_H_idx] != Hz[g_idx]) {
            //   std::cerr << "updateE, ";
            //   std::cerr << "Hz_shmem[s_H_idx] != Hz[g_idx]\n";
            //   std::exit(EXIT_FAILURE);
            // }
            // if(Hz_shmem[s_H_idx - 1] != Hz[g_idx - 1]) {
            //   std::cerr << "updateE, ";
            //   std::cerr << "Hz_shmem[s_H_idx - 1] != Hz[g_idx - 1]\n";
            //   std::exit(EXIT_FAILURE);
            // }
            // if(Hz_shmem[s_H_idx - BLX_EH] != Hz[g_idx - Nx]) {
            //   std::cerr << "updateE, ";
            //   std::cerr << "Hz_shmem[s_H_idx - BLX_EH] != Hz[g_idx - Nx]\n";
            //   std::exit(EXIT_FAILURE);
            // }

            // if(Hy[g_idx - 1] != Hy_shmem[s_H_idx - 1]) {
            //   std::cout << "Hy[g_idx - 1] = " << Hy[g_idx - 1] << ", Hy_shmem[s_H_idx - 1] = " << Hy_shmem[s_H_idx - 1] << "\n"; 
            //   std::cerr << "shared mem load wrong.\n";
            //   std::cerr << "t = " << t << "\n";
            //   std::cerr << "xx = " << xx << ", yy = " << yy << ", zz = " << zz << "\n";
            //   std::cerr << "(X, Y, Z) = (" << m_or_v_X << ", " << m_or_v_Y << ", " << m_or_v_Z << ")\n";  
            //   std::cout << "(local_x, local_y, local_z) = (" << local_x << ", " << local_y << ", " << local_y << ")\n";
            //   std::cout << "(global_x, global_y, global_z) = (" << global_x << ", " << global_y << ", " << global_y << ")\n";
            //   std::cout << "(shared_H_x, shared_H_y, shared_H_z) = (" << shared_H_x << ", " << shared_H_y << ", " << shared_H_y << ")\n";
            //   std::cout << "(shared_E_x, shared_E_y, shared_E_z) = (" << shared_E_x << ", " << shared_E_y << ", " << shared_E_y << ")\n";
            //   std::exit(EXIT_FAILURE);
            // }

            // Ex[g_idx] = Cax[g_idx] * Ex[g_idx] + Cbx[g_idx] *
            //           ((Hz[g_idx] - Hz[g_idx - Nx]) - (Hy[g_idx] - Hy[g_idx - Nx * Ny]) - Jx[g_idx] * dx);
            // Ey[g_idx] = Cay[g_idx] * Ey[g_idx] + Cby[g_idx] *
            //           ((Hx[g_idx] - Hx[g_idx - Nx * Ny]) - (Hz[g_idx] - Hz[g_idx - 1]) - Jy[g_idx] * dx);
            // Ez[g_idx] = Caz[g_idx] * Ez[g_idx] + Cbz[g_idx] *
            //           ((Hy[g_idx] - Hy[g_idx - 1]) - (Hx[g_idx] - Hx[g_idx - Nx]) - Jz[g_idx] * dx);
            Ex_shmem[s_E_idx] = Cax[g_idx] * Ex_shmem[s_E_idx] + Cbx[g_idx] *
                    ((Hz_shmem[s_H_idx] - Hz_shmem[s_H_idx - BLX_EH]) - (Hy_shmem[s_H_idx] - Hy_shmem[s_H_idx - BLX_EH * BLY_EH]) - Jx[g_idx] * dx);
            Ey_shmem[s_E_idx] = Cay[g_idx] * Ey_shmem[s_E_idx] + Cby[g_idx] *
                      ((Hx_shmem[s_H_idx] - Hx_shmem[s_H_idx - BLX_EH * BLY_EH]) - (Hz_shmem[s_H_idx] - Hz_shmem[s_H_idx - 1]) - Jy[g_idx] * dx);
            Ez_shmem[s_E_idx] = Caz[g_idx] * Ez_shmem[s_E_idx] + Cbz[g_idx] *
                      ((Hy_shmem[s_H_idx] - Hy_shmem[s_H_idx - 1]) - (Hx_shmem[s_H_idx] - Hx_shmem[s_H_idx - BLX_EH]) - Jz[g_idx] * dx);

            // if(Ex_shmem[s_E_idx] != Ex[g_idx]) {
            //   std::cerr << "after updateE, ";
            //   std::cerr << "Ex_shmem[s_E_idx] != Ex[g_idx]\n";
            //   std::exit(EXIT_FAILURE);
            // }
            // if(Ey_shmem[s_E_idx] != Ey[g_idx]) {
            //   std::cerr << "after updateE, ";
            //   std::cerr << "Ey_shmem[s_E_idx] != Ey[g_idx]\n";
            //   std::exit(EXIT_FAILURE);
            // }
            // if(Ez_shmem[s_E_idx] != Ez[g_idx]) {
            //   std::cerr << "after updateE, ";
            //   std::cerr << "Ez_shmem[s_E_idx] != Ez[g_idx]\n";
            //   std::exit(EXIT_FAILURE);
            // }
          }
        }
      }

      // update H 
      if(calculate_Hx & calculate_Hy & calculate_Hz) {
        for(size_t thread_id=0; thread_id<block_size; thread_id++) {
          int local_x = thread_id % BLX_GPU;                     // X coordinate within the tile
          int local_y = (thread_id / BLX_GPU) % BLY_GPU;     // Y coordinate within the tile
          int local_z = thread_id / (BLX_GPU * BLY_GPU);     // Z coordinate within the tile

          // Hhead is offset
          int global_x = indices_X[2] + local_x; // Global X coordinate
          int global_y = indices_Y[2] + local_y; // Global Y coordinate
          int global_z = indices_Z[2] + local_z; // Global Z coordinate

          if(global_x >= 1 && global_x <= Nx-2 && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2 &&
            global_x <= indices_X[3] &&
            global_y <= indices_Y[3] &&
            global_z <= indices_Z[3]) {

            // need to recalculate shared indices
            int Eoffset_X = indices_X[0] - xx_heads[xx];
            int Eoffset_Y = indices_Y[0] - yy_heads[yy];
            int Eoffset_Z = indices_Z[0] - zz_heads[zz];

            int Hoffset_X = indices_X[2] - xx_heads[xx];
            int Hoffset_Y = indices_Y[2] - yy_heads[yy];
            int Hoffset_Z = indices_Z[2] - zz_heads[zz];

            int shared_H_x = local_x + 1 + Hoffset_X;
            int shared_H_y = local_y + 1 + Hoffset_Y;
            int shared_H_z = local_z + 1 + Hoffset_Z;

            int shared_E_x = (m_or_v_X == 0 && xx != 0)? local_x + Eoffset_X - 1 : local_x + Eoffset_X;
            int shared_E_y = (m_or_v_Y == 0 && yy != 0)? local_y + Eoffset_Y - 1 : local_y + Eoffset_Y;
            int shared_E_z = (m_or_v_Z == 0 && zz != 0)? local_z + Eoffset_Z - 1 : local_z + Eoffset_Z;

            int s_H_idx = shared_H_x + shared_H_y * BLX_EH + shared_H_z * BLX_EH * BLY_EH; // shared memory idx for H
            int s_E_idx = shared_E_x + shared_E_y * BLX_EH + shared_E_z * BLX_EH * BLY_EH; // shared memory idx for E

            int g_idx = global_x + global_y * Nx + global_z * Nx * Ny; // global idx

            // if(Ex_shmem[s_E_idx] != Ex[g_idx]) {
            //   std::cerr << "updateH, ";
            //   std::cerr << "Ex_shmem[s_E_idx] != Ex[g_idx]\n";
            //   std::exit(EXIT_FAILURE);
            // }
            // if(Ex_shmem[s_E_idx + BLX_EH] != Ex[g_idx + Nx]) {
            //   std::cerr << "updateH, ";
            //   std::cerr << "Ex_shmem[s_E_idx + BLX_EH] != Ex[g_idx + Nx]\n";
            //   std::exit(EXIT_FAILURE);
            // }
            // if(Ex_shmem[s_E_idx + BLX_EH * BLY_EH] != Ex[g_idx + Nx * Ny]) {
            //   std::cerr << "updateH, ";
            //   std::cerr << "Ex_shmem[s_E_idx + BLX_EH * BLY_EH] != Ex[g_idx + Nx * Ny]\n";
            //   std::cerr << "Ex_shmem[s_E_idx + BLX_EH * BLY_EH] = " << Ex_shmem[s_E_idx + BLX_EH * BLY_EH] << ", Ex[g_idx + Nx * Ny] = " << Ex[g_idx + Nx * Ny] << "\n";
            //   std::exit(EXIT_FAILURE);
            // }
            // if(fabs(Ex[g_idx] + 0.0395334) <= 1e-8) {
            //   std::cerr << "found in Ex[g_idx].\n";
            // }
            // if(fabs(Ex[g_idx + Nx * Ny] + 0.0395334) <= 1e-8) {
            //   std::cerr << "found in Ex[g_idx + Nx * Ny].\n";
            // }
            // if(Ey_shmem[s_E_idx] != Ey[g_idx]) {
            //   std::cerr << "updateH, ";
            //   std::cerr << "Ey_shmem[s_E_idx] != Ey[g_idx]\n";
            //   std::exit(EXIT_FAILURE);
            // }
            // if(Ey_shmem[s_E_idx + 1] != Ey[g_idx + 1]) {
            //   std::cerr << "updateH, ";
            //   std::cerr << "Ey_shmem[s_E_idx + 1] != Ey[g_idx + 1]\n";
            //   std::exit(EXIT_FAILURE);
            // }
            // if(Ey_shmem[s_E_idx + BLX_EH * BLY_EH] != Ey[g_idx + Nx * Ny]) {
            //   std::cerr << "updateH, ";
            //   std::cerr << "Ey_shmem[s_E_idx + BLX_EH * BLY_EH] != Ey[g_idx + Nx * Ny]\n";
            //   std::exit(EXIT_FAILURE);
            // }
            // if(Ez_shmem[s_E_idx] != Ez[g_idx]) {
            //   std::cerr << "updateH, ";
            //   std::cerr << "Ez_shmem[s_E_idx] != Ez[g_idx]\n";
            //   std::exit(EXIT_FAILURE);
            // }
            // if(Ez_shmem[s_E_idx + 1] != Ez[g_idx + 1]) {
            //   std::cerr << "updateH, ";
            //   std::cerr << "Ez_shmem[s_E_idx + 1] != Ez[g_idx + 1]\n";
            //   std::exit(EXIT_FAILURE);
            // }
            // if(Ez_shmem[s_E_idx + BLX_EH] != Ez[g_idx + Nx]) {
            //   std::cerr << "updateH, ";
            //   std::cerr << "Ez_shmem[s_E_idx + BLX_EH] != Ez[g_idx + Nx]\n";
            //   std::exit(EXIT_FAILURE);
            // }
            
            // if(m_or_v_X == 0 && yy == 0 && zz == 0 && global_y == 3 && global_z == 3) {
            //   std::cout << "t = " << t << ", local_x = " << local_x << ", shared_H_x = " << shared_H_x << ", shared_E_x " << shared_E_x << "\n";
            // }
            // if(Hz[_source_idx] != Hz_shmem[_source_idx]) {
            //   std::cerr << "wrong!\n";
            //   std::cout << "t = " << t << ", local_x = " << local_x << ", shared_H_x = " << shared_H_x << ", shared_E_x " << shared_E_x << "\n";
            //   std::exit(EXIT_FAILURE);
            // }

            total_cal++;

            // Hx[g_idx] = Dax[g_idx] * Hx[g_idx] + Dbx[g_idx] *
            //           ((Ey[g_idx + Nx * Ny] - Ey[g_idx]) - (Ez[g_idx + Nx] - Ez[g_idx]) - Mx[g_idx] * dx);
            // Hy[g_idx] = Day[g_idx] * Hy[g_idx] + Dby[g_idx] *
            //           ((Ez[g_idx + 1] - Ez[g_idx]) - (Ex[g_idx + Nx * Ny] - Ex[g_idx]) - My[g_idx] * dx);
            // Hz[g_idx] = Daz[g_idx] * Hz[g_idx] + Dbz[g_idx] *
            //           ((Ex[g_idx + Nx] - Ex[g_idx]) - (Ey[g_idx + 1] - Ey[g_idx]) - Mz[g_idx] * dx);
            Hx_shmem[s_H_idx] = Dax[g_idx] * Hx_shmem[s_H_idx] + Dbx[g_idx] *
                    ((Ey_shmem[s_E_idx + BLX_EH * BLY_EH] - Ey_shmem[s_E_idx]) - (Ez_shmem[s_E_idx + BLX_EH] - Ez_shmem[s_E_idx]) - Mx[g_idx] * dx);
            Hy_shmem[s_H_idx] = Day[g_idx] * Hy_shmem[s_H_idx] + Dby[g_idx] *
                      ((Ez_shmem[s_E_idx + 1] - Ez_shmem[s_E_idx]) - (Ex_shmem[s_E_idx + BLX_EH * BLY_EH] - Ex_shmem[s_E_idx]) - My[g_idx] * dx);
            Hz_shmem[s_H_idx] = Daz[g_idx] * Hz_shmem[s_H_idx] + Dbz[g_idx] *
                      ((Ex_shmem[s_E_idx + BLX_EH] - Ex_shmem[s_E_idx]) - (Ey_shmem[s_E_idx + 1] - Ey_shmem[s_E_idx]) - Mz[g_idx] * dx);

            // if(Hx_shmem[s_H_idx] != Hx[g_idx]) {
            //   std::cerr << "after updateH, ";
            //   std::cerr << "Hx_shmem[s_H_idx] != Hx[g_idx]\n";
            //   std::exit(EXIT_FAILURE);
            // }
            // if(Hy_shmem[s_H_idx] != Hy[g_idx]) {
            //   std::cerr << "after updateH, ";
            //   std::cerr << "Hy_shmem[s_H_idx] != Hy[g_idx]\n";
            //   std::exit(EXIT_FAILURE);
            // }
            // if(Hz_shmem[s_H_idx] != Hz[g_idx]) {
            //   std::cerr << "after updateH, ";
            //   std::cerr << "Hz_shmem[s_H_idx] != Hz[g_idx]\n";
            //   std::exit(EXIT_FAILURE);
            // }
          }
        }
      }

    }

    // store E, H to global memory, no HALO needed
    for(size_t thread_id=0; thread_id<block_size; thread_id++) {
      int local_x = thread_id % BLX_GPU;                     // X coordinate within the tile
      int local_y = (thread_id / BLX_GPU) % BLY_GPU;     // Y coordinate within the tile
      int local_z = thread_id / (BLX_GPU * BLY_GPU);     // Z coordinate within the tile

      // Hhead is offset
      int global_x = xx_heads[xx] + local_x; // Global X coordinate
      int global_y = yy_heads[yy] + local_y; // Global Y coordinate
      int global_z = zz_heads[zz] + local_z; // Global Z coordinate

      if(global_x >= 1 && global_x <= Nx-2 && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2 &&
        global_x <= xx_tails[xx] &&
        global_y <= yy_tails[yy] &&
        global_z <= zz_tails[zz]) {

        int shared_H_x = local_x + 1;
        int shared_H_y = local_y + 1;
        int shared_H_z = local_z + 1;

        int shared_E_x = local_x;
        int shared_E_y = local_y;
        int shared_E_z = local_z;

        int s_H_idx = shared_H_x + shared_H_y * BLX_EH + shared_H_z * BLX_EH * BLY_EH; // shared memory idx for H
        int s_E_idx = shared_E_x + shared_E_y * BLX_EH + shared_E_z * BLX_EH * BLY_EH; // shared memory idx for E

        int g_idx = global_x + global_y * Nx + global_z * Nx * Ny; // global idx

        Ex[g_idx] = Ex_shmem[s_E_idx];
        Ey[g_idx] = Ey_shmem[s_E_idx];
        Ez[g_idx] = Ez_shmem[s_E_idx];
        Hx[g_idx] = Hx_shmem[s_H_idx];
        Hy[g_idx] = Hy_shmem[s_H_idx];
        Hz[g_idx] = Hz_shmem[s_H_idx];
      }
    }


  } 


}

void gDiamond::update_FDTD_gpu_fuse_kernel_globalmem(size_t num_timesteps) { // 3-D mapping, using diamond tiling to fuse kernels, global memory only

  size_t max_phases = 8;
  std::vector<int> mountain_heads_X;
  std::vector<int> mountain_tails_X;
  std::vector<int> mountain_heads_Y;
  std::vector<int> mountain_tails_Y;
  std::vector<int> mountain_heads_Z;
  std::vector<int> mountain_tails_Z;
  std::vector<int> valley_heads_X;
  std::vector<int> valley_tails_X;
  std::vector<int> valley_heads_Y;
  std::vector<int> valley_tails_Y;
  std::vector<int> valley_heads_Z;
  std::vector<int> valley_tails_Z;
  _setup_diamond_tiling_gpu(BLX_GPU, BLY_GPU, BLZ_GPU, BLT_GPU, max_phases);

  for(auto range : _Eranges_phases_X[0][0]) { 
    mountain_heads_X.push_back(range.first);
    mountain_tails_X.push_back(range.second);
  }
  for(auto range : _Eranges_phases_Y[0][0]) { 
    mountain_heads_Y.push_back(range.first);
    mountain_tails_Y.push_back(range.second);
  }
  for(auto range : _Eranges_phases_Z[0][0]) { 
    mountain_heads_Z.push_back(range.first);
    mountain_tails_Z.push_back(range.second);
  }
  // std::cout << "valley X range = ";
  for(auto range : _Hranges_phases_X[1][BLT_GPU-1]) { 
    // std::cout << "(" << range.first << ", " << range.second << ") ";
    valley_heads_X.push_back(range.first);
    valley_tails_X.push_back(range.second);
  }
  // std::cout << "\n";
  for(auto range : _Hranges_phases_Y[2][BLT_GPU-1]) { 
    valley_heads_Y.push_back(range.first);
    valley_tails_Y.push_back(range.second);
  }
  for(auto range : _Hranges_phases_Z[3][BLT_GPU-1]) { 
    valley_heads_Z.push_back(range.first);
    valley_tails_Z.push_back(range.second);
  }

  size_t num_mountains_X = mountain_heads_X.size();
  size_t num_mountains_Y = mountain_heads_Y.size();
  size_t num_mountains_Z = mountain_heads_Z.size();
  size_t num_valleys_X = valley_heads_X.size();
  size_t num_valleys_Y = valley_heads_Y.size();
  size_t num_valleys_Z = valley_heads_Z.size();

  // head and tail on device
  int *mountain_heads_X_d, *mountain_tails_X_d;
  int *mountain_heads_Y_d, *mountain_tails_Y_d;
  int *mountain_heads_Z_d, *mountain_tails_Z_d;
  int *valley_heads_X_d, *valley_tails_X_d;
  int *valley_heads_Y_d, *valley_tails_Y_d;
  int *valley_heads_Z_d, *valley_tails_Z_d;

  CUDACHECK(cudaMalloc(&mountain_heads_X_d, sizeof(int) * num_mountains_X));
  CUDACHECK(cudaMalloc(&mountain_tails_X_d, sizeof(int) * num_mountains_X));
  CUDACHECK(cudaMalloc(&mountain_heads_Y_d, sizeof(int) * num_mountains_Y));
  CUDACHECK(cudaMalloc(&mountain_tails_Y_d, sizeof(int) * num_mountains_Y));
  CUDACHECK(cudaMalloc(&mountain_heads_Z_d, sizeof(int) * num_mountains_Z));
  CUDACHECK(cudaMalloc(&mountain_tails_Z_d, sizeof(int) * num_mountains_Z));
  CUDACHECK(cudaMalloc(&valley_heads_X_d, sizeof(int) * num_valleys_X));
  CUDACHECK(cudaMalloc(&valley_tails_X_d, sizeof(int) * num_valleys_X));
  CUDACHECK(cudaMalloc(&valley_heads_Y_d, sizeof(int) * num_valleys_Y));
  CUDACHECK(cudaMalloc(&valley_tails_Y_d, sizeof(int) * num_valleys_Y));
  CUDACHECK(cudaMalloc(&valley_heads_Z_d, sizeof(int) * num_valleys_Z));
  CUDACHECK(cudaMalloc(&valley_tails_Z_d, sizeof(int) * num_valleys_Z));

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

  // transfer source
  for(size_t t=0; t<num_timesteps; t++) {
    float Mz_value = M_source_amp * std::sin(SOURCE_OMEGA * t * dt);
    CUDACHECK(cudaMemcpy(Mz + _source_idx, &Mz_value, sizeof(float), cudaMemcpyHostToDevice));
  }
  
  auto start = std::chrono::high_resolution_clock::now();

  // copy Ca, Cb, Da, Db
  CUDACHECK(cudaMemcpyAsync(Cax, _Cax.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Cay, _Cay.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Caz, _Caz.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Cbx, _Cbx.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Cby, _Cby.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Cbz, _Cbz.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Dax, _Dax.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Day, _Day.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Daz, _Daz.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Dbx, _Dbx.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Dby, _Dby.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Dbz, _Dbz.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));

  // copy heads and tails
  CUDACHECK(cudaMemcpyAsync(mountain_heads_X_d, mountain_heads_X.data(), sizeof(int) * num_mountains_X, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(mountain_tails_X_d, mountain_tails_X.data(), sizeof(int) * num_mountains_X, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(mountain_heads_Y_d, mountain_heads_Y.data(), sizeof(int) * num_mountains_Y, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(mountain_tails_Y_d, mountain_tails_Y.data(), sizeof(int) * num_mountains_Y, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(mountain_heads_Z_d, mountain_heads_Z.data(), sizeof(int) * num_mountains_Z, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(mountain_tails_Z_d, mountain_tails_Z.data(), sizeof(int) * num_mountains_Z, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(valley_heads_X_d, valley_heads_X.data(), sizeof(int) * num_valleys_X, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(valley_tails_X_d, valley_tails_X.data(), sizeof(int) * num_valleys_X, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(valley_heads_Y_d, valley_heads_Y.data(), sizeof(int) * num_valleys_Y, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(valley_tails_Y_d, valley_tails_Y.data(), sizeof(int) * num_valleys_Y, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(valley_heads_Z_d, valley_heads_Z.data(), sizeof(int) * num_valleys_Z, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(valley_tails_Z_d, valley_tails_Z.data(), sizeof(int) * num_valleys_Z, cudaMemcpyHostToDevice));

  // set block size 
  size_t block_size = BLX_GPU * BLY_GPU * BLZ_GPU;
  size_t grid_size;

  for(size_t t=0; t<num_timesteps/BLT_GPU; t++) {
 
    // phase 1: (m, m, m)
    grid_size = num_mountains_X * num_mountains_Y * num_mountains_Z;
    updateEH_phase_global_mem<<<grid_size, block_size>>>(Ex, Ey, Ez,
                        Hx, Hy, Hz,
                        Cax, Cbx,
                        Cay, Cby,
                        Caz, Cbz,
                        Dax, Dbx,
                        Day, Dby,
                        Daz, Dbz,
                        Jx, Jy, Jz,
                        Mx, My, Mz,
                        _dx, 
                        _Nx, _Ny, _Nz,
                        num_mountains_X, num_mountains_Y, num_mountains_Z, 
                        mountain_heads_X_d, mountain_heads_Y_d, mountain_heads_Z_d, 
                        mountain_tails_X_d, mountain_tails_Y_d, mountain_tails_Z_d, 
                        1, 1, 1,
                        block_size,
                        grid_size);

    // phase 2: (v, m, m)
    grid_size = num_valleys_X * num_mountains_Y * num_mountains_Z;
    updateEH_phase_global_mem<<<grid_size, block_size>>>(Ex, Ey, Ez,
                        Hx, Hy, Hz,
                        Cax, Cbx,
                        Cay, Cby,
                        Caz, Cbz,
                        Dax, Dbx,
                        Day, Dby,
                        Daz, Dbz,
                        Jx, Jy, Jz,
                        Mx, My, Mz,
                        _dx, 
                        _Nx, _Ny, _Nz,
                        num_valleys_X, num_mountains_Y, num_mountains_Z, 
                        valley_heads_X_d, mountain_heads_Y_d, mountain_heads_Z_d, 
                        valley_tails_X_d, mountain_tails_Y_d, mountain_tails_Z_d, 
                        0, 1, 1,
                        block_size,
                        grid_size);

    // phase 3: (m, v, m)
    grid_size = num_mountains_X * num_valleys_Y * num_mountains_Z;
    updateEH_phase_global_mem<<<grid_size, block_size>>>(Ex, Ey, Ez,
                        Hx, Hy, Hz,
                        Cax, Cbx,
                        Cay, Cby,
                        Caz, Cbz,
                        Dax, Dbx,
                        Day, Dby,
                        Daz, Dbz,
                        Jx, Jy, Jz,
                        Mx, My, Mz,
                        _dx, 
                        _Nx, _Ny, _Nz,
                        num_mountains_X, num_valleys_Y, num_mountains_Z, 
                        mountain_heads_X_d, valley_heads_Y_d, mountain_heads_Z_d, 
                        mountain_tails_X_d, valley_tails_Y_d, mountain_tails_Z_d, 
                        1, 0, 1,
                        block_size,
                        grid_size);

    // phase 4: (m, m, v)
    grid_size = num_mountains_X * num_mountains_Y * num_valleys_Z;
    updateEH_phase_global_mem<<<grid_size, block_size>>>(Ex, Ey, Ez,
                        Hx, Hy, Hz,
                        Cax, Cbx,
                        Cay, Cby,
                        Caz, Cbz,
                        Dax, Dbx,
                        Day, Dby,
                        Daz, Dbz,
                        Jx, Jy, Jz,
                        Mx, My, Mz,
                        _dx, 
                        _Nx, _Ny, _Nz,
                        num_mountains_X, num_mountains_Y, num_valleys_Z, 
                        mountain_heads_X_d, mountain_heads_Y_d, valley_heads_Z_d, 
                        mountain_tails_X_d, mountain_tails_Y_d, valley_tails_Z_d, 
                        1, 1, 0,
                        block_size,
                        grid_size);

    // phase 5: (v, v, m)
    grid_size = num_valleys_X * num_valleys_Y * num_mountains_Z;
    updateEH_phase_global_mem<<<grid_size, block_size>>>(Ex, Ey, Ez,
                        Hx, Hy, Hz,
                        Cax, Cbx,
                        Cay, Cby,
                        Caz, Cbz,
                        Dax, Dbx,
                        Day, Dby,
                        Daz, Dbz,
                        Jx, Jy, Jz,
                        Mx, My, Mz,
                        _dx, 
                        _Nx, _Ny, _Nz,
                        num_valleys_X, num_valleys_Y, num_mountains_Z, 
                        valley_heads_X_d, valley_heads_Y_d, mountain_heads_Z_d, 
                        valley_tails_X_d, valley_tails_Y_d, mountain_tails_Z_d, 
                        0, 0, 1,
                        block_size,
                        grid_size);

    // phase 6: (v, m, v)
    grid_size = num_valleys_X * num_mountains_Y * num_valleys_Z;
    updateEH_phase_global_mem<<<grid_size, block_size>>>(Ex, Ey, Ez,
                        Hx, Hy, Hz,
                        Cax, Cbx,
                        Cay, Cby,
                        Caz, Cbz,
                        Dax, Dbx,
                        Day, Dby,
                        Daz, Dbz,
                        Jx, Jy, Jz,
                        Mx, My, Mz,
                        _dx, 
                        _Nx, _Ny, _Nz,
                        num_valleys_X, num_mountains_Y, num_valleys_Z, 
                        valley_heads_X_d, mountain_heads_Y_d, valley_heads_Z_d, 
                        valley_tails_X_d, mountain_tails_Y_d, valley_tails_Z_d, 
                        0, 1, 0,
                        block_size,
                        grid_size);

    // phase 7: (m, v, v)
    grid_size = num_mountains_X * num_valleys_Y * num_valleys_Z;
    updateEH_phase_global_mem<<<grid_size, block_size>>>(Ex, Ey, Ez,
                        Hx, Hy, Hz,
                        Cax, Cbx,
                        Cay, Cby,
                        Caz, Cbz,
                        Dax, Dbx,
                        Day, Dby,
                        Daz, Dbz,
                        Jx, Jy, Jz,
                        Mx, My, Mz,
                        _dx, 
                        _Nx, _Ny, _Nz,
                        num_mountains_X, num_valleys_Y, num_valleys_Z, 
                        mountain_heads_X_d, valley_heads_Y_d, valley_heads_Z_d, 
                        mountain_tails_X_d, valley_tails_Y_d, valley_tails_Z_d, 
                        1, 0, 0,
                        block_size,
                        grid_size);

    // phase 8: (v, v, v)
    grid_size = num_valleys_X * num_valleys_Y * num_valleys_Z;
    updateEH_phase_global_mem<<<grid_size, block_size>>>(Ex, Ey, Ez,
                        Hx, Hy, Hz,
                        Cax, Cbx,
                        Cay, Cby,
                        Caz, Cbz,
                        Dax, Dbx,
                        Day, Dby,
                        Daz, Dbz,
                        Jx, Jy, Jz,
                        Mx, My, Mz,
                        _dx, 
                        _Nx, _Ny, _Nz,
                        num_valleys_X, num_valleys_Y, num_valleys_Z, 
                        valley_heads_X_d, valley_heads_Y_d, valley_heads_Z_d, 
                        valley_tails_X_d, valley_tails_Y_d, valley_tails_Z_d, 
                        0, 0, 0,
                        block_size,
                        grid_size);


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
  std::cout << "gpu runtime (3-D mapping, global memory only): " << std::chrono::duration<double>(end-start).count() << "s\n"; 
  std::cout << "gpu performance: " << (_Nx * _Ny * _Nz / 1.0e6 * num_timesteps) / std::chrono::duration<double>(end-start).count() << "Mcells/s\n";

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

  CUDACHECK(cudaFree(mountain_heads_X_d));
  CUDACHECK(cudaFree(mountain_tails_X_d));
  CUDACHECK(cudaFree(mountain_heads_Y_d));
  CUDACHECK(cudaFree(mountain_tails_Y_d));
  CUDACHECK(cudaFree(mountain_heads_Z_d));
  CUDACHECK(cudaFree(mountain_tails_Z_d));
  CUDACHECK(cudaFree(valley_heads_X_d));
  CUDACHECK(cudaFree(valley_tails_X_d));
  CUDACHECK(cudaFree(valley_heads_Y_d));
  CUDACHECK(cudaFree(valley_tails_Y_d));
  CUDACHECK(cudaFree(valley_heads_Z_d));
  CUDACHECK(cudaFree(valley_tails_Z_d));
}

void gDiamond::update_FDTD_gpu_fuse_kernel_shmem_EH(size_t num_timesteps) { // 3-D mapping, using diamond tiling to fuse kernels, put EH in shared memory 

  size_t max_phases = 8;
  std::vector<int> mountain_heads_X;
  std::vector<int> mountain_tails_X;
  std::vector<int> mountain_heads_Y;
  std::vector<int> mountain_tails_Y;
  std::vector<int> mountain_heads_Z;
  std::vector<int> mountain_tails_Z;
  std::vector<int> valley_heads_X;
  std::vector<int> valley_tails_X;
  std::vector<int> valley_heads_Y;
  std::vector<int> valley_tails_Y;
  std::vector<int> valley_heads_Z;
  std::vector<int> valley_tails_Z;
  _setup_diamond_tiling_gpu(BLX_GPU, BLY_GPU, BLZ_GPU, BLT_GPU, max_phases);

  for(auto range : _Eranges_phases_X[0][0]) { 
    mountain_heads_X.push_back(range.first);
    mountain_tails_X.push_back(range.second);
  }
  for(auto range : _Eranges_phases_Y[0][0]) { 
    mountain_heads_Y.push_back(range.first);
    mountain_tails_Y.push_back(range.second);
  }
  for(auto range : _Eranges_phases_Z[0][0]) { 
    mountain_heads_Z.push_back(range.first);
    mountain_tails_Z.push_back(range.second);
  }
  // std::cout << "valley X range = ";
  for(auto range : _Hranges_phases_X[1][BLT_GPU-1]) { 
    // std::cout << "(" << range.first << ", " << range.second << ") ";
    valley_heads_X.push_back(range.first);
    valley_tails_X.push_back(range.second);
  }
  // std::cout << "\n";
  for(auto range : _Hranges_phases_Y[2][BLT_GPU-1]) { 
    valley_heads_Y.push_back(range.first);
    valley_tails_Y.push_back(range.second);
  }
  for(auto range : _Hranges_phases_Z[3][BLT_GPU-1]) { 
    valley_heads_Z.push_back(range.first);
    valley_tails_Z.push_back(range.second);
  }

  size_t num_mountains_X = mountain_heads_X.size();
  size_t num_mountains_Y = mountain_heads_Y.size();
  size_t num_mountains_Z = mountain_heads_Z.size();
  size_t num_valleys_X = valley_heads_X.size();
  size_t num_valleys_Y = valley_heads_Y.size();
  size_t num_valleys_Z = valley_heads_Z.size();

  // head and tail on device
  int *mountain_heads_X_d, *mountain_tails_X_d;
  int *mountain_heads_Y_d, *mountain_tails_Y_d;
  int *mountain_heads_Z_d, *mountain_tails_Z_d;
  int *valley_heads_X_d, *valley_tails_X_d;
  int *valley_heads_Y_d, *valley_tails_Y_d;
  int *valley_heads_Z_d, *valley_tails_Z_d;

  CUDACHECK(cudaMalloc(&mountain_heads_X_d, sizeof(int) * num_mountains_X));
  CUDACHECK(cudaMalloc(&mountain_tails_X_d, sizeof(int) * num_mountains_X));
  CUDACHECK(cudaMalloc(&mountain_heads_Y_d, sizeof(int) * num_mountains_Y));
  CUDACHECK(cudaMalloc(&mountain_tails_Y_d, sizeof(int) * num_mountains_Y));
  CUDACHECK(cudaMalloc(&mountain_heads_Z_d, sizeof(int) * num_mountains_Z));
  CUDACHECK(cudaMalloc(&mountain_tails_Z_d, sizeof(int) * num_mountains_Z));
  CUDACHECK(cudaMalloc(&valley_heads_X_d, sizeof(int) * num_valleys_X));
  CUDACHECK(cudaMalloc(&valley_tails_X_d, sizeof(int) * num_valleys_X));
  CUDACHECK(cudaMalloc(&valley_heads_Y_d, sizeof(int) * num_valleys_Y));
  CUDACHECK(cudaMalloc(&valley_tails_Y_d, sizeof(int) * num_valleys_Y));
  CUDACHECK(cudaMalloc(&valley_heads_Z_d, sizeof(int) * num_valleys_Z));
  CUDACHECK(cudaMalloc(&valley_tails_Z_d, sizeof(int) * num_valleys_Z));

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

  // transfer source
  for(size_t t=0; t<num_timesteps; t++) {
    float Mz_value = M_source_amp * std::sin(SOURCE_OMEGA * t * dt);
    CUDACHECK(cudaMemcpy(Mz + _source_idx, &Mz_value, sizeof(float), cudaMemcpyHostToDevice));
  }
  
  auto start = std::chrono::high_resolution_clock::now();

  // copy Ca, Cb, Da, Db
  CUDACHECK(cudaMemcpyAsync(Cax, _Cax.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Cay, _Cay.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Caz, _Caz.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Cbx, _Cbx.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Cby, _Cby.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Cbz, _Cbz.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Dax, _Dax.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Day, _Day.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Daz, _Daz.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Dbx, _Dbx.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Dby, _Dby.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Dbz, _Dbz.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));

  // copy heads and tails
  CUDACHECK(cudaMemcpyAsync(mountain_heads_X_d, mountain_heads_X.data(), sizeof(int) * num_mountains_X, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(mountain_tails_X_d, mountain_tails_X.data(), sizeof(int) * num_mountains_X, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(mountain_heads_Y_d, mountain_heads_Y.data(), sizeof(int) * num_mountains_Y, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(mountain_tails_Y_d, mountain_tails_Y.data(), sizeof(int) * num_mountains_Y, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(mountain_heads_Z_d, mountain_heads_Z.data(), sizeof(int) * num_mountains_Z, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(mountain_tails_Z_d, mountain_tails_Z.data(), sizeof(int) * num_mountains_Z, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(valley_heads_X_d, valley_heads_X.data(), sizeof(int) * num_valleys_X, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(valley_tails_X_d, valley_tails_X.data(), sizeof(int) * num_valleys_X, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(valley_heads_Y_d, valley_heads_Y.data(), sizeof(int) * num_valleys_Y, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(valley_tails_Y_d, valley_tails_Y.data(), sizeof(int) * num_valleys_Y, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(valley_heads_Z_d, valley_heads_Z.data(), sizeof(int) * num_valleys_Z, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(valley_tails_Z_d, valley_tails_Z.data(), sizeof(int) * num_valleys_Z, cudaMemcpyHostToDevice));

  // set block size 
  size_t block_size = BLX_GPU * BLY_GPU * BLZ_GPU;
  size_t grid_size;

  for(size_t t=0; t<num_timesteps/BLT_GPU; t++) {
 
    // phase 1: (m, m, m)
    grid_size = num_mountains_X * num_mountains_Y * num_mountains_Z;
    updateEH_phase_shmem_EH<<<grid_size, block_size>>>(Ex, Ey, Ez,
                        Hx, Hy, Hz,
                        Cax, Cbx,
                        Cay, Cby,
                        Caz, Cbz,
                        Dax, Dbx,
                        Day, Dby,
                        Daz, Dbz,
                        Jx, Jy, Jz,
                        Mx, My, Mz,
                        _dx, 
                        _Nx, _Ny, _Nz,
                        num_mountains_X, num_mountains_Y, num_mountains_Z, 
                        mountain_heads_X_d, mountain_heads_Y_d, mountain_heads_Z_d, 
                        mountain_tails_X_d, mountain_tails_Y_d, mountain_tails_Z_d, 
                        1, 1, 1,
                        block_size,
                        grid_size);

    // phase 2: (v, m, m)
    grid_size = num_valleys_X * num_mountains_Y * num_mountains_Z;
    updateEH_phase_shmem_EH<<<grid_size, block_size>>>(Ex, Ey, Ez,
                        Hx, Hy, Hz,
                        Cax, Cbx,
                        Cay, Cby,
                        Caz, Cbz,
                        Dax, Dbx,
                        Day, Dby,
                        Daz, Dbz,
                        Jx, Jy, Jz,
                        Mx, My, Mz,
                        _dx, 
                        _Nx, _Ny, _Nz,
                        num_valleys_X, num_mountains_Y, num_mountains_Z, 
                        valley_heads_X_d, mountain_heads_Y_d, mountain_heads_Z_d, 
                        valley_tails_X_d, mountain_tails_Y_d, mountain_tails_Z_d, 
                        0, 1, 1,
                        block_size,
                        grid_size);

    // phase 3: (m, v, m)
    grid_size = num_mountains_X * num_valleys_Y * num_mountains_Z;
    updateEH_phase_shmem_EH<<<grid_size, block_size>>>(Ex, Ey, Ez,
                        Hx, Hy, Hz,
                        Cax, Cbx,
                        Cay, Cby,
                        Caz, Cbz,
                        Dax, Dbx,
                        Day, Dby,
                        Daz, Dbz,
                        Jx, Jy, Jz,
                        Mx, My, Mz,
                        _dx, 
                        _Nx, _Ny, _Nz,
                        num_mountains_X, num_valleys_Y, num_mountains_Z, 
                        mountain_heads_X_d, valley_heads_Y_d, mountain_heads_Z_d, 
                        mountain_tails_X_d, valley_tails_Y_d, mountain_tails_Z_d, 
                        1, 0, 1,
                        block_size,
                        grid_size);

    // phase 4: (m, m, v)
    grid_size = num_mountains_X * num_mountains_Y * num_valleys_Z;
    updateEH_phase_shmem_EH<<<grid_size, block_size>>>(Ex, Ey, Ez,
                        Hx, Hy, Hz,
                        Cax, Cbx,
                        Cay, Cby,
                        Caz, Cbz,
                        Dax, Dbx,
                        Day, Dby,
                        Daz, Dbz,
                        Jx, Jy, Jz,
                        Mx, My, Mz,
                        _dx, 
                        _Nx, _Ny, _Nz,
                        num_mountains_X, num_mountains_Y, num_valleys_Z, 
                        mountain_heads_X_d, mountain_heads_Y_d, valley_heads_Z_d, 
                        mountain_tails_X_d, mountain_tails_Y_d, valley_tails_Z_d, 
                        1, 1, 0,
                        block_size,
                        grid_size);

    // phase 5: (v, v, m)
    grid_size = num_valleys_X * num_valleys_Y * num_mountains_Z;
    updateEH_phase_shmem_EH<<<grid_size, block_size>>>(Ex, Ey, Ez,
                        Hx, Hy, Hz,
                        Cax, Cbx,
                        Cay, Cby,
                        Caz, Cbz,
                        Dax, Dbx,
                        Day, Dby,
                        Daz, Dbz,
                        Jx, Jy, Jz,
                        Mx, My, Mz,
                        _dx, 
                        _Nx, _Ny, _Nz,
                        num_valleys_X, num_valleys_Y, num_mountains_Z, 
                        valley_heads_X_d, valley_heads_Y_d, mountain_heads_Z_d, 
                        valley_tails_X_d, valley_tails_Y_d, mountain_tails_Z_d, 
                        0, 0, 1,
                        block_size,
                        grid_size);

    // phase 6: (v, m, v)
    grid_size = num_valleys_X * num_mountains_Y * num_valleys_Z;
    updateEH_phase_shmem_EH<<<grid_size, block_size>>>(Ex, Ey, Ez,
                        Hx, Hy, Hz,
                        Cax, Cbx,
                        Cay, Cby,
                        Caz, Cbz,
                        Dax, Dbx,
                        Day, Dby,
                        Daz, Dbz,
                        Jx, Jy, Jz,
                        Mx, My, Mz,
                        _dx, 
                        _Nx, _Ny, _Nz,
                        num_valleys_X, num_mountains_Y, num_valleys_Z, 
                        valley_heads_X_d, mountain_heads_Y_d, valley_heads_Z_d, 
                        valley_tails_X_d, mountain_tails_Y_d, valley_tails_Z_d, 
                        0, 1, 0,
                        block_size,
                        grid_size);

    // phase 7: (m, v, v)
    grid_size = num_mountains_X * num_valleys_Y * num_valleys_Z;
    updateEH_phase_shmem_EH<<<grid_size, block_size>>>(Ex, Ey, Ez,
                        Hx, Hy, Hz,
                        Cax, Cbx,
                        Cay, Cby,
                        Caz, Cbz,
                        Dax, Dbx,
                        Day, Dby,
                        Daz, Dbz,
                        Jx, Jy, Jz,
                        Mx, My, Mz,
                        _dx, 
                        _Nx, _Ny, _Nz,
                        num_mountains_X, num_valleys_Y, num_valleys_Z, 
                        mountain_heads_X_d, valley_heads_Y_d, valley_heads_Z_d, 
                        mountain_tails_X_d, valley_tails_Y_d, valley_tails_Z_d, 
                        1, 0, 0,
                        block_size,
                        grid_size);

    // phase 8: (v, v, v)
    grid_size = num_valleys_X * num_valleys_Y * num_valleys_Z;
    updateEH_phase_shmem_EH<<<grid_size, block_size>>>(Ex, Ey, Ez,
                        Hx, Hy, Hz,
                        Cax, Cbx,
                        Cay, Cby,
                        Caz, Cbz,
                        Dax, Dbx,
                        Day, Dby,
                        Daz, Dbz,
                        Jx, Jy, Jz,
                        Mx, My, Mz,
                        _dx, 
                        _Nx, _Ny, _Nz,
                        num_valleys_X, num_valleys_Y, num_valleys_Z, 
                        valley_heads_X_d, valley_heads_Y_d, valley_heads_Z_d, 
                        valley_tails_X_d, valley_tails_Y_d, valley_tails_Z_d, 
                        0, 0, 0,
                        block_size,
                        grid_size);


  }
  cudaDeviceSynchronize();

  // copy E, H back to host 
  CUDACHECK(cudaMemcpy(_Ex_gpu_shEH.data(), Ex, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Ey_gpu_shEH.data(), Ey, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Ez_gpu_shEH.data(), Ez, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Hx_gpu_shEH.data(), Hx, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Hy_gpu_shEH.data(), Hy, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Hz_gpu_shEH.data(), Hz, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));

  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "gpu runtime (3-D mapping, shared memory for EH only): " << std::chrono::duration<double>(end-start).count() << "s\n"; 
  std::cout << "gpu performance: " << (_Nx * _Ny * _Nz / 1.0e6 * num_timesteps) / std::chrono::duration<double>(end-start).count() << "Mcells/s\n";

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

  CUDACHECK(cudaFree(mountain_heads_X_d));
  CUDACHECK(cudaFree(mountain_tails_X_d));
  CUDACHECK(cudaFree(mountain_heads_Y_d));
  CUDACHECK(cudaFree(mountain_tails_Y_d));
  CUDACHECK(cudaFree(mountain_heads_Z_d));
  CUDACHECK(cudaFree(mountain_tails_Z_d));
  CUDACHECK(cudaFree(valley_heads_X_d));
  CUDACHECK(cudaFree(valley_tails_X_d));
  CUDACHECK(cudaFree(valley_heads_Y_d));
  CUDACHECK(cudaFree(valley_tails_Y_d));
  CUDACHECK(cudaFree(valley_heads_Z_d));
  CUDACHECK(cudaFree(valley_tails_Z_d));
}

void gDiamond::update_FDTD_gpu_fuse_kernel_shmem_EH_pt(size_t num_timesteps) { // 2-D mapping, using diamond tiling on X, Y dimension to fuse kernels, put EH in shared memory 

  size_t max_phases = 8;
  std::vector<int> mountain_heads_X;
  std::vector<int> mountain_tails_X;
  std::vector<int> mountain_heads_Y;
  std::vector<int> mountain_tails_Y;
  std::vector<int> mountain_heads_Z;
  std::vector<int> mountain_tails_Z;
  std::vector<int> valley_heads_X;
  std::vector<int> valley_tails_X;
  std::vector<int> valley_heads_Y;
  std::vector<int> valley_tails_Y;
  std::vector<int> valley_heads_Z;
  std::vector<int> valley_tails_Z;
  _setup_diamond_tiling_gpu(BLX_GPU_PT, BLY_GPU_PT, BLZ_GPU_PT, BLT_GPU_PT, max_phases);

  for(auto range : _Eranges_phases_X[0][0]) { 
    mountain_heads_X.push_back(range.first);
    mountain_tails_X.push_back(range.second);
  }
  for(auto range : _Eranges_phases_Y[0][0]) { 
    mountain_heads_Y.push_back(range.first);
    mountain_tails_Y.push_back(range.second);
  }
  // std::cout << "valley X range = ";
  for(auto range : _Hranges_phases_X[1][BLT_GPU_PT-1]) { 
    // std::cout << "(" << range.first << ", " << range.second << ") ";
    valley_heads_X.push_back(range.first);
    valley_tails_X.push_back(range.second);
  }
  // std::cout << "\n";
  for(auto range : _Hranges_phases_Y[2][BLT_GPU_PT-1]) { 
    valley_heads_Y.push_back(range.first);
    valley_tails_Y.push_back(range.second);
  }

  size_t num_mountains_X = mountain_heads_X.size();
  size_t num_mountains_Y = mountain_heads_Y.size();
  size_t num_mountains_Z = mountain_heads_Z.size();
  size_t num_valleys_X = valley_heads_X.size();
  size_t num_valleys_Y = valley_heads_Y.size();

  // head and tail on device
  int *mountain_heads_X_d, *mountain_tails_X_d;
  int *mountain_heads_Y_d, *mountain_tails_Y_d;
  int *valley_heads_X_d, *valley_tails_X_d;
  int *valley_heads_Y_d, *valley_tails_Y_d;

  CUDACHECK(cudaMalloc(&mountain_heads_X_d, sizeof(int) * num_mountains_X));
  CUDACHECK(cudaMalloc(&mountain_tails_X_d, sizeof(int) * num_mountains_X));
  CUDACHECK(cudaMalloc(&mountain_heads_Y_d, sizeof(int) * num_mountains_Y));
  CUDACHECK(cudaMalloc(&mountain_tails_Y_d, sizeof(int) * num_mountains_Y));
  CUDACHECK(cudaMalloc(&valley_heads_X_d, sizeof(int) * num_valleys_X));
  CUDACHECK(cudaMalloc(&valley_tails_X_d, sizeof(int) * num_valleys_X));
  CUDACHECK(cudaMalloc(&valley_heads_Y_d, sizeof(int) * num_valleys_Y));
  CUDACHECK(cudaMalloc(&valley_tails_Y_d, sizeof(int) * num_valleys_Y));

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

  // transfer source
  for(size_t t=0; t<num_timesteps; t++) {
    float Mz_value = M_source_amp * std::sin(SOURCE_OMEGA * t * dt);
    CUDACHECK(cudaMemcpy(Mz + _source_idx, &Mz_value, sizeof(float), cudaMemcpyHostToDevice));
  }
  
  auto start = std::chrono::high_resolution_clock::now();

  // copy Ca, Cb, Da, Db
  CUDACHECK(cudaMemcpyAsync(Cax, _Cax.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Cay, _Cay.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Caz, _Caz.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Cbx, _Cbx.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Cby, _Cby.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Cbz, _Cbz.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Dax, _Dax.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Day, _Day.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Daz, _Daz.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Dbx, _Dbx.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Dby, _Dby.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Dbz, _Dbz.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));

  // copy heads and tails
  CUDACHECK(cudaMemcpyAsync(mountain_heads_X_d, mountain_heads_X.data(), sizeof(int) * num_mountains_X, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(mountain_tails_X_d, mountain_tails_X.data(), sizeof(int) * num_mountains_X, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(mountain_heads_Y_d, mountain_heads_Y.data(), sizeof(int) * num_mountains_Y, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(mountain_tails_Y_d, mountain_tails_Y.data(), sizeof(int) * num_mountains_Y, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(valley_heads_X_d, valley_heads_X.data(), sizeof(int) * num_valleys_X, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(valley_tails_X_d, valley_tails_X.data(), sizeof(int) * num_valleys_X, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(valley_heads_Y_d, valley_heads_Y.data(), sizeof(int) * num_valleys_Y, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(valley_tails_Y_d, valley_tails_Y.data(), sizeof(int) * num_valleys_Y, cudaMemcpyHostToDevice));

  // set block size 
  size_t block_size = BLX_GPU_PT * BLY_GPU_PT;
  size_t grid_size;
  size_t num_para_Z = _Nz - BLT_GPU_PT; 
  
  for(size_t t=0; t<num_timesteps/BLT_GPU_PT; t++) {
 
    // phase 1. (m, m, *)
    grid_size = num_mountains_X * num_mountains_Y;
    updateEH_phase_shmem_EH_2D<<<grid_size, block_size>>>(Ex, Ey, Ez,
                                                            Hx, Hy, Hz,
                                                            Cax, Cbx,
                                                            Cay, Cby,
                                                            Caz, Cbz,
                                                            Dax, Dbx,
                                                            Day, Dby,
                                                            Daz, Dbz,
                                                            Jx, Jy, Jz,
                                                            Mx, My, Mz,
                                                            _dx, 
                                                            _Nx, _Ny, _Nz,
                                                            num_mountains_X, num_mountains_Y, num_para_Z, 
                                                            mountain_heads_X_d,
                                                            mountain_heads_Y_d,
                                                            mountain_tails_X_d,
                                                            mountain_tails_Y_d,
                                                            1, 1, 
                                                            block_size,
                                                            grid_size); 

    // phase 2. (v, m, *)
    grid_size = num_valleys_X * num_mountains_Y;
    updateEH_phase_shmem_EH_2D<<<grid_size, block_size>>>(Ex, Ey, Ez,
                                                            Hx, Hy, Hz,
                                                            Cax, Cbx,
                                                            Cay, Cby,
                                                            Caz, Cbz,
                                                            Dax, Dbx,
                                                            Day, Dby,
                                                            Daz, Dbz,
                                                            Jx, Jy, Jz,
                                                            Mx, My, Mz,
                                                            _dx, 
                                                            _Nx, _Ny, _Nz,
                                                            num_valleys_X, num_mountains_Y, num_para_Z, 
                                                            valley_heads_X_d,
                                                            mountain_heads_Y_d,
                                                            valley_tails_X_d,
                                                            mountain_tails_Y_d,
                                                            0, 1, 
                                                            block_size,
                                                            grid_size); 

    // phase 3. (m, v, *)
    grid_size = num_mountains_X * num_valleys_Y;
    updateEH_phase_shmem_EH_2D<<<grid_size, block_size>>>(Ex, Ey, Ez,
                                                            Hx, Hy, Hz,
                                                            Cax, Cbx,
                                                            Cay, Cby,
                                                            Caz, Cbz,
                                                            Dax, Dbx,
                                                            Day, Dby,
                                                            Daz, Dbz,
                                                            Jx, Jy, Jz,
                                                            Mx, My, Mz,
                                                            _dx, 
                                                            _Nx, _Ny, _Nz,
                                                            num_mountains_X, num_valleys_Y, num_para_Z, 
                                                            mountain_heads_X_d,
                                                            valley_heads_Y_d,
                                                            mountain_tails_X_d,
                                                            valley_tails_Y_d,
                                                            1, 0, 
                                                            block_size,
                                                            grid_size); 

    // phase 4. (v, v, *)
    grid_size = num_valleys_X * num_valleys_Y;
    updateEH_phase_shmem_EH_2D<<<grid_size, block_size>>>(Ex, Ey, Ez,
                                                            Hx, Hy, Hz,
                                                            Cax, Cbx,
                                                            Cay, Cby,
                                                            Caz, Cbz,
                                                            Dax, Dbx,
                                                            Day, Dby,
                                                            Daz, Dbz,
                                                            Jx, Jy, Jz,
                                                            Mx, My, Mz,
                                                            _dx, 
                                                            _Nx, _Ny, _Nz,
                                                            num_valleys_X, num_valleys_Y, num_para_Z, 
                                                            valley_heads_X_d,
                                                            valley_heads_Y_d,
                                                            valley_tails_X_d,
                                                            valley_tails_Y_d,
                                                            0, 0, 
                                                            block_size,
                                                            grid_size); 
     
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
  std::cout << "gpu runtime (2-D mapping, (dt on XY, pt on Z) shared memory on EH): " << std::chrono::duration<double>(end-start).count() << "s\n"; 
  std::cout << "gpu performance: " << (_Nx * _Ny * _Nz / 1.0e6 * num_timesteps) / std::chrono::duration<double>(end-start).count() << "Mcells/s\n";

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

  CUDACHECK(cudaFree(mountain_heads_X_d));
  CUDACHECK(cudaFree(mountain_tails_X_d));
  CUDACHECK(cudaFree(mountain_heads_Y_d));
  CUDACHECK(cudaFree(mountain_tails_Y_d));
  CUDACHECK(cudaFree(valley_heads_X_d));
  CUDACHECK(cudaFree(valley_tails_X_d));
  CUDACHECK(cudaFree(valley_heads_Y_d));
  CUDACHECK(cudaFree(valley_tails_Y_d));


}

void gDiamond::update_FDTD_gpu_simulation_2_D_globalmem(size_t num_timesteps) { // 2-D mapping, each thread finish the entire Z dimension,

  std::cout << "running update_FDTD_gpu_simulation_2_D_globalmem\n";

  // clear source Mz for experiments
  _Mz.clear();

  // transfer source
  for(size_t t=0; t<num_timesteps; t++) {
    float Mz_value = M_source_amp * std::sin(SOURCE_OMEGA * t * dt);
    _Mz[_source_idx] = Mz_value;
  }

  size_t max_phases = 8;
  std::vector<int> mountain_heads_X;
  std::vector<int> mountain_tails_X;
  std::vector<int> mountain_heads_Y;
  std::vector<int> mountain_tails_Y;
  std::vector<int> valley_heads_X;
  std::vector<int> valley_tails_X;
  std::vector<int> valley_heads_Y;
  std::vector<int> valley_tails_Y;
  _setup_diamond_tiling_gpu(BLX_GPU_PT, BLY_GPU_PT, BLZ_GPU_PT, BLT_GPU_PT, max_phases);

  for(auto range : _Eranges_phases_X[0][0]) { 
    mountain_heads_X.push_back(range.first);
    mountain_tails_X.push_back(range.second);
  }
  for(auto range : _Eranges_phases_Y[0][0]) { 
    mountain_heads_Y.push_back(range.first);
    mountain_tails_Y.push_back(range.second);
  }
  for(auto range : _Hranges_phases_X[1][BLT_GPU_PT-1]) { 
    valley_heads_X.push_back(range.first);
    valley_tails_X.push_back(range.second);
  }
  for(auto range : _Hranges_phases_Y[2][BLT_GPU_PT-1]) { 
    valley_heads_Y.push_back(range.first);
    valley_tails_Y.push_back(range.second);
  }

  size_t num_mountains_X = mountain_heads_X.size();
  size_t num_mountains_Y = mountain_heads_Y.size();
  size_t num_valleys_X = valley_heads_X.size();
  size_t num_valleys_Y = valley_heads_Y.size();

  size_t block_size = BLX_GPU_PT * BLY_GPU_PT;
  size_t grid_size;

  for(size_t t=0; t<num_timesteps/BLT_GPU_PT; t++) {
    
    // phase 1. (m, m, *)
    grid_size = num_mountains_X * num_mountains_Y;
    _updateEH_phase_seq_2D(_Ex_simu, _Ey_simu, _Ez_simu,
                           _Hx_simu, _Hy_simu, _Hz_simu,
                           _Cax, _Cbx,
                           _Cay, _Cby,
                           _Caz, _Cbz,
                           _Dax, _Dbx,
                           _Day, _Dby,
                           _Daz, _Dbz,
                           _Jx, _Jy, _Jz,
                           _Mx, _My, _Mz,
                           _dx, 
                           _Nx, _Ny, _Nz,
                           num_mountains_X, num_mountains_Y, 
                           mountain_heads_X,
                           mountain_heads_Y,
                           mountain_tails_X,
                           mountain_tails_Y,
                           1, 1, 
                           block_size,
                           grid_size); 

    // phase 2. (v, m, *)
    grid_size = num_valleys_X * num_mountains_Y;
    _updateEH_phase_seq_2D(_Ex_simu, _Ey_simu, _Ez_simu,
                           _Hx_simu, _Hy_simu, _Hz_simu,
                           _Cax, _Cbx,
                           _Cay, _Cby,
                           _Caz, _Cbz,
                           _Dax, _Dbx,
                           _Day, _Dby,
                           _Daz, _Dbz,
                           _Jx, _Jy, _Jz,
                           _Mx, _My, _Mz,
                           _dx, 
                           _Nx, _Ny, _Nz,
                           num_valleys_X, num_mountains_Y, 
                           valley_heads_X,
                           mountain_heads_Y,
                           valley_tails_X,
                           mountain_tails_Y,
                           0, 1, 
                           block_size,
                           grid_size); 

    // phase 3. (m, v, *)
    grid_size = num_mountains_X * num_valleys_Y;
    _updateEH_phase_seq_2D(_Ex_simu, _Ey_simu, _Ez_simu,
                           _Hx_simu, _Hy_simu, _Hz_simu,
                           _Cax, _Cbx,
                           _Cay, _Cby,
                           _Caz, _Cbz,
                           _Dax, _Dbx,
                           _Day, _Dby,
                           _Daz, _Dbz,
                           _Jx, _Jy, _Jz,
                           _Mx, _My, _Mz,
                           _dx, 
                           _Nx, _Ny, _Nz,
                           num_mountains_X, num_valleys_Y, 
                           mountain_heads_X,
                           valley_heads_Y,
                           mountain_tails_X,
                           valley_tails_Y,
                           1, 0, 
                           block_size,
                           grid_size); 

    // phase 4. (v, v, *)
    grid_size = num_valleys_X * num_valleys_Y;
    _updateEH_phase_seq_2D(_Ex_simu, _Ey_simu, _Ez_simu,
                           _Hx_simu, _Hy_simu, _Hz_simu,
                           _Cax, _Cbx,
                           _Cay, _Cby,
                           _Caz, _Cbz,
                           _Dax, _Dbx,
                           _Day, _Dby,
                           _Daz, _Dbz,
                           _Jx, _Jy, _Jz,
                           _Mx, _My, _Mz,
                           _dx, 
                           _Nx, _Ny, _Nz,
                           num_valleys_X, num_valleys_Y, 
                           valley_heads_X,
                           valley_heads_Y,
                           valley_tails_X,
                           valley_tails_Y,
                           0, 0, 
                           block_size,
                           grid_size); 
     
  }
}

void gDiamond::update_FDTD_gpu_simulation_2_D_shmem(size_t num_timesteps) { // 2-D mapping, each thread finish the entire Z dimension,

  std::cout << "running update_FDTD_gpu_simulation_2_D_shmem\n";

  // clear source Mz for experiments
  _Mz.clear();

  // transfer source
  for(size_t t=0; t<num_timesteps; t++) {
    float Mz_value = M_source_amp * std::sin(SOURCE_OMEGA * t * dt);
    _Mz[_source_idx] = Mz_value;
  }

  size_t max_phases = 8;
  std::vector<int> mountain_heads_X;
  std::vector<int> mountain_tails_X;
  std::vector<int> mountain_heads_Y;
  std::vector<int> mountain_tails_Y;
  std::vector<int> valley_heads_X;
  std::vector<int> valley_tails_X;
  std::vector<int> valley_heads_Y;
  std::vector<int> valley_tails_Y;
  _setup_diamond_tiling_gpu(BLX_GPU_PT, BLY_GPU_PT, BLZ_GPU_PT, BLT_GPU_PT, max_phases);

  for(auto range : _Eranges_phases_X[0][0]) { 
    mountain_heads_X.push_back(range.first);
    mountain_tails_X.push_back(range.second);
  }
  for(auto range : _Eranges_phases_Y[0][0]) { 
    mountain_heads_Y.push_back(range.first);
    mountain_tails_Y.push_back(range.second);
  }
  for(auto range : _Hranges_phases_X[1][BLT_GPU_PT-1]) { 
    valley_heads_X.push_back(range.first);
    valley_tails_X.push_back(range.second);
  }
  for(auto range : _Hranges_phases_Y[2][BLT_GPU_PT-1]) { 
    valley_heads_Y.push_back(range.first);
    valley_tails_Y.push_back(range.second);
  }

  size_t num_mountains_X = mountain_heads_X.size();
  size_t num_mountains_Y = mountain_heads_Y.size();
  size_t num_valleys_X = valley_heads_X.size();
  size_t num_valleys_Y = valley_heads_Y.size();

  size_t block_size = BLX_GPU_PT * BLY_GPU_PT;
  size_t grid_size;

  for(size_t t=0; t<num_timesteps/BLT_GPU_PT; t++) {

    // phase 1. (m, m, *)
    grid_size = num_mountains_X * num_mountains_Y;
    _updateEH_phase_seq_2D_shmem_EH(_Ex_simu, _Ey_simu, _Ez_simu,
                           _Hx_simu, _Hy_simu, _Hz_simu,
                           _Cax, _Cbx,
                           _Cay, _Cby,
                           _Caz, _Cbz,
                           _Dax, _Dbx,
                           _Day, _Dby,
                           _Daz, _Dbz,
                           _Jx, _Jy, _Jz,
                           _Mx, _My, _Mz,
                           _dx,
                           _Nx, _Ny, _Nz,
                           num_mountains_X, num_mountains_Y,
                           mountain_heads_X,
                           mountain_heads_Y,
                           mountain_tails_X,
                           mountain_tails_Y,
                           1, 1,
                           block_size,
                           grid_size);

    // phase 2. (v, m, *)
    grid_size = num_valleys_X * num_mountains_Y;
    _updateEH_phase_seq_2D_shmem_EH(_Ex_simu, _Ey_simu, _Ez_simu,
                           _Hx_simu, _Hy_simu, _Hz_simu,
                           _Cax, _Cbx,
                           _Cay, _Cby,
                           _Caz, _Cbz,
                           _Dax, _Dbx,
                           _Day, _Dby,
                           _Daz, _Dbz,
                           _Jx, _Jy, _Jz,
                           _Mx, _My, _Mz,
                           _dx,
                           _Nx, _Ny, _Nz,
                           num_valleys_X, num_mountains_Y,
                           valley_heads_X,
                           mountain_heads_Y,
                           valley_tails_X,
                           mountain_tails_Y,
                           0, 1,
                           block_size,
                           grid_size);

    // phase 3. (m, v, *)
    grid_size = num_mountains_X * num_valleys_Y;
    _updateEH_phase_seq_2D_shmem_EH(_Ex_simu, _Ey_simu, _Ez_simu,
                           _Hx_simu, _Hy_simu, _Hz_simu,
                           _Cax, _Cbx,
                           _Cay, _Cby,
                           _Caz, _Cbz,
                           _Dax, _Dbx,
                           _Day, _Dby,
                           _Daz, _Dbz,
                           _Jx, _Jy, _Jz,
                           _Mx, _My, _Mz,
                           _dx,
                           _Nx, _Ny, _Nz,
                           num_mountains_X, num_valleys_Y,
                           mountain_heads_X,
                           valley_heads_Y,
                           mountain_tails_X,
                           valley_tails_Y,
                           1, 0,
                           block_size,
                           grid_size);

    // phase 4. (v, v, *)
    grid_size = num_valleys_X * num_valleys_Y;
    _updateEH_phase_seq_2D_shmem_EH(_Ex_simu, _Ey_simu, _Ez_simu,
                           _Hx_simu, _Hy_simu, _Hz_simu,
                           _Cax, _Cbx,
                           _Cay, _Cby,
                           _Caz, _Cbz,
                           _Dax, _Dbx,
                           _Day, _Dby,
                           _Daz, _Dbz,
                           _Jx, _Jy, _Jz,
                           _Mx, _My, _Mz,
                           _dx,
                           _Nx, _Ny, _Nz,
                           num_valleys_X, num_valleys_Y,
                           valley_heads_X,
                           valley_heads_Y,
                           valley_tails_X,
                           valley_tails_Y,
                           0, 0,
                           block_size,
                           grid_size);
  }

}

void gDiamond::_updateEH_phase_seq_2D_shmem_EH(std::vector<float>& Ex, std::vector<float>& Ey, std::vector<float>& Ez,
                                               std::vector<float>& Hx, std::vector<float>& Hy, std::vector<float>& Hz,
                                               std::vector<float>& Cax, std::vector<float>& Cbx,
                                               std::vector<float>& Cay, std::vector<float>& Cby,
                                               std::vector<float>& Caz, std::vector<float>& Cbz,
                                               std::vector<float>& Dax, std::vector<float>& Dbx,
                                               std::vector<float>& Day, std::vector<float>& Dby,
                                               std::vector<float>& Daz, std::vector<float>& Dbz,
                                               std::vector<float>& Jx, std::vector<float>& Jy, std::vector<float>& Jz,
                                               std::vector<float>& Mx, std::vector<float>& My, std::vector<float>& Mz,
                                               float dx, 
                                               int Nx, int Ny, int Nz,
                                               int xx_num, int yy_num, 
                                               std::vector<int> xx_heads, 
                                               std::vector<int> yy_heads, 
                                               std::vector<int> xx_tails, 
                                               std::vector<int> yy_tails, 
                                               int m_or_v_X, int m_or_v_Y, 
                                               size_t block_size,
                                               size_t grid_size) {

  for(size_t block_id=0; block_id<grid_size; block_id++) {

    int xx = block_id % xx_num;
    int yy = block_id / xx_num;

    int num_zz = Nz - BLT_GPU_PT; // number of times to load shared memory

    // declare shared memory for each block
    float Ex_shmem[(BLX_GPU_PT + 1) * (BLY_GPU_PT + 1) * (BLT_GPU_PT + 1)];
    float Ey_shmem[(BLX_GPU_PT + 1) * (BLY_GPU_PT + 1) * (BLT_GPU_PT + 1)];
    float Ez_shmem[(BLX_GPU_PT + 1) * (BLY_GPU_PT + 1) * (BLT_GPU_PT + 1)];
    float Hx_shmem[(BLX_GPU_PT + 1) * (BLY_GPU_PT + 1) * (BLT_GPU_PT + 1)];
    float Hy_shmem[(BLX_GPU_PT + 1) * (BLY_GPU_PT + 1) * (BLT_GPU_PT + 1)];
    float Hz_shmem[(BLX_GPU_PT + 1) * (BLY_GPU_PT + 1) * (BLT_GPU_PT + 1)];

    for(int zz=0; zz<num_zz; zz++) {

      // load shared memory
      for(size_t thread_id=0; thread_id<block_size; thread_id++) {

        // thread index
        int local_x = thread_id % BLX_GPU_PT;
        int local_y = thread_id / BLX_GPU_PT;

        // global index
        int global_x = xx_heads[xx] + local_x; 
        int global_y = yy_heads[yy] + local_y;

        // shared index for H
        int shared_H_x = local_x + 1;
        int shared_H_y = local_y + 1;
        
        // shared index for E
        int shared_E_x = local_x;
        int shared_E_y = local_y;

        for(int local_z=0; local_z<BLT_GPU_PT+1; local_z++) { // each thread iterate Z dimension
          int global_z = local_z + zz;
          int shared_H_z = local_z;
          int shared_E_z = local_z;

          int global_idx = global_x + global_y * Nx + global_z * Nx * Ny;
          int shared_H_idx = shared_H_x + shared_H_y * BLX_EH_PT + shared_H_z * BLX_EH_PT * BLY_EH_PT;
          int shared_E_idx = shared_E_x + shared_E_y * BLX_EH_PT + shared_E_z * BLX_EH_PT * BLY_EH_PT;

          // load H
          if(global_x >= 0 && global_x < Nx && global_y >= 0 && global_y < Ny && global_z >= 0 && global_z < Nz) {

            // load core
            Hx_shmem[shared_H_idx] = Hx[global_idx];
            Hy_shmem[shared_H_idx] = Hy[global_idx];
            Hz_shmem[shared_H_idx] = Hz[global_idx];

            // load HALO region
            if(local_x == 0 && global_x > 0) {
              Hz_shmem[shared_H_x - 1 + shared_H_y * BLX_EH_PT + shared_H_z * BLX_EH_PT * BLY_EH_PT] = Hz[global_x - 1 + global_y * Nx + global_z * Nx * Ny];
              Hy_shmem[shared_H_x - 1 + shared_H_y * BLX_EH_PT + shared_H_z * BLX_EH_PT * BLY_EH_PT] = Hy[global_x - 1 + global_y * Nx + global_z * Nx * Ny];
              
            }
            if(local_y == 0 && global_y > 0) {
              Hx_shmem[shared_H_x + (shared_H_y - 1) * BLX_EH_PT + shared_H_z * BLX_EH_PT * BLY_EH_PT] = Hx[global_x + (global_y - 1) * Nx + global_z * Nx * Ny];
              Hz_shmem[shared_H_x + (shared_H_y - 1) * BLX_EH_PT + shared_H_z * BLX_EH_PT * BLY_EH_PT] = Hz[global_x + (global_y - 1) * Nx + global_z * Nx * Ny];
            }
          }

          // load E
          if(global_x >= 0 && global_x < Nx && global_y >= 0 && global_y < Ny && global_z >= 0 && global_z < Nz) {

            Ex_shmem[shared_E_idx] = Ex[global_idx];
            Ey_shmem[shared_E_idx] = Ey[global_idx];
            Ez_shmem[shared_E_idx] = Ez[global_idx];

            // load HALO region
            if(local_x == BLX_GPU - 1 && global_x < Nx - 1) {
              Ez_shmem[shared_E_x + 1 + shared_E_y * BLX_EH_PT + shared_E_z * BLX_EH_PT * BLY_EH_PT] = Ez[global_x + 1 + global_y * Nx + global_z * Nx * Ny];
              Ey_shmem[shared_E_x + 1 + shared_E_y * BLX_EH_PT + shared_E_z * BLX_EH_PT * BLY_EH_PT] = Ey[global_x + 1 + global_y * Nx + global_z * Nx * Ny];
            }
            if(local_y == BLY_GPU - 1 && global_y < Ny - 1) {
              Ex_shmem[shared_E_x + (shared_E_y + 1) * BLX_EH_PT + shared_E_z * BLX_EH_PT * BLY_EH_PT] = Ex[global_x + (global_y + 1) * Nx + global_z * Nx * Ny];
              Ez_shmem[shared_E_x + (shared_E_y + 1) * BLX_EH_PT + shared_E_z * BLX_EH_PT * BLY_EH_PT] = Ez[global_x + (global_y + 1) * Nx + global_z * Nx * Ny];
            }
          } 
        }
      }

      // do calculation
      int z_start = (zz == 0)? 0 : 4;  
      int z_bound = (zz == 0 || zz == num_zz - 1)? BLT_GPU_PT + 1 : 1;  
      for(size_t t=0; t<BLT_GPU_PT; t++) {
        for(int local_z=z_start; local_z<z_start+z_bound; local_z++) { // each thread iterate Z dimension
          int calculate_Ex = 1; // calculate this E tile or not
          int calculate_Hx = 1; // calculate this H tile or not
          int calculate_Ey = 1;
          int calculate_Hy = 1;

          // {Ehead, Etail, Hhead, Htail}
          std::vector<int> indices_X = _get_head_tail(BLX_GPU_PT, BLT_GPU_PT,
                                                      xx_heads, xx_tails,
                                                      xx, t,
                                                      m_or_v_X,
                                                      Nx,
                                                      &calculate_Ex, &calculate_Hx);

          std::vector<int> indices_Y = _get_head_tail(BLY_GPU_PT, BLT_GPU_PT,
                                                      yy_heads, yy_tails,
                                                      yy, t,
                                                      m_or_v_Y,
                                                      Ny,
                                                      &calculate_Ey, &calculate_Hy);

          // update E
          if(calculate_Ex & calculate_Ey) {
            for(size_t thread_id=0; thread_id<block_size; thread_id++) {
              int local_x = thread_id % BLX_GPU_PT;
              int local_y = thread_id / BLX_GPU_PT;

              // Ehead is offset
              int global_x = indices_X[0] + local_x; // Global X coordinate
              int global_y = indices_Y[0] + local_y; // Global Y coordinate

              int shared_E_z = _get_z_planeE_shmem(t, local_z, Nz);
              int global_E_z = shared_E_z + zz;

              if(global_x >= 1 && global_x <= Nx-2 && global_y >= 1 && global_y <= Ny-2 && global_E_z >= 1 && global_E_z <= Nz-2 &&
                global_x <= indices_X[1] &&
                global_y <= indices_Y[1] &&
                shared_E_z != -1) {

                // need to recalculate shared indices
                int Eoffset_X = indices_X[0] - xx_heads[xx];
                int Eoffset_Y = indices_Y[0] - yy_heads[yy];

                int Hoffset_X = indices_X[2] - xx_heads[xx];
                int Hoffset_Y = indices_Y[2] - yy_heads[yy];

                int shared_H_x = (m_or_v_X == 0 && xx != 0)? local_x + 1 + Hoffset_X + 1 : local_x + 1 + Hoffset_X;
                int shared_H_y = (m_or_v_Y == 0 && yy != 0)? local_y + 1 + Hoffset_Y + 1 : local_y + 1 + Hoffset_Y;

                int shared_E_x = local_x + Eoffset_X;
                int shared_E_y = local_y + Eoffset_Y;

                // notice that for H in Z dimension, it is using shared_E_z
                int s_H_idx = shared_H_x + shared_H_y * BLX_EH + shared_E_z * BLX_EH * BLY_EH; // shared memory idx for H
                int s_E_idx = shared_E_x + shared_E_y * BLX_EH + shared_E_z * BLX_EH * BLY_EH; // shared memory idx for E

                int g_idx = global_x + global_y * Nx + global_E_z * Nx * Ny; // global idx

                // Ex[g_idx] = Cax[g_idx] * Ex[g_idx] + Cbx[g_idx] *
                //           ((Hz[g_idx] - Hz[g_idx - Nx]) - (Hy[g_idx] - Hy[g_idx - Nx * Ny]) - Jx[g_idx] * dx);
                // Ey[g_idx] = Cay[g_idx] * Ey[g_idx] + Cby[g_idx] *
                //           ((Hx[g_idx] - Hx[g_idx - Nx * Ny]) - (Hz[g_idx] - Hz[g_idx - 1]) - Jy[g_idx] * dx);
                // Ez[g_idx] = Caz[g_idx] * Ez[g_idx] + Cbz[g_idx] *
                //           ((Hy[g_idx] - Hy[g_idx - 1]) - (Hx[g_idx] - Hx[g_idx - Nx]) - Jz[g_idx] * dx);

                Ex_shmem[s_E_idx] = Cax[g_idx] * Ex_shmem[s_E_idx] + Cbx[g_idx] *
                    ((Hz_shmem[s_H_idx] - Hz_shmem[s_H_idx - BLX_EH]) - (Hy_shmem[s_H_idx] - Hy_shmem[s_H_idx - BLX_EH * BLY_EH]) - Jx[g_idx] * dx);
                Ey_shmem[s_E_idx] = Cay[g_idx] * Ey_shmem[s_E_idx] + Cby[g_idx] *
                          ((Hx_shmem[s_H_idx] - Hx_shmem[s_H_idx - BLX_EH * BLY_EH]) - (Hz_shmem[s_H_idx] - Hz_shmem[s_H_idx - 1]) - Jy[g_idx] * dx);
                Ez_shmem[s_E_idx] = Caz[g_idx] * Ez_shmem[s_E_idx] + Cbz[g_idx] *
                          ((Hy_shmem[s_H_idx] - Hy_shmem[s_H_idx - 1]) - (Hx_shmem[s_H_idx] - Hx_shmem[s_H_idx - BLX_EH]) - Jz[g_idx] * dx);
              }
            }
          }

          // update H
          if(calculate_Hx & calculate_Hy) {
            for(size_t thread_id=0; thread_id<block_size; thread_id++) {
              int local_x = thread_id % BLX_GPU_PT;
              int local_y = thread_id / BLX_GPU_PT;

              // Hhead is offset
              int global_x = indices_X[2] + local_x; // Global X coordinate
              int global_y = indices_Y[2] + local_y; // Global Y coordinate

              int shared_H_z = _get_z_planeH_shmem(t, local_z, Nz);
              int global_H_z = shared_H_z + zz;

              if(global_x >= 1 && global_x <= Nx-2 && global_y >= 1 && global_y <= Ny-2 && global_H_z >= 1 && global_H_z <= Nz-2 &&
                global_x <= indices_X[3] &&
                global_y <= indices_Y[3] &&
                shared_H_z != -1) {

                // need to recalculate shared indices
                int Eoffset_X = indices_X[0] - xx_heads[xx];
                int Eoffset_Y = indices_Y[0] - yy_heads[yy];

                int Hoffset_X = indices_X[2] - xx_heads[xx];
                int Hoffset_Y = indices_Y[2] - yy_heads[yy];

                int shared_H_x = local_x + 1 + Hoffset_X;
                int shared_H_y = local_y + 1 + Hoffset_Y;

                int shared_E_x = (m_or_v_X == 0 && xx != 0)? local_x + Eoffset_X - 1 : local_x + Eoffset_X;
                int shared_E_y = (m_or_v_Y == 0 && yy != 0)? local_y + Eoffset_Y - 1 : local_y + Eoffset_Y;

                // notice that for E in Z dimension, it is using shared_H_z
                int s_H_idx = shared_H_x + shared_H_y * BLX_EH + shared_H_z * BLX_EH * BLY_EH; // shared memory idx for H
                int s_E_idx = shared_E_x + shared_E_y * BLX_EH + shared_H_z * BLX_EH * BLY_EH; // shared memory idx for E

                int g_idx = global_x + global_y * Nx + global_H_z * Nx * Ny; // global idx

                // Hx[g_idx] = Dax[g_idx] * Hx[g_idx] + Dbx[g_idx] *
                //           ((Ey[g_idx + Nx * Ny] - Ey[g_idx]) - (Ez[g_idx + Nx] - Ez[g_idx]) - Mx[g_idx] * dx);
                // Hy[g_idx] = Day[g_idx] * Hy[g_idx] + Dby[g_idx] *
                //           ((Ez[g_idx + 1] - Ez[g_idx]) - (Ex[g_idx + Nx * Ny] - Ex[g_idx]) - My[g_idx] * dx);
                // Hz[g_idx] = Daz[g_idx] * Hz[g_idx] + Dbz[g_idx] *
                //           ((Ex[g_idx + Nx] - Ex[g_idx]) - (Ey[g_idx + 1] - Ey[g_idx]) - Mz[g_idx] * dx);

                Hx_shmem[s_H_idx] = Dax[g_idx] * Hx_shmem[s_H_idx] + Dbx[g_idx] *
                    ((Ey_shmem[s_E_idx + BLX_EH * BLY_EH] - Ey_shmem[s_E_idx]) - (Ez_shmem[s_E_idx + BLX_EH] - Ez_shmem[s_E_idx]) - Mx[g_idx] * dx);
                Hy_shmem[s_H_idx] = Day[g_idx] * Hy_shmem[s_H_idx] + Dby[g_idx] *
                          ((Ez_shmem[s_E_idx + 1] - Ez_shmem[s_E_idx]) - (Ex_shmem[s_E_idx + BLX_EH * BLY_EH] - Ex_shmem[s_E_idx]) - My[g_idx] * dx);
                Hz_shmem[s_H_idx] = Daz[g_idx] * Hz_shmem[s_H_idx] + Dbz[g_idx] *
                          ((Ex_shmem[s_E_idx + BLX_EH] - Ex_shmem[s_E_idx]) - (Ey_shmem[s_E_idx + 1] - Ey_shmem[s_E_idx]) - Mz[g_idx] * dx);
              }
            }
          }
        }
      }

      // load back to global mem
      // store E, H to global memory, no HALO needed
      for(int local_z=0; local_z<BLT_GPU_PT+1; local_z++) {
        int global_z = local_z + zz;
        for(size_t thread_id=0; thread_id<block_size; thread_id++) {
          int local_x = thread_id % BLX_GPU;                     // X coordinate within the tile
          int local_y = (thread_id / BLX_GPU) % BLY_GPU;     // Y coordinate within the tile

          // Hhead is offset
          int global_x = xx_heads[xx] + local_x; // Global X coordinate
          int global_y = yy_heads[yy] + local_y; // Global Y coordinate

          if(global_x >= 1 && global_x <= Nx-2 && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2 &&
            global_x <= xx_tails[xx] &&
            global_y <= yy_tails[yy]) {

            int shared_H_x = local_x + 1;
            int shared_H_y = local_y + 1;
            int shared_H_z = local_z;

            int shared_E_x = local_x;
            int shared_E_y = local_y;
            int shared_E_z = local_z;

            int s_H_idx = shared_H_x + shared_H_y * BLX_EH + shared_H_z * BLX_EH * BLY_EH; // shared memory idx for H
            int s_E_idx = shared_E_x + shared_E_y * BLX_EH + shared_E_z * BLX_EH * BLY_EH; // shared memory idx for E

            int g_idx = global_x + global_y * Nx + global_z * Nx * Ny; // global idx

            Ex[g_idx] = Ex_shmem[s_E_idx];
            Ey[g_idx] = Ey_shmem[s_E_idx];
            Ez[g_idx] = Ez_shmem[s_E_idx];
            Hx[g_idx] = Hx_shmem[s_H_idx];
            Hy[g_idx] = Hy_shmem[s_H_idx];
            Hz[g_idx] = Hz_shmem[s_H_idx];
          }
        }
      }
    }
  }

}



void gDiamond::_updateEH_phase_seq_2D(std::vector<float>& Ex, std::vector<float>& Ey, std::vector<float>& Ez,
                                      std::vector<float>& Hx, std::vector<float>& Hy, std::vector<float>& Hz,
                                      std::vector<float>& Cax, std::vector<float>& Cbx,
                                      std::vector<float>& Cay, std::vector<float>& Cby,
                                      std::vector<float>& Caz, std::vector<float>& Cbz,
                                      std::vector<float>& Dax, std::vector<float>& Dbx,
                                      std::vector<float>& Day, std::vector<float>& Dby,
                                      std::vector<float>& Daz, std::vector<float>& Dbz,
                                      std::vector<float>& Jx, std::vector<float>& Jy, std::vector<float>& Jz,
                                      std::vector<float>& Mx, std::vector<float>& My, std::vector<float>& Mz,
                                      float dx, 
                                      int Nx, int Ny, int Nz,
                                      int xx_num, int yy_num, 
                                      std::vector<int> xx_heads, 
                                      std::vector<int> yy_heads, 
                                      std::vector<int> xx_tails, 
                                      std::vector<int> yy_tails, 
                                      int m_or_v_X, int m_or_v_Y, 
                                      size_t block_size,
                                      size_t grid_size) {

  for(size_t block_id=0; block_id<grid_size; block_id++) {
    int xx = block_id % xx_num;
    int yy = (block_id % (xx_num * yy_num)) / xx_num;

    int num_zz = Nz + BLT_GPU_PT; 

    for(int zz=0; zz<num_zz; zz++) {

      for(size_t t=0; t<BLT_GPU_PT; t++) {
        int calculate_Ex = 1; // calculate this E tile or not
        int calculate_Hx = 1; // calculate this H tile or not
        int calculate_Ey = 1; 
        int calculate_Hy = 1; 

        // {Ehead, Etail, Hhead, Htail}
        std::vector<int> indices_X = _get_head_tail(BLX_GPU_PT, BLT_GPU_PT,
                                                    xx_heads, xx_tails,
                                                    xx, t,
                                                    m_or_v_X,
                                                    Nx,
                                                    &calculate_Ex, &calculate_Hx);

        std::vector<int> indices_Y = _get_head_tail(BLY_GPU_PT, BLT_GPU_PT,
                                                    yy_heads, yy_tails,
                                                    yy, t,
                                                    m_or_v_Y,
                                                    Ny,
                                                    &calculate_Ey, &calculate_Hy);

        int global_z_E = _get_z_planeE(t, zz, Nz);
        int global_z_H = _get_z_planeH(t, zz, Nz); 

        // update E
        if(calculate_Ex & calculate_Ey) {
          for(size_t thread_id=0; thread_id<block_size; thread_id++) {
            int local_x = thread_id % BLX_GPU_PT;                     // X coordinate within the tile
            int local_y = (thread_id / BLX_GPU_PT) % BLY_GPU_PT;     // Y coordinate within the tile

            // Ehead is offset
            int global_x = indices_X[0] + local_x; // Global X coordinate
            int global_y = indices_Y[0] + local_y; // Global Y coordinate

            if(global_x >= 1 && global_x <= Nx-2 && global_y >= 1 && global_y <= Ny-2 && 
              global_x <= indices_X[1] &&
              global_y <= indices_Y[1] &&
              global_z_E != -1) {
              int g_idx = global_x + global_y * Nx + global_z_E * Nx * Ny; // global idx

              Ex[g_idx] = Cax[g_idx] * Ex[g_idx] + Cbx[g_idx] *
                        ((Hz[g_idx] - Hz[g_idx - Nx]) - (Hy[g_idx] - Hy[g_idx - Nx * Ny]) - Jx[g_idx] * dx);
              Ey[g_idx] = Cay[g_idx] * Ey[g_idx] + Cby[g_idx] *
                        ((Hx[g_idx] - Hx[g_idx - Nx * Ny]) - (Hz[g_idx] - Hz[g_idx - 1]) - Jy[g_idx] * dx);
              Ez[g_idx] = Caz[g_idx] * Ez[g_idx] + Cbz[g_idx] *
                        ((Hy[g_idx] - Hy[g_idx - 1]) - (Hx[g_idx] - Hx[g_idx - Nx]) - Jz[g_idx] * dx);
            }
          }
        }

        // update H
        if(calculate_Hx & calculate_Hy) {
          for(size_t thread_id=0; thread_id<block_size; thread_id++) {
            int local_x = thread_id % BLX_GPU_PT;                     // X coordinate within the tile
            int local_y = (thread_id / BLX_GPU_PT) % BLY_GPU_PT;     // Y coordinate within the tile

            // Hhead is offset
            int global_x = indices_X[2] + local_x; // Global X coordinate
            int global_y = indices_Y[2] + local_y; // Global Y coordinate

            if(global_x >= 1 && global_x <= Nx-2 && global_y >= 1 && global_y <= Ny-2 &&
              global_x <= indices_X[3] &&
              global_y <= indices_Y[3] &&
              global_z_H != -1) {
              int g_idx = global_x + global_y * Nx + global_z_H * Nx * Ny; // global idx

              Hx[g_idx] = Dax[g_idx] * Hx[g_idx] + Dbx[g_idx] *
                        ((Ey[g_idx + Nx * Ny] - Ey[g_idx]) - (Ez[g_idx + Nx] - Ez[g_idx]) - Mx[g_idx] * dx);
              Hy[g_idx] = Day[g_idx] * Hy[g_idx] + Dby[g_idx] *
                        ((Ez[g_idx + 1] - Ez[g_idx]) - (Ex[g_idx + Nx * Ny] - Ex[g_idx]) - My[g_idx] * dx);
              Hz[g_idx] = Daz[g_idx] * Hz[g_idx] + Dbz[g_idx] *
                        ((Ex[g_idx + Nx] - Ex[g_idx]) - (Ey[g_idx + 1] - Ey[g_idx]) - Mz[g_idx] * dx);
            }
          }
        }

      }
    }
  }

}

void gDiamond::update_FDTD_gpu_simulation_1_D_pt(size_t num_timesteps) { // CPU single thread 1-D simulation of GPU workflow 
  
  // write 1 dimension just to check
  std::vector<float> E_simu(_Nz, 1);
  std::vector<float> H_simu(_Nz, 1);
  std::vector<float> E_seq(_Nz, 1);
  std::vector<float> H_seq(_Nz, 1);
  size_t total_timesteps = 4;

  // seq version
  for(size_t t=0; t<total_timesteps; t++) {

    // update E
    for(size_t z=1; z<_Nz-1; z++) {
      E_seq[z] = H_seq[z-1] + H_seq[z] * 2; 
    }

    std::cout << "t = " << t << ", E_seq =";
    for(size_t z=0; z<_Nz; z++) {
      std::cout << E_seq[z] << " ";
    }
    std::cout << "\n";

    // update H 
    for(size_t z=1; z<_Nz-1; z++) {
      H_seq[z] = E_seq[z+1] + E_seq[z] * 2; 
    }
  }

  // tiling version
  size_t num_zz = _Nz + BLT_GPU_PT;
  for(size_t tt=0; tt<total_timesteps/BLT_GPU_PT; tt++) {
    for(size_t zz=0; zz<num_zz; zz++) {
      for(size_t t=0; t<BLT_GPU_PT; t++) {
        int z_E = _get_z_planeE(t, zz, _Nz);
        int z_H = _get_z_planeH(t, zz, _Nz); 
        if(z_E != -1) {
          E_simu[z_E] = H_simu[z_E-1] + H_simu[z_E] * 2; 
        }
        if(z_H != -1) {
          H_simu[z_H] = E_simu[z_H+1] + E_simu[z_H] * 2; 
        }
      }
    }
  }

  /*
  size_t num_zz = _Nz - BLT_GPU_PT;
  for(size_t tt=0; tt<total_timesteps/BLT_GPU_PT; tt++) {
    for(size_t zz=0; zz<num_zz; zz++) {
      size_t z_start = (zz == 0)? 0 : 4;  
      size_t z_bound = (zz == 0 || zz == num_zz - 1)? BLT_GPU_PT + 1 : 1;  
      for(size_t z=z_start; z<z_start+z_bound; z++) {
        std::cout << "z+zz = " << z+zz << "\n";
        for(size_t t=0; t<BLT_GPU_PT; t++) {
          int z_E = _get_z_planeE(t, z+zz, _Nz);
          int z_H = _get_z_planeH(t, z+zz, _Nz); 
          if(z_E != -1) {
            E_simu[z_E] = H_simu[z_E-1] + H_simu[z_E] * 2; 
          }
          if(z_H != -1) {
            H_simu[z_H] = E_simu[z_H+1] + E_simu[z_H] * 2; 
          }
        }
      }
    }
  }
  */
  
  std::cout << "E_seq = ";
  for(size_t z=0; z<_Nz; z++) {
    std::cout << E_seq[z] << " ";
  }
  std::cout << "\n";

  std::cout << "E_simu = ";
  for(size_t z=0; z<_Nz; z++) {
    std::cout << E_simu[z] << " ";
  }
  std::cout << "\n";

  for(size_t z=0; z<_Nz; z++) {
    if(E_seq[z] != E_simu[z] || H_seq[z] != H_simu[z]) {
      std::cerr << "1-D demo results mismatch.\n";
      std::exit(EXIT_FAILURE);
    }
  }
}

void gDiamond::update_FDTD_gpu_simulation_1_D_pt_shmem(size_t num_timesteps) { // CPU single thread 1-D simulation of GPU workflow 
  
  // write 1 dimension just to check
  std::vector<float> E_simu(_Nz, 1);
  std::vector<float> H_simu(_Nz, 1);
  std::vector<float> E_seq(_Nz, 1);
  std::vector<float> H_seq(_Nz, 1);
  int total_timesteps = 4;

  int Nz = _Nz;
  // seq version
  for(int t=0; t<total_timesteps; t++) {

    // update E
    for(int z=1; z<Nz-1; z++) {
      E_seq[z] = H_seq[z-1] + H_seq[z] * 2; 
    }

    std::cout << "t = " << t << ", E_seq =";
    for(int z=0; z<Nz; z++) {
      std::cout << E_seq[z] << " ";
    }
    std::cout << "\n";

    // update H 
    for(int z=1; z<Nz-1; z++) {
      H_seq[z] = E_seq[z+1] + E_seq[z] * 2; 
    }
  }

  // tiling version
  // int num_z = Nz + BLT_GPU_PT; // total number of z tiles

  int num_zz = Nz - BLT_GPU_PT;
  for(int tt=0; tt<total_timesteps/BLT_GPU_PT; tt++) {
    for(int zz=0; zz<num_zz; zz++) {

      float E_shmem[BLT_GPU_PT + 1];
      float H_shmem[BLT_GPU_PT + 1];

      // load shmem
      for(int z=0; z<BLT_GPU_PT+1; z++) {
        E_shmem[z] = E_simu[z+zz]; 
        H_shmem[z] = H_simu[z+zz];
      }

      int z_start = (zz == 0)? 0 : 4;  
      int z_bound = (zz == 0 || zz == num_zz - 1)? BLT_GPU_PT + 1 : 1;  
      for(int t=0; t<BLT_GPU_PT; t++) {
        for(int z=z_start; z<z_start+z_bound; z++) {
          int s_z_E = _get_z_planeE_shmem(t, z, Nz);
          int s_z_H = _get_z_planeH_shmem(t, z, Nz); 
          // if(zz == num_zz - 1 && s_z_E != -1 && s_z_H != -1) {
          //   std::cout << "z = " << z << ", t = " << t << ", s_z_E = " << s_z_E << ", s_z_H = " << s_z_H << "\n";
          // }
          int g_z_E = s_z_E + zz;
          int g_z_H = s_z_H + zz;
          if(s_z_E != -1 && g_z_E >= 1 && g_z_E <= Nz-2) {
            E_shmem[s_z_E] = H_shmem[s_z_E-1] + H_shmem[s_z_E] * 2; 
            // E_simu[g_z_E] = H_simu[g_z_E-1] + H_simu[g_z_E] * 2; 
          }
          if(s_z_H != -1 && g_z_H >= 1 && g_z_H <= Nz-2) {
            H_shmem[s_z_H] = E_shmem[s_z_H+1] + E_shmem[s_z_H] * 2; 
            // H_simu[g_z_H] = E_simu[g_z_H+1] + E_simu[g_z_H] * 2; 
          }
        }
      }

      // store globalmem
      for(int z=0; z<BLT_GPU_PT+1; z++) {
        E_simu[z+zz] = E_shmem[z]; 
        H_simu[z+zz] = H_shmem[z];
      }

      /*
      int z_start = (zz == 0)? 0 : 4;  
      int z_bound = (zz == 0 || zz == num_zz - 1)? BLT_GPU_PT + 1 : 1;  
      for(int z=z_start; z<z_start+z_bound; z++) {
        for(int t=0; t<BLT_GPU_PT; t++) {
          int g_z_E = _get_z_planeE(t, z+zz, Nz);
          int g_z_H = _get_z_planeH(t, z+zz, Nz); 
          if(g_z_E != -1) {
            E_simu[g_z_E] = H_simu[g_z_E-1] + H_simu[g_z_E] * 2; 
          }
          if(g_z_H != -1) {
            H_simu[g_z_H] = E_simu[g_z_H+1] + E_simu[g_z_H] * 2; 
          }
        }
      }
      */


    }
  }
  
  std::cout << "E_seq = ";
  for(int z=0; z<Nz; z++) {
    std::cout << E_seq[z] << " ";
  }
  std::cout << "\n";

  std::cout << "E_simu = ";
  for(int z=0; z<Nz; z++) {
    std::cout << E_simu[z] << " ";
  }
  std::cout << "\n";

  for(int z=0; z<Nz; z++) {
    if(E_seq[z] != E_simu[z] || H_seq[z] != H_simu[z]) {
      std::cerr << "1-D demo results mismatch.\n";
      std::exit(EXIT_FAILURE);
    }
  }
}

int gDiamond::_get_z_planeE(int t, int zz, int Nz) {
  int result = zz - t;
  // return (result >= 0 && result <= Nz - 1)? result : -1; 
  return (result >= 1 && result <= Nz - 2)? result : -1; 
}

int gDiamond::_get_z_planeH(int t, int zz, int Nz) {
  int result = zz - t - 1;
  // return (result >= 0 && result <= Nz - 1)? result : -1;
  return (result >= 1 && result <= Nz - 2)? result : -1;
}

int gDiamond::_get_z_planeE_shmem(int t, int zz, int Nz) {
  int result = zz - t;
  return (result >= 0 && result <= Nz - 1)? result : -1; 
}

int gDiamond::_get_z_planeH_shmem(int t, int zz, int Nz) {
  int result = zz - t - 1;
  return (result >= 0 && result <= Nz - 1)? result : -1;
}



void gDiamond::update_FDTD_gpu_fuse_kernel_globalmem_pt(size_t num_timesteps) { // 2-D mapping, using diamond tiling on X, Y dimension to fuse kernels, 

  size_t max_phases = 8;
  std::vector<int> mountain_heads_X;
  std::vector<int> mountain_tails_X;
  std::vector<int> mountain_heads_Y;
  std::vector<int> mountain_tails_Y;
  std::vector<int> mountain_heads_Z;
  std::vector<int> mountain_tails_Z;
  std::vector<int> valley_heads_X;
  std::vector<int> valley_tails_X;
  std::vector<int> valley_heads_Y;
  std::vector<int> valley_tails_Y;
  std::vector<int> valley_heads_Z;
  std::vector<int> valley_tails_Z;
  _setup_diamond_tiling_gpu(BLX_GPU_PT, BLY_GPU_PT, BLZ_GPU_PT, BLT_GPU_PT, max_phases);

  for(auto range : _Eranges_phases_X[0][0]) { 
    mountain_heads_X.push_back(range.first);
    mountain_tails_X.push_back(range.second);
  }
  for(auto range : _Eranges_phases_Y[0][0]) { 
    mountain_heads_Y.push_back(range.first);
    mountain_tails_Y.push_back(range.second);
  }
  // std::cout << "valley X range = ";
  for(auto range : _Hranges_phases_X[1][BLT_GPU_PT-1]) { 
    // std::cout << "(" << range.first << ", " << range.second << ") ";
    valley_heads_X.push_back(range.first);
    valley_tails_X.push_back(range.second);
  }
  // std::cout << "\n";
  for(auto range : _Hranges_phases_Y[2][BLT_GPU_PT-1]) { 
    valley_heads_Y.push_back(range.first);
    valley_tails_Y.push_back(range.second);
  }

  size_t num_mountains_X = mountain_heads_X.size();
  size_t num_mountains_Y = mountain_heads_Y.size();
  size_t num_mountains_Z = mountain_heads_Z.size();
  size_t num_valleys_X = valley_heads_X.size();
  size_t num_valleys_Y = valley_heads_Y.size();

  // head and tail on device
  int *mountain_heads_X_d, *mountain_tails_X_d;
  int *mountain_heads_Y_d, *mountain_tails_Y_d;
  int *valley_heads_X_d, *valley_tails_X_d;
  int *valley_heads_Y_d, *valley_tails_Y_d;

  CUDACHECK(cudaMalloc(&mountain_heads_X_d, sizeof(int) * num_mountains_X));
  CUDACHECK(cudaMalloc(&mountain_tails_X_d, sizeof(int) * num_mountains_X));
  CUDACHECK(cudaMalloc(&mountain_heads_Y_d, sizeof(int) * num_mountains_Y));
  CUDACHECK(cudaMalloc(&mountain_tails_Y_d, sizeof(int) * num_mountains_Y));
  CUDACHECK(cudaMalloc(&valley_heads_X_d, sizeof(int) * num_valleys_X));
  CUDACHECK(cudaMalloc(&valley_tails_X_d, sizeof(int) * num_valleys_X));
  CUDACHECK(cudaMalloc(&valley_heads_Y_d, sizeof(int) * num_valleys_Y));
  CUDACHECK(cudaMalloc(&valley_tails_Y_d, sizeof(int) * num_valleys_Y));

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

  // transfer source
  for(size_t t=0; t<num_timesteps; t++) {
    float Mz_value = M_source_amp * std::sin(SOURCE_OMEGA * t * dt);
    CUDACHECK(cudaMemcpy(Mz + _source_idx, &Mz_value, sizeof(float), cudaMemcpyHostToDevice));
  }
  
  auto start = std::chrono::high_resolution_clock::now();

  // copy Ca, Cb, Da, Db
  CUDACHECK(cudaMemcpyAsync(Cax, _Cax.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Cay, _Cay.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Caz, _Caz.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Cbx, _Cbx.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Cby, _Cby.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Cbz, _Cbz.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Dax, _Dax.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Day, _Day.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Daz, _Daz.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Dbx, _Dbx.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Dby, _Dby.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Dbz, _Dbz.data(), sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyHostToDevice));

  // copy heads and tails
  CUDACHECK(cudaMemcpyAsync(mountain_heads_X_d, mountain_heads_X.data(), sizeof(int) * num_mountains_X, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(mountain_tails_X_d, mountain_tails_X.data(), sizeof(int) * num_mountains_X, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(mountain_heads_Y_d, mountain_heads_Y.data(), sizeof(int) * num_mountains_Y, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(mountain_tails_Y_d, mountain_tails_Y.data(), sizeof(int) * num_mountains_Y, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(valley_heads_X_d, valley_heads_X.data(), sizeof(int) * num_valleys_X, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(valley_tails_X_d, valley_tails_X.data(), sizeof(int) * num_valleys_X, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(valley_heads_Y_d, valley_heads_Y.data(), sizeof(int) * num_valleys_Y, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(valley_tails_Y_d, valley_tails_Y.data(), sizeof(int) * num_valleys_Y, cudaMemcpyHostToDevice));

  // set block size 
  size_t block_size = BLX_GPU_PT * BLY_GPU_PT;
  size_t grid_size;
  size_t num_para_Z = _Nz + BLT_GPU_PT; 
  
  for(size_t t=0; t<num_timesteps/BLT_GPU_PT; t++) {
 
    // phase 1. (m, m, *)
    grid_size = num_mountains_X * num_mountains_Y;
    updateEH_phase_global_mem_2D<<<grid_size, block_size>>>(Ex, Ey, Ez,
                                                            Hx, Hy, Hz,
                                                            Cax, Cbx,
                                                            Cay, Cby,
                                                            Caz, Cbz,
                                                            Dax, Dbx,
                                                            Day, Dby,
                                                            Daz, Dbz,
                                                            Jx, Jy, Jz,
                                                            Mx, My, Mz,
                                                            _dx, 
                                                            _Nx, _Ny, _Nz,
                                                            num_mountains_X, num_mountains_Y, num_para_Z, 
                                                            mountain_heads_X_d,
                                                            mountain_heads_Y_d,
                                                            mountain_tails_X_d,
                                                            mountain_tails_Y_d,
                                                            1, 1, 
                                                            block_size,
                                                            grid_size); 

    // phase 2. (v, m, *)
    grid_size = num_valleys_X * num_mountains_Y;
    updateEH_phase_global_mem_2D<<<grid_size, block_size>>>(Ex, Ey, Ez,
                                                            Hx, Hy, Hz,
                                                            Cax, Cbx,
                                                            Cay, Cby,
                                                            Caz, Cbz,
                                                            Dax, Dbx,
                                                            Day, Dby,
                                                            Daz, Dbz,
                                                            Jx, Jy, Jz,
                                                            Mx, My, Mz,
                                                            _dx, 
                                                            _Nx, _Ny, _Nz,
                                                            num_valleys_X, num_mountains_Y, num_para_Z, 
                                                            valley_heads_X_d,
                                                            mountain_heads_Y_d,
                                                            valley_tails_X_d,
                                                            mountain_tails_Y_d,
                                                            0, 1, 
                                                            block_size,
                                                            grid_size); 

    // phase 3. (m, v, *)
    grid_size = num_mountains_X * num_valleys_Y;
    updateEH_phase_global_mem_2D<<<grid_size, block_size>>>(Ex, Ey, Ez,
                                                            Hx, Hy, Hz,
                                                            Cax, Cbx,
                                                            Cay, Cby,
                                                            Caz, Cbz,
                                                            Dax, Dbx,
                                                            Day, Dby,
                                                            Daz, Dbz,
                                                            Jx, Jy, Jz,
                                                            Mx, My, Mz,
                                                            _dx, 
                                                            _Nx, _Ny, _Nz,
                                                            num_mountains_X, num_valleys_Y, num_para_Z, 
                                                            mountain_heads_X_d,
                                                            valley_heads_Y_d,
                                                            mountain_tails_X_d,
                                                            valley_tails_Y_d,
                                                            1, 0, 
                                                            block_size,
                                                            grid_size); 

    // phase 4. (v, v, *)
    grid_size = num_valleys_X * num_valleys_Y;
    updateEH_phase_global_mem_2D<<<grid_size, block_size>>>(Ex, Ey, Ez,
                                                            Hx, Hy, Hz,
                                                            Cax, Cbx,
                                                            Cay, Cby,
                                                            Caz, Cbz,
                                                            Dax, Dbx,
                                                            Day, Dby,
                                                            Daz, Dbz,
                                                            Jx, Jy, Jz,
                                                            Mx, My, Mz,
                                                            _dx, 
                                                            _Nx, _Ny, _Nz,
                                                            num_valleys_X, num_valleys_Y, num_para_Z, 
                                                            valley_heads_X_d,
                                                            valley_heads_Y_d,
                                                            valley_tails_X_d,
                                                            valley_tails_Y_d,
                                                            0, 0, 
                                                            block_size,
                                                            grid_size); 
     
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
  std::cout << "gpu runtime (2-D mapping, (dt on XY, pt on Z) global memory only): " << std::chrono::duration<double>(end-start).count() << "s\n"; 
  std::cout << "gpu performance: " << (_Nx * _Ny * _Nz / 1.0e6 * num_timesteps) / std::chrono::duration<double>(end-start).count() << "Mcells/s\n";

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

  CUDACHECK(cudaFree(mountain_heads_X_d));
  CUDACHECK(cudaFree(mountain_tails_X_d));
  CUDACHECK(cudaFree(mountain_heads_Y_d));
  CUDACHECK(cudaFree(mountain_tails_Y_d));
  CUDACHECK(cudaFree(valley_heads_X_d));
  CUDACHECK(cudaFree(valley_tails_X_d));
  CUDACHECK(cudaFree(valley_heads_Y_d));
  CUDACHECK(cudaFree(valley_tails_Y_d));

}

} // end of namespace gdiamond

#endif































