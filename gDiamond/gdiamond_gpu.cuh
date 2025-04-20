#ifndef GDIAMOND_GPU_CUH
#define GDIAMOND_GPU_CUH

#include "gdiamond.hpp"
#include "kernels.cuh"
#include <cuda_runtime.h>

#define BLOCK_SIZE 512 

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

  // dft for Ey
  // Monitor frequencies: from 0.96 to 1.04, step size 0.004
  float freq_monitor_start = 0.9 * SOURCE_FREQUENCY;
  float freq_monitor_end = 1.15 * SOURCE_FREQUENCY;
  float freq_monitor_step = 0.01 * SOURCE_FREQUENCY;
  int num_freqs = (int)((freq_monitor_end - freq_monitor_start) / freq_monitor_step) + 1;
  printf("num_freqs = %d\n", num_freqs); 

  // frequency monitors
  float *freq_monitors = (float *)malloc(num_freqs * sizeof(float));
  float *freq_monitors_device;
  cudaMalloc((void **)&freq_monitors_device, num_freqs * sizeof(float));
  for (int f = 0; f < num_freqs; ++f)
  {
      freq_monitors[f] = freq_monitor_start + f * freq_monitor_step;
  }
  cudaMemcpy(freq_monitors_device, freq_monitors, num_freqs * sizeof(float), cudaMemcpyHostToDevice);

  // to record real and imag figures
  float* real_host = (float *)malloc(_Nx * _Ny * sizeof(float));
  float* imag_host = (float *)malloc(_Nx * _Ny * sizeof(float));
  float *Ex_output_real_monitor, *Ex_output_imag_monitor;
  int save_len = _Nx * _Ny;
  cudaMalloc((void **)&Ex_output_real_monitor, num_freqs * save_len * sizeof(float));
  cudaMalloc((void **)&Ex_output_imag_monitor, num_freqs * save_len * sizeof(float));
  cudaMemset(Ex_output_real_monitor, 0, num_freqs * save_len * sizeof(float));
  cudaMemset(Ex_output_imag_monitor, 0, num_freqs * save_len * sizeof(float));

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
  size_t block_size = BLOCK_SIZE;
  size_t grid_size = (_Nx*_Ny*_Nz + block_size - 1) / block_size;
  size_t block_size_fft = BLOCK_SIZE;
  size_t grid_size_fft = (_Nx * _Ny + block_size - 1) / block_size;

  for(size_t t=0; t<num_timesteps; t++) {

    auto start1 = std::chrono::high_resolution_clock::now();

    // // Current source
    // float Mz_value = M_source_amp * std::sin(SOURCE_OMEGA * t * dt);

    // CUDACHECK(cudaMemcpy(Mz + _source_idx, &Mz_value, sizeof(float), cudaMemcpyHostToDevice));

    // Current source
    // float Jx_value = J_source_amp * std::sin(SOURCE_OMEGA * t * dt);
    float Jx_value = J_source_amp * std::exp(-((t * dt - t_peak) * (t * dt - t_peak)) / (2 * t_sigma * t_sigma)) * std::sin(SOURCE_OMEGA * t * dt);
    // float Jx_value = J_source_amp * std::sin(SOURCE_OMEGA * t * dt);

    CUDACHECK(cudaMemcpy(Jx + _source_idx, &Jx_value, sizeof(float), cudaMemcpyHostToDevice));
    
    // update E
    updateE_3Dmap_fix<<<grid_size, block_size, 0>>>(Ex, Ey, Ez,
          Hx, Hy, Hz, Cax, Cbx, Cay, Cby, Caz, Cbz,
          Jx, Jy, Jz, _dx, _Nx, _Ny, _Nz);

    // update H
    updateH_3Dmap_fix<<<grid_size, block_size, 0>>>(Ex, Ey, Ez,
          Hx, Hy, Hz, Dax, Dbx, Day, Dby, Daz, Dbz,
          Mx, My, Mz, _dx, _Nx, _Ny, _Nz);

    // calculate DFT for Ey
    update_field_FFT_xy<<<grid_size_fft, block_size_fft>>>(Ex, _Nz / 2, _Nx, _Ny, _Nz, Ex_output_real_monitor,
          Ex_output_imag_monitor, freq_monitors_device, num_freqs, t * dt, 50.0f / num_timesteps);

    auto end1 = std::chrono::high_resolution_clock::now();

    gpu_runtime += end1 - start1;

    // Record the field using a monitor, once in a while
    if (t % (num_timesteps/20) == 99)
    {
      printf("Iter: %ld / %ld \n", t, num_timesteps);

      // ------------ plotting time domain
      float *E_time_monitor_xy;
      E_time_monitor_xy = (float *)malloc(_Nx * _Ny * sizeof(float));
      memset(E_time_monitor_xy, 0, _Nx * _Ny * sizeof(float));

      // ------------ plotting time domain
      // File name initialization
      char field_filename[50];
      size_t slice_pitch = _Nx * sizeof(float); // The size in bytes of the 2D slice row
      size_t k = _Nz / 2;  // Assuming you want the middle slice
      for (size_t j = 0; j < _Ny; ++j)
      {
        float* device_ptr = Ex + j * _Nx + k * _Nx * _Ny; // Pointer to the start of the row in the desired slice
        float* host_ptr = E_time_monitor_xy + j * _Nx;  // Pointer to the host memory
        cudaMemcpy(host_ptr, device_ptr, slice_pitch, cudaMemcpyDeviceToHost);
      }

      snprintf(field_filename, sizeof(field_filename), "gpu_figures/Ex_naive_gpu_%04ld.png", (t+1));
      // save_field_png(E_time_monitor_xy, field_filename, _Nx, _Ny, 1.0 / sqrt(mu0 / eps0));
      save_field_png(E_time_monitor_xy, field_filename, _Nx, _Ny, 1);

      free(E_time_monitor_xy);

      // ------------ plotting frequency domain
      int freq_check = 11;
      cudaMemcpy(real_host, Ex_output_real_monitor + freq_check * _Nx * _Ny, _Nx * _Ny * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(imag_host, Ex_output_imag_monitor + freq_check * _Nx * _Ny, _Nx * _Ny * sizeof(float), cudaMemcpyDeviceToHost);

      char real_freq_filename[50];
      snprintf(real_freq_filename, sizeof(real_freq_filename), "gpu_figures/fft_Ex_real_%04ld.png", (t+1));
      save_field_png(real_host, real_freq_filename, _Nx, _Ny, 10);
      char imag_freq_filename[50];
      snprintf(imag_freq_filename, sizeof(imag_freq_filename), "gpu_figures/fft_Ex_imag_%04ld.png", (t+1));
      save_field_png(imag_host, imag_freq_filename, _Nx, _Ny, 10);
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

  free(freq_monitors);
  free(real_host);
  free(imag_host);
  CUDACHECK(cudaFree(freq_monitors_device));
  CUDACHECK(cudaFree(Ex_output_real_monitor));
  CUDACHECK(cudaFree(Ex_output_imag_monitor));

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

void gDiamond::update_FDTD_gpu_fq(size_t num_timesteps) { // GPU fuse equation try 

  // E, H, J, M on device 
  float *Ex, *Ey, *Ez, *Hx, *Hy, *Hz, *Jx, *Jy, *Jz, *Mx, *My, *Mz;
  float *Hx_temp, *Hy_temp, *Hz_temp;

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

  CUDACHECK(cudaMalloc(&Hx_temp, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Hy_temp, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMalloc(&Hz_temp, sizeof(float) * _Nx * _Ny * _Nz));

  // initialize E, H as 0 
  CUDACHECK(cudaMemset(Ex, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Ey, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Ez, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Hx, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Hy, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Hz, 0, sizeof(float) * _Nx * _Ny * _Nz));

  CUDACHECK(cudaMemset(Hx_temp, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Hy_temp, 0, sizeof(float) * _Nx * _Ny * _Nz));
  CUDACHECK(cudaMemset(Hz_temp, 0, sizeof(float) * _Nx * _Ny * _Nz));

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

    updateEH_3Dmap_fq<<<grid_size, BLOCK_SIZE, 0>>>(Ex, Ey, Ez,
          Hx, Hy, Hz, 
          Hx_temp, Hy_temp, Hz_temp,
          Cax, Cbx, Cay, Cby, Caz, Cbz,
          Jx, Jy, Jz, 
          Dax, Dbx, Day, Dby, Daz, Dbz,
          Mx, My, Mz, 
          _dx, _Nx, _Ny, _Nz);

    CUDACHECK(cudaMemcpy(Hx_temp, Hx, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToDevice));
    CUDACHECK(cudaMemcpy(Hy_temp, Hy, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToDevice));
    CUDACHECK(cudaMemcpy(Hz_temp, Hz, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToDevice));

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

  CUDACHECK(cudaFree(Hx_temp));
  CUDACHECK(cudaFree(Hy_temp));
  CUDACHECK(cudaFree(Hz_temp));



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
  CUDACHECK(cudaMemcpy(_Ex_gpu.data(), Ex, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Ey_gpu.data(), Ey, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Ez_gpu.data(), Ez, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Hx_gpu.data(), Hx, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Hy_gpu.data(), Hy, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Hz_gpu.data(), Hz, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));

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
  CUDACHECK(cudaMemcpy(_Ex_gpu.data(), Ex, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Ey_gpu.data(), Ey, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Ez_gpu.data(), Ez, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Hx_gpu.data(), Hx, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Hy_gpu.data(), Hy, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Hz_gpu.data(), Hz, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));

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

void gDiamond::update_FDTD_gpu_fuse_kernel_shmem_EH_mil(size_t num_timesteps) { // 3-D mapping, using more is less tiling on X, Y, Z dimension to fuse kernels, put EH in shared memory 

  std::cout << "running update_FDTD_gpu_fuse_kernel_shmem_EH_mil\n";

  int Nx = _Nx;
  int Ny = _Ny;
  int Nz = _Nz;

  // get xx_num, yy_num, zz_num
  int xx_top_length = BLX_MIL - 2 * (BLT_MIL - 1) - 1; // length of mountain top
  int yy_top_length = BLY_MIL - 2 * (BLT_MIL - 1) - 1; 
  int zz_top_length = BLZ_MIL - 2 * (BLT_MIL - 1) - 1; 
  int xx_num = (Nx + xx_top_length - 1) / xx_top_length;
  int yy_num = (Ny + yy_top_length - 1) / yy_top_length;
  int zz_num = (Nz + zz_top_length - 1) / zz_top_length;

  // heads and tails for mountain bottom
  std::vector<int> xx_heads(xx_num);
  std::vector<int> xx_tails(xx_num);
  std::vector<int> yy_heads(yy_num);
  std::vector<int> yy_tails(yy_num);
  std::vector<int> zz_heads(zz_num);
  std::vector<int> zz_tails(zz_num);
  // heads and tails for mountain top 
  std::vector<int> xx_top_heads(xx_num);
  std::vector<int> xx_top_tails(xx_num);
  std::vector<int> yy_top_heads(yy_num);
  std::vector<int> yy_top_tails(yy_num);
  std::vector<int> zz_top_heads(zz_num);
  std::vector<int> zz_top_tails(zz_num);

  // fill xx_top_heads and xx_top_tails
  for(int i=0; i<xx_num; i++) {
    xx_top_heads[i] = i * xx_top_length;
  }
  for(int i=0; i<xx_num; i++) {
    int temp = xx_top_heads[i] + xx_top_length - 1;
    xx_top_tails[i] = (temp > Nx - 1)? Nx - 1 : temp;
  }
  for(int i=0; i<yy_num; i++) {
    yy_top_heads[i] = i * yy_top_length;
  }
  for(int i=0; i<yy_num; i++) {
    int temp = yy_top_heads[i] + yy_top_length - 1;
    yy_top_tails[i] = (temp > Ny - 1)? Ny - 1 : temp;
  }
  for(int i=0; i<zz_num; i++) {
    zz_top_heads[i] = i * zz_top_length;
  }
  for(int i=0; i<zz_num; i++) {
    int temp = zz_top_heads[i] + zz_top_length - 1;
    zz_top_tails[i] = (temp > Nz - 1)? Nz - 1 : temp;
  }

  // fill xx_heads and xx_tails
  for(int i=0; i<xx_num; i++) {
    int temp = xx_top_heads[i] - (BLT_MIL - 1);
    xx_heads[i] = (temp < 0)? 0 : temp;
  }
  for(int i=0; i<xx_num; i++) {
    int temp = xx_top_tails[i] + BLT_MIL;
    xx_tails[i] = (temp > Nx - 1)? Nx - 1 : temp;
  }
  for(int i=0; i<yy_num; i++) {
    int temp = yy_top_heads[i] - (BLT_MIL - 1);
    yy_heads[i] = (temp < 0)? 0 : temp;
  }
  for(int i=0; i<yy_num; i++) {
    int temp = yy_top_tails[i] + BLT_MIL;
    yy_tails[i] = (temp > Ny - 1)? Ny - 1 : temp;
  }
  for(int i=0; i<zz_num; i++) {
    int temp = zz_top_heads[i] - (BLT_MIL - 1);
    zz_heads[i] = (temp < 0)? 0 : temp;
  }
  for(int i=0; i<zz_num; i++) {
    int temp = zz_top_tails[i] + BLT_MIL;
    zz_tails[i] = (temp > Nz - 1)? Nz - 1 : temp;
  }

  // E, H, J, M on device 
  float *Ex, *Ey, *Ez, *Hx, *Hy, *Hz, *Jx, *Jy, *Jz, *Mx, *My, *Mz;

  // Ca, Cb, Da, Db on device
  float *Cax, *Cay, *Caz, *Cbx, *Cby, *Cbz;
  float *Dax, *Day, *Daz, *Dbx, *Dby, *Dbz;

  // mil parameters on device
  int *xx_heads_d, *yy_heads_d, *zz_heads_d;
  int *xx_tails_d, *yy_tails_d, *zz_tails_d;
  int *xx_top_heads_d, *yy_top_heads_d, *zz_top_heads_d;
  int *xx_top_tails_d, *yy_top_tails_d, *zz_top_tails_d;
  int *shmem_load_finish;

  CUDACHECK(cudaMalloc(&xx_heads_d, sizeof(int) * xx_num));
  CUDACHECK(cudaMalloc(&yy_heads_d, sizeof(int) * yy_num));
  CUDACHECK(cudaMalloc(&zz_heads_d, sizeof(int) * zz_num));
  CUDACHECK(cudaMalloc(&xx_tails_d, sizeof(int) * xx_num));
  CUDACHECK(cudaMalloc(&yy_tails_d, sizeof(int) * yy_num));
  CUDACHECK(cudaMalloc(&zz_tails_d, sizeof(int) * zz_num));
  CUDACHECK(cudaMalloc(&xx_top_heads_d, sizeof(int) * xx_num));
  CUDACHECK(cudaMalloc(&yy_top_heads_d, sizeof(int) * yy_num));
  CUDACHECK(cudaMalloc(&zz_top_heads_d, sizeof(int) * zz_num));
  CUDACHECK(cudaMalloc(&xx_top_tails_d, sizeof(int) * xx_num));
  CUDACHECK(cudaMalloc(&yy_top_tails_d, sizeof(int) * yy_num));
  CUDACHECK(cudaMalloc(&zz_top_tails_d, sizeof(int) * zz_num));

  CUDACHECK(cudaMalloc(&shmem_load_finish, sizeof(int) * xx_num * yy_num * zz_num));

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

  // initialize shmem_load_finish
  CUDACHECK(cudaMemset(shmem_load_finish, 0, sizeof(int) * xx_num * yy_num * zz_num));

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
  CUDACHECK(cudaMemcpyAsync(xx_heads_d, xx_heads.data(), sizeof(int) * xx_num, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(yy_heads_d, yy_heads.data(), sizeof(int) * yy_num, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(zz_heads_d, zz_heads.data(), sizeof(int) * zz_num, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(xx_tails_d, xx_tails.data(), sizeof(int) * xx_num, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(yy_tails_d, yy_tails.data(), sizeof(int) * yy_num, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(zz_tails_d, zz_tails.data(), sizeof(int) * zz_num, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(xx_top_heads_d, xx_top_heads.data(), sizeof(int) * xx_num, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(yy_top_heads_d, yy_top_heads.data(), sizeof(int) * yy_num, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(zz_top_heads_d, zz_top_heads.data(), sizeof(int) * zz_num, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(xx_top_tails_d, xx_top_tails.data(), sizeof(int) * xx_num, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(yy_top_tails_d, yy_top_tails.data(), sizeof(int) * yy_num, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(zz_top_tails_d, zz_top_tails.data(), sizeof(int) * zz_num, cudaMemcpyHostToDevice));

  // calculation
  size_t block_size = BLX_MIL * BLY_MIL * BLZ_MIL; 
  size_t grid_size = xx_num * yy_num * zz_num;
  std::cout << "block_size = " << block_size << ", grid_size = " << grid_size << "\n";
  for(size_t t=0; t<num_timesteps/BLT_MIL; t++) {
    updateEH_mil<<<grid_size, block_size>>>(Ex, Ey, Ez,
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
                                            Nx, Ny, Nz,
                                            xx_num, yy_num, zz_num, 
                                            xx_heads_d, yy_heads_d, zz_heads_d,
                                            xx_tails_d, yy_tails_d, zz_tails_d,
                                            xx_top_heads_d, yy_top_heads_d, zz_top_heads_d,
                                            xx_top_tails_d, yy_top_tails_d, zz_top_tails_d,
                                            shmem_load_finish,
                                            block_size,
                                            grid_size); 
    cudaError_t err = cudaGetLastError(); // Check for launch errors
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }

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
  std::cout << "gpu runtime (mil, 3-D mapping, shared memory for EH only): " << std::chrono::duration<double>(end-start).count() << "s\n";
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

  CUDACHECK(cudaFree(xx_heads_d));
  CUDACHECK(cudaFree(yy_heads_d));
  CUDACHECK(cudaFree(zz_heads_d));
  CUDACHECK(cudaFree(xx_tails_d));
  CUDACHECK(cudaFree(yy_tails_d));
  CUDACHECK(cudaFree(zz_tails_d));
  CUDACHECK(cudaFree(xx_top_heads_d));
  CUDACHECK(cudaFree(yy_top_heads_d));
  CUDACHECK(cudaFree(zz_top_heads_d));
  CUDACHECK(cudaFree(xx_top_tails_d));
  CUDACHECK(cudaFree(yy_top_tails_d));
  CUDACHECK(cudaFree(zz_top_tails_d));

  CUDACHECK(cudaFree(shmem_load_finish));

}

void gDiamond::update_FDTD_gpu_simulation_3_D_mil(size_t num_timesteps) { // CPU single thread 3-D simulation of GPU workflow, more is less tiling

  std::cout << "running update_FDTD_gpu_simulation_3_D_mil\n";

  // clear source Mz for experiments
  _Mz.clear();

  // transfer source
  for(size_t t=0; t<num_timesteps; t++) {
    float Mz_value = M_source_amp * std::sin(SOURCE_OMEGA * t * dt);
    _Mz[_source_idx] = Mz_value;
  }

  int Nx = _Nx;
  int Ny = _Ny;
  int Nz = _Nz;

  // get xx_num, yy_num, zz_num
  int xx_top_length = BLX_MIL - 2 * (BLT_MIL - 1) - 1; // length of mountain top
  int yy_top_length = BLY_MIL - 2 * (BLT_MIL - 1) - 1; 
  int zz_top_length = BLZ_MIL - 2 * (BLT_MIL - 1) - 1; 
  int xx_num = (Nx + xx_top_length - 1) / xx_top_length;
  int yy_num = (Ny + yy_top_length - 1) / yy_top_length;
  int zz_num = (Nz + zz_top_length - 1) / zz_top_length;

  // heads and tails for mountain bottom
  std::vector<int> xx_heads(xx_num);
  std::vector<int> xx_tails(xx_num);
  std::vector<int> yy_heads(yy_num);
  std::vector<int> yy_tails(yy_num);
  std::vector<int> zz_heads(zz_num);
  std::vector<int> zz_tails(zz_num);
  // heads and tails for mountain top 
  std::vector<int> xx_top_heads(xx_num);
  std::vector<int> xx_top_tails(xx_num);
  std::vector<int> yy_top_heads(yy_num);
  std::vector<int> yy_top_tails(yy_num);
  std::vector<int> zz_top_heads(zz_num);
  std::vector<int> zz_top_tails(zz_num);

  // fill xx_top_heads and xx_top_tails
  for(int i=0; i<xx_num; i++) {
    xx_top_heads[i] = i * xx_top_length;
  }
  for(int i=0; i<xx_num; i++) {
    int temp = xx_top_heads[i] + xx_top_length - 1;
    xx_top_tails[i] = (temp > Nx - 1)? Nx - 1 : temp;
  }
  for(int i=0; i<yy_num; i++) {
    yy_top_heads[i] = i * yy_top_length;
  }
  for(int i=0; i<yy_num; i++) {
    int temp = yy_top_heads[i] + yy_top_length - 1;
    yy_top_tails[i] = (temp > Ny - 1)? Ny - 1 : temp;
  }
  for(int i=0; i<zz_num; i++) {
    zz_top_heads[i] = i * zz_top_length;
  }
  for(int i=0; i<zz_num; i++) {
    int temp = zz_top_heads[i] + zz_top_length - 1;
    zz_top_tails[i] = (temp > Nz - 1)? Nz - 1 : temp;
  }

  // fill xx_heads and xx_tails
  for(int i=0; i<xx_num; i++) {
    int temp = xx_top_heads[i] - (BLT_MIL - 1);
    xx_heads[i] = (temp < 0)? 0 : temp;
  }
  for(int i=0; i<xx_num; i++) {
    int temp = xx_top_tails[i] + BLT_MIL;
    xx_tails[i] = (temp > Nx - 1)? Nx - 1 : temp;
  }
  for(int i=0; i<yy_num; i++) {
    int temp = yy_top_heads[i] - (BLT_MIL - 1);
    yy_heads[i] = (temp < 0)? 0 : temp;
  }
  for(int i=0; i<yy_num; i++) {
    int temp = yy_top_tails[i] + BLT_MIL;
    yy_tails[i] = (temp > Ny - 1)? Ny - 1 : temp;
  }
  for(int i=0; i<zz_num; i++) {
    int temp = zz_top_heads[i] - (BLT_MIL - 1);
    zz_heads[i] = (temp < 0)? 0 : temp;
  }
  for(int i=0; i<zz_num; i++) {
    int temp = zz_top_tails[i] + BLT_MIL;
    zz_tails[i] = (temp > Nz - 1)? Nz - 1 : temp;
  }

  size_t block_size = BLX_MIL * BLY_MIL * BLZ_MIL;
  size_t grid_size = xx_num * yy_num * zz_num;
  for(size_t tt=0; tt<num_timesteps/BLT_MIL; tt++) {
    _updateEH_mil_seq(_Ex_simu_init, _Ey_simu_init, _Ez_simu_init,
                      _Hx_simu_init, _Hy_simu_init, _Hz_simu_init,
                      _Ex_simu, _Ey_simu, _Ez_simu,
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
                      Nx, Ny, Nz,
                      xx_num, yy_num, zz_num, // number of tiles in each dimensions
                      xx_heads, yy_heads, zz_heads,
                      xx_tails, yy_tails, zz_tails,
                      xx_top_heads, yy_top_heads, zz_top_heads,
                      xx_top_tails, yy_top_tails, zz_top_tails,
                      block_size,
                      grid_size); 

    // memory transfer from EH to EH init
    for(int i=0; i<Nx*Ny*Nz; i++) {
      _Ex_simu_init[i] = _Ex_simu[i];
      _Ey_simu_init[i] = _Ey_simu[i];
      _Ez_simu_init[i] = _Ez_simu[i];
      _Hx_simu_init[i] = _Hx_simu[i];
      _Hy_simu_init[i] = _Hy_simu[i];
      _Hz_simu_init[i] = _Hz_simu[i];
    }
  }

}

void gDiamond::_updateEH_mil_seq(std::vector<float>& Ex_init, std::vector<float>& Ey_init, std::vector<float>& Ez_init,
                                 std::vector<float>& Hx_init, std::vector<float>& Hy_init, std::vector<float>& Hz_init,
                                 std::vector<float>& Ex, std::vector<float>& Ey, std::vector<float>& Ez,
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
                                 std::vector<int> xx_heads, std::vector<int> yy_heads, std::vector<int> zz_heads,
                                 std::vector<int> xx_tails, std::vector<int> yy_tails, std::vector<int> zz_tails,
                                 std::vector<int> xx_top_heads, std::vector<int> yy_top_heads, std::vector<int> zz_top_heads,
                                 std::vector<int> xx_top_tails, std::vector<int> yy_top_tails, std::vector<int> zz_top_tails,
                                 size_t block_size,
                                 size_t grid_size) {

  for(size_t block_id=0; block_id<grid_size; block_id++) {
    int xx = block_id % xx_num;
    int yy = (block_id % (xx_num * yy_num)) / xx_num;
    int zz = block_id / (xx_num * yy_num); 

    // declare shared memory
    float Ex_shmem[BLX_MIL_EH * BLY_MIL_EH * BLZ_MIL_EH];
    float Ey_shmem[BLX_MIL_EH * BLY_MIL_EH * BLZ_MIL_EH];
    float Ez_shmem[BLX_MIL_EH * BLY_MIL_EH * BLZ_MIL_EH];
    float Hx_shmem[BLX_MIL_EH * BLY_MIL_EH * BLZ_MIL_EH];
    float Hy_shmem[BLX_MIL_EH * BLY_MIL_EH * BLZ_MIL_EH];
    float Hz_shmem[BLX_MIL_EH * BLY_MIL_EH * BLZ_MIL_EH];

    // load shared memory
    for(size_t thread_id=0; thread_id<block_size; thread_id++) {
      int local_x = thread_id % BLX_MIL;                     
      int local_y = (thread_id / BLX_MIL) % BLY_MIL;     
      int local_z = thread_id / (BLX_MIL * BLY_MIL);     

      int global_x = xx_heads[xx] + local_x; 
      int global_y = yy_heads[yy] + local_y; 
      int global_z = zz_heads[zz] + local_z; 
      int global_idx = global_x + global_y * Nx + global_z * Nx * Ny;

      // load H, stencil pattern x-1, y-1, z-1
      int shared_H_x = local_x + 1;
      int shared_H_y = local_y + 1;
      int shared_H_z = local_z + 1;
      int shared_H_idx = shared_H_x + shared_H_y * BLX_MIL_EH + shared_H_z * BLX_MIL_EH * BLY_MIL_EH;
      
      // load E, stencil pattern x+1, y+1, z+1
      // the padding does not affect origins of local idx and shared_E_idx
      // local idx and shared_E_idx still have the same origin
      int shared_E_x = local_x;
      int shared_E_y = local_y;
      int shared_E_z = local_z;
      int shared_E_idx = shared_E_x + shared_E_y * BLX_MIL_EH + shared_E_z * BLX_MIL_EH * BLY_MIL_EH;

      if(global_x < Nx && global_y < Ny && global_z < Nz) {

        // load core
        Hx_shmem[shared_H_idx] = Hx_init[global_idx];
        Hy_shmem[shared_H_idx] = Hy_init[global_idx];
        Hz_shmem[shared_H_idx] = Hz_init[global_idx];

        // load HALO region
        if(local_x == 0 && global_x > 0) {
          Hz_shmem[shared_H_x - 1 + shared_H_y * BLX_MIL_EH + shared_H_z * BLX_MIL_EH * BLY_MIL_EH] = Hz_init[global_x - 1 + global_y * Nx + global_z * Nx * Ny];
          Hy_shmem[shared_H_x - 1 + shared_H_y * BLX_MIL_EH + shared_H_z * BLX_MIL_EH * BLY_MIL_EH] = Hy_init[global_x - 1 + global_y * Nx + global_z * Nx * Ny];

        }
        if(local_y == 0 && global_y > 0) {
          Hx_shmem[shared_H_x + (shared_H_y - 1) * BLX_MIL_EH + shared_H_z * BLX_MIL_EH * BLY_MIL_EH] = Hx_init[global_x + (global_y - 1) * Nx + global_z * Nx * Ny];
          Hz_shmem[shared_H_x + (shared_H_y - 1) * BLX_MIL_EH + shared_H_z * BLX_MIL_EH * BLY_MIL_EH] = Hz_init[global_x + (global_y - 1) * Nx + global_z * Nx * Ny];
        }
        if(local_z == 0 && global_z > 0) {
          Hx_shmem[shared_H_x + shared_H_y * BLX_MIL_EH + (shared_H_z - 1) * BLX_MIL_EH * BLY_MIL_EH] = Hx_init[global_x + global_y * Nx + (global_z - 1) * Nx * Ny];
          Hy_shmem[shared_H_x + shared_H_y * BLX_MIL_EH + (shared_H_z - 1) * BLX_MIL_EH * BLY_MIL_EH] = Hy_init[global_x + global_y * Nx + (global_z - 1) * Nx * Ny];
        }
      }

      if(global_x < Nx && global_y < Ny && global_z < Nz) {

        Ex_shmem[shared_E_idx] = Ex_init[global_idx];
        Ey_shmem[shared_E_idx] = Ey_init[global_idx];
        Ez_shmem[shared_E_idx] = Ez_init[global_idx];

        // load HALO region
        if(local_x == BLX_GPU - 1 && global_x < Nx - 1) {
          Ez_shmem[shared_E_x + 1 + shared_E_y * BLX_MIL_EH + shared_E_z * BLX_MIL_EH * BLY_MIL_EH] = Ez_init[global_x + 1 + global_y * Nx + global_z * Nx * Ny];
          Ey_shmem[shared_E_x + 1 + shared_E_y * BLX_MIL_EH + shared_E_z * BLX_MIL_EH * BLY_MIL_EH] = Ey_init[global_x + 1 + global_y * Nx + global_z * Nx * Ny];
        }
        if(local_y == BLY_GPU - 1 && global_y < Ny - 1) {
          Ex_shmem[shared_E_x + (shared_E_y + 1) * BLX_MIL_EH + shared_E_z * BLX_MIL_EH * BLY_MIL_EH] = Ex_init[global_x + (global_y + 1) * Nx + global_z * Nx * Ny];
          Ez_shmem[shared_E_x + (shared_E_y + 1) * BLX_MIL_EH + shared_E_z * BLX_MIL_EH * BLY_MIL_EH] = Ez_init[global_x + (global_y + 1) * Nx + global_z * Nx * Ny];
        }
        if(local_z == BLZ_GPU - 1 && global_z < Nz - 1) {
          Ex_shmem[shared_E_x + shared_E_y * BLX_MIL_EH + (shared_E_z + 1) * BLX_MIL_EH * BLY_MIL_EH] = Ex_init[global_x + global_y * Nx + (global_z + 1) * Nx * Ny];
          Ey_shmem[shared_E_x + shared_E_y * BLX_MIL_EH + (shared_E_z + 1) * BLX_MIL_EH * BLY_MIL_EH] = Ey_init[global_x + global_y * Nx + (global_z + 1) * Nx * Ny];
        }
      }

    }

    // calculation 
    for(size_t t=0; t<BLT_MIL; t++) {

      // update E
      for(size_t thread_id=0; thread_id<block_size; thread_id++) {
        int local_x = thread_id % BLX_MIL;                     
        int local_y = (thread_id / BLX_MIL) % BLY_MIL;     
        int local_z = thread_id / (BLX_MIL * BLY_MIL);     

        int global_x = xx_heads[xx] + local_x; 
        int global_y = yy_heads[yy] + local_y; 
        int global_z = zz_heads[zz] + local_z; 

        int shared_H_x = local_x + 1;
        int shared_H_y = local_y + 1;
        int shared_H_z = local_z + 1;
        
        int shared_E_x = local_x;
        int shared_E_y = local_y;
        int shared_E_z = local_z;

        int s_H_idx = shared_H_x + shared_H_y * BLX_MIL_EH + shared_H_z * BLX_MIL_EH * BLY_MIL_EH; // shared memory idx for H
        int s_E_idx = shared_E_x + shared_E_y * BLX_MIL_EH + shared_E_z * BLX_MIL_EH * BLY_MIL_EH; // shared memory idx for E
        int g_idx = global_x + global_y * Nx + global_z * Nx * Ny; // global idx

        if(global_x >= 1 && global_x <= Nx-2 && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2 &&
           global_x <= xx_tails[xx] &&
           global_y <= yy_tails[yy] &&
           global_z <= zz_tails[zz]) {

          Ex_shmem[s_E_idx] = Cax[g_idx] * Ex_shmem[s_E_idx] + Cbx[g_idx] *
                    ((Hz_shmem[s_H_idx] - Hz_shmem[s_H_idx - BLX_MIL_EH]) - (Hy_shmem[s_H_idx] - Hy_shmem[s_H_idx - BLX_MIL_EH * BLY_MIL_EH]) - Jx[g_idx] * dx);
          Ey_shmem[s_E_idx] = Cay[g_idx] * Ey_shmem[s_E_idx] + Cby[g_idx] *
                    ((Hx_shmem[s_H_idx] - Hx_shmem[s_H_idx - BLX_MIL_EH * BLY_MIL_EH]) - (Hz_shmem[s_H_idx] - Hz_shmem[s_H_idx - 1]) - Jy[g_idx] * dx);
          Ez_shmem[s_E_idx] = Caz[g_idx] * Ez_shmem[s_E_idx] + Cbz[g_idx] *
                    ((Hy_shmem[s_H_idx] - Hy_shmem[s_H_idx - 1]) - (Hx_shmem[s_H_idx] - Hx_shmem[s_H_idx - BLX_MIL_EH]) - Jz[g_idx] * dx);
        }
      }

      // update H
      for(size_t thread_id=0; thread_id<block_size; thread_id++) {
        int local_x = thread_id % BLX_MIL;                     
        int local_y = (thread_id / BLX_MIL) % BLY_MIL;     
        int local_z = thread_id / (BLX_MIL * BLY_MIL);     

        int global_x = xx_heads[xx] + local_x; 
        int global_y = yy_heads[yy] + local_y; 
        int global_z = zz_heads[zz] + local_z; 

        int shared_H_x = local_x + 1;
        int shared_H_y = local_y + 1;
        int shared_H_z = local_z + 1;
        
        int shared_E_x = local_x;
        int shared_E_y = local_y;
        int shared_E_z = local_z;

        int s_H_idx = shared_H_x + shared_H_y * BLX_MIL_EH + shared_H_z * BLX_MIL_EH * BLY_MIL_EH; // shared memory idx for H
        int s_E_idx = shared_E_x + shared_E_y * BLX_MIL_EH + shared_E_z * BLX_MIL_EH * BLY_MIL_EH; // shared memory idx for E
        int g_idx = global_x + global_y * Nx + global_z * Nx * Ny; // global idx

        if(global_x >= 1 && global_x <= Nx-2 && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2 &&
           global_x <= xx_tails[xx] &&
           global_y <= yy_tails[yy] &&
           global_z <= zz_tails[zz]) {

          Hx_shmem[s_H_idx] = Dax[g_idx] * Hx_shmem[s_H_idx] + Dbx[g_idx] *
                    ((Ey_shmem[s_E_idx + BLX_MIL_EH * BLY_MIL_EH] - Ey_shmem[s_E_idx]) - (Ez_shmem[s_E_idx + BLX_MIL_EH] - Ez_shmem[s_E_idx]) - Mx[g_idx] * dx);
          Hy_shmem[s_H_idx] = Day[g_idx] * Hy_shmem[s_H_idx] + Dby[g_idx] *
                    ((Ez_shmem[s_E_idx + 1] - Ez_shmem[s_E_idx]) - (Ex_shmem[s_E_idx + BLX_MIL_EH * BLY_MIL_EH] - Ex_shmem[s_E_idx]) - My[g_idx] * dx);
          Hz_shmem[s_H_idx] = Daz[g_idx] * Hz_shmem[s_H_idx] + Dbz[g_idx] *
                    ((Ex_shmem[s_E_idx + BLX_MIL_EH] - Ex_shmem[s_E_idx]) - (Ey_shmem[s_E_idx + 1] - Ey_shmem[s_E_idx]) - Mz[g_idx] * dx);
        }
      } 
    }

    // store back to global memory
    for(size_t thread_id=0; thread_id<block_size; thread_id++) {
      int local_x = thread_id % BLX_MIL;                     
      int local_y = (thread_id / BLX_MIL) % BLY_MIL;     
      int local_z = thread_id / (BLX_MIL * BLY_MIL);     

      int global_x = xx_heads[xx] + local_x; 
      int global_y = yy_heads[yy] + local_y; 
      int global_z = zz_heads[zz] + local_z; 

      int shared_H_x = local_x + 1;
      int shared_H_y = local_y + 1;
      int shared_H_z = local_z + 1;
      
      int shared_E_x = local_x;
      int shared_E_y = local_y;
      int shared_E_z = local_z;

      int s_H_idx = shared_H_x + shared_H_y * BLX_MIL_EH + shared_H_z * BLX_MIL_EH * BLY_MIL_EH; // shared memory idx for H
      int s_E_idx = shared_E_x + shared_E_y * BLX_MIL_EH + shared_E_z * BLX_MIL_EH * BLY_MIL_EH; // shared memory idx for E
      int g_idx = global_x + global_y * Nx + global_z * Nx * Ny; // global idx

      if(global_x >= 1 && global_x <= Nx-2 && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2 &&
         global_x >= xx_top_heads[xx] && global_x <= xx_top_tails[xx] &&
         global_y >= yy_top_heads[yy] && global_y <= yy_top_tails[yy] &&
         global_z >= zz_top_heads[zz] && global_z <= zz_top_tails[zz]) {
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

void gDiamond::update_FDTD_gpu_simulation_1_D_pt_pl(size_t num_tiemsteps) { // CPU single thread 1-D simulation of GPU workflow, parallelogram tiling, pipeline



}

void gDiamond::update_FDTD_gpu_simulation_1_D_mil(size_t num_timesteps) { // CPU single thread 1-D simulation of GPU workflow, more is less tiling
  
  // write 1 dimension just to check
  std::vector<float> E_simu(_Nx, 1);
  std::vector<float> H_simu(_Nx, 1);
  std::vector<float> E_simu_init(_Nx, 1);
  std::vector<float> H_simu_init(_Nx, 1);
  std::vector<float> E_seq(_Nx, 1);
  std::vector<float> H_seq(_Nx, 1);

  int Nx = _Nx;
  // seq version
  for(size_t t=0; t<num_timesteps; t++) {

    // update E
    for(int x=1; x<Nx-1; x++) {
      E_seq[x] = H_seq[x-1] + H_seq[x] * 2; 
    }

    std::cout << "t = " << t << ", E_seq =";
    for(int x=0; x<Nx; x++) {
      std::cout << E_seq[x] << " ";
    }
    std::cout << "\n";

    // update H 
    for(int x=1; x<Nx-1; x++) {
      H_seq[x] = E_seq[x+1] + E_seq[x] * 2; 
    }
  }

  // tiling version
  int xx_top_length = BLX_MIL - 2 * (BLT_MIL - 1) - 1; // length of mountain top
  int xx_num = (Nx + xx_top_length - 1) / xx_top_length;

  // std::cout << "xx_num = " << xx_num << "\n";

  // heads and tails for mountain bottom
  std::vector<int> xx_heads(xx_num);
  std::vector<int> xx_tails(xx_num);

  // heads and tails for mountain top 
  std::vector<int> xx_top_heads(xx_num);
  std::vector<int> xx_top_tails(xx_num);

  // fill xx_top_heads and xx_top_tails
  for(int i=0; i<xx_num; i++) {
    xx_top_heads[i] = i * xx_top_length;
  }
  for(int i=0; i<xx_num; i++) {
    int temp = xx_top_heads[i] + xx_top_length - 1;
    xx_top_tails[i] = (temp > Nx - 1)? Nx - 1 : temp;
  }

  // fill xx_heads and xx_tails
  for(int i=0; i<xx_num; i++) {
    int temp = xx_top_heads[i] - (BLT_MIL - 1);
    xx_heads[i] = (temp < 0)? 0 : temp;
  }
  for(int i=0; i<xx_num; i++) {
    int temp = xx_top_tails[i] + BLT_MIL;
    xx_tails[i] = (temp > Nx - 1)? Nx - 1 : temp;
  }

  // check bounds  
  // for(int i=0; i<xx_num; i++) {
  //   printf("xx_heads[%d] = %d, xx_tails[%d] = %d, xx_top_heads[%d] = %d, xx_top_tails[%d] = %d\n", 
  //           i, xx_heads[i], i, xx_tails[i], i, xx_top_heads[i], i, xx_top_tails[i]);
  // }

  int grid_size = xx_num; // we will launch xx_num blocks
  for(size_t tt=0; tt<num_timesteps/BLT_MIL; tt++) {

    
    // launching kernel
    for(int xx=0; xx<grid_size; xx++) { // xx is block id
      
      // declare shmem 
      float E_shmem[BLX_MIL_EH] = {0}; 
      float H_shmem[BLX_MIL_EH] = {0};

      // load shmem
      for(int local_x=0; local_x<BLX_MIL; local_x++) { // local_x is thread_id
        int global_x = xx_heads[xx] + local_x;
        int shared_H_x = local_x + 1;
        int shared_E_x = local_x;
        
        if(global_x < Nx) {
          H_shmem[shared_H_x] = H_simu_init[global_x];
          // load HALO
          if(local_x == 0 && global_x > 0) {
            H_shmem[shared_H_x - 1] = H_simu_init[global_x - 1];
          }
        }

        if(global_x < Nx) {
          E_shmem[shared_E_x] = E_simu_init[global_x];
          // load HALO
          if(local_x == BLX_MIL-1 && global_x < Nx-1) {
            E_shmem[shared_E_x + 1] = E_simu_init[global_x + 1];
          }
        }
      }

      // calculation
      for(size_t t=0; t<BLT_MIL; t++) {

        // update E
        for(int local_x=0; local_x<BLX_MIL; local_x++) { // local_x is thread_id
          int global_x = xx_heads[xx] + local_x; 
          int shared_H_x = local_x + 1;
          int shared_E_x = local_x;
          if(global_x >= 1 && global_x <= Nx - 2 && global_x <= xx_tails[xx]) {
            E_shmem[shared_E_x] = H_shmem[shared_H_x - 1] + H_shmem[shared_H_x] * 2;
          }
        }

        // update H 
        for(int local_x=0; local_x<BLX_MIL; local_x++) { // local_x is thread_id
          int global_x = xx_heads[xx] + local_x; 
          int shared_H_x = local_x + 1;
          int shared_E_x = local_x;
          if(global_x >= 1 && global_x <= Nx - 2 && global_x <= xx_tails[xx]) {
            H_shmem[shared_H_x] = E_shmem[shared_E_x + 1] + E_shmem[shared_E_x] * 2;
          }
        }
      }

      // store back to global mem
      for(int local_x=0; local_x<BLX_MIL; local_x++) { // local_x is thread_id
        int global_x = xx_heads[xx] + local_x;
        int shared_H_x = local_x + 1;
        int shared_E_x = local_x;
        if(global_x >= 1 && global_x <= Nx-2 && 
           global_x >= xx_top_heads[xx] && global_x <= xx_top_tails[xx]) {

          H_simu[global_x] = H_shmem[shared_H_x];
          E_simu[global_x] = E_shmem[shared_E_x];
          printf("xx = %d, E_simu[%d] = %f\n", xx, global_x, E_simu[global_x]);
        }
      }
      std::cout << "after xx = " << xx << ", E_simu = ";
      for(auto e : E_simu) {
        std::cout << e << " ";
      }
      std::cout << "\n";
    }

    // memory transfer from dest to init
    for(int i=0; i<Nx; i++) {
      E_simu_init[i] = E_simu[i];
      H_simu_init[i] = H_simu[i];
    }
  }
    
  std::cout << "E_seq = ";
  for(int x=0; x<Nx; x++) {
    std::cout << E_seq[x] << " ";
  }
  std::cout << "\n";

  std::cout << "E_simu = ";
  for(int x=0; x<Nx; x++) {
    std::cout << E_simu[x] << " ";
  }
  std::cout << "\n";

  for(int x=0; x<Nx; x++) {
    if(E_seq[x] != E_simu[x] || H_seq[x] != H_simu[x]) {
      std::cerr << "1-D demo results mismatch.\n";
      std::exit(EXIT_FAILURE);
    }
  }
}

void gDiamond::update_FDTD_gpu_simulation_1_D_pt_pipeline(size_t num_timesteps) { // CPU single thread simulation of GPU workflow, parallelogram tiling with pipeline
  
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
  int num_zz = (Nz - (BLZ_GPU_PT - BLT_GPU_PT - 1) + BLZ_GPU_PT - 1) / BLZ_GPU_PT;

  std::cout << "num_zz = " << num_zz << "\n";
    
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

void gDiamond::update_FDTD_cpu_simulation_1_D_pt(size_t num_timesteps) { // CPU single thread 1-D simulation of parallelogram tiling

  // write 1 dimension just to check
  std::vector<float> E_simu(_Nx, 1);
  std::vector<float> H_simu(_Nx, 1);
  std::vector<float> E_seq(_Nx, 1);
  std::vector<float> H_seq(_Nx, 1);

  int Nx = _Nx;

  // seq version
  for(size_t t=0; t<num_timesteps; t++) {

    // update E
    for(int x=1; x<Nx-1; x++) {
      E_seq[x] = H_seq[x-1] + H_seq[x] * 2; 
    }

    std::cout << "t = " << t << ", E_seq =";
    for(int x=0; x<Nx; x++) {
      std::cout << E_seq[x] << " ";
    }
    std::cout << "\n";

    // update H 
    for(int x=1; x<Nx-1; x++) {
      H_seq[x] = E_seq[x+1] + E_seq[x] * 2; 
    }
  }

  // tiling version
  int num_pts = (Nx - (BLX_PT - BLT_PT) + BLX_PT - 1) / BLX_PT + 1; 
  std::cout << "num_pts = " << num_pts << "\n";

  // we will need to calculate the head and tail index for each parallelogram tile
  std::vector<int> pt_heads(num_pts, 0);
  std::vector<int> pt_tails(num_pts);
  for(int idx=1; idx<num_pts; idx++) {
    int step = (idx == 1)? BLX_PT - BLT_PT : BLX_PT;
    pt_heads[idx] = pt_heads[idx-1] + step;
  } 
  for(int idx=0; idx<num_pts; idx++) {
    int step = (idx == 0)? BLX_PT - 1 : BLX_PT + BLT_PT - 1;
    pt_tails[idx] = (pt_heads[idx] + step > Nx - 1)? Nx - 1 : pt_heads[idx] + step;
  }
  pt_heads[0] = 0 - BLT_PT;

  std::cout << "pt_heads = [";
  for(auto idx : pt_heads) {
    std::cout << idx << " ";
  }
  std::cout << "}\n";

  std::cout << "pt_tails = [";
  for(auto idx : pt_tails) {
    std::cout << idx << " ";
  }
  std::cout << "}\n";

  int block_size = 8;

  for(size_t tt=0; tt<num_timesteps/BLT_PT; tt++) {
    
    // compute each parallelogram tile one by one
    for(int pt=0; pt<num_pts; pt++) {

      // head and tail for this parallelogram tile
      int head = pt_heads[pt];
      int tail = pt_tails[pt];

      // declare shared memory
      std::vector<float> E_shmem(BLX_PT + BLT_PT); 
      std::vector<float> H_shmem(BLX_PT + BLT_PT); 

      // load shared memory
      for(int tid=0; tid<block_size; tid++) {
        for(int local_idx=tid; local_idx<(BLX_PT + BLT_PT); local_idx+=BLX_PT) {
          int global_idx = local_idx + pt_heads[pt]; 
          if(global_idx >= 0 && global_idx <= Nx-1) {
            E_shmem[local_idx] = E_simu[global_idx];
            H_shmem[local_idx] = H_simu[global_idx];
          }
        }
      }

      // check shmem
      if(pt == 0) {
        std::cout << "before calculation: \n";
        std::cout << "E_shmem = ";
        for(auto e : E_shmem) {
          std::cout << e << " ";
        }
        std::cout << "\n";

        std::cout << "H_shmem = ";
        for(auto e : H_shmem) {
          std::cout << e << " ";
        }
        std::cout << "\n";
      }

      // update  
      for(int t=0; t<BLT_PT; t++) {
        int offset = BLT_PT - t;

        // update E
        for(int local_idx=0; local_idx<block_size; local_idx++) {
          int global_idx = local_idx + offset + pt_heads[pt];
          int shared_E_idx = local_idx + offset; 
          if(global_idx >= 1 && global_idx <= Nx-2) {
            E_shmem[shared_E_idx] = H_shmem[shared_E_idx-1] + H_shmem[shared_E_idx] * 2; 
          }
        }

        // update H
        for(int local_idx=0; local_idx<block_size; local_idx++) {
          int global_idx = local_idx + offset + pt_heads[pt] - 1;
          int shared_H_idx = local_idx + offset - 1; 
          if(global_idx >= 1 && global_idx <= Nx-2) {
            H_shmem[shared_H_idx] = E_shmem[shared_H_idx+1] + E_shmem[shared_H_idx] * 2; 
          }
        }
      }

      if(pt == 0) {
        std::cout << "after calculation: \n";
        std::cout << "E_shmem = ";
        for(auto e : E_shmem) {
          std::cout << e << " ";
        }
        std::cout << "\n";

        std::cout << "H_shmem = ";
        for(auto e : H_shmem) {
          std::cout << e << " ";
        }
        std::cout << "\n";
      }

      // store global memory
      for(int tid=0; tid<block_size; tid++) {
        for(int local_idx=tid; local_idx<(BLX_PT + BLT_PT); local_idx+=BLX_PT) {
          int global_idx = local_idx + pt_heads[pt]; 
          if(global_idx >= 0 && global_idx <= Nx-1) {
            E_simu[global_idx] = E_shmem[local_idx];
            H_simu[global_idx] = H_shmem[local_idx];
          }
        }
      }

    }

  }
 
  std::cout << "E_seq = ";
  for(int x=0; x<Nx; x++) {
    std::cout << E_seq[x] << " ";
  }
  std::cout << "\n";

  std::cout << "E_simu = ";
  for(int x=0; x<Nx; x++) {
    std::cout << E_simu[x] << " ";
  }
  std::cout << "\n";

  for(int x=0; x<Nx; x++) {
    if(E_seq[x] != E_simu[x] || H_seq[x] != H_simu[x]) {
      std::cerr << "1-D demo results mismatch.\n";
      std::exit(EXIT_FAILURE);
    }
  }
}

void gDiamond::_find_diagonal_hyperplanes(int Nx, int Ny, int Nz, 
                                          std::vector<Pt_idx>& hyperplanes, 
                                          std::vector<int>& hyperplane_heads, 
                                          std::vector<int>& hyperplanes_sizes) {
  int total_pixels = 0;

  // Generate hyperplanes and store them in a flattened array
  for (int d = 0; d < Nx + Ny + Nz - 2; ++d) {
    hyperplane_heads.push_back(total_pixels); // Start index of current hyperplane
    int count = 0;
    for (int x = 0; x < Nx; ++x) {
      for (int y = 0; y < Ny; ++y) {
        for (int z = 0; z < Nz; ++z) {
          if (x + y + z == d) {
            hyperplanes.push_back({x, y, z});
            count++;
          }
        }
      }
    }
    hyperplanes_sizes.push_back(count); // Store number of pixels in this hyperplane
    total_pixels += count;
  }

  // Print all hyperplanes
  // for (size_t d = 0; d < hyperplane_heads.size(); ++d) {
  //   std::cout << "Hyperplane " << d << ": ";
  //   int startIdx = hyperplane_heads[d];
  //   int size = hyperplanes_sizes[d];
  //   for (int i = 0; i < size; ++i) {
  //     Pt_idx p = hyperplanes[startIdx + i];
  //     std::cout << "(" << p.x << ", " << p.y << ", " << p.z << ") ";
  //   }
  //   std::cout << std::endl;
  // }

}

void gDiamond::update_FDTD_cpu_simulation_3_D_pt(size_t num_timesteps) { // CPU single thread 3-D simulation of parallelogram tiling

  // clear source Mz for experiments
  _Mz.clear();

  // transfer source
  for(size_t t=0; t<num_timesteps; t++) {
    float Mz_value = M_source_amp * std::sin(SOURCE_OMEGA * t * dt);
    _Mz[_source_idx] = Mz_value;
  }

  int Nx = _Nx;
  int Ny = _Ny;
  int Nz = _Nz;

  // number of parallelogram tiles in each dimension 
  int xx_num = (Nx - (BLX_PT - BLT_PT) + BLX_PT - 1) / BLX_PT + 1; 
  int yy_num = (Ny - (BLY_PT - BLT_PT) + BLY_PT - 1) / BLY_PT + 1; 
  int zz_num = (Nz - (BLZ_PT - BLT_PT) + BLZ_PT - 1) / BLZ_PT + 1; 

  std::vector<int> xx_heads(xx_num, 0); 
  for(int idx=1; idx<xx_num; idx++) {
    int step = (idx == 1)? BLX_PT - BLT_PT : BLX_PT;
    xx_heads[idx] = xx_heads[idx-1] + step;
  } 
  xx_heads[0] = 0 - BLT_PT;

  std::vector<int> yy_heads(yy_num, 0); 
  for(int idx=1; idx<yy_num; idx++) {
    int step = (idx == 1)? BLY_PT - BLT_PT : BLY_PT;
    yy_heads[idx] = yy_heads[idx-1] + step;
  } 
  yy_heads[0] = 0 - BLT_PT;

  std::vector<int> zz_heads(zz_num, 0); 
  for(int idx=1; idx<zz_num; idx++) {
    int step = (idx == 1)? BLZ_PT - BLT_PT : BLZ_PT;
    zz_heads[idx] = zz_heads[idx-1] + step;
  } 
  zz_heads[0] = 0 - BLT_PT;

  // find hyperplanes of tiles given Nx, Ny, Nz
  std::vector<Pt_idx> hyperplanes;
  std::vector<int> hyperplane_heads;
  std::vector<int> hyperplane_sizes;
  _find_diagonal_hyperplanes(xx_num, yy_num, zz_num, 
                             hyperplanes, 
                             hyperplane_heads, 
                             hyperplane_sizes);

  size_t block_size = BLX_PT * BLY_PT * BLZ_PT;
  size_t grid_size;  

  for(size_t tt=0; tt<num_timesteps/BLT_PT; tt++) {
    size_t num_hyperplanes = hyperplane_heads.size();
    for(size_t h=0; h<num_hyperplanes; h++) {
      grid_size = hyperplane_sizes[h];
      _updateEH_pt_seq(_Ex_simu, _Ey_simu, _Ez_simu,
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
                       Nx, Ny, Nz,
                       xx_num, yy_num, zz_num, // number of tiles in each dimensions
                       xx_heads, 
                       yy_heads, 
                       zz_heads,
                       hyperplanes,
                       hyperplane_heads[h],
                       block_size,
                       grid_size); 

    }
  }


}

void gDiamond::_updateEH_pt_seq(std::vector<float>& Ex, std::vector<float>& Ey, std::vector<float>& Ez,
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
                                const std::vector<Pt_idx>& hyperplanes,
                                int hyperplane_head,
                                size_t block_size,
                                // grid_size = hyperplane_size
                                size_t grid_size) {

  for(size_t block_id=0; block_id<grid_size; block_id++) {
    
    // map each tile on hyperplane to a block
    Pt_idx p = hyperplanes[hyperplane_head + block_id];
    int xx = p.x;
    int yy = p.y;
    int zz = p.z;

    // std::cout << "grid_size = " << grid_size << ", x = " << xx << ", y = " << yy << ", zz = " << zz << "\n";

    // declare shared memory
    float Ex_shmem[(BLX_PT + BLT_PT) * (BLY_PT + BLT_PT) * (BLZ_PT + BLT_PT)] = {};
    float Ey_shmem[(BLX_PT + BLT_PT) * (BLY_PT + BLT_PT) * (BLZ_PT + BLT_PT)] = {};
    float Ez_shmem[(BLX_PT + BLT_PT) * (BLY_PT + BLT_PT) * (BLZ_PT + BLT_PT)] = {};
    float Hx_shmem[(BLX_PT + BLT_PT) * (BLY_PT + BLT_PT) * (BLZ_PT + BLT_PT)] = {};
    float Hy_shmem[(BLX_PT + BLT_PT) * (BLY_PT + BLT_PT) * (BLZ_PT + BLT_PT)] = {};
    float Hz_shmem[(BLX_PT + BLT_PT) * (BLY_PT + BLT_PT) * (BLZ_PT + BLT_PT)] = {};

    // load shared memory
    for(size_t thread_id=0; thread_id<block_size; thread_id++) {
      for(int local_z = thread_id / (BLX_PT * BLY_PT); local_z < (BLZ_PT + BLT_PT); local_z += BLZ_PT) {
        for(int local_y = (thread_id / BLX_PT) % BLY_PT; local_y < (BLY_PT + BLT_PT); local_y += BLY_PT) {
          for(int local_x = thread_id % BLX_PT; local_x < (BLX_PT + BLT_PT); local_x += BLX_PT) {
            int global_x = local_x + xx_heads[xx];
            int global_y = local_y + yy_heads[yy];
            int global_z = local_z + zz_heads[zz];

            int global_idx = global_x + global_y * Nx + global_z * Nx * Ny;
            int local_idx = local_x + local_y * (BLX_PT + BLT_PT) + local_z * (BLX_PT + BLT_PT) * (BLY_PT + BLT_PT);

            if(global_x >= 0 && global_x <= Nx-1 &&
               global_y >= 0 && global_y <= Ny-1 &&
               global_z >= 0 && global_z <= Nz-1) {
              Ex_shmem[local_idx] = Ex[global_idx];
              Ey_shmem[local_idx] = Ey[global_idx];
              Ez_shmem[local_idx] = Ez[global_idx];
              Hx_shmem[local_idx] = Hx[global_idx];
              Hy_shmem[local_idx] = Hy[global_idx];
              Hz_shmem[local_idx] = Hz[global_idx];
            }
          }
        }
      }
    }

    // update 
    for(int t=0; t<BLT_PT; t++) {
      int offset = BLT_PT - t;

      // update E
      for(size_t thread_id=0; thread_id<block_size; thread_id++) {
        int local_x = thread_id % BLX_PT;
        int local_y = (thread_id / BLX_PT) % BLY_PT;
        int local_z = thread_id / (BLX_PT * BLY_PT);

        int shared_E_x = local_x + offset;
        int shared_E_y = local_y + offset;
        int shared_E_z = local_z + offset;

        int global_x = local_x + offset + xx_heads[xx];
        int global_y = local_y + offset + yy_heads[yy];
        int global_z = local_z + offset + zz_heads[zz];

        int shared_E_idx = shared_E_x + shared_E_y * (BLX_PT + BLT_PT) + shared_E_z * (BLX_PT + BLT_PT) * (BLY_PT + BLT_PT); 
        int global_idx = global_x + global_y * Nx + global_z * Nx * Ny;

        if(global_x >= 1 && global_x <= Nx-2 &&
           global_y >= 1 && global_y <= Ny-2 &&
           global_z >= 1 && global_z <= Nz-2) {

          Ex_shmem[shared_E_idx] = Cax[global_idx] * Ex_shmem[shared_E_idx] + Cbx[global_idx] *
                    ((Hz_shmem[shared_E_idx] - Hz_shmem[shared_E_idx - (BLX_PT + BLT_PT)]) - (Hy_shmem[shared_E_idx] - Hy_shmem[shared_E_idx - (BLX_PT + BLT_PT) * (BLY_PT + BLT_PT)]) - Jx[global_idx] * dx);
          Ey_shmem[shared_E_idx] = Cay[global_idx] * Ey_shmem[shared_E_idx] + Cby[global_idx] *
                    ((Hx_shmem[shared_E_idx] - Hx_shmem[shared_E_idx - (BLX_PT + BLT_PT) * (BLY_PT + BLT_PT)]) - (Hz_shmem[shared_E_idx] - Hz_shmem[shared_E_idx - 1]) - Jy[global_idx] * dx);
          Ez_shmem[shared_E_idx] = Caz[global_idx] * Ez_shmem[shared_E_idx] + Cbz[global_idx] *
                    ((Hy_shmem[shared_E_idx] - Hy_shmem[shared_E_idx - 1]) - (Hx_shmem[shared_E_idx] - Hx_shmem[shared_E_idx - (BLX_PT + BLT_PT)]) - Jz[global_idx] * dx);
        }
      }

      // update H
      for(size_t thread_id=0; thread_id<block_size; thread_id++) {
        int local_x = thread_id % BLX_PT;
        int local_y = (thread_id / BLX_PT) % BLY_PT;
        int local_z = thread_id / (BLX_PT * BLY_PT);

        int shared_H_x = local_x + offset - 1;
        int shared_H_y = local_y + offset - 1;
        int shared_H_z = local_z + offset - 1;

        int global_x = local_x + offset + xx_heads[xx] - 1;
        int global_y = local_y + offset + yy_heads[yy] - 1;
        int global_z = local_z + offset + zz_heads[zz] - 1;

        int shared_H_idx = shared_H_x + shared_H_y * (BLX_PT + BLT_PT) + shared_H_z * (BLX_PT + BLT_PT) * (BLY_PT + BLT_PT); 
        int global_idx = global_x + global_y * Nx + global_z * Nx * Ny;

        if(global_x >= 1 && global_x <= Nx-2 &&
           global_y >= 1 && global_y <= Ny-2 &&
           global_z >= 1 && global_z <= Nz-2) {
        
          Hx_shmem[shared_H_idx] = Dax[global_idx] * Hx_shmem[shared_H_idx] + Dbx[global_idx] *
                    ((Ey_shmem[shared_H_idx + (BLX_PT + BLT_PT) * (BLY_PT + BLT_PT)] - Ey_shmem[shared_H_idx]) - (Ez_shmem[shared_H_idx + (BLX_PT + BLT_PT)] - Ez_shmem[shared_H_idx]) - Mx[global_idx] * dx);
          Hy_shmem[shared_H_idx] = Day[global_idx] * Hy_shmem[shared_H_idx] + Dby[global_idx] *
                    ((Ez_shmem[shared_H_idx + 1] - Ez_shmem[shared_H_idx]) - (Ex_shmem[shared_H_idx + (BLX_PT + BLT_PT) * (BLY_PT + BLT_PT)] - Ex_shmem[shared_H_idx]) - My[global_idx] * dx);
          Hz_shmem[shared_H_idx] = Daz[global_idx] * Hz_shmem[shared_H_idx] + Dbz[global_idx] *
                    ((Ex_shmem[shared_H_idx + (BLX_PT + BLT_PT)] - Ex_shmem[shared_H_idx]) - (Ey_shmem[shared_H_idx + 1] - Ey_shmem[shared_H_idx]) - Mz[global_idx] * dx);
        }
      }
    }

    // store global memory
    for(size_t thread_id=0; thread_id<block_size; thread_id++) {
      for(int local_z = thread_id / (BLX_PT * BLY_PT); local_z < (BLZ_PT + BLT_PT); local_z += BLZ_PT) {
        for(int local_y = (thread_id / BLX_PT) % BLY_PT; local_y < (BLY_PT + BLT_PT); local_y += BLY_PT) {
          for(int local_x = thread_id % BLX_PT; local_x < (BLX_PT + BLT_PT); local_x += BLX_PT) {
            int global_x = local_x + xx_heads[xx];
            int global_y = local_y + yy_heads[yy];
            int global_z = local_z + zz_heads[zz];

            int global_idx = global_x + global_y * Nx + global_z * Nx * Ny;
            int local_idx = local_x + local_y * (BLX_PT + BLT_PT) + local_z * (BLX_PT + BLT_PT) * (BLY_PT + BLT_PT);

            if(global_x >= 0 && global_x <= Nx-1 &&
               global_y >= 0 && global_y <= Ny-1 &&
               global_z >= 0 && global_z <= Nz-1) {
              Ex[global_idx] = Ex_shmem[local_idx];
              Ey[global_idx] = Ey_shmem[local_idx];
              Ez[global_idx] = Ez_shmem[local_idx];
              Hx[global_idx] = Hx_shmem[local_idx];
              Hy[global_idx] = Hy_shmem[local_idx];
              Hz[global_idx] = Hz_shmem[local_idx];
            }
          }
        }
      }
    }

  }

}

void gDiamond::update_FDTD_gpu_pt(size_t num_timesteps) { // GPU 3-D implementation of parallelogram tiling

  // clear source Mz for experiments
  _Mz.clear();

  // transfer source
  for(size_t t=0; t<num_timesteps; t++) {
    float Mz_value = M_source_amp * std::sin(SOURCE_OMEGA * t * dt);
    _Mz[_source_idx] = Mz_value;
  }

  int Nx = _Nx;
  int Ny = _Ny;
  int Nz = _Nz;

  // number of parallelogram tiles in each dimension 
  int xx_num = (Nx - (BLX_PT - BLT_PT) + BLX_PT - 1) / BLX_PT + 1; 
  int yy_num = (Ny - (BLY_PT - BLT_PT) + BLY_PT - 1) / BLY_PT + 1; 
  int zz_num = (Nz - (BLZ_PT - BLT_PT) + BLZ_PT - 1) / BLZ_PT + 1; 

  std::vector<int> xx_heads(xx_num, 0); 
  for(int idx=1; idx<xx_num; idx++) {
    int step = (idx == 1)? BLX_PT - BLT_PT : BLX_PT;
    xx_heads[idx] = xx_heads[idx-1] + step;
  } 
  xx_heads[0] = 0 - BLT_PT;

  std::vector<int> yy_heads(yy_num, 0); 
  for(int idx=1; idx<yy_num; idx++) {
    int step = (idx == 1)? BLY_PT - BLT_PT : BLY_PT;
    yy_heads[idx] = yy_heads[idx-1] + step;
  } 
  yy_heads[0] = 0 - BLT_PT;

  std::vector<int> zz_heads(zz_num, 0); 
  for(int idx=1; idx<zz_num; idx++) {
    int step = (idx == 1)? BLZ_PT - BLT_PT : BLZ_PT;
    zz_heads[idx] = zz_heads[idx-1] + step;
  } 
  zz_heads[0] = 0 - BLT_PT;

  // find hyperplanes of tiles given Nx, Ny, Nz
  std::vector<Pt_idx> hyperplanes;
  std::vector<int> hyperplane_heads;
  std::vector<int> hyperplane_sizes;
  _find_diagonal_hyperplanes(xx_num, yy_num, zz_num, 
                             hyperplanes, 
                             hyperplane_heads, 
                             hyperplane_sizes);

  // E, H, J, M on device 
  float *Ex, *Ey, *Ez, *Hx, *Hy, *Hz, *Jx, *Jy, *Jz, *Mx, *My, *Mz;

  // Ca, Cb, Da, Db on device
  float *Cax, *Cay, *Caz, *Cbx, *Cby, *Cbz;
  float *Dax, *Day, *Daz, *Dbx, *Dby, *Dbz;

  // data for parallelogram tiling
  Pt_idx *hyperplanes_d;  
  int *xx_heads_d, *yy_heads_d, *zz_heads_d;

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
  CUDACHECK(cudaMalloc(&hyperplanes_d, sizeof(Pt_idx) * hyperplanes.size()));
  CUDACHECK(cudaMalloc(&xx_heads_d, sizeof(int) * xx_heads.size()));
  CUDACHECK(cudaMalloc(&yy_heads_d, sizeof(int) * yy_heads.size()));
  CUDACHECK(cudaMalloc(&zz_heads_d, sizeof(int) * zz_heads.size()));

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

  // copy data for parallelogram tiling
  CUDACHECK(cudaMemcpyAsync(hyperplanes_d, hyperplanes.data(), sizeof(Pt_idx) * hyperplanes.size(), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(xx_heads_d, xx_heads.data(), sizeof(int) * xx_heads.size(), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(yy_heads_d, yy_heads.data(), sizeof(int) * yy_heads.size(), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(zz_heads_d, zz_heads.data(), sizeof(int) * zz_heads.size(), cudaMemcpyHostToDevice));

  size_t block_size = BLX_PT * BLY_PT * BLZ_PT;
  size_t grid_size;

  for(size_t tt=0; tt<num_timesteps/BLT_PT; tt++) {
    size_t num_hyperplanes = hyperplane_heads.size();
    for(size_t h=0; h<num_hyperplanes; h++) {
      grid_size = hyperplane_sizes[h];
      updateEH_pt<<<grid_size, block_size>>>(Ex, Ey, Ez,
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
                                             Nx, Ny, Nz,
                                             xx_num, yy_num, zz_num, // number of tiles in each dimensions
                                             xx_heads_d, 
                                             yy_heads_d, 
                                             zz_heads_d,
                                             hyperplanes_d,
                                             hyperplane_heads[h]); 
    }
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
  std::cout << "gpu runtime (parallelogram tiling 3-D mapping, shared memory for EH only): " << std::chrono::duration<double>(end-start).count() << "s\n";
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

  CUDACHECK(cudaFree(hyperplanes_d));
  CUDACHECK(cudaFree(xx_heads_d));
  CUDACHECK(cudaFree(yy_heads_d));
  CUDACHECK(cudaFree(zz_heads_d));

}

void gDiamond::update_FDTD_gpu_shmem_no_deps_obeyed(size_t num_timesteps) { // GPU 3-D implementation naive, with shared memory, but does not obey dependencies

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

  // get xx_num, yy_num, zz_num
  int xx_num = (_Nx + BLX_UB - 1) / BLX_UB;
  int yy_num = (_Ny + BLY_UB - 1) / BLY_UB;
  int zz_num = (_Nz + BLZ_UB - 1) / BLZ_UB;

  // get xx_heads, yy_heads, zz_heads
  std::vector<int> xx_heads(xx_num, 0);
  std::vector<int> yy_heads(yy_num, 0);
  std::vector<int> zz_heads(zz_num, 0);
  for(int i=0; i<xx_num; i++) {
    xx_heads[i] = i * BLX_UB;
  }
  for(int i=0; i<yy_num; i++) {
    yy_heads[i] = i * BLY_UB;
  }
  for(int i=0; i<zz_num; i++) {
    zz_heads[i] = i * BLZ_UB;
  }

  int *xx_heads_d, *yy_heads_d, *zz_heads_d;

  CUDACHECK(cudaMalloc(&xx_heads_d, sizeof(int) * xx_num));
  CUDACHECK(cudaMalloc(&yy_heads_d, sizeof(int) * yy_num));
  CUDACHECK(cudaMalloc(&zz_heads_d, sizeof(int) * zz_num));
  
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
  
  // copy upper bound parameters
  CUDACHECK(cudaMemcpy(xx_heads_d, xx_heads.data(), sizeof(int) * xx_num, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(yy_heads_d, yy_heads.data(), sizeof(int) * yy_num, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(zz_heads_d, zz_heads.data(), sizeof(int) * zz_num, cudaMemcpyHostToDevice));

  // set block and grid
  size_t block_size = BLX_UB * BLY_UB * BLZ_UB;
  size_t grid_size = xx_num * yy_num * zz_num;

  for(size_t t=0; t<num_timesteps; t++) {
    updateE_ub_globalmem_only<<<grid_size, block_size>>>(Ex, Ey, Ez,
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
                                                         xx_num, yy_num, zz_num,
                                                         xx_heads_d, yy_heads_d, zz_heads_d);

    updateH_ub_globalmem_only<<<grid_size, block_size>>>(Ex, Ey, Ez,
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
                                                         xx_num, yy_num, zz_num,
                                                         xx_heads_d, yy_heads_d, zz_heads_d);
  
  }
  
  // for(size_t t=0; t<num_timesteps/BLT_UB; t++) {
  //   updateEH_ub<<<grid_size, block_size>>>(Ex, Ey, Ez,
  //                                          Hx, Hy, Hz,
  //                                          Cax, Cbx,
  //                                          Cay, Cby,
  //                                          Caz, Cbz,
  //                                          Dax, Dbx,
  //                                          Day, Dby,
  //                                          Daz, Dbz,
  //                                          Jx, Jy, Jz,
  //                                          Mx, My, Mz,
  //                                          _dx, 
  //                                          _Nx, _Ny, _Nz,
  //                                          xx_num, yy_num, zz_num,
  //                                          xx_heads_d, yy_heads_d, zz_heads_d);

  //   cudaError_t err = cudaGetLastError();
  //   if (err != cudaSuccess) {
  //       std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
  //   }

  // }

  cudaDeviceSynchronize();

  // copy E, H back to host 
  CUDACHECK(cudaMemcpy(_Ex_gpu.data(), Ex, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Ey_gpu.data(), Ey, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Ez_gpu.data(), Ez, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Hx_gpu.data(), Hx, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Hy_gpu.data(), Hy, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(_Hz_gpu.data(), Hz, sizeof(float) * _Nx * _Ny * _Nz, cudaMemcpyDeviceToHost));

  auto end = std::chrono::high_resolution_clock::now();
  // std::cout << "gpu runtime (3-D mapping, naive, upper bound, shared memory in E, H): " << std::chrono::duration<double>(end-start).count() << "s\n"; 
  std::cout << "gpu runtime (3-D mapping, naive, upper bound, global memory): " << std::chrono::duration<double>(end-start).count() << "s\n"; 
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

  CUDACHECK(cudaFree(xx_heads_d));
  CUDACHECK(cudaFree(yy_heads_d));
  CUDACHECK(cudaFree(zz_heads_d));

}

void gDiamond::update_FDTD_cpu_simulation_dt_1_D(size_t num_timesteps, size_t Tx, size_t Ty, size_t Tz) { // CPU single thread 1-D simulation of diamond tiling, reimplemented

  // write 1 dimension just to check
  std::vector<float> E_simu(_Nx, 1);
  std::vector<float> H_simu(_Nx, 1);
  std::vector<float> E_seq(_Nx, 1);
  std::vector<float> H_seq(_Nx, 1);

  int Nx = _Nx;

  // seq version
  for(size_t t=0; t<num_timesteps; t++) {

    // update E
    for(int x=1; x<Nx-1; x++) {
      E_seq[x] = H_seq[x-1] + H_seq[x] * 2; 
    }

    std::cout << "t = " << t << ", E_seq =";
    for(int x=0; x<Nx; x++) {
      std::cout << E_seq[x] << " ";
    }
    std::cout << "\n";

    // update H 
    for(int x=1; x<Nx-1; x++) {
      H_seq[x] = E_seq[x+1] + E_seq[x] * 2; 
    }
  }

  // diamond tiling

  // before tiling, zero padding E and H
  int left_pad = BLT_UB - 1 + 1; // + 1 for shmem load
  int right_pad = BLT_UB - 1;
  int Nx_pad = Nx + left_pad + right_pad;

  std::vector<float> E_pad(Nx_pad, 0);
  std::vector<float> H_pad(Nx_pad, 0);

  // copy E, H to padded array
  for(int i=0; i<Nx; i++) {
    E_pad[i + left_pad] = E_simu[i]; 
    H_pad[i + left_pad] = H_simu[i]; 
  }

  std::cout << "E_pad = ";
  for(const auto& data : E_pad) {
    std::cout << data << " ";
  }
  std::cout << "\n";

  int xx_num_mountains = 1 + Tx;
  int xx_num_valleys = Tx + 1;

  // xx_heads_mountain[xx] is 1 element left offset to the actual mountain
  std::vector<int> xx_heads_mountain(xx_num_mountains, 0);
  std::vector<int> xx_heads_valley(xx_num_valleys, 0);

  for(int index=0; index<xx_num_mountains; index++) {
    xx_heads_mountain[index] = (index == 0)? 0 :
                               xx_heads_mountain[index-1] + (11 + 4);
  }

  for(int index=0; index<xx_num_valleys; index++) {
    xx_heads_valley[index] = (index == 0)? 8 : 
                             xx_heads_valley[index-1] + (11 + 4);
  }

  std::cout << "xx_heads_mountain = ";
  for(const auto& index : xx_heads_mountain) {
    std::cout << index << " ";
  }
  std::cout << "\n";

  std::cout << "xx_heads_valley = ";
  for(const auto& index : xx_heads_valley) {
    std::cout << index << " ";
  }
  std::cout << "\n";

  int block_size = 4; // each block has 4 threads
  bool is_mountain; // true : mountain, false : valley
  int shmem_size = block_size + 2 * (BLT_UB - 1) + 1 + 1; // BLX + 1
  for(size_t tt=0; tt<num_timesteps/BLT_UB; tt++) {
    // phase 1: mountain
    is_mountain = true;
    for(int xx=0; xx<xx_num_mountains; xx++) {
 
      // declare shared memory
      float E_shmem[shmem_size];
      float H_shmem[shmem_size];

      // load shared memory
      for(int tid=0; tid<block_size; tid++) {
        for(int shared_idx=tid; shared_idx<shmem_size; shared_idx+=block_size) {
          int global_idx = xx_heads_mountain[xx] + shared_idx;
          E_shmem[shared_idx] = E_pad[global_idx];
          H_shmem[shared_idx] = H_pad[global_idx];
        }
      }

      // calculation
      int cal_offsetE, cal_offsetH;
      int cal_boundE, cal_boundH;
      for(int t=0; t<BLT_UB; t++) {
        cal_offsetE = (is_mountain)? t + 1 : BLT_UB - t;
        cal_offsetH = (is_mountain)? cal_offsetE : cal_offsetE - 1;
        cal_boundE = (is_mountain)? shmem_size - t : shmem_size - (BLT_UB - t);
        cal_boundH = (is_mountain)? cal_boundE - 1 : cal_boundE;

        // update E
        for(int tid=0; tid<block_size; tid++) {
          for(int shared_idx=tid+cal_offsetE; shared_idx<cal_boundE; shared_idx+=block_size) {
            int global_idx = xx_heads_mountain[xx] + shared_idx;
            // std::cout << "t = " << t << ", ";
            // std::cout << "xx = " << xx << ", shared_idx = " << shared_idx << ", global_idx = " << global_idx << "\n";
            if(global_idx >= 1 + left_pad && global_idx <= Nx - 2 + left_pad) {
              E_shmem[shared_idx] = H_shmem[shared_idx-1] + H_shmem[shared_idx] * 2;
            }
          }
        }

        // update H
        for(int tid=0; tid<block_size; tid++) {
          for(int shared_idx=tid+cal_offsetH; shared_idx<cal_boundH; shared_idx+=block_size) {
            int global_idx = xx_heads_mountain[xx] + shared_idx;
            // std::cout << "t = " << t << ", ";
            // std::cout << "xx = " << xx << ", shared_idx = " << shared_idx << ", global_idx = " << global_idx << "\n";
            if(global_idx >= 1 + left_pad && global_idx <= Nx - 2 + left_pad) {
              H_shmem[shared_idx] = E_shmem[shared_idx+1] + E_shmem[shared_idx] * 2;
            }
          }
        }
      }

      // store to global memory
      int store_offsetE, store_offsetH;
      int store_boundE, store_boundH;
      store_offsetE = 1;
      store_offsetH = (is_mountain)? 1 : 0;
      store_boundE = (is_mountain)? shmem_size : shmem_size - 1;
      store_boundH = shmem_size - 1;
      for(int tid=0; tid<block_size; tid++) {

        // store E
        for(int shared_idx=tid + store_offsetE; shared_idx<store_boundE; shared_idx+=block_size) {
          int global_idx = xx_heads_mountain[xx] + shared_idx;
          E_pad[global_idx] = E_shmem[shared_idx];
        }

        // update H
        for(int shared_idx=tid + store_offsetH; shared_idx<store_boundH; shared_idx+=block_size) {
          int global_idx = xx_heads_mountain[xx] + shared_idx;
          H_pad[global_idx] = H_shmem[shared_idx];
        }

      }
    }

    // phase 2: valley 
    is_mountain = false;
    for(int xx=0; xx<xx_num_valleys; xx++) {
 
      // declare shared memory
      float E_shmem[shmem_size];
      float H_shmem[shmem_size];

      // load shared memory
      for(int tid=0; tid<block_size; tid++) {
        for(int shared_idx=tid; shared_idx<shmem_size; shared_idx+=block_size) {
          int global_idx = xx_heads_valley[xx] + shared_idx;
          E_shmem[shared_idx] = E_pad[global_idx];
          H_shmem[shared_idx] = H_pad[global_idx];
        }
      }

      // calculation
      int cal_offsetE, cal_offsetH;
      int cal_boundE, cal_boundH;
      for(int t=0; t<BLT_UB; t++) {
        cal_offsetE = (is_mountain)? t + 1 : BLT_UB - t;
        cal_offsetH = (is_mountain)? cal_offsetE : cal_offsetE - 1;
        cal_boundE = (is_mountain)? shmem_size - t : shmem_size - (BLT_UB - t);
        cal_boundH = (is_mountain)? cal_boundE - 1 : cal_boundE;

        // update E
        for(int tid=0; tid<block_size; tid++) {
          for(int shared_idx=tid+cal_offsetE; shared_idx<cal_boundE; shared_idx+=block_size) {
            int global_idx = xx_heads_valley[xx] + shared_idx;
            if(global_idx >= 1 + left_pad && global_idx <= Nx - 2 + left_pad) {
              E_shmem[shared_idx] = H_shmem[shared_idx-1] + H_shmem[shared_idx] * 2;
            }
          }
        }

        // update H
        for(int tid=0; tid<block_size; tid++) {
          for(int shared_idx=tid+cal_offsetH; shared_idx<cal_boundH; shared_idx+=block_size) {
            int global_idx = xx_heads_valley[xx] + shared_idx;
            // std::cout << "t = " << t << ", ";
            // std::cout << "xx = " << xx << ", shared_idx = " << shared_idx << ", global_idx = " << global_idx << "\n";
            if(global_idx >= 1 + left_pad && global_idx <= Nx - 2 + left_pad) {
              H_shmem[shared_idx] = E_shmem[shared_idx+1] + E_shmem[shared_idx] * 2;
            }
          }
        }
      }

      // store to global memory
      int store_offsetE, store_offsetH;
      int store_boundE, store_boundH;
      store_offsetE = 1;
      store_offsetH = (is_mountain)? 1 : 0;
      store_boundE = (is_mountain)? shmem_size : shmem_size - 1;
      store_boundH = shmem_size - 1;
      for(int tid=0; tid<block_size; tid++) {

        // store E
        for(int shared_idx=tid + store_offsetE; shared_idx<store_boundE; shared_idx+=block_size) {
          int global_idx = xx_heads_valley[xx] + shared_idx;
          E_pad[global_idx] = E_shmem[shared_idx];
        }

        // update H
        for(int shared_idx=tid + store_offsetH; shared_idx<store_boundH; shared_idx+=block_size) {
          int global_idx = xx_heads_valley[xx] + shared_idx;
          H_pad[global_idx] = H_shmem[shared_idx];
        }

      }
    }
  }

  // copy E_pad back to E_simu
  for(int i=0; i<Nx; i++) {
    E_simu[i] = E_pad[i + left_pad];
    H_simu[i] = H_pad[i + left_pad];
  } 

  std::cout << "E_seq = ";
  for(int x=0; x<Nx; x++) {
    std::cout << E_seq[x] << " ";
  }
  std::cout << "\n";

  std::cout << "E_simu = ";
  for(int x=0; x<Nx; x++) {
    std::cout << E_simu[x] << " ";
  }
  std::cout << "\n";

  for(int x=0; x<Nx; x++) {
    if(E_seq[x] != E_simu[x] || H_seq[x] != H_simu[x]) {
      std::cerr << "1-D demo results mismatch.\n";
      std::exit(EXIT_FAILURE);
    }
  }
}

void gDiamond::update_FDTD_cpu_simulation_dt_3_D(size_t num_timesteps, size_t Tx, size_t Ty, size_t Tz) { // CPU single thread 3-D simulation of diamond tiling, reimplemented
                                                                                                // only apply tiling on X, Y dimension

  // clear source Mz for experiments
  _Mz.clear();

  // transfer source
  for(size_t t=0; t<num_timesteps; t++) {
    float Mz_value = M_source_amp * std::sin(SOURCE_OMEGA * t * dt);
    _Mz[_source_idx] = Mz_value;
  }

  // tiling parameter pre-processing 
  int Nx = _Nx;
  int Ny = _Ny;
  int Nz = _Nz;
  int left_pad = BLT_UB - 1 + 1; // + 1 for shmem load
  int right_pad = BLT_UB - 1;
  int Nx_pad = Nx + left_pad + right_pad;
  int Ny_pad = Ny + left_pad + right_pad;

  int xx_num_mountains = 1 + Tx;
  int xx_num_valleys = Tx + 1;
  int yy_num_mountains = 1 + Ty;
  int yy_num_valleys = Ty + 1;

  // xx_heads_mountain[xx] is 1 element left offset to the actual mountain
  std::vector<int> xx_heads_mountain(xx_num_mountains, 0);
  std::vector<int> xx_heads_valley(xx_num_valleys, 0);
  std::vector<int> yy_heads_mountain(yy_num_mountains, 0);
  std::vector<int> yy_heads_valley(yy_num_valleys, 0);

  for(int index=0; index<xx_num_mountains; index++) {
    xx_heads_mountain[index] = (index == 0)? 0 :
                               xx_heads_mountain[index-1] + (BLX_DTR + NTX);
  }
  for(int index=0; index<xx_num_valleys; index++) {
    xx_heads_valley[index] = (index == 0)? BLX_DTR - (BLT_DTR - 1) : 
                             xx_heads_valley[index-1] + (BLX_DTR + NTX);
  }
  for(int index=0; index<yy_num_mountains; index++) {
    yy_heads_mountain[index] = (index == 0)? 0 :
                               yy_heads_mountain[index-1] + (BLY_DTR + NTY);
  }
  for(int index=0; index<yy_num_valleys; index++) {
    yy_heads_valley[index] = (index == 0)? BLY_DTR - (BLT_DTR - 1) : 
                             yy_heads_valley[index-1] + (BLY_DTR + NTY);
  }

  std::cout << "xx_heads_mountain = ";
  for(const auto& index : xx_heads_mountain) {
    std::cout << index << " ";
  }
  std::cout << "\n";

  std::cout << "xx_heads_valley = ";
  for(const auto& index : xx_heads_valley) {
    std::cout << index << " ";
  }
  std::cout << "\n";

  std::cout << "yy_heads_mountain = ";
  for(const auto& index : yy_heads_mountain) {
    std::cout << index << " ";
  }
  std::cout << "\n";

  std::cout << "yy_heads_valley = ";
  for(const auto& index : yy_heads_valley) {
    std::cout << index << " ";
  }
  std::cout << "\n";

  std::cout << "SHX = " << SHX << "\n";
  std::cout << "SHY = " << SHY << "\n";

  // padded E, H
  std::vector<float> Ex_src(Nx_pad * Ny_pad * Nz, 0); 
  std::vector<float> Ey_src(Nx_pad * Ny_pad * Nz, 0); 
  std::vector<float> Ez_src(Nx_pad * Ny_pad * Nz, 0); 
  std::vector<float> Hx_src(Nx_pad * Ny_pad * Nz, 0); 
  std::vector<float> Hy_src(Nx_pad * Ny_pad * Nz, 0); 
  std::vector<float> Hz_src(Nx_pad * Ny_pad * Nz, 0); 

  std::vector<float> Ex_dst(Nx_pad * Ny_pad * Nz, 0); 
  std::vector<float> Ey_dst(Nx_pad * Ny_pad * Nz, 0); 
  std::vector<float> Ez_dst(Nx_pad * Ny_pad * Nz, 0); 
  std::vector<float> Hx_dst(Nx_pad * Ny_pad * Nz, 0); 
  std::vector<float> Hy_dst(Nx_pad * Ny_pad * Nz, 0); 
  std::vector<float> Hz_dst(Nx_pad * Ny_pad * Nz, 0); 

  // padded Ca, Cb, Da, Db, J, M
  std::vector<float> Cax_pad(Nx_pad * Ny_pad * Nz, 0); 
  std::vector<float> Cay_pad(Nx_pad * Ny_pad * Nz, 0); 
  std::vector<float> Caz_pad(Nx_pad * Ny_pad * Nz, 0); 
  std::vector<float> Cbx_pad(Nx_pad * Ny_pad * Nz, 0); 
  std::vector<float> Cby_pad(Nx_pad * Ny_pad * Nz, 0); 
  std::vector<float> Cbz_pad(Nx_pad * Ny_pad * Nz, 0); 
  std::vector<float> Dax_pad(Nx_pad * Ny_pad * Nz, 0); 
  std::vector<float> Day_pad(Nx_pad * Ny_pad * Nz, 0); 
  std::vector<float> Daz_pad(Nx_pad * Ny_pad * Nz, 0); 
  std::vector<float> Dbx_pad(Nx_pad * Ny_pad * Nz, 0); 
  std::vector<float> Dby_pad(Nx_pad * Ny_pad * Nz, 0); 
  std::vector<float> Dbz_pad(Nx_pad * Ny_pad * Nz, 0); 
  std::vector<float> Jx_pad(Nx_pad * Ny_pad * Nz, 0); 
  std::vector<float> Jy_pad(Nx_pad * Ny_pad * Nz, 0); 
  std::vector<float> Jz_pad(Nx_pad * Ny_pad * Nz, 0); 
  std::vector<float> Mx_pad(Nx_pad * Ny_pad * Nz, 0); 
  std::vector<float> My_pad(Nx_pad * Ny_pad * Nz, 0); 
  std::vector<float> Mz_pad(Nx_pad * Ny_pad * Nz, 0); 

  padXY_1D_col_major(_Cax, Cax_pad, Nx, Ny, Nz, left_pad, right_pad); 
  padXY_1D_col_major(_Cay, Cay_pad, Nx, Ny, Nz, left_pad, right_pad); 
  padXY_1D_col_major(_Caz, Caz_pad, Nx, Ny, Nz, left_pad, right_pad); 
  padXY_1D_col_major(_Cbx, Cbx_pad, Nx, Ny, Nz, left_pad, right_pad); 
  padXY_1D_col_major(_Cby, Cby_pad, Nx, Ny, Nz, left_pad, right_pad); 
  padXY_1D_col_major(_Cbz, Cbz_pad, Nx, Ny, Nz, left_pad, right_pad); 
  padXY_1D_col_major(_Dax, Dax_pad, Nx, Ny, Nz, left_pad, right_pad); 
  padXY_1D_col_major(_Day, Day_pad, Nx, Ny, Nz, left_pad, right_pad); 
  padXY_1D_col_major(_Daz, Daz_pad, Nx, Ny, Nz, left_pad, right_pad); 
  padXY_1D_col_major(_Dbx, Dbx_pad, Nx, Ny, Nz, left_pad, right_pad); 
  padXY_1D_col_major(_Dby, Dby_pad, Nx, Ny, Nz, left_pad, right_pad); 
  padXY_1D_col_major(_Dbz, Dbz_pad, Nx, Ny, Nz, left_pad, right_pad); 
  padXY_1D_col_major(_Jx, Jx_pad, Nx, Ny, Nz, left_pad, right_pad); 
  padXY_1D_col_major(_Jy, Jy_pad, Nx, Ny, Nz, left_pad, right_pad); 
  padXY_1D_col_major(_Jz, Jz_pad, Nx, Ny, Nz, left_pad, right_pad); 
  padXY_1D_col_major(_Mx, Mx_pad, Nx, Ny, Nz, left_pad, right_pad); 
  padXY_1D_col_major(_My, My_pad, Nx, Ny, Nz, left_pad, right_pad); 
  padXY_1D_col_major(_Mz, Mz_pad, Nx, Ny, Nz, left_pad, right_pad); 

  // define block size and grid size
  size_t block_size = NTX * NTY;
  size_t grid_size; // grid_size = xx_num * yy_num * Nz;

  // // initialize E to check shared memory load
  // float data = 1;
  // for(int z=0; z<Nz; z++) {
  //   for(int y=0; y<Ny; y++) {
  //     for(int x=0; x<Nx; x++) {
  //       int idx = (x + left_pad) + (y + left_pad) * Nx_pad + z * Nx_pad * Ny_pad;
  //       Ex_src[idx] = data;
  //       Ey_src[idx] = data;
  //       Ez_src[idx] = data;
  //       Hx_src[idx] = data;
  //       Hy_src[idx] = data;
  //       Hz_src[idx] = data;
  //       data++;
  //     }
  //   }
  // }

  // initialize E to check shared memory load
  for(int z=0; z<Nz; z++) {
    for(int y=0; y<Ny_pad; y++) {
      for(int x=0; x<Nx_pad; x++) {
        int idx = x + y * Nx_pad + z * Nx_pad * Ny_pad;
        float data = 1.0 * idx;
        Ex_src[idx] = data;
        Ey_src[idx] = data;
        Ez_src[idx] = data;
        Hx_src[idx] = data;
        Hy_src[idx] = data;
        Hz_src[idx] = data;
      }
    }
  }

  /*
  for(size_t tt=0; tt<num_timesteps/BLT_DTR; tt++) {

    // phase 1: (m, m)
    grid_size = xx_num_mountains * yy_num_mountains * Nz; 
    _updateEH_dt_2D_seq(Ex_src, Ey_src, Ez_src,
                        Hx_src, Hy_src, Hz_src,
                        Ex_dst, Ey_dst, Ez_dst,
                        Hx_dst, Hy_dst, Hz_dst,
                        Cax_pad, Cbx_pad,
                        Cay_pad, Cby_pad,
                        Caz_pad, Cbz_pad,
                        Dax_pad, Dbx_pad,
                        Day_pad, Dby_pad,
                        Daz_pad, Dbz_pad,
                        Jx_pad, Jy_pad, Jz_pad,
                        Mx_pad, My_pad, Mz_pad,
                        _dx, 
                        Nx, Ny, Nz,
                        Nx_pad, Ny_pad,
                        xx_num_mountains, yy_num_mountains, // number of tiles in each dimensions
                        true, true,
                        xx_heads_mountain, 
                        yy_heads_mountain, 
                        left_pad,
                        block_size,
                        grid_size); 

    std::swap(Ex_src, Ex_dst);
    std::swap(Ey_src, Ey_dst);
    std::swap(Ez_src, Ez_dst);
    std::swap(Hx_src, Hx_dst);
    std::swap(Hy_src, Hy_dst);
    std::swap(Hz_src, Hz_dst);
  
    // phase 2: (v, m)
    grid_size = xx_num_valleys * yy_num_mountains * Nz; 
    _updateEH_dt_2D_seq(Ex_src, Ey_src, Ez_src,
                        Hx_src, Hy_src, Hz_src,
                        Ex_dst, Ey_dst, Ez_dst,
                        Hx_dst, Hy_dst, Hz_dst,
                        Cax_pad, Cbx_pad,
                        Cay_pad, Cby_pad,
                        Caz_pad, Cbz_pad,
                        Dax_pad, Dbx_pad,
                        Day_pad, Dby_pad,
                        Daz_pad, Dbz_pad,
                        Jx_pad, Jy_pad, Jz_pad,
                        Mx_pad, My_pad, Mz_pad,
                        _dx, 
                        Nx, Ny, Nz,
                        Nx_pad, Ny_pad,
                        xx_num_valleys, yy_num_mountains, // number of tiles in each dimensions
                        false, true,
                        xx_heads_valley, 
                        yy_heads_mountain, 
                        left_pad,
                        block_size,
                        grid_size); 

    std::swap(Ex_src, Ex_dst);
    std::swap(Ey_src, Ey_dst);
    std::swap(Ez_src, Ez_dst);
    std::swap(Hx_src, Hx_dst);
    std::swap(Hy_src, Hy_dst);
    std::swap(Hz_src, Hz_dst);

    // phase 3: (m, v)
    grid_size = xx_num_mountains * yy_num_valleys * Nz; 
    _updateEH_dt_2D_seq(Ex_src, Ey_src, Ez_src,
                        Hx_src, Hy_src, Hz_src,
                        Ex_dst, Ey_dst, Ez_dst,
                        Hx_dst, Hy_dst, Hz_dst,
                        Cax_pad, Cbx_pad,
                        Cay_pad, Cby_pad,
                        Caz_pad, Cbz_pad,
                        Dax_pad, Dbx_pad,
                        Day_pad, Dby_pad,
                        Daz_pad, Dbz_pad,
                        Jx_pad, Jy_pad, Jz_pad,
                        Mx_pad, My_pad, Mz_pad,
                        _dx, 
                        Nx, Ny, Nz,
                        Nx_pad, Ny_pad,
                        xx_num_mountains, yy_num_valleys, // number of tiles in each dimensions
                        true, false,
                        xx_heads_mountain, 
                        yy_heads_valley, 
                        left_pad,
                        block_size,
                        grid_size); 

    std::swap(Ex_src, Ex_dst);
    std::swap(Ey_src, Ey_dst);
    std::swap(Ez_src, Ez_dst);
    std::swap(Hx_src, Hx_dst);
    std::swap(Hy_src, Hy_dst);
    std::swap(Hz_src, Hz_dst);

    // phase 4: (v, v)
    grid_size = xx_num_valleys * yy_num_valleys * Nz; 
    _updateEH_dt_2D_seq(Ex_src, Ey_src, Ez_src,
                        Hx_src, Hy_src, Hz_src,
                        Ex_dst, Ey_dst, Ez_dst,
                        Hx_dst, Hy_dst, Hz_dst,
                        Cax_pad, Cbx_pad,
                        Cay_pad, Cby_pad,
                        Caz_pad, Cbz_pad,
                        Dax_pad, Dbx_pad,
                        Day_pad, Dby_pad,
                        Daz_pad, Dbz_pad,
                        Jx_pad, Jy_pad, Jz_pad,
                        Mx_pad, My_pad, Mz_pad,
                        _dx, 
                        Nx, Ny, Nz,
                        Nx_pad, Ny_pad,
                        xx_num_valleys, yy_num_valleys, // number of tiles in each dimensions
                        false, false,
                        xx_heads_valley, 
                        yy_heads_valley, 
                        left_pad,
                        block_size,
                        grid_size); 

    std::swap(Ex_src, Ex_dst);
    std::swap(Ey_src, Ey_dst);
    std::swap(Ez_src, Ez_dst);
    std::swap(Hx_src, Hx_dst);
    std::swap(Hy_src, Hy_dst);
    std::swap(Hz_src, Hz_dst);

  }
  */
      
  std::cout << "A : Ex_src[1164] = " << Ex_src[1164] << "\n";

  for(size_t tt=0; tt<num_timesteps/BLT_DTR; tt++) {

    // phase 1: (m, m)
    grid_size = xx_num_mountains * yy_num_mountains * Nz; 
    _updateEH_dt_2D_seq(Ex_src, Ey_src, Ez_src,
                        Hx_src, Hy_src, Hz_src,
                        Ex_dst, Ey_dst, Ez_dst,
                        Hx_dst, Hy_dst, Hz_dst,
                        _Cax, _Cbx,
                        _Cay, _Cby,
                        _Caz, _Cbz,
                        _Dax, _Dbx,
                        _Day, _Dby,
                        _Daz, _Dbz,
                        _Jx, _Jy, _Jz,
                        _Mx, _My, _Mz,
                        _dx, 
                        Nx, Ny, Nz,
                        Nx_pad, Ny_pad,
                        xx_num_mountains, yy_num_mountains, // number of tiles in each dimensions
                        true, true,
                        xx_heads_mountain, 
                        yy_heads_mountain, 
                        left_pad,
                        block_size,
                        grid_size); 

    std::swap(Ex_src, Ex_dst);
    std::swap(Ey_src, Ey_dst);
    std::swap(Ez_src, Ez_dst);
    std::swap(Hx_src, Hx_dst);
    std::swap(Hy_src, Hy_dst);
    std::swap(Hz_src, Hz_dst);

    std::cout << "phase 1 : Ex_src[1164] = " << Ex_src[1164] << "\n";
  
    // // phase 2: (v, m)
    // grid_size = xx_num_valleys * yy_num_mountains * Nz; 
    // _updateEH_dt_2D_seq(Ex_src, Ey_src, Ez_src,
    //                     Hx_src, Hy_src, Hz_src,
    //                     Ex_dst, Ey_dst, Ez_dst,
    //                     Hx_dst, Hy_dst, Hz_dst,
    //                     _Cax, _Cbx,
    //                     _Cay, _Cby,
    //                     _Caz, _Cbz,
    //                     _Dax, _Dbx,
    //                     _Day, _Dby,
    //                     _Daz, _Dbz,
    //                     _Jx, _Jy, _Jz,
    //                     _Mx, _My, _Mz,
    //                     _dx, 
    //                     Nx, Ny, Nz,
    //                     Nx_pad, Ny_pad,
    //                     xx_num_valleys, yy_num_mountains, // number of tiles in each dimensions
    //                     false, true,
    //                     xx_heads_valley, 
    //                     yy_heads_mountain, 
    //                     left_pad,
    //                     block_size,
    //                     grid_size); 

    // std::swap(Ex_src, Ex_dst);
    // std::swap(Ey_src, Ey_dst);
    // std::swap(Ez_src, Ez_dst);
    // std::swap(Hx_src, Hx_dst);
    // std::swap(Hy_src, Hy_dst);
    // std::swap(Hz_src, Hz_dst);

    // std::cout << "phase 2 : Ex_src[1164] = " << Ex_src[1164] << "\n";

    // // phase 3: (m, v)
    // grid_size = xx_num_mountains * yy_num_valleys * Nz; 
    // _updateEH_dt_2D_seq(Ex_src, Ey_src, Ez_src,
    //                     Hx_src, Hy_src, Hz_src,
    //                     Ex_dst, Ey_dst, Ez_dst,
    //                     Hx_dst, Hy_dst, Hz_dst,
    //                     _Cax, _Cbx,
    //                     _Cay, _Cby,
    //                     _Caz, _Cbz,
    //                     _Dax, _Dbx,
    //                     _Day, _Dby,
    //                     _Daz, _Dbz,
    //                     _Jx, _Jy, _Jz,
    //                     _Mx, _My, _Mz,
    //                     _dx, 
    //                     Nx, Ny, Nz,
    //                     Nx_pad, Ny_pad,
    //                     xx_num_mountains, yy_num_valleys, // number of tiles in each dimensions
    //                     true, false,
    //                     xx_heads_mountain, 
    //                     yy_heads_valley, 
    //                     left_pad,
    //                     block_size,
    //                     grid_size); 

    // std::swap(Ex_src, Ex_dst);
    // std::swap(Ey_src, Ey_dst);
    // std::swap(Ez_src, Ez_dst);
    // std::swap(Hx_src, Hx_dst);
    // std::swap(Hy_src, Hy_dst);
    // std::swap(Hz_src, Hz_dst);

    // std::cout << "phase 3 : Ex_src[1164] = " << Ex_src[1164] << "\n";

    // // phase 4: (v, v)
    // grid_size = xx_num_valleys * yy_num_valleys * Nz; 
    // _updateEH_dt_2D_seq(Ex_src, Ey_src, Ez_src,
    //                     Hx_src, Hy_src, Hz_src,
    //                     Ex_dst, Ey_dst, Ez_dst,
    //                     Hx_dst, Hy_dst, Hz_dst,
    //                     _Cax, _Cbx,
    //                     _Cay, _Cby,
    //                     _Caz, _Cbz,
    //                     _Dax, _Dbx,
    //                     _Day, _Dby,
    //                     _Daz, _Dbz,
    //                     _Jx, _Jy, _Jz,
    //                     _Mx, _My, _Mz,
    //                     _dx, 
    //                     Nx, Ny, Nz,
    //                     Nx_pad, Ny_pad,
    //                     xx_num_valleys, yy_num_valleys, // number of tiles in each dimensions
    //                     false, false,
    //                     xx_heads_valley, 
    //                     yy_heads_valley, 
    //                     left_pad,
    //                     block_size,
    //                     grid_size); 

    // std::swap(Ex_src, Ex_dst);
    // std::swap(Ey_src, Ey_dst);
    // std::swap(Ez_src, Ez_dst);
    // std::swap(Hx_src, Hx_dst);
    // std::swap(Hy_src, Hy_dst);
    // std::swap(Hz_src, Hz_dst);

    // std::cout << "phase 4 : Ex_src[1164] = " << Ex_src[1164] << "\n";

  }

  // std::cout << "Ex_src = ";
  // for(int i=0; i<Nx_pad*Ny_pad*Nz; i++) {
  //   if(Ex_src[i] != 0) { 
  //     std::cout << Ex_src[i] << " ";
  //   }
  // }

  _extract_original_from_padded(Ex_src, _Ex_simu, Nx, Ny, Nz, Nx_pad, Ny_pad, left_pad);
  _extract_original_from_padded(Ey_src, _Ey_simu, Nx, Ny, Nz, Nx_pad, Ny_pad, left_pad);
  _extract_original_from_padded(Ez_src, _Ez_simu, Nx, Ny, Nz, Nx_pad, Ny_pad, left_pad);
  _extract_original_from_padded(Hx_src, _Hx_simu, Nx, Ny, Nz, Nx_pad, Ny_pad, left_pad);
  _extract_original_from_padded(Hy_src, _Hy_simu, Nx, Ny, Nz, Nx_pad, Ny_pad, left_pad);
  _extract_original_from_padded(Hz_src, _Hz_simu, Nx, Ny, Nz, Nx_pad, Ny_pad, left_pad);

}

void gDiamond::_updateEH_dt_2D_seq(const std::vector<float>& Ex_src, const std::vector<float>& Ey_src, const std::vector<float>& Ez_src,
                                   const std::vector<float>& Hx_src, const std::vector<float>& Hy_src, const std::vector<float>& Hz_src,
                                   std::vector<float>& Ex_dst, std::vector<float>& Ey_dst, std::vector<float>& Ez_dst,
                                   std::vector<float>& Hx_dst, std::vector<float>& Hy_dst, std::vector<float>& Hz_dst,
                                   const std::vector<float>& Cax, const std::vector<float>& Cbx,
                                   const std::vector<float>& Cay, const std::vector<float>& Cby,
                                   const std::vector<float>& Caz, const std::vector<float>& Cbz,
                                   const std::vector<float>& Dax, const std::vector<float>& Dbx,
                                   const std::vector<float>& Day, const std::vector<float>& Dby,
                                   const std::vector<float>& Daz, const std::vector<float>& Dbz,
                                   const std::vector<float>& Jx, const std::vector<float>& Jy, const std::vector<float>& Jz,
                                   const std::vector<float>& Mx, const std::vector<float>& My, const std::vector<float>& Mz,
                                   float dx, 
                                   int Nx, int Ny, int Nz,
                                   int Nx_pad, int Ny_pad,
                                   int xx_num, int yy_num, // number of tiles in each dimensions
                                   bool is_mountain_X, bool is_mountain_Y,
                                   std::vector<int> xx_heads, 
                                   std::vector<int> yy_heads, 
                                   int left_pad,
                                   size_t block_size,
                                   size_t grid_size) {

  for(size_t block_id=0; block_id<grid_size; block_id++) {
    const int zz = block_id / (xx_num * yy_num);
    const int rem = block_id % (xx_num * yy_num); 
    const int xx = rem % xx_num;
    const int yy = rem / xx_num;

    const int global_z = zz; // global_z is always zz

    if(global_z < 1 || global_z > Nz-2) {
      continue;
    }

    // declare shared memory
    float Ex_shmem[SHX * SHY * 2] = {0};
    float Ey_shmem[SHX * SHY * 2] = {0};
    float Ez_shmem[SHX * SHY * 2] = {0};
    float Hx_shmem[SHX * SHY * 2] = {0};
    float Hy_shmem[SHX * SHY * 2] = {0};
    float Hz_shmem[SHX * SHY * 2] = {0};

    // load shared memory
    // In E_shmem, we store z and z + 1 plane
    // In H_shmem, we store z - 1 and z plane
    for(size_t tid=0; tid<block_size; tid++) {
      int local_x = tid % NTX; 
      int local_y = tid / NTY;
      for(int shared_x=local_x; shared_x<SHX; shared_x+=NTX) {
        for(int shared_y=local_y; shared_y<SHY; shared_y+=NTY) {

          int shared_z_E, shared_z_H;
          int shared_idx_E, shared_idx_H;
          int global_x = xx_heads[xx] + shared_x; 
          int global_y = yy_heads[yy] + shared_y;
          int global_idx;

          if(!is_mountain_X && !is_mountain_Y && xx == 0 && yy == 0 && zz == 1 && shared_y == 5) {
            std::cout << "(xx, yy, zz) = " << xx << ", " << yy << ", " << zz << ", ";
            std::cout << "local_x = " << local_x << ", local_y = " << local_y << ", ";
            std::cout << "shared_x = " << shared_x << ", ";
            std::cout << "global_x = " << global_x << "\n";
          }

          // load z plane for E_shmem, H_shmem 
          shared_z_E = 0; // z plane at first layer
          shared_z_H = 1; // z plane at second layer 
          shared_idx_E = shared_x + shared_y * SHX + shared_z_E * SHX * SHY;
          shared_idx_H = shared_x + shared_y * SHX + shared_z_H * SHX * SHY;

          global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad; 

          Ex_shmem[shared_idx_E] = Ex_src[global_idx];
          Ey_shmem[shared_idx_E] = Ey_src[global_idx];
          Ez_shmem[shared_idx_E] = Ez_src[global_idx];
          Hx_shmem[shared_idx_H] = Hx_src[global_idx];
          Hy_shmem[shared_idx_H] = Hy_src[global_idx];
          Hz_shmem[shared_idx_H] = Hz_src[global_idx];

          // load z + 1 plane for E_shmem
          shared_z_E = 1;
          shared_idx_E = shared_x + shared_y * SHX + shared_z_E * SHX * SHY;
          global_idx = global_x + global_y * Nx_pad + (global_z + 1) * Nx_pad * Ny_pad;

          Ex_shmem[shared_idx_E] = Ex_src[global_idx];
          Ey_shmem[shared_idx_E] = Ey_src[global_idx];
          Ez_shmem[shared_idx_E] = Ez_src[global_idx];

          // load z - 1 plane for H_shmem
          shared_z_H = 0;
          shared_idx_H = shared_x + shared_y * SHX + shared_z_H * SHX * SHY;
          global_idx = global_x + global_y * Nx_pad + (global_z - 1) * Nx_pad * Ny_pad;

          Hx_shmem[shared_idx_H] = Hx_src[global_idx];
          Hy_shmem[shared_idx_H] = Hy_src[global_idx];
          Hz_shmem[shared_idx_H] = Hz_src[global_idx];
        }
      }
    }

    // check shared memory
    // if(!is_mountain_X && !is_mountain_Y) {
    //   std::cout << "Ex_src[1164] = " << Ex_src[1164] << "\n";
    //   std::cout << "check shared memory\n";
    // }

    /*
    // calculation
    int cal_offsetX_E, cal_offsetY_E, cal_offsetX_H, cal_offsetY_H;
    int cal_boundX_E, cal_boundY_E, cal_boundX_H, cal_boundY_H;
    for(int t=0; t<BLT_DTR; t++) {
      cal_offsetX_E = (is_mountain_X)? t + 1 : BLT_DTR - t;
      cal_offsetX_H = (is_mountain_X)? cal_offsetX_E : cal_offsetX_E - 1;
      cal_offsetY_E = (is_mountain_Y)? t + 1 : BLT_DTR - t;
      cal_offsetY_H = (is_mountain_Y)? cal_offsetY_E : cal_offsetY_E - 1;
      cal_boundX_E = (is_mountain_X)? SHX - t : SHX - (BLT_DTR - t);
      cal_boundX_H = (is_mountain_X)? cal_boundX_E - 1 : cal_boundX_E; 
      cal_boundY_E = (is_mountain_Y)? SHY - t : SHY - (BLT_DTR - t);
      cal_boundY_H = (is_mountain_Y)? cal_boundY_E - 1 : cal_boundY_E; 

      for(size_t tid=0; tid<block_size; tid++) {
        int local_x = tid % NTX; 
        int local_y = tid / NTY;

        // update E
        for(int shared_x=local_x+cal_offsetX_E; shared_x<cal_boundX_E; shared_x+=NTX) {
          for(int shared_y=local_y+cal_offsetY_E; shared_y<cal_boundY_E; shared_y+=NTY) {

            int shared_z_E = 0; // z plane
            int shared_z_H = 1; // z plane
            int shared_idx_E = shared_x + shared_y * SHX + shared_z_E * SHX * SHY;
            int shared_idx_H = shared_x + shared_y * SHX + shared_z_H * SHX * SHY;
            int global_x = xx_heads[xx] + shared_x - left_pad; // - left_pad since constant arrays has not been padded  
            int global_y = yy_heads[yy] + shared_y - left_pad;
            int global_idx = global_x + global_y * Nx + global_z * Nx * Ny; // notice that here we are accessing the unpadded constant array
                                                                            // so we are using the original array size

            // if(!is_mountain_X && !is_mountain_Y) {
            //   std::cout << "t = " << t << ", ";
            //   std::cout << "(xx, yy, zz) = " << xx << ", " << yy << ", " << zz << ", ";
            //   std::cout << "local_x = " << local_x << ", local_y = " << local_y << ", ";
            //   std::cout << "shared_x = " << shared_x << ", shared_y = " << shared_y << ", ";
            //   std::cout << "global_x = " << global_x << ", global_y = " << global_y << "\n";
            // }

            if(global_x >= 1 && global_x <= Nx-2 && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2) {

              Ex_shmem[shared_idx_E] = Cax[global_idx] * Ex_shmem[shared_idx_E] + Cbx[global_idx] *
                        ((Hz_shmem[shared_idx_H] - Hz_shmem[shared_idx_H - SHX]) - (Hy_shmem[shared_idx_H] - Hy_shmem[shared_idx_H - SHX * SHY]) - Jx[global_idx] * dx);
              Ey_shmem[shared_idx_E] = Cay[global_idx] * Ey_shmem[shared_idx_E] + Cby[global_idx] *
                        ((Hx_shmem[shared_idx_H] - Hx_shmem[shared_idx_H - SHX * SHY]) - (Hz_shmem[shared_idx_H] - Hz_shmem[shared_idx_H - 1]) - Jy[global_idx] * dx);
              Ez_shmem[shared_idx_E] = Caz[global_idx] * Ez_shmem[shared_idx_E] + Cbz[global_idx] *
                        ((Hy_shmem[shared_idx_H] - Hy_shmem[shared_idx_H - 1]) - (Hx_shmem[shared_idx_H] - Hx_shmem[shared_idx_H - SHX]) - Jz[global_idx] * dx);
            }
          }
        }

        // update H
        for(int shared_x=local_x+cal_offsetX_H; shared_x<cal_boundX_H; shared_x+=NTX) {
          for(int shared_y=local_y+cal_offsetY_H; shared_y<cal_boundY_H; shared_y+=NTY) {

            int shared_z_E = 0; // z plane
            int shared_z_H = 1; // z plane
            int shared_idx_E = shared_x + shared_y * SHX + shared_z_E * SHX * SHY;
            int shared_idx_H = shared_x + shared_y * SHX + shared_z_H * SHX * SHY;
            int global_x = xx_heads[xx] + shared_x - left_pad; // - left_pad since constant arrays has not been padded  
            int global_y = yy_heads[yy] + shared_y - left_pad;
            int global_idx = global_x + global_y * Nx + global_z * Nx * Ny; // notice that here we are accessing the unpadded constant array
                                                                            // so we are using the original array size

            if(global_x >= 1 && global_x <= Nx-2 && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2) {
             
              Hx_shmem[shared_idx_H] = Dax[global_idx] * Hx_shmem[shared_idx_H] + Dbx[global_idx] *
                        ((Ey_shmem[shared_idx_E + SHX * SHY] - Ey_shmem[shared_idx_E]) - (Ez_shmem[shared_idx_E + SHX] - Ez_shmem[shared_idx_E]) - Mx[global_idx] * dx);
              Hy_shmem[shared_idx_H] = Day[global_idx] * Hy_shmem[shared_idx_H] + Dby[global_idx] *
                        ((Ez_shmem[shared_idx_E + 1] - Ez_shmem[shared_idx_E]) - (Ex_shmem[shared_idx_E + SHX * SHY] - Ex_shmem[shared_idx_E]) - My[global_idx] * dx);
              Hz_shmem[shared_idx_H] = Daz[global_idx] * Hz_shmem[shared_idx_H] + Dbz[global_idx] *
                        ((Ex_shmem[shared_idx_E + SHX] - Ex_shmem[shared_idx_E]) - (Ey_shmem[shared_idx_E + 1] - Ey_shmem[shared_idx_E]) - Mz[global_idx] * dx);
            }
          }
        }
      }
    }
    */

    // store to global memory
    int store_offsetX_E, store_offsetY_E, store_offsetX_H, store_offsetY_H;
    int store_boundX_E, store_boundY_E, store_boundX_H, store_boundY_H;
    store_offsetX_E = 1;
    store_offsetX_H = (is_mountain_X)? 1 : 0;
    store_offsetY_E = 1;
    store_offsetY_H = (is_mountain_Y)? 1 : 0;
    store_boundX_E = (is_mountain_X)? SHX : SHX - 1;
    store_boundX_H = SHX - 1;
    store_boundY_E = (is_mountain_Y)? SHY : SHY - 1;
    store_boundY_H = SHY - 1;
    for(size_t tid=0; tid<block_size; tid++) {
      int local_x = tid % NTX; 
      int local_y = tid / NTY;

      // store E
      for(int shared_x=local_x+store_offsetX_E; shared_x<store_boundX_E; shared_x+=NTX) {
        for(int shared_y=local_y+store_offsetY_E; shared_y<store_boundY_E; shared_y+=NTY) {
          int shared_z_E = 0; // z plane
          int shared_idx = shared_x + shared_y * SHX + shared_z_E * SHX * SHY;
          int global_x = xx_heads[xx] + shared_x;   
          int global_y = yy_heads[yy] + shared_y;
          int global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad; 

          if(is_mountain_X && is_mountain_Y && xx == 0 && yy == 0 && zz == 1 && shared_y == 1) {
            std::cout << "(xx, yy, zz) = " << xx << ", " << yy << ", " << zz << ", ";
            std::cout << "local_x = " << local_x << ", local_y = " << local_y << ", ";
            std::cout << "shared_x = " << shared_x << ", ";
            std::cout << "global_x = " << global_x << "\n";
          }

          if(global_x >= 1 + left_pad && global_x <= Nx-2 + left_pad && global_y >= 1 + left_pad && global_y <= Ny-2 + left_pad && global_z >= 1 && global_z <= Nz-2) {
            Ex_dst[global_idx] = Ex_shmem[shared_idx];
            Ey_dst[global_idx] = Ey_shmem[shared_idx];
            Ez_dst[global_idx] = Ez_shmem[shared_idx];
          }
        }
      }

      // store H
      for(int shared_x=local_x+store_offsetX_H; shared_x<store_boundX_H; shared_x+=NTX) {
        for(int shared_y=local_y+store_offsetY_H; shared_y<store_boundY_H; shared_y+=NTY) {
          int shared_z_H = 1; // z plane
          int shared_idx = shared_x + shared_y * SHX + shared_z_H * SHX * SHY;
          int global_x = xx_heads[xx] + shared_x;   
          int global_y = yy_heads[yy] + shared_y;
          int global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad; 

          if(global_x >= 1 + left_pad && global_x <= Nx-2 + left_pad && global_y >= 1 + left_pad && global_y <= Ny-2 + left_pad && global_z >= 1 && global_z <= Nz-2) {
            Hx_dst[global_idx] = Hx_shmem[shared_idx];
            Hy_dst[global_idx] = Hy_shmem[shared_idx];
            Hz_dst[global_idx] = Hz_shmem[shared_idx];
          }
        }
      }

    }

  }

}

void gDiamond::update_FDTD_cpu_simulation_dt_1_D_sdf(size_t num_timesteps, size_t Tx, size_t Ty, size_t Tz) { // CPU single thread 1-D simulation of diamond tiling, reimplemented

  // write 1 dimension just to check
  std::vector<float> E_simu(_Nx, 1);
  std::vector<float> H_simu(_Nx, 1);
  std::vector<float> E_seq(_Nx, 1);
  std::vector<float> H_seq(_Nx, 1);

  int Nx = _Nx;

  // seq version
  for(size_t t=0; t<num_timesteps; t++) {

    // update E
    for(int x=1; x<Nx-1; x++) {
      E_seq[x] = H_seq[x-1] + H_seq[x] * 2; 
    }

    std::cout << "t = " << t << ", E_seq =";
    for(int x=0; x<Nx; x++) {
      std::cout << E_seq[x] << " ";
    }
    std::cout << "\n";

    // update H 
    for(int x=1; x<Nx-1; x++) {
      H_seq[x] = E_seq[x+1] + E_seq[x] * 2; 
    }
  }

  // diamond tiling
  int Nx_pad = Nx + LEFT_PAD + RIGHT_PAD;
  
  // src, dst, final
  std::vector<float> E_src(Nx_pad, 1);
  std::vector<float> H_src(Nx_pad, 1);
  std::vector<float> E_dst(Nx_pad, 1);
  std::vector<float> H_dst(Nx_pad, 1);
  std::vector<float> E_final(Nx_pad, 1);
  std::vector<float> H_final(Nx_pad, 1);
  std::cout << "copy E, H to padded array\n";
  for(int i=0; i<Nx; i++) {
    E_src[i + LEFT_PAD] = E_simu[i];
    H_src[i + LEFT_PAD] = H_simu[i];
  }

  std::cout << "E_src = ";
  for(const auto& data : E_src) {
    std::cout << data << " ";
  }
  std::cout << "\n";

  // xx_heads_mountain[xx] is 1 element left offset to the actual mountain
  int xx_num_mountains = 1 + Tx;
  int xx_num_valleys = Tx + 1;
  std::vector<int> xx_heads_mountain(xx_num_mountains, 0);
  std::vector<int> xx_heads_valley(xx_num_valleys, 0);

  for(int index=0; index<xx_num_mountains; index++) {
    xx_heads_mountain[index] = (index == 0)? 0 :
                               xx_heads_mountain[index-1] + (BLX_DTR + NTX);
  }
  for(int index=0; index<xx_num_valleys; index++) {
    xx_heads_valley[index] = (index == 0)? BLX_DTR - (BLT_DTR - 1) :
                             xx_heads_valley[index-1] + (BLX_DTR + NTX);
  }
  std::cout << "xx_heads_mountain = ";
  for(const auto& index : xx_heads_mountain) {
    std::cout << index << " ";
  }
  std::cout << "\n";

  std::cout << "xx_heads_valley = ";
  for(const auto& index : xx_heads_valley) {
    std::cout << index << " ";
  }
  std::cout << "\n";

  int block_size = 4; // each block has 4 threads
  for(size_t tt=0; tt<num_timesteps/BLT_UB; tt++) {

    // phase 1.
    for(int xx=0; xx<xx_num_mountains; xx++) {
      
      // declare shared memory
      float E_shmem[SHX];
      float H_shmem[SHX]; 

      // load shared memory
      for(int tid=0; tid<block_size; tid++) {
        for(int shared_idx=tid; shared_idx<SHX; shared_idx+=block_size) {
          int global_idx = xx_heads_mountain[xx] + shared_idx;
          E_shmem[shared_idx] = E_src[global_idx];
          H_shmem[shared_idx] = H_src[global_idx];
        }
      }

      // calculation
      for(int t=0; t<BLT_DTR; t++) {
        int cal_offsetE = t + 1;
        int cal_offsetH = cal_offsetE;
        int cal_boundE = SHX - t;
        int cal_boundH = cal_boundE - 1;

        // update E
        for(int tid=0; tid<block_size; tid++) {
          for(int shared_idx=tid+cal_offsetE; shared_idx<cal_boundE; shared_idx+=block_size) {
            int global_idx = xx_heads_mountain[xx] + shared_idx;
            if(global_idx >= 1 + LEFT_PAD && global_idx <= Nx - 2 + LEFT_PAD) {
              E_shmem[shared_idx] = H_shmem[shared_idx-1] + H_shmem[shared_idx] * 2;
            }
          }
        }
        
        // update H
        for(int tid=0; tid<block_size; tid++) {
          for(int shared_idx=tid+cal_offsetH; shared_idx<cal_boundH; shared_idx+=block_size) {
            int global_idx = xx_heads_mountain[xx] + shared_idx;
            if(global_idx >= 1 + LEFT_PAD && global_idx <= Nx - 2 + LEFT_PAD) {
              H_shmem[shared_idx] = E_shmem[shared_idx+1] + E_shmem[shared_idx] * 2;
            }
          }
        }
      }

      // store to global memory
      // we store final results to final, and partial results to dst for valley 
      // final results means data are already in the final time steps

      // store final
      int final_offsetE = BLT_DTR; 
      int final_offsetH = final_offsetE;
      int final_boundE = final_offsetE + NTX + 1;
      int final_boundH = final_boundE - 1;
      for(int tid=0; tid<block_size; tid++) {
        for(int shared_idx=tid+final_offsetE; shared_idx<final_boundE; shared_idx+=NTX) {
          int global_idx = xx_heads_mountain[xx] + shared_idx;
          E_final[global_idx] = E_shmem[shared_idx];
        }
        for(int shared_idx=tid+final_offsetH; shared_idx<final_boundH; shared_idx+=NTX) {
          int global_idx = xx_heads_mountain[xx] + shared_idx;
          H_final[global_idx] = H_shmem[shared_idx];
        }
      }

      // store dst
      // dst has left and right part. 
      // tid = 0, 1, 2, ... are in charge of left part
      // tid = block_size - 1, block_size - 2, ... are in charge of right part 
      int left_dst_offsetE = 1;
      int left_dst_offsetH = left_dst_offsetE;
      int left_dst_boundE = left_dst_offsetE + BLT_DTR;
      int left_dst_boundH = left_dst_boundE - 1;
      for(int tid=0; tid<block_size; tid++) {
        if(tid < left_dst_boundE) { // asking left_dst_boundE threads to store left part to dst
          for(int shared_idx=tid+left_dst_offsetE; shared_idx<left_dst_boundE; shared_idx+=NTX) {
            int global_idx = xx_heads_mountain[xx] + shared_idx;
            E_dst[global_idx] = E_shmem[shared_idx];
          }
          for(int shared_idx=tid+left_dst_offsetH; shared_idx<left_dst_boundH; shared_idx+=NTX) {
            int global_idx = xx_heads_mountain[xx] + shared_idx;
            H_dst[global_idx] = H_shmem[shared_idx];
          }
        }
      }

      int right_dst_offsetE = BLT_DTR + NTX;
      int right_dst_offsetH = right_dst_offsetE;
      int right_dst_boundE = SHX; 
      int right_dst_boundH = right_dst_boundE - 1;
      for(int tid=0; tid<block_size; tid++) {
        // asking (right_dst_boundE - right_dst_offsetE) threads to store right part to dst
        // cannot demo on this since we only has 4 threads per block
        for(int shared_idx=tid+right_dst_offsetE; shared_idx<right_dst_boundE; shared_idx+=NTX) {
          int global_idx = xx_heads_mountain[xx] + shared_idx;
          E_dst[global_idx] = E_shmem[shared_idx];
        }
        for(int shared_idx=tid+right_dst_offsetH; shared_idx<right_dst_boundH; shared_idx+=NTX) {
          int global_idx = xx_heads_mountain[xx] + shared_idx;
          H_dst[global_idx] = H_shmem[shared_idx];
        }
      }
    }

    // phase 2
    for(int xx=0; xx<xx_num_valleys; xx++) {

      // declare shared memory
      float E_shmem[SHX];
      float H_shmem[SHX];

      // load shared memory
      for(int tid=0; tid<block_size; tid++) {
        for(int shared_idx=tid; shared_idx<SHX; shared_idx+=block_size) {
          int global_idx = xx_heads_valley[xx] + shared_idx;
          E_shmem[shared_idx] = E_dst[global_idx];
          H_shmem[shared_idx] = H_dst[global_idx];
        }
      }

      // calculation
      int cal_offsetE, cal_offsetH;
      int cal_boundE, cal_boundH;
      for(int t=0; t<BLT_DTR; t++) {
        cal_offsetE = BLT_DTR - t;
        cal_offsetH = cal_offsetE - 1;
        cal_boundE = SHX - (BLT_DTR - t);
        cal_boundH = cal_boundE; 

        // update E
        for(int tid=0; tid<block_size; tid++) {
          for(int shared_idx=tid+cal_offsetE; shared_idx<cal_boundE; shared_idx+=block_size) {
            int global_idx = xx_heads_valley[xx] + shared_idx;
            if(global_idx >= 1 + LEFT_PAD && global_idx <= Nx - 2 + LEFT_PAD) {
              E_shmem[shared_idx] = H_shmem[shared_idx-1] + H_shmem[shared_idx] * 2;
            }
          }
        }

        // update H
        for(int tid=0; tid<block_size; tid++) {
          for(int shared_idx=tid+cal_offsetH; shared_idx<cal_boundH; shared_idx+=block_size) {
            int global_idx = xx_heads_valley[xx] + shared_idx;
            // std::cout << "t = " << t << ", ";
            // std::cout << "xx = " << xx << ", shared_idx = " << shared_idx << ", global_idx = " << global_idx << "\n";
            if(global_idx >= 1 + LEFT_PAD && global_idx <= Nx - 2 + LEFT_PAD) {
              H_shmem[shared_idx] = E_shmem[shared_idx+1] + E_shmem[shared_idx] * 2;
            }
          }
        }
      }

      // store to global memory
      int store_offsetE = 1;
      int store_offsetH = 0;
      int store_boundE = SHX - 1;
      int store_boundH = store_boundE;
      for(int tid=0; tid<block_size; tid++) {
        // store E
        for(int shared_idx=tid + store_offsetE; shared_idx<store_boundE; shared_idx+=block_size) {
          int global_idx = xx_heads_valley[xx] + shared_idx;
          E_final[global_idx] = E_shmem[shared_idx];
        }

        // store H
        for(int shared_idx=tid + store_offsetH; shared_idx<store_boundH; shared_idx+=block_size) {
          int global_idx = xx_heads_valley[xx] + shared_idx;
          H_final[global_idx] = H_shmem[shared_idx];
        }
      }
    }

  }

  std::cout << "E_final = ";
  for(const auto& data : E_final) {
    std::cout << data << " ";
  }
  std::cout << "\n";

  std::cout << "E_dst = ";
  for(const auto& data : E_dst) {
    std::cout << data << " ";
  }
  std::cout << "\n";

  // copy results from final to simu
  for(int x=0; x<Nx; x++) {
    E_simu[x] = E_final[x+LEFT_PAD];
    H_simu[x] = H_final[x+LEFT_PAD];
  }

  std::cout << "E_seq = ";
  for(int x=0; x<Nx; x++) {
    std::cout << E_seq[x] << " ";
  }
  std::cout << "\n";

  std::cout << "E_simu = ";
  for(int x=0; x<Nx; x++) {
    std::cout << E_simu[x] << " ";
  }
  std::cout << "\n";

  for(int x=0; x<Nx; x++) {
    if(E_seq[x] != E_simu[x] || H_seq[x] != H_simu[x]) {
      std::cerr << "1-D demo results mismatch.\n";
      std::exit(EXIT_FAILURE);
    }
  }

}

void gDiamond::update_FDTD_cpu_simulation_dt_3_D_sdf(size_t num_timesteps, size_t Tx) { // CPU single thread 3-D simulation of diamond tiling, reimplemented
                                                                                                              // sdf stands for src, dst, final array

  // clear source Mz for experiments
  _Mz.clear();

  // transfer source
  for(size_t t=0; t<num_timesteps; t++) {
    float Mz_value = M_source_amp * std::sin(SOURCE_OMEGA * t * dt);
    _Mz[_source_idx] = Mz_value;
  }

  // tiling parameter pre-processing
  int Nx = _Nx;
  int Ny = _Ny;
  int Nz = _Nz;
  int Nx_pad = Nx + LEFT_PAD + RIGHT_PAD;

  int xx_num_mountains = 1 + Tx;
  int xx_num_valleys = Tx + 1;

  // xx_heads_mountain[xx] is 1 element left offset to the actual mountain
  std::vector<int> xx_heads_mountain(xx_num_mountains, 0);
  std::vector<int> xx_heads_valley(xx_num_valleys, 0);

  for(int index=0; index<xx_num_mountains; index++) {
    xx_heads_mountain[index] = (index == 0)? 0 :
                               xx_heads_mountain[index-1] + (BLX_DTR + NTX);
  }
  for(int index=0; index<xx_num_valleys; index++) {
    xx_heads_valley[index] = (index == 0)? BLX_DTR - (BLT_DTR - 1) :
                             xx_heads_valley[index-1] + (BLX_DTR + NTX);
  }

  std::cout << "xx_heads_mountain = ";
  for(const auto& index : xx_heads_mountain) {
    std::cout << index << " ";
  }
  std::cout << "\n";

  std::cout << "xx_heads_valley = ";
  for(const auto& index : xx_heads_valley) {
    std::cout << index << " ";
  }
  std::cout << "\n";

  // padded E, H
  std::vector<float> Ex_src(Nx_pad * Ny * Nz, 0);
  std::vector<float> Ey_src(Nx_pad * Ny * Nz, 0);
  std::vector<float> Ez_src(Nx_pad * Ny * Nz, 0);
  std::vector<float> Hx_src(Nx_pad * Ny * Nz, 0);
  std::vector<float> Hy_src(Nx_pad * Ny * Nz, 0);
  std::vector<float> Hz_src(Nx_pad * Ny * Nz, 0);

  std::vector<float> Ex_dst(Nx_pad * Ny * Nz, 0);
  std::vector<float> Ey_dst(Nx_pad * Ny * Nz, 0);
  std::vector<float> Ez_dst(Nx_pad * Ny * Nz, 0);
  std::vector<float> Hx_dst(Nx_pad * Ny * Nz, 0);
  std::vector<float> Hy_dst(Nx_pad * Ny * Nz, 0);
  std::vector<float> Hz_dst(Nx_pad * Ny * Nz, 0);

  std::vector<float> Ex_final(Nx_pad * Ny * Nz, 0);
  std::vector<float> Ey_final(Nx_pad * Ny * Nz, 0);
  std::vector<float> Ez_final(Nx_pad * Ny * Nz, 0);
  std::vector<float> Hx_final(Nx_pad * Ny * Nz, 0);
  std::vector<float> Hy_final(Nx_pad * Ny * Nz, 0);
  std::vector<float> Hz_final(Nx_pad * Ny * Nz, 0);

  // define block size and grid size
  size_t block_size = NTX;
  size_t grid_size; // grid_size = xx_num * Ny * Nz;

  // initialize E to check shared memory load
  // for(int z=0; z<Nz; z++) {
  //   for(int y=0; y<Ny; y++) {
  //     for(int x=0; x<Nx_pad; x++) {
  //       int idx = x + y * Nx_pad + z * Nx_pad * Ny;
  //       float data = 1.0 * idx;
  //       Ex_src[idx] = data;
  //       Ey_src[idx] = data;
  //       Ez_src[idx] = data;
  //       Hx_src[idx] = data;
  //       Hy_src[idx] = data;
  //       Hz_src[idx] = data;
  //     }
  //   }
  // }
  // 
  // for(int z=0; z<Nz; z++) {
  //   for(int y=0; y<Ny; y++) {
  //     for(int x=0; x<Nx; x++) {
  //       int idx = x + y * Nx + z * Nx * Ny;
  //       float data = 1.0 * idx;
  //       _Cax[idx] = data;
  //       _Dax[idx] = data;
  //     }
  //   }
  // }


  for(size_t tt=0; tt<num_timesteps/BLT_DTR; tt++) {
    std::cout << "running\n";
    grid_size = xx_num_mountains * Ny * Nz; 
    _updateEH_dt_1D_mountain_seq(Ex_src, Ey_src, Ez_src,
                                 Hx_src, Hy_src, Hz_src,
                                 Ex_final, Ey_final, Ez_final,
                                 Hx_final, Hy_final, Hz_final,
                                 _Cax, _Cbx,
                                 _Cay, _Cby,
                                 _Caz, _Cbz,
                                 _Dax, _Dbx,
                                 _Day, _Dby,
                                 _Daz, _Dbz,
                                 _Jx, _Jy, _Jz,
                                 _Mx, _My, _Mz,
                                 _dx, 
                                 Nx, Ny, Nz,
                                 Nx_pad, 
                                 xx_num_mountains, // number of tiles in each dimensions
                                 xx_heads_mountain, 
                                 block_size,
                                 grid_size, 
                                 tt); 

    grid_size = xx_num_valleys * Ny * Nz; 
    _updateEH_dt_1D_valley_seq(Ex_src, Ey_src, Ez_src,
                               Hx_src, Hy_src, Hz_src,
                               Ex_final, Ey_final, Ez_final,
                               Hx_final, Hy_final, Hz_final,
                               _Cax, _Cbx,
                               _Cay, _Cby,
                               _Caz, _Cbz,
                               _Dax, _Dbx,
                               _Day, _Dby,
                               _Daz, _Dbz,
                               _Jx, _Jy, _Jz,
                               _Mx, _My, _Mz,
                               _dx, 
                               Nx, Ny, Nz,
                               Nx_pad, 
                               xx_num_valleys, // number of tiles in each dimensions
                               xx_heads_valley, 
                               block_size,
                               grid_size,
                               tt); 

    std::swap(Ex_final, Ex_src);
    std::swap(Ey_final, Ey_src);
    std::swap(Ez_final, Ez_src);
    std::swap(Hx_final, Hx_src);
    std::swap(Hy_final, Hy_src);
    std::swap(Hz_final, Hz_src);

  }

  _extract_original_from_padded_1D(Ex_src, _Ex_simu, Nx, Ny, Nz, Nx_pad, LEFT_PAD);
  _extract_original_from_padded_1D(Ey_src, _Ey_simu, Nx, Ny, Nz, Nx_pad, LEFT_PAD);
  _extract_original_from_padded_1D(Ez_src, _Ez_simu, Nx, Ny, Nz, Nx_pad, LEFT_PAD);
  _extract_original_from_padded_1D(Hx_src, _Hx_simu, Nx, Ny, Nz, Nx_pad, LEFT_PAD);
  _extract_original_from_padded_1D(Hy_src, _Hy_simu, Nx, Ny, Nz, Nx_pad, LEFT_PAD);
  _extract_original_from_padded_1D(Hz_src, _Hz_simu, Nx, Ny, Nz, Nx_pad, LEFT_PAD);

}

void gDiamond::_updateEH_dt_1D_mountain_seq(std::vector<float>& Ex_src, std::vector<float>& Ey_src, std::vector<float>& Ez_src,
                                            std::vector<float>& Hx_src, std::vector<float>& Hy_src, std::vector<float>& Hz_src,
                                            std::vector<float>& Ex_final, std::vector<float>& Ey_final, std::vector<float>& Ez_final,
                                            std::vector<float>& Hx_final, std::vector<float>& Hy_final, std::vector<float>& Hz_final,
                                            const std::vector<float>& Cax, const std::vector<float>& Cbx,
                                            const std::vector<float>& Cay, const std::vector<float>& Cby,
                                            const std::vector<float>& Caz, const std::vector<float>& Cbz,
                                            const std::vector<float>& Dax, const std::vector<float>& Dbx,
                                            const std::vector<float>& Day, const std::vector<float>& Dby,
                                            const std::vector<float>& Daz, const std::vector<float>& Dbz,
                                            const std::vector<float>& Jx, const std::vector<float>& Jy, const std::vector<float>& Jz,
                                            const std::vector<float>& Mx, const std::vector<float>& My, const std::vector<float>& Mz,
                                            float dx, 
                                            int Nx, int Ny, int Nz,
                                            int Nx_pad, 
                                            int xx_num, // number of tiles in each dimensions
                                            std::vector<int> xx_heads, 
                                            size_t block_size,
                                            size_t grid_size,
                                            size_t tt) {

  // std::cout << "xx_num = " << xx_num << "\n";

  for(size_t block_id=0; block_id<grid_size; block_id++) {
    int xx = block_id % xx_num;
    int temp = block_id / xx_num;
    int yy = temp % Ny;
    int zz = temp / Ny;

    // std::cout << "(xx, yy, zz) = " << xx << ", " << yy << ", " << zz << "\n"; 

    const int global_z = zz; // global_z is always zz
    const int global_y = yy; // global_y is always yy

    // if(global_y < 1 || global_y > Ny-2 || global_z < 1 || global_z > Nz-2) {
    //   continue;
    // }

    // declare shared memory
    float Ex_shmem[SHX * 3] = {0};
    float Ey_shmem[SHX * 3] = {0};
    float Ez_shmem[SHX * 3] = {0};
    float Hx_shmem[SHX * 3] = {0};
    float Hy_shmem[SHX * 3] = {0};
    float Hz_shmem[SHX * 3] = {0};

    // load shared memory
    // In E_shmem, we store, in order, (y, z), (y+1, z), (y, z+1) stride
    // In H_shmem, we store, in order, (y, z-1), (y-1, z), (y, z) stride
    for(size_t tid=0; tid<block_size; tid++) {
      int local_x = tid;
      for(int shared_x=local_x; shared_x<SHX; shared_x+=NTX) {
        int shared_order_E, shared_order_H;
        int shared_idx_E, shared_idx_H;
        int global_x = xx_heads[xx] + shared_x;
        int global_idx;

        // load (y, z) stride for E_shmem, H_shmem
        // (y, z) is 1st stride for E_shmem
        // (y, z) is 3rd stride for H_shmem
        shared_order_E = 0;
        shared_order_H = 2;
        shared_idx_E = shared_x + shared_order_E * SHX; 
        shared_idx_H = shared_x + shared_order_H * SHX;
        global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny;

        // if(xx == 1 && yy == 1 && zz == 1) {
        //   std::cout << "(xx, yy, zz) = " << xx << ", " << yy << ", " << zz << ", ";
        //   std::cout << "local_x = " << local_x << ", ";
        //   std::cout << "shared_x = " << shared_x << ", ";
        //   std::cout << "global_x = " << global_x << ", ";
        //   std::cout << "global_idx = " << global_idx << ", ";
        //   std::cout << "Ex_src[global_idx] = " << Ex_src[global_idx] << "\n";
        // }

        Ex_shmem[shared_idx_E] = Ex_src[global_idx];
        Ey_shmem[shared_idx_E] = Ey_src[global_idx];
        Ez_shmem[shared_idx_E] = Ez_src[global_idx];
        Hx_shmem[shared_idx_H] = Hx_src[global_idx];
        Hy_shmem[shared_idx_H] = Hy_src[global_idx];
        Hz_shmem[shared_idx_H] = Hz_src[global_idx];

        // load (y+1, z) stride for E_shmem
        shared_order_E = 1;
        shared_idx_E = shared_x + shared_order_E * SHX; 
        global_idx = global_x + (global_y + 1) * Nx_pad + global_z * Nx_pad * Ny;
        Ex_shmem[shared_idx_E] = Ex_src[global_idx];
        Ey_shmem[shared_idx_E] = Ey_src[global_idx];
        Ez_shmem[shared_idx_E] = Ez_src[global_idx];

        // load (y, z+1) stride for E_shmem
        shared_order_E = 2;
        shared_idx_E = shared_x + shared_order_E * SHX; 
        global_idx = global_x + global_y * Nx_pad + (global_z + 1) * Nx_pad * Ny;
        Ex_shmem[shared_idx_E] = Ex_src[global_idx];
        Ey_shmem[shared_idx_E] = Ey_src[global_idx];
        Ez_shmem[shared_idx_E] = Ez_src[global_idx];

        // load (y, z-1) stride for H_shmem
        shared_order_H = 0;
        shared_idx_H = shared_x + shared_order_H * SHX;
        global_idx = global_x + global_y * Nx_pad + (global_z - 1) * Nx_pad * Ny;
        Hx_shmem[shared_idx_H] = Hx_src[global_idx];
        Hy_shmem[shared_idx_H] = Hy_src[global_idx];
        Hz_shmem[shared_idx_H] = Hz_src[global_idx];

        // load (y-1, z) stride for H_shmem
        shared_order_H = 1;
        shared_idx_H = shared_x + shared_order_H * SHX;
        global_idx = global_x + (global_y - 1) * Nx_pad + global_z * Nx_pad * Ny;
        Hx_shmem[shared_idx_H] = Hx_src[global_idx];
        Hy_shmem[shared_idx_H] = Hy_src[global_idx];
        Hz_shmem[shared_idx_H] = Hz_src[global_idx];
      }
    }

    // calculation
    int cal_offsetX_E, cal_offsetX_H;
    int cal_boundX_E, cal_boundX_H;
    for(int t=0; t<BLT_DTR; t++) {
      cal_offsetX_E = t + 1;
      cal_offsetX_H = cal_offsetX_E;
      cal_boundX_E = SHX - t;
      cal_boundX_H = cal_boundX_E - 1; 

      for(size_t tid=0; tid<block_size; tid++) {
        int local_x = tid;

        // In E_shmem, we store, in order, (y, z), (y+1, z), (y, z+1) stride
        // In H_shmem, we store, in order, (y, z-1), (y-1, z), (y, z) stride

        // update E
        for(int shared_x=local_x+cal_offsetX_E; shared_x<cal_boundX_E; shared_x+=NTX) {
          int shared_order_E = 0; // (y, z) stride 
          int shared_order_H = 2; // (y, z) stride
          int shared_idx_E = shared_x + shared_order_E * SHX;
          int shared_idx_H = shared_x + shared_order_H * SHX;
          int global_x = xx_heads[xx] + shared_x - LEFT_PAD; // - LEFT_PAD since constant arrays has not been padded  
          int global_idx = global_x + global_y * Nx + global_z * Nx * Ny; // notice that here we are accessing the unpadded constant array

          if(global_x >= 1 && global_x <= Nx-2 && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2) {
            // (y, z-1) for Hx and Hy, (y-1, z) for Hx, Hz
            Ex_shmem[shared_idx_E] = Cax[global_idx] * Ex_shmem[shared_idx_E] + Cbx[global_idx] *
                        ((Hz_shmem[shared_idx_H] - Hz_shmem[shared_idx_H - SHX]) - (Hy_shmem[shared_idx_H] - Hy_shmem[shared_idx_H - 2 * SHX]) - Jx[global_idx] * dx);
            Ey_shmem[shared_idx_E] = Cay[global_idx] * Ey_shmem[shared_idx_E] + Cby[global_idx] *
                      ((Hx_shmem[shared_idx_H] - Hx_shmem[shared_idx_H - 2 * SHX]) - (Hz_shmem[shared_idx_H] - Hz_shmem[shared_idx_H - 1]) - Jy[global_idx] * dx);
            Ez_shmem[shared_idx_E] = Caz[global_idx] * Ez_shmem[shared_idx_E] + Cbz[global_idx] *
                      ((Hy_shmem[shared_idx_H] - Hy_shmem[shared_idx_H - 1]) - (Hx_shmem[shared_idx_H] - Hx_shmem[shared_idx_H - SHX]) - Jz[global_idx] * dx);
          }
        }
      }

      for(size_t tid=0; tid<block_size; tid++) {
        int local_x = tid;
        // update H
        for(int shared_x=local_x+cal_offsetX_H; shared_x<cal_boundX_H; shared_x+=NTX) {
          int shared_order_E = 0; // (y, z) stride 
          int shared_order_H = 2; // (y, z) stride
          int shared_idx_E = shared_x + shared_order_E * SHX;
          int shared_idx_H = shared_x + shared_order_H * SHX;
          int global_x = xx_heads[xx] + shared_x - LEFT_PAD; // - LEFT_PAD since constant arrays has not been padded  
          int global_idx = global_x + global_y * Nx + global_z * Nx * Ny; // notice that here we are accessing the unpadded constant array
  
          if(global_x >= 1 && global_x <= Nx-2 && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2) {
            // (y+1, z) for Ex, Ez, (y, z+1) for Ex, Ey
            Hx_shmem[shared_idx_H] = Dax[global_idx] * Hx_shmem[shared_idx_H] + Dbx[global_idx] *
                        ((Ey_shmem[shared_idx_E + 2 * SHX] - Ey_shmem[shared_idx_E]) - (Ez_shmem[shared_idx_E + SHX] - Ez_shmem[shared_idx_E]) - Mx[global_idx] * dx);
            Hy_shmem[shared_idx_H] = Day[global_idx] * Hy_shmem[shared_idx_H] + Dby[global_idx] *
                      ((Ez_shmem[shared_idx_E + 1] - Ez_shmem[shared_idx_E]) - (Ex_shmem[shared_idx_E + 2 * SHX] - Ex_shmem[shared_idx_E]) - My[global_idx] * dx);
            Hz_shmem[shared_idx_H] = Daz[global_idx] * Hz_shmem[shared_idx_H] + Dbz[global_idx] *
                      ((Ex_shmem[shared_idx_E + SHX] - Ex_shmem[shared_idx_E]) - (Ey_shmem[shared_idx_E + 1] - Ey_shmem[shared_idx_E]) - Mz[global_idx] * dx);
          }
        }

      }
    }

    // store global memory
    // we store final results to final, and partial results to dst for valley
    // final results means data are already in the final time steps

    // store final, bounds are exclusive
    int final_offsetE = BLT_DTR;
    int final_offsetH = final_offsetE;
    int final_boundE = final_offsetE + NTX + 1;
    int final_boundH = final_boundE - 1;
    for(size_t tid=0; tid<block_size; tid++) {
      int local_x = tid;

      // store E
      for(int shared_x=local_x+final_offsetE; shared_x<final_boundE; shared_x+=NTX) {
        int shared_order_E = 0; // (y, z) stride 
        int shared_idx_E = shared_x + shared_order_E * SHX;
        int global_x = xx_heads[xx] + shared_x;   
        int global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny;
        if(global_x >= 1 + LEFT_PAD && global_x <= Nx-2 + LEFT_PAD && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2) {
          Ex_final[global_idx] = Ex_shmem[shared_idx_E];
          Ey_final[global_idx] = Ey_shmem[shared_idx_E];
          Ez_final[global_idx] = Ez_shmem[shared_idx_E];
        }
      }

      // store H
      for(int shared_x=local_x+final_offsetH; shared_x<final_boundH; shared_x+=NTX) {
        int shared_order_H = 2; // (y, z) stride 
        int shared_idx_H = shared_x + shared_order_H * SHX;
        int global_x = xx_heads[xx] + shared_x;   
        int global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny;
        if(global_x >= 1 + LEFT_PAD && global_x <= Nx-2 + LEFT_PAD && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2) {
          Hx_final[global_idx] = Hx_shmem[shared_idx_H];
          Hy_final[global_idx] = Hy_shmem[shared_idx_H];
          Hz_final[global_idx] = Hz_shmem[shared_idx_H];
        }
      }
    }

    // store dst
    // dst has left and right part.
    // tid = 0, 1, 2, ... are in charge of left part
    // tid = block_size - 1, block_size - 2, ... are in charge of right part
    // for now, let tid = 0, 1, 2 be in charge of left and right
    int left_dst_offsetE = 1;
    int left_dst_offsetH = left_dst_offsetE;
    int left_dst_boundE = left_dst_offsetE + BLT_DTR;
    int left_dst_boundH = left_dst_boundE - 1;
    for(size_t tid=0; tid<block_size; tid++) {
      int local_x = tid;

      // store E
      for(int shared_x=local_x+left_dst_offsetE; shared_x<left_dst_boundE; shared_x+=NTX) {
        int shared_order_E = 0; // (y, z) stride 
        int shared_idx_E = shared_x + shared_order_E * SHX;
        int global_x = xx_heads[xx] + shared_x;   
        int global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny;
        if(global_x >= 1 + LEFT_PAD && global_x <= Nx-2 + LEFT_PAD && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2) {
          Ex_src[global_idx] = Ex_shmem[shared_idx_E];
          Ey_src[global_idx] = Ey_shmem[shared_idx_E];
          Ez_src[global_idx] = Ez_shmem[shared_idx_E];
        }
      }

      // store H
      for(int shared_x=local_x+left_dst_offsetH; shared_x<left_dst_boundH; shared_x+=NTX) {
        int shared_order_H = 2; // (y, z) stride 
        int shared_idx_H = shared_x + shared_order_H * SHX;
        int global_x = xx_heads[xx] + shared_x;   
        int global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny;
        if(global_x >= 1 + LEFT_PAD && global_x <= Nx-2 + LEFT_PAD && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2) {
          Hx_src[global_idx] = Hx_shmem[shared_idx_H];
          Hy_src[global_idx] = Hy_shmem[shared_idx_H];
          Hz_src[global_idx] = Hz_shmem[shared_idx_H];
        }
      }
    }

    int right_dst_offsetE = BLT_DTR + NTX;
    int right_dst_offsetH = right_dst_offsetE;
    int right_dst_boundE = SHX;
    int right_dst_boundH = right_dst_boundE - 1;
    for(size_t tid=0; tid<block_size; tid++) {
      int local_x = tid;

      // store E
      for(int shared_x=local_x+right_dst_offsetE; shared_x<right_dst_boundE; shared_x+=NTX) {
        int shared_order_E = 0; // (y, z) stride 
        int shared_idx_E = shared_x + shared_order_E * SHX;
        int global_x = xx_heads[xx] + shared_x;   
        int global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny;
        if(global_x >= 1 + LEFT_PAD && global_x <= Nx-2 + LEFT_PAD && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2) {
          Ex_src[global_idx] = Ex_shmem[shared_idx_E];
          Ey_src[global_idx] = Ey_shmem[shared_idx_E];
          Ez_src[global_idx] = Ez_shmem[shared_idx_E];
        }
      }

      // store H
      for(int shared_x=local_x+right_dst_offsetH; shared_x<right_dst_boundH; shared_x+=NTX) {
        int shared_order_H = 2; // (y, z) stride 
        int shared_idx_H = shared_x + shared_order_H * SHX;
        int global_x = xx_heads[xx] + shared_x;   
        int global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny;
        if(global_x >= 1 + LEFT_PAD && global_x <= Nx-2 + LEFT_PAD && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2) {
          Hx_src[global_idx] = Hx_shmem[shared_idx_H];
          Hy_src[global_idx] = Hy_shmem[shared_idx_H];
          Hz_src[global_idx] = Hz_shmem[shared_idx_H];
        }
      }
    }
  }
} 

void gDiamond::_updateEH_dt_1D_valley_seq(const std::vector<float>& Ex_src, const std::vector<float>& Ey_src, const std::vector<float>& Ez_src,
                                          const std::vector<float>& Hx_src, const std::vector<float>& Hy_src, const std::vector<float>& Hz_src,
                                          std::vector<float>& Ex_final, std::vector<float>& Ey_final, std::vector<float>& Ez_final,
                                          std::vector<float>& Hx_final, std::vector<float>& Hy_final, std::vector<float>& Hz_final,
                                          const std::vector<float>& Cax, const std::vector<float>& Cbx,
                                          const std::vector<float>& Cay, const std::vector<float>& Cby,
                                          const std::vector<float>& Caz, const std::vector<float>& Cbz,
                                          const std::vector<float>& Dax, const std::vector<float>& Dbx,
                                          const std::vector<float>& Day, const std::vector<float>& Dby,
                                          const std::vector<float>& Daz, const std::vector<float>& Dbz,
                                          const std::vector<float>& Jx, const std::vector<float>& Jy, const std::vector<float>& Jz,
                                          const std::vector<float>& Mx, const std::vector<float>& My, const std::vector<float>& Mz,
                                          float dx, 
                                          int Nx, int Ny, int Nz,
                                          int Nx_pad, 
                                          int xx_num, // number of tiles in each dimensions
                                          std::vector<int> xx_heads, 
                                          size_t block_size,
                                          size_t grid_size,
                                          size_t tt) {

  for(size_t block_id=0; block_id<grid_size; block_id++) {
    int xx = block_id % xx_num;
    int temp = block_id / xx_num;
    int yy = temp % Ny;
    int zz = temp / Ny;

    const int global_z = zz; // global_z is always zz
    const int global_y = yy; // global_y is always yy

    // if(global_y < 1 || global_y > Ny-2 || global_z < 1 || global_z > Nz-2) {
    //   continue;
    // }
  
    // declare shared memory
    float Ex_shmem[SHX * 3] = {0};
    float Ey_shmem[SHX * 3] = {0};
    float Ez_shmem[SHX * 3] = {0};
    float Hx_shmem[SHX * 3] = {0};
    float Hy_shmem[SHX * 3] = {0};
    float Hz_shmem[SHX * 3] = {0};

    // load shared memory
    // In E_shmem, we store, in order, (y, z), (y+1, z), (y, z+1) stride
    // In H_shmem, we store, in order, (y, z-1), (y-1, z), (y, z) stride
    for(size_t tid=0; tid<block_size; tid++) {
      int local_x = tid;
      for(int shared_x=local_x; shared_x<SHX; shared_x+=NTX) {
        int shared_order_E, shared_order_H;
        int shared_idx_E, shared_idx_H;
        int global_x = xx_heads[xx] + shared_x;
        int global_idx;

        // load (y, z) stride for E_shmem, H_shmem
        // (y, z) is 1st stride for E_shmem
        // (y, z) is 3rd stride for H_shmem
        shared_order_E = 0;
        shared_order_H = 2;
        shared_idx_E = shared_x + shared_order_E * SHX;
        shared_idx_H = shared_x + shared_order_H * SHX;
        global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny;
        Ex_shmem[shared_idx_E] = Ex_src[global_idx];
        Ey_shmem[shared_idx_E] = Ey_src[global_idx];
        Ez_shmem[shared_idx_E] = Ez_src[global_idx];
        Hx_shmem[shared_idx_H] = Hx_src[global_idx];
        Hy_shmem[shared_idx_H] = Hy_src[global_idx];
        Hz_shmem[shared_idx_H] = Hz_src[global_idx];

        // load (y+1, z) stride for E_shmem
        shared_order_E = 1;
        shared_idx_E = shared_x + shared_order_E * SHX;
        global_idx = global_x + (global_y + 1) * Nx_pad + global_z * Nx_pad * Ny;
        Ex_shmem[shared_idx_E] = Ex_src[global_idx];
        Ey_shmem[shared_idx_E] = Ey_src[global_idx];
        Ez_shmem[shared_idx_E] = Ez_src[global_idx];

        // load (y, z+1) stride for E_shmem
        shared_order_E = 2;
        shared_idx_E = shared_x + shared_order_E * SHX;
        global_idx = global_x + global_y * Nx_pad + (global_z + 1) * Nx_pad * Ny;
        Ex_shmem[shared_idx_E] = Ex_src[global_idx];
        Ey_shmem[shared_idx_E] = Ey_src[global_idx];
        Ez_shmem[shared_idx_E] = Ez_src[global_idx];

        // load (y, z-1) stride for H_shmem
        shared_order_H = 0;
        shared_idx_H = shared_x + shared_order_H * SHX;
        global_idx = global_x + global_y * Nx_pad + (global_z - 1) * Nx_pad * Ny;
        Hx_shmem[shared_idx_H] = Hx_src[global_idx];
        Hy_shmem[shared_idx_H] = Hy_src[global_idx];
        Hz_shmem[shared_idx_H] = Hz_src[global_idx];

        // load (y-1, z) stride for H_shmem
        shared_order_H = 1;
        shared_idx_H = shared_x + shared_order_H * SHX;
        global_idx = global_x + (global_y - 1) * Nx_pad + global_z * Nx_pad * Ny;
        Hx_shmem[shared_idx_H] = Hx_src[global_idx];
        Hy_shmem[shared_idx_H] = Hy_src[global_idx];
        Hz_shmem[shared_idx_H] = Hz_src[global_idx];
      }
    }

    // calculation
    int cal_offsetX_E, cal_offsetX_H;
    int cal_boundX_E, cal_boundX_H;
    for(int t=0; t<BLT_DTR; t++) {
      cal_offsetX_E = BLT_DTR - t;
      cal_offsetX_H = cal_offsetX_E - 1;
      cal_boundX_E = SHX - (BLT_DTR - t);
      cal_boundX_H = cal_boundX_E; 
      for(size_t tid=0; tid<block_size; tid++) {
        int local_x = tid;

        // In E_shmem, we store, in order, (y, z), (y+1, z), (y, z+1) stride
        // In H_shmem, we store, in order, (y, z-1), (y-1, z), (y, z) stride

        // update E
        for(int shared_x=local_x+cal_offsetX_E; shared_x<cal_boundX_E; shared_x+=NTX) {
          int shared_order_E = 0; // (y, z) stride
          int shared_order_H = 2; // (y, z) stride
          int shared_idx_E = shared_x + shared_order_E * SHX;
          int shared_idx_H = shared_x + shared_order_H * SHX;
          int global_x = xx_heads[xx] + shared_x - LEFT_PAD; // - LEFT_PAD since constant arrays has not been padded
          int global_idx = global_x + global_y * Nx + global_z * Nx * Ny; // notice that here we are accessing the unpadded constant array

          if(global_x >= 1 && global_x <= Nx-2 && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2) {
            // (y, z-1) for Hx and Hy, (y-1, z) for Hx, Hz
            Ex_shmem[shared_idx_E] = Cax[global_idx] * Ex_shmem[shared_idx_E] + Cbx[global_idx] *
                        ((Hz_shmem[shared_idx_H] - Hz_shmem[shared_idx_H - SHX]) - (Hy_shmem[shared_idx_H] - Hy_shmem[shared_idx_H - 2 * SHX]) - Jx[global_idx] * dx);
            Ey_shmem[shared_idx_E] = Cay[global_idx] * Ey_shmem[shared_idx_E] + Cby[global_idx] *
                      ((Hx_shmem[shared_idx_H] - Hx_shmem[shared_idx_H - 2 * SHX]) - (Hz_shmem[shared_idx_H] - Hz_shmem[shared_idx_H - 1]) - Jy[global_idx] * dx);
            Ez_shmem[shared_idx_E] = Caz[global_idx] * Ez_shmem[shared_idx_E] + Cbz[global_idx] *
                      ((Hy_shmem[shared_idx_H] - Hy_shmem[shared_idx_H - 1]) - (Hx_shmem[shared_idx_H] - Hx_shmem[shared_idx_H - SHX]) - Jz[global_idx] * dx);
          }
        }
      }
      for(size_t tid=0; tid<block_size; tid++) {
        int local_x = tid;

        // update H
        for(int shared_x=local_x+cal_offsetX_H; shared_x<cal_boundX_H; shared_x+=NTX) {
          int shared_order_E = 0; // (y, z) stride
          int shared_order_H = 2; // (y, z) stride
          int shared_idx_E = shared_x + shared_order_E * SHX;
          int shared_idx_H = shared_x + shared_order_H * SHX;
          int global_x = xx_heads[xx] + shared_x - LEFT_PAD; // - LEFT_PAD since constant arrays has not been padded
          int global_idx = global_x + global_y * Nx + global_z * Nx * Ny; // notice that here we are accessing the unpadded constant array

          if(global_x >= 1 && global_x <= Nx-2 && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2) {
            // (y+1, z) for Ex, Ez, (y, z+1) for Ex, Ey
            Hx_shmem[shared_idx_H] = Dax[global_idx] * Hx_shmem[shared_idx_H] + Dbx[global_idx] *
                        ((Ey_shmem[shared_idx_E + 2 * SHX] - Ey_shmem[shared_idx_E]) - (Ez_shmem[shared_idx_E + SHX] - Ez_shmem[shared_idx_E]) - Mx[global_idx] * dx);
            Hy_shmem[shared_idx_H] = Day[global_idx] * Hy_shmem[shared_idx_H] + Dby[global_idx] *
                      ((Ez_shmem[shared_idx_E + 1] - Ez_shmem[shared_idx_E]) - (Ex_shmem[shared_idx_E + 2 * SHX] - Ex_shmem[shared_idx_E]) - My[global_idx] * dx);
            Hz_shmem[shared_idx_H] = Daz[global_idx] * Hz_shmem[shared_idx_H] + Dbz[global_idx] *
                      ((Ex_shmem[shared_idx_E + SHX] - Ex_shmem[shared_idx_E]) - (Ey_shmem[shared_idx_E + 1] - Ey_shmem[shared_idx_E]) - Mz[global_idx] * dx);
          }
        }
      }
    }

    // store to global memory
    int store_offsetE = 1;
    int store_offsetH = 0;
    int store_boundE = SHX - 1;
    int store_boundH = store_boundE;
    for(size_t tid=0; tid<block_size; tid++) {
      int local_x = tid; 
      // store E
      for(int shared_x=local_x+store_offsetE; shared_x<store_boundE; shared_x+=NTX) {
        int shared_order_E = 0; // (y, z) stride 
        int shared_idx_E = shared_x + shared_order_E * SHX;
        int global_x = xx_heads[xx] + shared_x;   
        int global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny;
        if(global_x >= 1 + LEFT_PAD && global_x <= Nx-2 + LEFT_PAD && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2) {
          Ex_final[global_idx] = Ex_shmem[shared_idx_E];
          Ey_final[global_idx] = Ey_shmem[shared_idx_E];
          Ez_final[global_idx] = Ez_shmem[shared_idx_E];
        }
      }
      // store H
      for(int shared_x=local_x+store_offsetH; shared_x<store_boundH; shared_x+=NTX) {
        int shared_order_H = 2; // (y, z) stride 
        int shared_idx_H = shared_x + shared_order_H * SHX;
        int global_x = xx_heads[xx] + shared_x;   
        int global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny;
        if(global_x >= 1 + LEFT_PAD && global_x <= Nx-2 + LEFT_PAD && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2) {
          Hx_final[global_idx] = Hx_shmem[shared_idx_H];
          Hy_final[global_idx] = Hy_shmem[shared_idx_H];
          Hz_final[global_idx] = Hz_shmem[shared_idx_H];
        }

      }
    }
  }

} 

void gDiamond::update_FDTD_cpu_simulation_dt_1_D_extra_copy(size_t num_timesteps, size_t Tx) { // CPU single thread 1-D simulation of diamond tiling, reimplemented

  // write 1 dimension just to check
  std::vector<float> E_simu(_Nx, 0.1);
  std::vector<float> H_simu(_Nx, 0.1);
  std::vector<float> E_seq(_Nx, 0.1);
  std::vector<float> H_seq(_Nx, 0.1);

  int Nx = _Nx;

  // seq version
  for(size_t t=0; t<num_timesteps; t++) {

    // update E
    for(int x=1; x<Nx-1; x++) {
      E_seq[x] = H_seq[x-1] + H_seq[x] * 2; 
    }

    std::cout << "t = " << t << ", E_seq =";
    for(int x=0; x<Nx; x++) {
      std::cout << E_seq[x] << " ";
    }
    std::cout << "\n";

    // update H 
    for(int x=1; x<Nx-1; x++) {
      H_seq[x] = E_seq[x+1] + E_seq[x] * 2; 
    }
  }

  // diamond tiling
  int Nx_pad = Nx + LEFT_PAD + RIGHT_PAD;
  
  // src, dst
  std::vector<float> E_src(Nx_pad, 0.1);
  std::vector<float> H_src(Nx_pad, 0.1);
  std::vector<float> E_dst(Nx_pad, 0.1);
  std::vector<float> H_dst(Nx_pad, 0.1);
  std::cout << "copy E, H to padded array\n";
  for(int i=0; i<Nx; i++) {
    E_src[i + LEFT_PAD] = E_simu[i];
    H_src[i + LEFT_PAD] = H_simu[i];
  }

  std::cout << "E_src = ";
  for(const auto& data : E_src) {
    std::cout << data << " ";
  }
  std::cout << "\n";

  // xx_heads_mountain[xx] is 1 element left offset to the actual mountain
  int xx_num_mountains = 1 + Tx;
  int xx_num_valleys = Tx + 1;
  std::vector<int> xx_heads_mountain(xx_num_mountains, 0);
  std::vector<int> xx_heads_valley(xx_num_valleys, 0);

  for(int index=0; index<xx_num_mountains; index++) {
    xx_heads_mountain[index] = (index == 0)? 0 :
                               xx_heads_mountain[index-1] + (BLX_DTR + NTX);
  }
  for(int index=0; index<xx_num_valleys; index++) {
    xx_heads_valley[index] = (index == 0)? BLX_DTR - (BLT_DTR - 1) :
                             xx_heads_valley[index-1] + (BLX_DTR + NTX);
  }
  std::cout << "xx_heads_mountain = ";
  for(const auto& index : xx_heads_mountain) {
    std::cout << index << " ";
  }
  std::cout << "\n";

  std::cout << "xx_heads_valley = ";
  for(const auto& index : xx_heads_valley) {
    std::cout << index << " ";
  }
  std::cout << "\n";

  int block_size = 4; // each block has 4 threads
  for(size_t tt=0; tt<num_timesteps/BLT_DTR; tt++) {

    // phase 1
    for(int xx=0; xx<xx_num_mountains; xx++) {
      // declare shared memory
      float E_shmem[SHX];
      float H_shmem[SHX]; 

      // load shared memory
      for(int tid=0; tid<block_size; tid++) {
        for(int shared_idx=tid; shared_idx<SHX; shared_idx+=block_size) {
          int global_idx = xx_heads_mountain[xx] + shared_idx;
          E_shmem[shared_idx] = E_src[global_idx];
          H_shmem[shared_idx] = H_src[global_idx];
        }
      }

      // calculation
      for(int t=0; t<BLT_DTR; t++) {
        int cal_offsetE = t + 1;
        int cal_offsetH = cal_offsetE;
        int cal_boundE = SHX - t;
        int cal_boundH = cal_boundE - 1;

        // update E
        for(int tid=0; tid<block_size; tid++) {
          for(int shared_idx=tid+cal_offsetE; shared_idx<cal_boundE; shared_idx+=block_size) {
            int global_idx = xx_heads_mountain[xx] + shared_idx;
            if(global_idx >= 1 + LEFT_PAD && global_idx <= Nx - 2 + LEFT_PAD) {
              E_shmem[shared_idx] = H_shmem[shared_idx-1] + H_shmem[shared_idx] * 2;
            }
          }
        }
        
        // update H
        for(int tid=0; tid<block_size; tid++) {
          for(int shared_idx=tid+cal_offsetH; shared_idx<cal_boundH; shared_idx+=block_size) {
            int global_idx = xx_heads_mountain[xx] + shared_idx;
            if(global_idx >= 1 + LEFT_PAD && global_idx <= Nx - 2 + LEFT_PAD) {
              H_shmem[shared_idx] = E_shmem[shared_idx+1] + E_shmem[shared_idx] * 2;
            }
          }
        }
      }

      // store global memory
      int store_offsetE = 1;
      int store_offsetH = store_offsetE;
      int store_boundE = SHX;
      int store_boundH = store_boundE - 1;
      for(int tid=0; tid<block_size; tid++) {

        // store E
        for(int shared_idx=tid + store_offsetE; shared_idx<store_boundE; shared_idx+=block_size) {
          int global_idx = xx_heads_mountain[xx] + shared_idx;
          if(global_idx >= 1 + LEFT_PAD && global_idx <= Nx - 2 + LEFT_PAD) {
            E_dst[global_idx] = E_shmem[shared_idx];
          }
        }

        // update H
        for(int shared_idx=tid + store_offsetH; shared_idx<store_boundH; shared_idx+=block_size) {
          int global_idx = xx_heads_mountain[xx] + shared_idx;
          if(global_idx >= 1 + LEFT_PAD && global_idx <= Nx - 2 + LEFT_PAD) {
            H_dst[global_idx] = H_shmem[shared_idx];
          }
        }

      }

      // extra copy to dst
      int extra_offsetE = BLT_DTR;
      int extra_offsetH = extra_offsetE - 1;
      int extra_boundE = extra_offsetE + NTX;
      int extra_boundH = extra_boundE + 1;
      for(int tid=0; tid<block_size; tid++) {
        for(int shared_idx=tid+extra_offsetE; shared_idx<extra_boundE; shared_idx+=NTX) {
          int global_idx = xx_heads_valley[xx] + shared_idx;
          if(global_idx >= 1 + LEFT_PAD && global_idx <= Nx - 2 + LEFT_PAD) {
            E_dst[global_idx] = E_src[global_idx];
          }
        }
        for(int shared_idx=tid+extra_offsetH; shared_idx<extra_boundH; shared_idx+=NTX) {
          int global_idx = xx_heads_valley[xx] + shared_idx;
          if(global_idx >= 1 + LEFT_PAD && global_idx <= Nx - 2 + LEFT_PAD) {
            H_dst[global_idx] = H_src[global_idx];
          }
        }
      }


    }

    // swap src and dst
    std::swap(E_src, E_dst);
    std::swap(H_src, H_dst);

    // phase 2
    for(int xx=0; xx<xx_num_valleys; xx++) {
      // declare shared memory
      float E_shmem[SHX];
      float H_shmem[SHX];

      // load shared memory
      for(int tid=0; tid<block_size; tid++) {
        for(int shared_idx=tid; shared_idx<SHX; shared_idx+=block_size) {
          int global_idx = xx_heads_valley[xx] + shared_idx;
          E_shmem[shared_idx] = E_src[global_idx];
          H_shmem[shared_idx] = H_src[global_idx];
        }
      }

      // calculation
      int cal_offsetE, cal_offsetH;
      int cal_boundE, cal_boundH;
      for(int t=0; t<BLT_DTR; t++) {
        cal_offsetE = BLT_DTR - t;
        cal_offsetH = cal_offsetE - 1;
        cal_boundE = SHX - (BLT_DTR - t);
        cal_boundH = cal_boundE; 

        // update E
        for(int tid=0; tid<block_size; tid++) {
          for(int shared_idx=tid+cal_offsetE; shared_idx<cal_boundE; shared_idx+=block_size) {
            int global_idx = xx_heads_valley[xx] + shared_idx;
            if(global_idx >= 1 + LEFT_PAD && global_idx <= Nx - 2 + LEFT_PAD) {
              E_shmem[shared_idx] = H_shmem[shared_idx-1] + H_shmem[shared_idx] * 2;
            }
          }
        }

        // update H
        for(int tid=0; tid<block_size; tid++) {
          for(int shared_idx=tid+cal_offsetH; shared_idx<cal_boundH; shared_idx+=block_size) {
            int global_idx = xx_heads_valley[xx] + shared_idx;
            // std::cout << "t = " << t << ", ";
            // std::cout << "xx = " << xx << ", shared_idx = " << shared_idx << ", global_idx = " << global_idx << "\n";
            if(global_idx >= 1 + LEFT_PAD && global_idx <= Nx - 2 + LEFT_PAD) {
              H_shmem[shared_idx] = E_shmem[shared_idx+1] + E_shmem[shared_idx] * 2;
            }
          }
        }
      }

      // store global memory 
      int store_offsetE = 1;
      int store_offsetH = 0;
      int store_boundE = SHX - 1;
      int store_boundH = store_boundE;
      for(int tid=0; tid<block_size; tid++) {

        // store E
        for(int shared_idx=tid + store_offsetE; shared_idx<store_boundE; shared_idx+=block_size) {
          int global_idx = xx_heads_valley[xx] + shared_idx;
          if(global_idx >= 1 + LEFT_PAD && global_idx <= Nx - 2 + LEFT_PAD) {
            E_dst[global_idx] = E_shmem[shared_idx];
          }
        }

        // update H
        for(int shared_idx=tid + store_offsetH; shared_idx<store_boundH; shared_idx+=block_size) {
          int global_idx = xx_heads_valley[xx] + shared_idx;
          if(global_idx >= 1 + LEFT_PAD && global_idx <= Nx - 2 + LEFT_PAD) {
            H_dst[global_idx] = H_shmem[shared_idx];
          }
        }

      }

      // extra store to dst
      int extra_offsetE = BLT_DTR;
      int extra_offsetH = extra_offsetE;
      int extra_boundE = extra_offsetE + NTX + 1;
      int extra_boundH = extra_boundE - 1;
      for(int tid=0; tid<block_size; tid++) {
        for(int shared_idx=tid+extra_offsetE; shared_idx<extra_boundE; shared_idx+=NTX) {
          int global_idx = xx_heads_mountain[xx] + shared_idx;
          if(global_idx >= 1 + LEFT_PAD && global_idx <= Nx - 2 + LEFT_PAD) {
            E_dst[global_idx] = E_src[global_idx];
          }
        }
        for(int shared_idx=tid+extra_offsetH; shared_idx<extra_boundH; shared_idx+=NTX) {
          int global_idx = xx_heads_mountain[xx] + shared_idx;
          if(global_idx >= 1 + LEFT_PAD && global_idx <= Nx - 2 + LEFT_PAD) {
            H_dst[global_idx] = H_src[global_idx];
          }
        }
      }
    }

    // swap src and dst
    std::swap(E_src, E_dst);
    std::swap(H_src, H_dst);

  }

  std::cout << "E_src = ";
  for(int x=0; x<Nx_pad; x++) {
    std::cout << E_src[x] << " ";
  }
  std::cout << "\n";

  // copy results from src to simu
  for(int x=0; x<Nx; x++) {
    E_simu[x] = E_src[x+LEFT_PAD];
    H_simu[x] = H_src[x+LEFT_PAD];
  }

  std::cout << "E_seq = ";
  for(int x=0; x<Nx; x++) {
    std::cout << E_seq[x] << " ";
  }
  std::cout << "\n";

  std::cout << "E_simu = ";
  for(int x=0; x<Nx; x++) {
    std::cout << E_simu[x] << " ";
  }
  std::cout << "\n";

  for(int x=0; x<Nx; x++) {
    if(E_seq[x] != E_simu[x] || H_seq[x] != H_simu[x]) {
      std::cerr << "1-D demo results mismatch.\n";
      std::exit(EXIT_FAILURE);
    }
  }

}

void gDiamond::update_FDTD_cpu_simulation_dt_3_D_extra_copy(size_t num_timesteps, size_t Tx) { // CPU single thread 3-D simulation of diamond tiling, reimplemented

  // clear source Mz for experiments
  _Mz.clear();

  // transfer source
  for(size_t t=0; t<num_timesteps; t++) {
    float Mz_value = M_source_amp * std::sin(SOURCE_OMEGA * t * dt);
    _Mz[_source_idx] = Mz_value;
  }

  // tiling parameter pre-processing
  int Nx = _Nx;
  int Ny = _Ny;
  int Nz = _Nz;
  int Nx_pad = Nx + LEFT_PAD + RIGHT_PAD;

  int xx_num_mountains = 1 + Tx;
  int xx_num_valleys = Tx + 1;

  // xx_heads_mountain[xx] is 1 element left offset to the actual mountain
  std::vector<int> xx_heads_mountain(xx_num_mountains, 0);
  std::vector<int> xx_heads_valley(xx_num_valleys, 0);

  for(int index=0; index<xx_num_mountains; index++) {
    xx_heads_mountain[index] = (index == 0)? 0 :
                               xx_heads_mountain[index-1] + (BLX_DTR + NTX);
  }
  for(int index=0; index<xx_num_valleys; index++) {
    xx_heads_valley[index] = (index == 0)? BLX_DTR - (BLT_DTR - 1) :
                             xx_heads_valley[index-1] + (BLX_DTR + NTX);
  }

  std::cout << "xx_heads_mountain = ";
  for(const auto& index : xx_heads_mountain) {
    std::cout << index << " ";
  }
  std::cout << "\n";

  std::cout << "xx_heads_valley = ";
  for(const auto& index : xx_heads_valley) {
    std::cout << index << " ";
  }
  std::cout << "\n";

  // padded E, H
  std::vector<float> Ex_src(Nx_pad * Ny * Nz, 0);
  std::vector<float> Ey_src(Nx_pad * Ny * Nz, 0);
  std::vector<float> Ez_src(Nx_pad * Ny * Nz, 0);
  std::vector<float> Hx_src(Nx_pad * Ny * Nz, 0);
  std::vector<float> Hy_src(Nx_pad * Ny * Nz, 0);
  std::vector<float> Hz_src(Nx_pad * Ny * Nz, 0);

  std::vector<float> Ex_dst(Nx_pad * Ny * Nz, 0);
  std::vector<float> Ey_dst(Nx_pad * Ny * Nz, 0);
  std::vector<float> Ez_dst(Nx_pad * Ny * Nz, 0);
  std::vector<float> Hx_dst(Nx_pad * Ny * Nz, 0);
  std::vector<float> Hy_dst(Nx_pad * Ny * Nz, 0);
  std::vector<float> Hz_dst(Nx_pad * Ny * Nz, 0);

  // define block size and grid size
  size_t block_size = NTX;
  size_t grid_size; // grid_size = xx_num * Ny * Nz;

  for(size_t tt=0; tt<num_timesteps/BLT_DTR; tt++) {
    std::cout << "running\n";
    grid_size = xx_num_mountains * Ny * Nz; 
    _updateEH_dt_1D_mountain_seq_extra_copy(Ex_src, Ey_src, Ez_src,
                                 Hx_src, Hy_src, Hz_src,
                                 Ex_dst, Ey_dst, Ez_dst,
                                 Hx_dst, Hy_dst, Hz_dst,
                                 _Cax, _Cbx,
                                 _Cay, _Cby,
                                 _Caz, _Cbz,
                                 _Dax, _Dbx,
                                 _Day, _Dby,
                                 _Daz, _Dbz,
                                 _Jx, _Jy, _Jz,
                                 _Mx, _My, _Mz,
                                 _dx, 
                                 Nx, Ny, Nz,
                                 Nx_pad, 
                                 xx_num_mountains, // number of tiles in each dimensions
                                 xx_heads_mountain, 
                                 xx_heads_valley, 
                                 block_size,
                                 grid_size, 
                                 tt); 

    if(1) {
      std::cout << "check data\n";
    }

    std::swap(Ex_dst, Ex_src);
    std::swap(Ey_dst, Ey_src);
    std::swap(Ez_dst, Ez_src);
    std::swap(Hx_dst, Hx_src);
    std::swap(Hy_dst, Hy_src);
    std::swap(Hz_dst, Hz_src);

    grid_size = xx_num_valleys * Ny * Nz;
    _updateEH_dt_1D_valley_seq_extra_copy(Ex_src, Ey_src, Ez_src,
                                 Hx_src, Hy_src, Hz_src,
                                 Ex_dst, Ey_dst, Ez_dst,
                                 Hx_dst, Hy_dst, Hz_dst,
                                 _Cax, _Cbx,
                                 _Cay, _Cby,
                                 _Caz, _Cbz,
                                 _Dax, _Dbx,
                                 _Day, _Dby,
                                 _Daz, _Dbz,
                                 _Jx, _Jy, _Jz,
                                 _Mx, _My, _Mz,
                                 _dx, 
                                 Nx, Ny, Nz,
                                 Nx_pad, 
                                 xx_num_valleys, // number of tiles in each dimensions
                                 xx_heads_mountain, 
                                 xx_heads_valley, 
                                 block_size,
                                 grid_size, 
                                 tt); 

    std::swap(Ex_dst, Ex_src);
    std::swap(Ey_dst, Ey_src);
    std::swap(Ez_dst, Ez_src);
    std::swap(Hx_dst, Hx_src);
    std::swap(Hy_dst, Hy_src);
    std::swap(Hz_dst, Hz_src);
  }

  std::cout << "Ex_src = \n";
  for(int k=0; k<Nz; k++) {
    for(int j=0; j<Ny; j++) {
      for(int i=0; i<Nx_pad; i++) {
        int idx = i + j*Nx_pad + k*(Nx_pad*Ny);
        if(Ex_src[idx] != 0) { 
          std::cout << "(x, y, z) = " << i << ", " << j << ", " << k << ", ";
          std::cout << "Ex_src[idx] = " << Ex_src[idx] << "\n";
        }
      }
    }
  }

  _extract_original_from_padded_1D(Ex_src, _Ex_simu, Nx, Ny, Nz, Nx_pad, LEFT_PAD);
  _extract_original_from_padded_1D(Ey_src, _Ey_simu, Nx, Ny, Nz, Nx_pad, LEFT_PAD);
  _extract_original_from_padded_1D(Ez_src, _Ez_simu, Nx, Ny, Nz, Nx_pad, LEFT_PAD);
  _extract_original_from_padded_1D(Hx_src, _Hx_simu, Nx, Ny, Nz, Nx_pad, LEFT_PAD);
  _extract_original_from_padded_1D(Hy_src, _Hy_simu, Nx, Ny, Nz, Nx_pad, LEFT_PAD);
  _extract_original_from_padded_1D(Hz_src, _Hz_simu, Nx, Ny, Nz, Nx_pad, LEFT_PAD);

}

void gDiamond::_updateEH_dt_1D_mountain_seq_extra_copy(const std::vector<float>& Ex_src, const std::vector<float>& Ey_src, const std::vector<float>& Ez_src,
                                      const std::vector<float>& Hx_src, const std::vector<float>& Hy_src, const std::vector<float>& Hz_src,
                                      std::vector<float>& Ex_dst, std::vector<float>& Ey_dst, std::vector<float>& Ez_dst,
                                      std::vector<float>& Hx_dst, std::vector<float>& Hy_dst, std::vector<float>& Hz_dst,
                                      const std::vector<float>& Cax, const std::vector<float>& Cbx,
                                      const std::vector<float>& Cay, const std::vector<float>& Cby,
                                      const std::vector<float>& Caz, const std::vector<float>& Cbz,
                                      const std::vector<float>& Dax, const std::vector<float>& Dbx,
                                      const std::vector<float>& Day, const std::vector<float>& Dby,
                                      const std::vector<float>& Daz, const std::vector<float>& Dbz,
                                      const std::vector<float>& Jx, const std::vector<float>& Jy, const std::vector<float>& Jz,
                                      const std::vector<float>& Mx, const std::vector<float>& My, const std::vector<float>& Mz,
                                      float dx, 
                                      int Nx, int Ny, int Nz,
                                      int Nx_pad, 
                                      int xx_num, // number of tiles in each dimensions
                                      std::vector<int> xx_heads_mountain, 
                                      std::vector<int> xx_heads_valley, 
                                      size_t block_size,
                                      size_t grid_size, 
                                      size_t tt) {

  for(size_t block_id=0; block_id<grid_size; block_id++) {
    
    int xx = block_id % xx_num;
    int temp = block_id / xx_num;
    int yy = temp % Ny;
    int zz = temp / Ny;

    // std::cout << "(xx, yy, zz) = " << xx << ", " << yy << ", " << zz << "\n"; 

    const int global_z = zz; // global_z is always zz
    const int global_y = yy; // global_y is always yy

    // declare shared memory
    float Ex_shmem[SHX * 3] = {0};
    float Ey_shmem[SHX * 3] = {0};
    float Ez_shmem[SHX * 3] = {0};
    float Hx_shmem[SHX * 3] = {0};
    float Hy_shmem[SHX * 3] = {0};
    float Hz_shmem[SHX * 3] = {0};

    // load shared memory
    // In E_shmem, we store, in order, (y, z), (y+1, z), (y, z+1) stride
    // In H_shmem, we store, in order, (y, z-1), (y-1, z), (y, z) stride
    for(size_t tid=0; tid<block_size; tid++) {
      int local_x = tid;
      for(int shared_x=local_x; shared_x<SHX; shared_x+=NTX) {
        int shared_order_E, shared_order_H;
        int shared_idx_E, shared_idx_H;
        int global_x = xx_heads_mountain[xx] + shared_x;
        int global_idx;

        // load (y, z) stride for E_shmem, H_shmem
        // (y, z) is 1st stride for E_shmem
        // (y, z) is 3rd stride for H_shmem
        shared_order_E = 0;
        shared_order_H = 2;
        shared_idx_E = shared_x + shared_order_E * SHX;
        shared_idx_H = shared_x + shared_order_H * SHX;
        global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny;

        // if(xx == 1 && yy == 1 && zz == 1) {
        //   std::cout << "(xx, yy, zz) = " << xx << ", " << yy << ", " << zz << ", ";
        //   std::cout << "local_x = " << local_x << ", ";
        //   std::cout << "shared_x = " << shared_x << ", ";
        //   std::cout << "global_x = " << global_x << ", ";
        //   std::cout << "global_idx = " << global_idx << ", ";
        //   std::cout << "Ex_src[global_idx] = " << Ex_src[global_idx] << "\n";
        // }

        Ex_shmem[shared_idx_E] = Ex_src[global_idx];
        Ey_shmem[shared_idx_E] = Ey_src[global_idx];
        Ez_shmem[shared_idx_E] = Ez_src[global_idx];
        Hx_shmem[shared_idx_H] = Hx_src[global_idx];
        Hy_shmem[shared_idx_H] = Hy_src[global_idx];
        Hz_shmem[shared_idx_H] = Hz_src[global_idx];

        // load (y+1, z) stride for E_shmem
        shared_order_E = 1;
        shared_idx_E = shared_x + shared_order_E * SHX;
        global_idx = global_x + (global_y + 1) * Nx_pad + global_z * Nx_pad * Ny;
        Ex_shmem[shared_idx_E] = Ex_src[global_idx];
        Ey_shmem[shared_idx_E] = Ey_src[global_idx];
        Ez_shmem[shared_idx_E] = Ez_src[global_idx];

        // load (y, z+1) stride for E_shmem
        shared_order_E = 2;
        shared_idx_E = shared_x + shared_order_E * SHX;
        global_idx = global_x + global_y * Nx_pad + (global_z + 1) * Nx_pad * Ny;
        Ex_shmem[shared_idx_E] = Ex_src[global_idx];
        Ey_shmem[shared_idx_E] = Ey_src[global_idx];
        Ez_shmem[shared_idx_E] = Ez_src[global_idx];

        // load (y, z-1) stride for H_shmem
        shared_order_H = 0;
        shared_idx_H = shared_x + shared_order_H * SHX;
        global_idx = global_x + global_y * Nx_pad + (global_z - 1) * Nx_pad * Ny;
        Hx_shmem[shared_idx_H] = Hx_src[global_idx];
        Hy_shmem[shared_idx_H] = Hy_src[global_idx];
        Hz_shmem[shared_idx_H] = Hz_src[global_idx];

        // load (y-1, z) stride for H_shmem
        shared_order_H = 1;
        shared_idx_H = shared_x + shared_order_H * SHX;
        global_idx = global_x + (global_y - 1) * Nx_pad + global_z * Nx_pad * Ny;
        Hx_shmem[shared_idx_H] = Hx_src[global_idx];
        Hy_shmem[shared_idx_H] = Hy_src[global_idx];
        Hz_shmem[shared_idx_H] = Hz_src[global_idx];
      }
    }

    // calculation
    int cal_offsetX_E, cal_offsetX_H;
    int cal_boundX_E, cal_boundX_H;
    for(int t=0; t<BLT_DTR; t++) {
      cal_offsetX_E = t + 1;
      cal_offsetX_H = cal_offsetX_E;
      cal_boundX_E = SHX - t;
      cal_boundX_H = cal_boundX_E - 1;

      for(size_t tid=0; tid<block_size; tid++) {
        int local_x = tid;

        // In E_shmem, we store, in order, (y, z), (y+1, z), (y, z+1) stride
        // In H_shmem, we store, in order, (y, z-1), (y-1, z), (y, z) stride

        // update E
        for(int shared_x=local_x+cal_offsetX_E; shared_x<cal_boundX_E; shared_x+=NTX) {
          int shared_order_E = 0; // (y, z) stride
          int shared_order_H = 2; // (y, z) stride
          int shared_idx_E = shared_x + shared_order_E * SHX;
          int shared_idx_H = shared_x + shared_order_H * SHX;
          int global_x = xx_heads_mountain[xx] + shared_x - LEFT_PAD; // - LEFT_PAD since constant arrays has not been padded
          int global_idx = global_x + global_y * Nx + global_z * Nx * Ny; // notice that here we are accessing the unpadded constant array

          if(global_x >= 1 && global_x <= Nx-2 && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2) {
            // (y, z-1) for Hx and Hy, (y-1, z) for Hx, Hz
            Ex_shmem[shared_idx_E] = Cax[global_idx] * Ex_shmem[shared_idx_E] + Cbx[global_idx] *
                        ((Hz_shmem[shared_idx_H] - Hz_shmem[shared_idx_H - SHX]) - (Hy_shmem[shared_idx_H] - Hy_shmem[shared_idx_H - 2 * SHX]) - Jx[global_idx] * dx);
            Ey_shmem[shared_idx_E] = Cay[global_idx] * Ey_shmem[shared_idx_E] + Cby[global_idx] *
                      ((Hx_shmem[shared_idx_H] - Hx_shmem[shared_idx_H - 2 * SHX]) - (Hz_shmem[shared_idx_H] - Hz_shmem[shared_idx_H - 1]) - Jy[global_idx] * dx);
            Ez_shmem[shared_idx_E] = Caz[global_idx] * Ez_shmem[shared_idx_E] + Cbz[global_idx] *
                      ((Hy_shmem[shared_idx_H] - Hy_shmem[shared_idx_H - 1]) - (Hx_shmem[shared_idx_H] - Hx_shmem[shared_idx_H - SHX]) - Jz[global_idx] * dx);
            
            if(t == 1 && global_x == 13 && global_y == 2 && global_z == 2) {
              std::cout << "but here, Ex_shmem[shared_idx_E] = " << Ex_shmem[shared_idx_E]
                        << ", Hz_shmem[shared_idx_H] = " << Hz_shmem[shared_idx_H]
                        << ", Hz_shmem[shared_idx_H - SHX] = " << Hz_shmem[shared_idx_H - SHX]
                        << ", Hy_shmem[shared_idx_H] = " << Hy_shmem[shared_idx_H]
                        << ", Hy_shmem[shared_idx_H - 2 * SHX] = " << Hy_shmem[shared_idx_H - 2 * SHX]
                        << ", Jx[global_idx] = " << Jx[global_idx]
                        << ", shared_idx_E = " << shared_idx_E
                        << "\n";
            }
          
          }
        }
      }

      for(size_t tid=0; tid<block_size; tid++) {
        int local_x = tid;
        // update H
        for(int shared_x=local_x+cal_offsetX_H; shared_x<cal_boundX_H; shared_x+=NTX) {
          int shared_order_E = 0; // (y, z) stride
          int shared_order_H = 2; // (y, z) stride
          int shared_idx_E = shared_x + shared_order_E * SHX;
          int shared_idx_H = shared_x + shared_order_H * SHX;
          int global_x = xx_heads_mountain[xx] + shared_x - LEFT_PAD; // - LEFT_PAD since constant arrays has not been padded
          int global_idx = global_x + global_y * Nx + global_z * Nx * Ny; // notice that here we are accessing the unpadded constant array

          if(global_x >= 1 && global_x <= Nx-2 && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2) {
            // (y+1, z) for Ex, Ez, (y, z+1) for Ex, Ey
            Hx_shmem[shared_idx_H] = Dax[global_idx] * Hx_shmem[shared_idx_H] + Dbx[global_idx] *
                        ((Ey_shmem[shared_idx_E + 2 * SHX] - Ey_shmem[shared_idx_E]) - (Ez_shmem[shared_idx_E + SHX] - Ez_shmem[shared_idx_E]) - Mx[global_idx] * dx);
            Hy_shmem[shared_idx_H] = Day[global_idx] * Hy_shmem[shared_idx_H] + Dby[global_idx] *
                      ((Ez_shmem[shared_idx_E + 1] - Ez_shmem[shared_idx_E]) - (Ex_shmem[shared_idx_E + 2 * SHX] - Ex_shmem[shared_idx_E]) - My[global_idx] * dx);
            Hz_shmem[shared_idx_H] = Daz[global_idx] * Hz_shmem[shared_idx_H] + Dbz[global_idx] *
                      ((Ex_shmem[shared_idx_E + SHX] - Ex_shmem[shared_idx_E]) - (Ey_shmem[shared_idx_E + 1] - Ey_shmem[shared_idx_E]) - Mz[global_idx] * dx);

            if(t == 1 && global_x == 13 && global_y == 1 && global_z == 2) {
              std::cout << "but here, Hz_shmem[shared_idx_H] = " << Hz_shmem[shared_idx_H] 
                        << ", Ex_shmem[shared_idx_E + SHX] = " << Ex_shmem[shared_idx_E + SHX]
                        << ", Ex_shmem[shared_idx_E] = " << Ex_shmem[shared_idx_E]
                        << ", Ey_shmem[shared_idx_E + 1] = " << Ey_shmem[shared_idx_E + 1]
                        << ", Ey_shmem[shared_idx_E] = " << Ey_shmem[shared_idx_E]
                        << ", Mz[global_idx] = " << Mz[global_idx]
                        << ", shared_idx_E + SHX = " << shared_idx_E + SHX
                        << "\n";
            }

          }
        }

      }
    }

    // store global memory
    int store_offsetE = 1;
    int store_offsetH = store_offsetE;
    int store_boundE = SHX;
    int store_boundH = store_boundE - 1;
    for(size_t tid=0; tid<block_size; tid++) {
      int local_x = tid;

      // store E
      for(int shared_x=local_x+store_offsetE; shared_x<store_boundE; shared_x+=NTX) {
        int shared_order_E = 0; // (y, z) stride
        int shared_idx_E = shared_x + shared_order_E * SHX;
        int global_x = xx_heads_mountain[xx] + shared_x;
        int global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny;
        if(global_x >= 1 + LEFT_PAD && global_x <= Nx-2 + LEFT_PAD && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2) {
          Ex_dst[global_idx] = Ex_shmem[shared_idx_E];
          Ey_dst[global_idx] = Ey_shmem[shared_idx_E];
          Ez_dst[global_idx] = Ez_shmem[shared_idx_E];
        }
      }

      // store H
      for(int shared_x=local_x+store_offsetH; shared_x<store_boundH; shared_x+=NTX) {
        int shared_order_H = 2; // (y, z) stride
        int shared_idx_H = shared_x + shared_order_H * SHX;
        int global_x = xx_heads_mountain[xx] + shared_x;
        int global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny;
        if(global_x >= 1 + LEFT_PAD && global_x <= Nx-2 + LEFT_PAD && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2) {
          Hx_dst[global_idx] = Hx_shmem[shared_idx_H];
          Hy_dst[global_idx] = Hy_shmem[shared_idx_H];
          Hz_dst[global_idx] = Hz_shmem[shared_idx_H];
        }
      }
    }
    
    // extra store to dst
    int extra_offsetE = BLT_DTR;
    int extra_offsetH = extra_offsetE - 1;
    int extra_boundE = extra_offsetE + NTX;
    int extra_boundH = extra_boundE + 1;
    for(size_t tid=0; tid<block_size; tid++) {
      int local_x = tid;

      // store E
      for(int shared_x=local_x+extra_offsetE; shared_x<extra_boundE; shared_x+=NTX) {
        int global_x = xx_heads_valley[xx] + shared_x;
        int global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny;
        if(global_x >= 1 + LEFT_PAD && global_x <= Nx-2 + LEFT_PAD && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2) {
          Ex_dst[global_idx] = Ex_src[global_idx];
          Ey_dst[global_idx] = Ey_src[global_idx];
          Ez_dst[global_idx] = Ez_src[global_idx];
        }
      }

      // store H
      for(int shared_x=local_x+extra_offsetH; shared_x<extra_boundH; shared_x+=NTX) {
        int global_x = xx_heads_valley[xx] + shared_x;
        int global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny;
        if(global_x >= 1 + LEFT_PAD && global_x <= Nx-2 + LEFT_PAD && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2) {
          Hx_dst[global_idx] = Hx_src[global_idx];
          Hy_dst[global_idx] = Hy_src[global_idx];
          Hz_dst[global_idx] = Hz_src[global_idx];
        }
      }
    }

    if(xx == 1 && yy == 2 && zz == 2) {
      std::cout << "2nd check, Ex_dst[357] = " << Ex_dst[357] << "\n";
    }

  }

} 

void gDiamond::_updateEH_dt_1D_valley_seq_extra_copy(const std::vector<float>& Ex_src, const std::vector<float>& Ey_src, const std::vector<float>& Ez_src,
                                      const std::vector<float>& Hx_src, const std::vector<float>& Hy_src, const std::vector<float>& Hz_src,
                                      std::vector<float>& Ex_dst, std::vector<float>& Ey_dst, std::vector<float>& Ez_dst,
                                      std::vector<float>& Hx_dst, std::vector<float>& Hy_dst, std::vector<float>& Hz_dst,
                                      const std::vector<float>& Cax, const std::vector<float>& Cbx,
                                      const std::vector<float>& Cay, const std::vector<float>& Cby,
                                      const std::vector<float>& Caz, const std::vector<float>& Cbz,
                                      const std::vector<float>& Dax, const std::vector<float>& Dbx,
                                      const std::vector<float>& Day, const std::vector<float>& Dby,
                                      const std::vector<float>& Daz, const std::vector<float>& Dbz,
                                      const std::vector<float>& Jx, const std::vector<float>& Jy, const std::vector<float>& Jz,
                                      const std::vector<float>& Mx, const std::vector<float>& My, const std::vector<float>& Mz,
                                      float dx, 
                                      int Nx, int Ny, int Nz,
                                      int Nx_pad, 
                                      int xx_num, // number of tiles in each dimensions
                                      std::vector<int> xx_heads_mountain, 
                                      std::vector<int> xx_heads_valley, 
                                      size_t block_size,
                                      size_t grid_size, 
                                      size_t tt) {

  for(size_t block_id=0; block_id<grid_size; block_id++) {
    int xx = block_id % xx_num;
    int temp = block_id / xx_num;
    int yy = temp % Ny;
    int zz = temp / Ny;

    const int global_z = zz; // global_z is always zz
    const int global_y = yy; // global_y is always yy

    // if(global_y < 1 || global_y > Ny-2 || global_z < 1 || global_z > Nz-2) {
    //   continue;
    // }
  
    // declare shared memory
    float Ex_shmem[SHX * 3] = {0};
    float Ey_shmem[SHX * 3] = {0};
    float Ez_shmem[SHX * 3] = {0};
    float Hx_shmem[SHX * 3] = {0};
    float Hy_shmem[SHX * 3] = {0};
    float Hz_shmem[SHX * 3] = {0};

    // load shared memory
    // In E_shmem, we store, in order, (y, z), (y+1, z), (y, z+1) stride
    // In H_shmem, we store, in order, (y, z-1), (y-1, z), (y, z) stride
    for(size_t tid=0; tid<block_size; tid++) {
      int local_x = tid;
      for(int shared_x=local_x; shared_x<SHX; shared_x+=NTX) {
        int shared_order_E, shared_order_H;
        int shared_idx_E, shared_idx_H;
        int global_x = xx_heads_valley[xx] + shared_x;
        int global_idx;

        // load (y, z) stride for E_shmem, H_shmem
        // (y, z) is 1st stride for E_shmem
        // (y, z) is 3rd stride for H_shmem
        shared_order_E = 0;
        shared_order_H = 2;
        shared_idx_E = shared_x + shared_order_E * SHX;
        shared_idx_H = shared_x + shared_order_H * SHX;
        global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny;
        Ex_shmem[shared_idx_E] = Ex_src[global_idx];
        Ey_shmem[shared_idx_E] = Ey_src[global_idx];
        Ez_shmem[shared_idx_E] = Ez_src[global_idx];
        Hx_shmem[shared_idx_H] = Hx_src[global_idx];
        Hy_shmem[shared_idx_H] = Hy_src[global_idx];
        Hz_shmem[shared_idx_H] = Hz_src[global_idx];

        // load (y+1, z) stride for E_shmem
        shared_order_E = 1;
        shared_idx_E = shared_x + shared_order_E * SHX;
        global_idx = global_x + (global_y + 1) * Nx_pad + global_z * Nx_pad * Ny;
        Ex_shmem[shared_idx_E] = Ex_src[global_idx];
        Ey_shmem[shared_idx_E] = Ey_src[global_idx];
        Ez_shmem[shared_idx_E] = Ez_src[global_idx];

        // load (y, z+1) stride for E_shmem
        shared_order_E = 2;
        shared_idx_E = shared_x + shared_order_E * SHX;
        global_idx = global_x + global_y * Nx_pad + (global_z + 1) * Nx_pad * Ny;
        Ex_shmem[shared_idx_E] = Ex_src[global_idx];
        Ey_shmem[shared_idx_E] = Ey_src[global_idx];
        Ez_shmem[shared_idx_E] = Ez_src[global_idx];

        // load (y, z-1) stride for H_shmem
        shared_order_H = 0;
        shared_idx_H = shared_x + shared_order_H * SHX;
        global_idx = global_x + global_y * Nx_pad + (global_z - 1) * Nx_pad * Ny;
        Hx_shmem[shared_idx_H] = Hx_src[global_idx];
        Hy_shmem[shared_idx_H] = Hy_src[global_idx];
        Hz_shmem[shared_idx_H] = Hz_src[global_idx];

        // load (y-1, z) stride for H_shmem
        shared_order_H = 1;
        shared_idx_H = shared_x + shared_order_H * SHX;
        global_idx = global_x + (global_y - 1) * Nx_pad + global_z * Nx_pad * Ny;
        Hx_shmem[shared_idx_H] = Hx_src[global_idx];
        Hy_shmem[shared_idx_H] = Hy_src[global_idx];
        Hz_shmem[shared_idx_H] = Hz_src[global_idx];
      }
    }

    // calculation
    int cal_offsetX_E, cal_offsetX_H;
    int cal_boundX_E, cal_boundX_H;
    for(int t=0; t<BLT_DTR; t++) {
      cal_offsetX_E = BLT_DTR - t;
      cal_offsetX_H = cal_offsetX_E - 1;
      cal_boundX_E = SHX - (BLT_DTR - t);
      cal_boundX_H = cal_boundX_E; 
      for(size_t tid=0; tid<block_size; tid++) {
        int local_x = tid;

        // In E_shmem, we store, in order, (y, z), (y+1, z), (y, z+1) stride
        // In H_shmem, we store, in order, (y, z-1), (y-1, z), (y, z) stride

        // update E
        for(int shared_x=local_x+cal_offsetX_E; shared_x<cal_boundX_E; shared_x+=NTX) {
          int shared_order_E = 0; // (y, z) stride
          int shared_order_H = 2; // (y, z) stride
          int shared_idx_E = shared_x + shared_order_E * SHX;
          int shared_idx_H = shared_x + shared_order_H * SHX;
          int global_x = xx_heads_valley[xx] + shared_x - LEFT_PAD; // - LEFT_PAD since constant arrays has not been padded
          int global_idx = global_x + global_y * Nx + global_z * Nx * Ny; // notice that here we are accessing the unpadded constant array

          if(global_x >= 1 && global_x <= Nx-2 && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2) {
            // (y, z-1) for Hx and Hy, (y-1, z) for Hx, Hz
            Ex_shmem[shared_idx_E] = Cax[global_idx] * Ex_shmem[shared_idx_E] + Cbx[global_idx] *
                        ((Hz_shmem[shared_idx_H] - Hz_shmem[shared_idx_H - SHX]) - (Hy_shmem[shared_idx_H] - Hy_shmem[shared_idx_H - 2 * SHX]) - Jx[global_idx] * dx);
            Ey_shmem[shared_idx_E] = Cay[global_idx] * Ey_shmem[shared_idx_E] + Cby[global_idx] *
                      ((Hx_shmem[shared_idx_H] - Hx_shmem[shared_idx_H - 2 * SHX]) - (Hz_shmem[shared_idx_H] - Hz_shmem[shared_idx_H - 1]) - Jy[global_idx] * dx);
            Ez_shmem[shared_idx_E] = Caz[global_idx] * Ez_shmem[shared_idx_E] + Cbz[global_idx] *
                      ((Hy_shmem[shared_idx_H] - Hy_shmem[shared_idx_H - 1]) - (Hx_shmem[shared_idx_H] - Hx_shmem[shared_idx_H - SHX]) - Jz[global_idx] * dx);
        
            
          }
        }
      }
      for(size_t tid=0; tid<block_size; tid++) {
        int local_x = tid;

        // update H
        for(int shared_x=local_x+cal_offsetX_H; shared_x<cal_boundX_H; shared_x+=NTX) {
          int shared_order_E = 0; // (y, z) stride
          int shared_order_H = 2; // (y, z) stride
          int shared_idx_E = shared_x + shared_order_E * SHX;
          int shared_idx_H = shared_x + shared_order_H * SHX;
          int global_x = xx_heads_valley[xx] + shared_x - LEFT_PAD; // - LEFT_PAD since constant arrays has not been padded
          int global_idx = global_x + global_y * Nx + global_z * Nx * Ny; // notice that here we are accessing the unpadded constant array

          if(global_x >= 1 && global_x <= Nx-2 && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2) {
            // (y+1, z) for Ex, Ez, (y, z+1) for Ex, Ey
            Hx_shmem[shared_idx_H] = Dax[global_idx] * Hx_shmem[shared_idx_H] + Dbx[global_idx] *
                        ((Ey_shmem[shared_idx_E + 2 * SHX] - Ey_shmem[shared_idx_E]) - (Ez_shmem[shared_idx_E + SHX] - Ez_shmem[shared_idx_E]) - Mx[global_idx] * dx);
            Hy_shmem[shared_idx_H] = Day[global_idx] * Hy_shmem[shared_idx_H] + Dby[global_idx] *
                      ((Ez_shmem[shared_idx_E + 1] - Ez_shmem[shared_idx_E]) - (Ex_shmem[shared_idx_E + 2 * SHX] - Ex_shmem[shared_idx_E]) - My[global_idx] * dx);
            Hz_shmem[shared_idx_H] = Daz[global_idx] * Hz_shmem[shared_idx_H] + Dbz[global_idx] *
                      ((Ex_shmem[shared_idx_E + SHX] - Ex_shmem[shared_idx_E]) - (Ey_shmem[shared_idx_E + 1] - Ey_shmem[shared_idx_E]) - Mz[global_idx] * dx);
          }
        }
      }
    }

    // store global memory
    int store_offsetE = 1;
    int store_offsetH = 0;
    int store_boundE = SHX - 1;
    int store_boundH = store_boundE;
    for(size_t tid=0; tid<block_size; tid++) {
      int local_x = tid;
      // store E
      for(int shared_x=local_x+store_offsetE; shared_x<store_boundE; shared_x+=NTX) {
        int shared_order_E = 0; // (y, z) stride
        int shared_idx_E = shared_x + shared_order_E * SHX;
        int global_x = xx_heads_valley[xx] + shared_x;
        int global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny;
        if(global_x >= 1 + LEFT_PAD && global_x <= Nx-2 + LEFT_PAD && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2) {
          Ex_dst[global_idx] = Ex_shmem[shared_idx_E];
          Ey_dst[global_idx] = Ey_shmem[shared_idx_E];
          Ez_dst[global_idx] = Ez_shmem[shared_idx_E];
        }
      }
      // store H
      for(int shared_x=local_x+store_offsetH; shared_x<store_boundH; shared_x+=NTX) {
        int shared_order_H = 2; // (y, z) stride
        int shared_idx_H = shared_x + shared_order_H * SHX;
        int global_x = xx_heads_valley[xx] + shared_x;
        int global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny;
        if(global_x >= 1 + LEFT_PAD && global_x <= Nx-2 + LEFT_PAD && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2) {
          Hx_dst[global_idx] = Hx_shmem[shared_idx_H];
          Hy_dst[global_idx] = Hy_shmem[shared_idx_H];
          Hz_dst[global_idx] = Hz_shmem[shared_idx_H];
        }
      }
    }

    // extra store to dst
    int extra_offsetE = BLT_DTR;
    int extra_offsetH = extra_offsetE;
    int extra_boundE = extra_offsetE + NTX + 1;
    int extra_boundH = extra_boundE - 1;
    for(size_t tid=0; tid<block_size; tid++) {
      int local_x = tid;

      // store E
      for(int shared_x=local_x+extra_offsetE; shared_x<extra_boundE; shared_x+=NTX) {
        int global_x = xx_heads_mountain[xx] + shared_x;
        int global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny;
        if(global_x >= 1 + LEFT_PAD && global_x <= Nx-2 + LEFT_PAD && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2) {
          Ex_dst[global_idx] = Ex_src[global_idx];
          Ey_dst[global_idx] = Ey_src[global_idx];
          Ez_dst[global_idx] = Ez_src[global_idx];
        }
      }

      // store H
      for(int shared_x=local_x+extra_offsetH; shared_x<extra_boundH; shared_x+=NTX) {
        int global_x = xx_heads_mountain[xx] + shared_x;
        int global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny;
        if(global_x >= 1 + LEFT_PAD && global_x <= Nx-2 + LEFT_PAD && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2) {
          Hx_dst[global_idx] = Hx_src[global_idx];
          Hy_dst[global_idx] = Hy_src[global_idx];
          Hz_dst[global_idx] = Hz_src[global_idx];
        }
      }
    }


  } 

} 



} // end of namespace gdiamond

#endif































