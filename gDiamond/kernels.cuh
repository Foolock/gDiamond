#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "gdiamond.hpp"

// for diamond tiling 
#define BLX_GPU 8 
#define BLY_GPU 8
#define BLZ_GPU 8
#define BLT_GPU 2 
#define BLX_EH (BLX_GPU + 1)
#define BLY_EH (BLY_GPU + 1)
#define BLZ_EH (BLZ_GPU + 1)

// for parallelogram tiling
#define BLX_GPU_PT 8
#define BLY_GPU_PT 8
#define BLT_GPU_PT 4 
#define BLZ_GPU_PT (BLT_GPU_PT + 1) // ?, not sure 
#define BLX_EH_PT (BLX_GPU_PT + 1)
#define BLY_EH_PT (BLY_GPU_PT + 1)

// for more is less tiling
#define BLX_MIL 8
#define BLY_MIL 8
#define BLZ_MIL 8
#define BLT_MIL 2 
#define BLX_MIL_EH (BLX_MIL + 1)
#define BLY_MIL_EH (BLY_MIL + 1)
#define BLZ_MIL_EH (BLZ_MIL + 1)

// parallelogram tiling for pipeline idea
#define BLX_PT 8
#define BLY_PT 8
#define BLZ_PT 8
#define BLT_PT 3 

// upper bound check
#define BLX_UB 32 
#define BLY_UB 16 
#define BLZ_UB 1 
#define BLT_UB 4

// reimplemented diamond tiling
// for NTX = 32, NTY = 16, shared mem requirement = 40 * 24 * 2 * 4 * 6 = 46080 
#define NTX 4 // number of threads in X dimension
#define NTY 4 // number of threads in Y dimension
#define BLT_DTR 4
#define BLX_DTR (NTX + 2 * (BLT_DTR - 1) + 1) // tile length, mountain bottom
#define BLY_DTR (NTY + 2 * (BLT_DTR - 1) + 1) 
#define SHX (BLX_DTR + 1)
#define SHY (BLY_DTR + 1)

//
// ----------------------------------- dft -----------------------------------
// 
// Update rule for frequency-domain field monitor
__global__ void update_field_FFT_yz(float *E, int i, int Nx, int Ny, int Nz, float *E_real,
    float *E_imag, float *freq_monitors, int num_freqs, float t, float scale_factor)
{
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  unsigned int j = tid % Ny;
  unsigned int k = tid / Ny;

  // Ensure indices are within bounds
  if (j < Ny && k < Nz)
  {
      // 2D index for the yz-plane at given i
      int idx_2D = j + k * Ny;

      // 3D index for the full grid
      int idx_3D = i + j * Nx + k * (Nx * Ny);

      // Update the frequency-domain field for each frequency
      for (int f = 0; f < num_freqs; ++f)
      {
          float omega = 2 * PI * freq_monitors[f];
          E_real[f * Ny * Nz + idx_2D] += E[idx_3D] * cos(omega * t) * scale_factor;
          E_imag[f * Ny * Nz + idx_2D] -= E[idx_3D] * sin(omega * t) * scale_factor;
      }
  }
}

__global__ void update_field_FFT_xy(float *E, int k, int Nx, int Ny, int Nz, float *E_real,
    float *E_imag, float *freq_monitors, int num_freqs, float t, float scale_factor)
{
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  unsigned int i = tid % Nx;
  unsigned int j = tid / Nx;

  // Ensure indices are within bounds
  if (i < Nx && j < Ny)
  {
      // 2D index for the xy-plane at given k 
      int idx_2D = i + j * Nx;

      // 3D index for the full grid
      int idx_3D = i + j * Nx + k * (Nx * Ny);

      // Update the frequency-domain field for each frequency
      for (int f = 0; f < num_freqs; ++f)
      {
          float omega = 2 * PI * freq_monitors[f];

          E_real[f * Nx * Ny + idx_2D] += E[idx_3D] * cos(omega * t) * scale_factor;
          E_imag[f * Nx * Ny + idx_2D] -= E[idx_3D] * sin(omega * t) * scale_factor;
      }
  }
}

//
// ----------------------------------- device function -----------------------------------
// 

__device__ void get_head_tail(size_t BLX, size_t BLT,
                              int *xx_heads, int *xx_tails,
                              size_t xx, size_t t,
                              int mountain_or_valley, // 1 = mountain, 0 = valley
                              int Nx,
                              int *calculate_E, int *calculate_H,
                              int *results) 
{

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
}

__device__ int get_z_planeE(int t, int zz, int Nz) {
  int result = zz - t;
  // return (result >= 0 && result <= Nz - 1)? result : -1; 
  return (result >= 1 && result <= Nz - 2)? result : -1; 
}

__device__ int get_z_planeH(int t, int zz, int Nz) {
  int result = zz - t - 1;
  // return (result >= 0 && result <= Nz - 1)? result : -1;
  return (result >= 1 && result <= Nz - 2)? result : -1;
}

__device__ int get_z_planeE_shmem(int t, int zz, int Nz) {
  int result = zz - t;
  return (result >= 0 && result <= Nz - 1)? result : -1; 
}

__device__ int get_z_planeH_shmem(int t, int zz, int Nz) {
  int result = zz - t - 1;
  return (result >= 0 && result <= Nz - 1)? result : -1;
}



//
// ----------------------------------- 2-D mapping, no pipeline, no diamond tiling ------------------------------------------
//

__global__ void updateE_2Dmap(float * Ex, float * Ey, float * Ez,
                        float * Hx, float * Hy, float * Hz,
                        float * Cax, float * Cbx, float * Cay,
                        float * Cby, float * Caz, float * Cbz,
                        float * Jx, float * Jy, float * Jz,
                        float dx, int Nx, int Ny, int Nz)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= 1 && i < Nx - 1 && j >= 1 && j < Ny - 1)
  {
    // Iterate over z direction
    for (int k = 1; k < Nz - 1; ++k)
    {
      int idx = i + j * Nx + k * (Nx * Ny);

      Ex[idx] = Cax[idx] * Ex[idx] + Cbx[idx] *
                ((Hz[idx] - Hz[idx - Nx]) - (Hy[idx] - Hy[idx - Nx * Ny]) - Jx[idx] * dx);

      Ey[idx] = Cay[idx] * Ey[idx] + Cby[idx] *
                ((Hx[idx] - Hx[idx - Nx * Ny]) - (Hz[idx] - Hz[idx - 1]) - Jy[idx] * dx);

      Ez[idx] = Caz[idx] * Ez[idx] + Cbz[idx] *
                ((Hy[idx] - Hy[idx - 1]) - (Hx[idx] - Hx[idx - Nx]) - Jz[idx] * dx);
    }
  }
}

__global__ void updateH_2Dmap(float * Ex, float * Ey, float * Ez,
                        float * Hx, float * Hy, float * Hz,
                        float * Dax, float * Dbx,
                        float * Day, float * Dby,
                        float * Daz, float * Dbz,
                        float * Mx, float * My, float * Mz,
                        float dx, int Nx, int Ny, int Nz)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= 1 && i < Nx - 1 && j >= 1 && j < Ny - 1)
  {
    // Iterate over z direction
    for (int k = 1; k < Nz - 1; ++k)
    {
      int idx = i + j * Nx + k * (Nx * Ny);

      Hx[idx] = Dax[idx] * Hx[idx] + Dbx[idx] *
                ((Ey[idx + Nx * Ny] - Ey[idx]) - (Ez[idx + Nx] - Ez[idx]) - Mx[idx] * dx);

      Hy[idx] = Day[idx] * Hy[idx] + Dby[idx] *
                ((Ez[idx + 1] - Ez[idx]) - (Ex[idx + Nx * Ny] - Ex[idx]) - My[idx] * dx);

      Hz[idx] = Daz[idx] * Hz[idx] + Dbz[idx] *
                ((Ex[idx + Nx] - Ex[idx]) - (Ey[idx + 1] - Ey[idx]) - Mz[idx] * dx);
    }
  }
}

//
// ----------------------------------- 3-D mapping, has warp underutilization issue ------------------------------------------
//

__global__ void updateE_3Dmap(float * Ex, float * Ey, float * Ez,
                        float * Hx, float * Hy, float * Hz,
                        float * Cax, float * Cbx, float * Cay,
                        float * Cby, float * Caz, float * Cbz,
                        float * Jx, float * Jy, float * Jz,
                        float dx, int Nx, int Ny, int Nz)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int k = blockIdx.z * blockDim.z + threadIdx.z; 

  // Jx source at _source_idx

  if (i >= 1 && i < Nx - 1 && j >= 1 && j < Ny - 1 && k >= 1 && k < Nz - 1)
  {
    int idx = i + j * Nx + k * (Nx * Ny);

    Ex[idx] = Cax[idx] * Ex[idx] + Cbx[idx] *
              ((Hz[idx] - Hz[idx - Nx]) - (Hy[idx] - Hy[idx - Nx * Ny]) - Jx[idx] * dx);

    Ey[idx] = Cay[idx] * Ey[idx] + Cby[idx] *
              ((Hx[idx] - Hx[idx - Nx * Ny]) - (Hz[idx] - Hz[idx - 1]) - Jy[idx] * dx);

    Ez[idx] = Caz[idx] * Ez[idx] + Cbz[idx] *
              ((Hy[idx] - Hy[idx - 1]) - (Hx[idx] - Hx[idx - Nx]) - Jz[idx] * dx);
  }
}

__global__ void updateH_3Dmap(float * Ex, float * Ey, float * Ez,
                        float * Hx, float * Hy, float * Hz,
                        float * Dax, float * Dbx,
                        float * Day, float * Dby,
                        float * Daz, float * Dbz,
                        float * Mx, float * My, float * Mz,
                        float dx, int Nx, int Ny, int Nz)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int k = blockIdx.z * blockDim.z + threadIdx.z; 

  if (i >= 1 && i < Nx - 1 && j >= 1 && j < Ny - 1 && k >= 1 && k < Nz - 1)
  {
    int idx = i + j * Nx + k * (Nx * Ny);

    Hx[idx] = Dax[idx] * Hx[idx] + Dbx[idx] *
              ((Ey[idx + Nx * Ny] - Ey[idx]) - (Ez[idx + Nx] - Ez[idx]) - Mx[idx] * dx);

    Hy[idx] = Day[idx] * Hy[idx] + Dby[idx] *
              ((Ez[idx + 1] - Ez[idx]) - (Ex[idx + Nx * Ny] - Ex[idx]) - My[idx] * dx);

    Hz[idx] = Daz[idx] * Hz[idx] + Dbz[idx] *
              ((Ex[idx + Nx] - Ex[idx]) - (Ey[idx + 1] - Ey[idx]) - Mz[idx] * dx);
  }
}

//
// ----------------------------------- 3-D mapping, fix warp underutilization issue ------------------------------------------
//

/*

  unsigned int i = blockIdx.x % xx;
  unsigned int j = (blockIdx.x % (xx * yy)) / xx;
  unsigned int k = blockIdx.x / (xx * yy);

*/

__global__ void updateE_3Dmap_fix(float * Ex, float * Ey, float * Ez,
                        float * Hx, float * Hy, float * Hz,
                        float * Cax, float * Cbx, float * Cay,
                        float * Cby, float * Caz, float * Cbz,
                        float * Jx, float * Jy, float * Jz,
                        float dx, int Nx, int Ny, int Nz)
{

  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  unsigned int i = tid % Nx;
  unsigned int j = (tid % (Nx * Ny)) / Nx;
  unsigned int k = tid / (Nx * Ny);

  if (i >= 1 && i < Nx - 1 && j >= 1 && j < Ny - 1 && k >= 1 && k < Nz - 1)
  {
    int idx = i + j * Nx + k * (Nx * Ny);

    Ex[idx] = Cax[idx] * Ex[idx] + Cbx[idx] *
              ((Hz[idx] - Hz[idx - Nx]) - (Hy[idx] - Hy[idx - Nx * Ny]) - Jx[idx] * dx);

    Ey[idx] = Cay[idx] * Ey[idx] + Cby[idx] *
              ((Hx[idx] - Hx[idx - Nx * Ny]) - (Hz[idx] - Hz[idx - 1]) - Jy[idx] * dx);

    Ez[idx] = Caz[idx] * Ez[idx] + Cbz[idx] *
              ((Hy[idx] - Hy[idx - 1]) - (Hx[idx] - Hx[idx - Nx]) - Jz[idx] * dx);
  }
}

__global__ void updateH_3Dmap_fix(float * Ex, float * Ey, float * Ez,
                        float * Hx, float * Hy, float * Hz,
                        float * Dax, float * Dbx,
                        float * Day, float * Dby,
                        float * Daz, float * Dbz,
                        float * Mx, float * My, float * Mz,
                        float dx, int Nx, int Ny, int Nz)
{

  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  unsigned int i = tid % Nx;
  unsigned int j = (tid % (Nx * Ny)) / Nx;
  unsigned int k = tid / (Nx * Ny);

  if (i >= 1 && i < Nx - 1 && j >= 1 && j < Ny - 1 && k >= 1 && k < Nz - 1)
  {
    int idx = i + j * Nx + k * (Nx * Ny);

    Hx[idx] = Dax[idx] * Hx[idx] + Dbx[idx] *
              ((Ey[idx + Nx * Ny] - Ey[idx]) - (Ez[idx + Nx] - Ez[idx]) - Mx[idx] * dx);

    Hy[idx] = Day[idx] * Hy[idx] + Dby[idx] *
              ((Ez[idx + 1] - Ez[idx]) - (Ex[idx + Nx * Ny] - Ex[idx]) - My[idx] * dx);

    Hz[idx] = Daz[idx] * Hz[idx] + Dbz[idx] *
              ((Ex[idx + Nx] - Ex[idx]) - (Ey[idx + 1] - Ey[idx]) - Mz[idx] * dx);
  }
}

__global__ void updateEH_3Dmap_fq(float * Ex, float * Ey, float * Ez,
                                  float * Hx, float * Hy, float * Hz,
                                  float * Hx_temp, float * Hy_temp, float * Hz_temp,
                                  float * Cax, float * Cbx, float * Cay,
                                  float * Cby, float * Caz, float * Cbz,
                                  float * Jx, float * Jy, float * Jz,
                                  float * Dax, float * Dbx,
                                  float * Day, float * Dby,
                                  float * Daz, float * Dbz,
                                  float * Mx, float * My, float * Mz,
                                  float dx, int Nx, int Ny, int Nz) 
{

  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  unsigned int i = tid % Nx;
  unsigned int j = (tid % (Nx * Ny)) / Nx;
  unsigned int k = tid / (Nx * Ny);

  if (i >= 1 && i < Nx - 1 && j >= 1 && j < Ny - 1 && k >= 1 && k < Nz - 1)
  {
    int idx = i + j * Nx + k * (Nx * Ny);

    // E are registers
    // new Ex[idx]
    float Ex_reg1 = Cax[idx] * Ex[idx] + Cbx[idx] *
              ((Hz_temp[idx] - Hz_temp[idx - Nx]) - (Hy_temp[idx] - Hy_temp[idx - Nx * Ny]) - Jx[idx] * dx);

    // new Ex[idx + Nx]
    float Ex_reg2 = Cax[idx + Nx] * Ex[idx + Nx] + Cbx[idx + Nx] *
              ((Hz_temp[idx + Nx] - Hz_temp[idx]) - (Hy_temp[idx + Nx] - Hy_temp[idx + Nx - Nx * Ny]) - Jx[idx + Nx] * dx);

    // new Ex[idx + Nx * Ny]
    float Ex_reg3 = Cax[idx + Nx * Ny] * Ex[idx + Nx * Ny] + Cbx[idx + Nx * Ny] *
              ((Hz_temp[idx + Nx * Ny] - Hz_temp[idx + Nx * Ny - Nx]) - (Hy_temp[idx + Nx * Ny] - Hy_temp[idx]) - Jx[idx + Nx * Ny] * dx);

    // new Ey[idx]
    float Ey_reg1 = Cay[idx] * Ey[idx] + Cby[idx] *
              ((Hx_temp[idx] - Hx_temp[idx - Nx * Ny]) - (Hz_temp[idx] - Hz_temp[idx - 1]) - Jy[idx] * dx);

    // new Ey[idx + 1]
    float Ey_reg2 = Cay[idx + 1] * Ey[idx + 1] + Cby[idx + 1] *
              ((Hx_temp[idx + 1] - Hx_temp[idx + 1 - Nx * Ny]) - (Hz_temp[idx + 1] - Hz_temp[idx]) - Jy[idx + 1] * dx);

    // new Ey[idx + Nx * Ny]
    float Ey_reg3 = Cay[idx + Nx * Ny] * Ey[idx + Nx * Ny] + Cby[idx + Nx * Ny] *
              ((Hx_temp[idx + Nx * Ny] - Hx_temp[idx]) - (Hz_temp[idx + Nx * Ny] - Hz_temp[idx + Nx * Ny - 1]) - Jy[idx + Nx * Ny] * dx);

    // new Ez[idx]
    float Ez_reg1 = Caz[idx] * Ez[idx] + Cbz[idx] *
              ((Hy_temp[idx] - Hy_temp[idx - 1]) - (Hx_temp[idx] - Hx_temp[idx - Nx]) - Jz[idx] * dx);

    // new Ez[idx + Nx]
    float Ez_reg2 = Caz[idx + Nx] * Ez[idx + Nx] + Cbz[idx + Nx] *
              ((Hy_temp[idx + Nx] - Hy_temp[idx + Nx - 1]) - (Hx_temp[idx + Nx] - Hx_temp[idx]) - Jz[idx + Nx] * dx);

    // new Ez[idx + 1]
    float Ez_reg3 = Caz[idx + 1] * Ez[idx + 1] + Cbz[idx + 1] *
              ((Hy_temp[idx + 1] - Hy_temp[idx]) - (Hx_temp[idx + 1] - Hx_temp[idx + 1 - Nx]) - Jz[idx + 1] * dx);

    // H are from global memory
    Hx[idx] = Dax[idx] * Hx[idx] + Dbx[idx] *
              ((Ey_reg3 - Ey_reg1) - (Ez_reg2 - Ez_reg1) - Mx[idx] * dx);

    Hy[idx] = Day[idx] * Hy[idx] + Dby[idx] *
              ((Ez_reg3 - Ez_reg1) - (Ex_reg3 - Ex_reg1) - My[idx] * dx);

    Hz[idx] = Daz[idx] * Hz[idx] + Dbz[idx] *
              ((Ex_reg2 - Ex_reg1) - (Ey_reg2 - Ey_reg1) - Mz[idx] * dx);

    // we will need to store Ex[idx], Ex[idx + Nx], Ex[idx + Nx * Ny]
    //                       Ey[idx], Ey[idx + 1], Ey[idx + Nx * Ny]
    //                       Ez[idx], Ez[idx + Nx], Ez[idx + 1] 
    // back to global memory
    Ex[idx] = Ex_reg1;
    Ey[idx] = Ey_reg1;
    Ez[idx] = Ez_reg1;
  }

}

//
// ----------------------------------- diamond tiling kernel ------------------------------------------
//

__global__ void updateEH_phase(float *Ex, float *Ey, float *Ez,
                               float *Hx, float *Hy, float *Hz,
                               float *Cax, float *Cbx,
                               float *Cay, float *Cby,
                               float *Caz, float *Cbz,
                               float *Dax, float *Dbx,
                               float *Day, float *Dby,
                               float *Daz, float *Dbz,
                               float *Jx, float *Jy, float *Jz,
                               float *Mx, float *My, float *Mz,
                               float dx, 
                               int Nx, int Ny, int Nz,
                               int xx_num, int yy_num, int zz_num, // number of tiles in each dimensions
                               int *xx_heads, 
                               int *yy_heads, 
                               int *zz_heads 
                               ) 
{
  // first we map each (xx, yy, zz) to a block
  int xx = blockIdx.x % xx_num;
  int yy = (blockIdx.x % (xx_num * yy_num)) / xx_num;
  int zz = blockIdx.x / (xx_num * yy_num);

  // for constant arrays, I will need a shared memory size of BLX * BLY * BLZ
  __shared__ float Cax_shmem[BLX_GPU * BLY_GPU * BLZ_GPU];
  __shared__ float Cay_shmem[BLX_GPU * BLY_GPU * BLZ_GPU];
  __shared__ float Caz_shmem[BLX_GPU * BLY_GPU * BLZ_GPU];
  __shared__ float Cbx_shmem[BLX_GPU * BLY_GPU * BLZ_GPU];
  __shared__ float Cby_shmem[BLX_GPU * BLY_GPU * BLZ_GPU];
  __shared__ float Cbz_shmem[BLX_GPU * BLY_GPU * BLZ_GPU];
  __shared__ float Dax_shmem[BLX_GPU * BLY_GPU * BLZ_GPU];
  __shared__ float Day_shmem[BLX_GPU * BLY_GPU * BLZ_GPU];
  __shared__ float Daz_shmem[BLX_GPU * BLY_GPU * BLZ_GPU];
  __shared__ float Dbx_shmem[BLX_GPU * BLY_GPU * BLZ_GPU];
  __shared__ float Dby_shmem[BLX_GPU * BLY_GPU * BLZ_GPU];
  __shared__ float Dbz_shmem[BLX_GPU * BLY_GPU * BLZ_GPU];

  // leave J, M in global memory to save space for E, H
  // __shared__ float Jx_shmem[BLX_GPU * BLY_GPU * BLZ_GPU];
  // __shared__ float Jy_shmem[BLX_GPU * BLY_GPU * BLZ_GPU];
  // __shared__ float Jz_shmem[BLX_GPU * BLY_GPU * BLZ_GPU];
  // __shared__ float Mx_shmem[BLX_GPU * BLY_GPU * BLZ_GPU];
  // __shared__ float My_shmem[BLX_GPU * BLY_GPU * BLZ_GPU];
  // __shared__ float Mz_shmem[BLX_GPU * BLY_GPU * BLZ_GPU];

  // E, H array needs extra HALO space since stencil
  __shared__ float Ex_shmem[BLX_EH * BLY_EH * BLZ_EH];
  __shared__ float Ey_shmem[BLX_EH * BLY_EH * BLZ_EH];
  __shared__ float Ez_shmem[BLX_EH * BLY_EH * BLZ_EH];
  __shared__ float Hx_shmem[BLX_EH * BLY_EH * BLZ_EH];
  __shared__ float Hy_shmem[BLX_EH * BLY_EH * BLZ_EH];
  __shared__ float Hz_shmem[BLX_EH * BLY_EH * BLZ_EH];

  // map each thread in the block to a global index
  int local_x = threadIdx.x % BLX_GPU;                     // X coordinate within the tile
  int local_y = (threadIdx.x / BLX_GPU) % BLY_GPU;     // Y coordinate within the tile
  int local_z = threadIdx.x / (BLX_GPU * BLY_GPU);     // Z coordinate within the tile

  int global_x = xx_heads[xx] + local_x; // Global X coordinate
  int global_y = yy_heads[yy] + local_y; // Global Y coordinate
  int global_z = zz_heads[zz] + local_z; // Global Z coordinate

  int global_idx = global_x + global_y * Nx + global_z * Nx * Ny;
  int local_idx = local_x + local_y * BLX_GPU + local_z * BLX_GPU * BLY_GPU;

  // load constant
  if(global_x >= 0 && global_x < Nx && global_y >= 0 && global_y < Ny && global_z >= 0 && global_z < Nz) {
    Cax_shmem[local_idx] = Cax[global_idx];  
    Cay_shmem[local_idx] = Cay[global_idx];  
    Caz_shmem[local_idx] = Caz[global_idx];  
    Cbx_shmem[local_idx] = Cbx[global_idx];  
    Cby_shmem[local_idx] = Cby[global_idx];  
    Cbz_shmem[local_idx] = Cbz[global_idx];  
    Dax_shmem[local_idx] = Dax[global_idx];  
    Day_shmem[local_idx] = Day[global_idx];  
    Daz_shmem[local_idx] = Daz[global_idx];  
    Dbx_shmem[local_idx] = Dbx[global_idx];  
    Dby_shmem[local_idx] = Dby[global_idx];  
    Dbz_shmem[local_idx] = Dbz[global_idx];  
  }

  // load H, stencil pattern x-1, y-1, z-1
  int shared_H_x = local_x + 1;  
  int shared_H_y = local_y + 1;
  int shared_H_z = local_z + 1;

  int shared_H_idx = shared_H_x + shared_H_y * BLX_EH + shared_H_z * BLX_EH * BLY_EH;

  if(global_x >= 0 && global_x < Nx && global_y >= 0 && global_y < Ny && global_z >= 0 && global_z < Nz) {
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

  __syncthreads();

  if(global_x >= 1 && global_x <= Nx-2 && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2) {
    for(int t=0; t<BLT_GPU; t++) { // we will do BLT_GPU time steps in one kernel 
      int g_idx = global_x + global_y * Nx + global_z * Nx * Ny; // global idx
      int l_idx = local_x + local_y * BLX_GPU + local_z * BLX_GPU * BLY_GPU; // local idx in each block, also shared memory idx for constant 
      int s_H_idx = shared_H_x + shared_H_y * BLX_EH + shared_H_z * BLX_EH * BLY_EH; // shared memory idx for H
      int s_E_idx = shared_E_x + shared_E_y * BLX_EH + shared_E_z * BLX_EH * BLY_EH; // shared memory idx for E

      // update E
      Ex_shmem[s_E_idx] = Cax_shmem[l_idx] * Ex_shmem[s_E_idx] + Cbx_shmem[l_idx] *
                ((Hz_shmem[s_H_idx] - Hz_shmem[s_H_idx - BLX_EH]) - (Hy_shmem[s_H_idx] - Hy_shmem[s_H_idx - BLX_EH * BLY_EH]) - Jx[g_idx] * dx);
      Ey_shmem[s_E_idx] = Cay_shmem[l_idx] * Ey_shmem[s_E_idx] + Cby_shmem[l_idx] *
                ((Hx_shmem[s_H_idx] - Hx_shmem[s_H_idx - BLX_EH * BLY_EH]) - (Hz_shmem[s_H_idx] - Hz_shmem[s_H_idx - 1]) - Jy[g_idx] * dx);
      Ez_shmem[s_E_idx] = Caz_shmem[l_idx] * Ez_shmem[s_E_idx] + Cbz_shmem[l_idx] *
                ((Hy_shmem[s_H_idx] - Hy_shmem[s_H_idx - 1]) - (Hx_shmem[s_H_idx] - Hx_shmem[s_H_idx - BLX_EH]) - Jz[g_idx] * dx);
                
      __syncthreads();

      // update H
      Hx_shmem[s_H_idx] = Dax_shmem[l_idx] * Hx_shmem[s_H_idx] + Dbx_shmem[l_idx] *
                ((Ey_shmem[s_E_idx + BLX_EH * BLY_EH] - Ey_shmem[s_E_idx]) - (Ez_shmem[s_E_idx + BLX_EH] - Ez_shmem[s_E_idx]) - Mx[g_idx] * dx);
      Hy_shmem[s_H_idx] = Day_shmem[l_idx] * Hy_shmem[s_H_idx] + Dby_shmem[l_idx] *
                ((Ez_shmem[s_E_idx + 1] - Ez_shmem[s_E_idx]) - (Ex_shmem[s_E_idx + BLX_EH * BLY_EH] - Ex_shmem[s_E_idx]) - My[g_idx] * dx);
      Hz_shmem[s_H_idx] = Daz_shmem[l_idx] * Hz_shmem[s_H_idx] + Dbz_shmem[l_idx] *
                ((Ex_shmem[s_E_idx + BLX_EH] - Ex_shmem[s_E_idx]) - (Ey_shmem[s_E_idx + 1] - Ey_shmem[s_E_idx]) - Mz[g_idx] * dx);
                
      __syncthreads();
    }
  }

  // store E, H to global memory, no HALO needed
  if(global_x >= 0 && global_x < Nx && global_y >= 0 && global_y < Ny && global_z >= 0 && global_z < Nz) {
     Ex[global_idx] = Ex_shmem[shared_E_idx]; 
     Ey[global_idx] = Ey_shmem[shared_E_idx];
     Ez[global_idx] = Ez_shmem[shared_E_idx];
     Hx[global_idx] = Hx_shmem[shared_H_idx]; 
     Hy[global_idx] = Hy_shmem[shared_H_idx];
     Hz[global_idx] = Hz_shmem[shared_H_idx];
  }
}

__global__ void updateEH_phase_EH_shared_only(float *Ex, float *Ey, float *Ez,
                               float *Hx, float *Hy, float *Hz,
                               float *Cax, float *Cbx,
                               float *Cay, float *Cby,
                               float *Caz, float *Cbz,
                               float *Dax, float *Dbx,
                               float *Day, float *Dby,
                               float *Daz, float *Dbz,
                               float *Jx, float *Jy, float *Jz,
                               float *Mx, float *My, float *Mz,
                               float dx, 
                               int Nx, int Ny, int Nz,
                               int xx_num, int yy_num, int zz_num, // number of tiles in each dimensions
                               int *xx_heads, 
                               int *yy_heads, 
                               int *zz_heads, 
                               int *xx_tails, 
                               int *yy_tails, 
                               int *zz_tails 
                               ) 
{
  // first we map each (xx, yy, zz) to a block
  int xx = blockIdx.x % xx_num;
  int yy = (blockIdx.x % (xx_num * yy_num)) / xx_num;
  int zz = blockIdx.x / (xx_num * yy_num);

  // E, H array needs extra HALO space since stencil
  __shared__ float Ex_shmem[BLX_EH * BLY_EH * BLZ_EH];
  __shared__ float Ey_shmem[BLX_EH * BLY_EH * BLZ_EH];
  __shared__ float Ez_shmem[BLX_EH * BLY_EH * BLZ_EH];
  __shared__ float Hx_shmem[BLX_EH * BLY_EH * BLZ_EH];
  __shared__ float Hy_shmem[BLX_EH * BLY_EH * BLZ_EH];
  __shared__ float Hz_shmem[BLX_EH * BLY_EH * BLZ_EH];

  // map each thread in the block to a global index
  int local_x = threadIdx.x % BLX_GPU;                     // X coordinate within the tile
  int local_y = (threadIdx.x / BLX_GPU) % BLY_GPU;     // Y coordinate within the tile
  int local_z = threadIdx.x / (BLX_GPU * BLY_GPU);     // Z coordinate within the tile

  int global_x = xx_heads[xx] + local_x; // Global X coordinate
  int global_y = yy_heads[yy] + local_y; // Global Y coordinate
  int global_z = zz_heads[zz] + local_z; // Global Z coordinate

  int global_idx = global_x + global_y * Nx + global_z * Nx * Ny;

  // load H, stencil pattern x-1, y-1, z-1
  int shared_H_x = local_x + 1;  
  int shared_H_y = local_y + 1;
  int shared_H_z = local_z + 1;

  int shared_H_idx = shared_H_x + shared_H_y * BLX_EH + shared_H_z * BLX_EH * BLY_EH;

  if(global_x >= 0 && global_x < Nx && global_y >= 0 && global_y < Ny && global_z >= 0 && global_z < Nz &&
     local_x >= xx_heads[xx] && local_x <= xx_tails[xx] && 
     local_y >= yy_heads[yy] && local_y <= yy_tails[yy] && 
     local_z >= zz_heads[zz] && local_z <= zz_tails[zz]) {
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

  // if(global_x >= 0 && global_x < Nx && global_y >= 0 && global_y < Ny && global_z >= 0 && global_z < Nz) {
  if(global_x >= 0 && global_x < Nx && global_y >= 0 && global_y < Ny && global_z >= 0 && global_z < Nz &&
     local_x >= xx_heads[xx] && local_x <= xx_tails[xx] && 
     local_y >= yy_heads[yy] && local_y <= yy_tails[yy] && 
     local_z >= zz_heads[zz] && local_z <= zz_tails[zz]) {

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

  __syncthreads();

  // if(global_x >= 1 && global_x <= Nx-2 && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2) {
  if(global_x >= 1 && global_x <= Nx-2 && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2 &&
     local_x >= xx_heads[xx] && local_x <= xx_tails[xx] && 
     local_y >= yy_heads[yy] && local_y <= yy_tails[yy] && 
     local_z >= zz_heads[zz] && local_z <= zz_tails[zz]) {
    for(int t=0; t<BLT_GPU; t++) { // we will do BLT_GPU time steps in one kernel 
      int g_idx = global_x + global_y * Nx + global_z * Nx * Ny; // global idx
      int s_H_idx = shared_H_x + shared_H_y * BLX_EH + shared_H_z * BLX_EH * BLY_EH; // shared memory idx for H
      int s_E_idx = shared_E_x + shared_E_y * BLX_EH + shared_E_z * BLX_EH * BLY_EH; // shared memory idx for E

      // update E
      Ex_shmem[s_E_idx] = Cax[g_idx] * Ex_shmem[s_E_idx] + Cbx[g_idx] *
                ((Hz_shmem[s_H_idx] - Hz_shmem[s_H_idx - BLX_EH]) - (Hy_shmem[s_H_idx] - Hy_shmem[s_H_idx - BLX_EH * BLY_EH]) - Jx[g_idx] * dx);
      Ey_shmem[s_E_idx] = Cay[g_idx] * Ey_shmem[s_E_idx] + Cby[g_idx] *
                ((Hx_shmem[s_H_idx] - Hx_shmem[s_H_idx - BLX_EH * BLY_EH]) - (Hz_shmem[s_H_idx] - Hz_shmem[s_H_idx - 1]) - Jy[g_idx] * dx);
      Ez_shmem[s_E_idx] = Caz[g_idx] * Ez_shmem[s_E_idx] + Cbz[g_idx] *
                ((Hy_shmem[s_H_idx] - Hy_shmem[s_H_idx - 1]) - (Hx_shmem[s_H_idx] - Hx_shmem[s_H_idx - BLX_EH]) - Jz[g_idx] * dx);

      __syncthreads();

      // update H
      Hx_shmem[s_H_idx] = Dax[g_idx] * Hx_shmem[s_H_idx] + Dbx[g_idx] *
                ((Ey_shmem[s_E_idx + BLX_EH * BLY_EH] - Ey_shmem[s_E_idx]) - (Ez_shmem[s_E_idx + BLX_EH] - Ez_shmem[s_E_idx]) - Mx[g_idx] * dx);
      Hy_shmem[s_H_idx] = Day[g_idx] * Hy_shmem[s_H_idx] + Dby[g_idx] *
                ((Ez_shmem[s_E_idx + 1] - Ez_shmem[s_E_idx]) - (Ex_shmem[s_E_idx + BLX_EH * BLY_EH] - Ex_shmem[s_E_idx]) - My[g_idx] * dx);
      Hz_shmem[s_H_idx] = Daz[g_idx] * Hz_shmem[s_H_idx] + Dbz[g_idx] *
                ((Ex_shmem[s_E_idx + BLX_EH] - Ex_shmem[s_E_idx]) - (Ey_shmem[s_E_idx + 1] - Ey_shmem[s_E_idx]) - Mz[g_idx] * dx);
                
      __syncthreads();
    }
  }

  // store E, H to global memory, no HALO needed
  // if(global_x >= 0 && global_x < Nx && global_y >= 0 && global_y < Ny && global_z >= 0 && global_z < Nz) {
  if(global_x >= 1 && global_x <= Nx-2 && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2 &&
     local_x >= xx_heads[xx] && local_x <= xx_tails[xx] && 
     local_y >= yy_heads[yy] && local_y <= yy_tails[yy] && 
     local_z >= zz_heads[zz] && local_z <= zz_tails[zz]) {
     Ex[global_idx] = Ex_shmem[shared_E_idx]; 
     Ey[global_idx] = Ey_shmem[shared_E_idx];
     Ez[global_idx] = Ez_shmem[shared_E_idx];
     Hx[global_idx] = Hx_shmem[shared_H_idx]; 
     Hy[global_idx] = Hy_shmem[shared_H_idx];
     Hz[global_idx] = Hz_shmem[shared_H_idx];
  }
}

__global__ void updateEH_phase_E_only(float *Ex, float *Ey, float *Ez,
                               float *Hx, float *Hy, float *Hz,
                               float *Cax, float *Cbx,
                               float *Cay, float *Cby,
                               float *Caz, float *Cbz,
                               float *Dax, float *Dbx,
                               float *Day, float *Dby,
                               float *Daz, float *Dbz,
                               float *Jx, float *Jy, float *Jz,
                               float *Mx, float *My, float *Mz,
                               float dx, 
                               int Nx, int Ny, int Nz,
                               int xx_num, int yy_num, int zz_num, // number of tiles in each dimensions
                               int *xx_heads, 
                               int *yy_heads, 
                               int *zz_heads 
                               ) 
{
  // first we map each (xx, yy, zz) to a block
  int xx = blockIdx.x % xx_num;
  int yy = (blockIdx.x % (xx_num * yy_num)) / xx_num;
  int zz = blockIdx.x / (xx_num * yy_num);

  // map each thread in the block to a global index
  int local_x = threadIdx.x % BLX_GPU;                     // X coordinate within the tile
  int local_y = (threadIdx.x / BLX_GPU) % BLY_GPU;     // Y coordinate within the tile
  int local_z = threadIdx.x / (BLX_GPU * BLY_GPU);     // Z coordinate within the tile

  int global_x = xx_heads[xx] + local_x; // Global X coordinate
  int global_y = yy_heads[yy] + local_y; // Global Y coordinate
  int global_z = zz_heads[zz] + local_z; // Global Z coordinate

  if(global_x >= 1 && global_x <= Nx-2 && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2) {
    for(int t=0; t<1; t++) { // we will do 1 time steps in one kernel, just to verify 
      int g_idx = global_x + global_y * Nx + global_z * Nx * Ny; // global idx

      // update E
      Ex[g_idx] = Cax[g_idx] * Ex[g_idx] + Cbx[g_idx] *
                ((Hz[g_idx] - Hz[g_idx - Nx]) - (Hy[g_idx] - Hy[g_idx - Nx * Ny]) - Jx[g_idx] * dx);
      Ey[g_idx] = Cay[g_idx] * Ey[g_idx] + Cby[g_idx] *
                ((Hx[g_idx] - Hx[g_idx - Nx * Ny]) - (Hz[g_idx] - Hz[g_idx - 1]) - Jy[g_idx] * dx);
      Ez[g_idx] = Caz[g_idx] * Ez[g_idx] + Cbz[g_idx] *
                ((Hy[g_idx] - Hy[g_idx - 1]) - (Hx[g_idx] - Hx[g_idx - Nx]) - Jz[g_idx] * dx);
    }
  }
}

__global__ void updateEH_phase_H_only(float *Ex, float *Ey, float *Ez,
                               float *Hx, float *Hy, float *Hz,
                               float *Cax, float *Cbx,
                               float *Cay, float *Cby,
                               float *Caz, float *Cbz,
                               float *Dax, float *Dbx,
                               float *Day, float *Dby,
                               float *Daz, float *Dbz,
                               float *Jx, float *Jy, float *Jz,
                               float *Mx, float *My, float *Mz,
                               float dx, 
                               int Nx, int Ny, int Nz,
                               int xx_num, int yy_num, int zz_num, // number of tiles in each dimensions
                               int *xx_heads, 
                               int *yy_heads, 
                               int *zz_heads 
                               ) 
{
  // first we map each (xx, yy, zz) to a block
  int xx = blockIdx.x % xx_num;
  int yy = (blockIdx.x % (xx_num * yy_num)) / xx_num;
  int zz = blockIdx.x / (xx_num * yy_num);

  // map each thread in the block to a global index
  int local_x = threadIdx.x % BLX_GPU;                     // X coordinate within the tile
  int local_y = (threadIdx.x / BLX_GPU) % BLY_GPU;     // Y coordinate within the tile
  int local_z = threadIdx.x / (BLX_GPU * BLY_GPU);     // Z coordinate within the tile

  int global_x = xx_heads[xx] + local_x; // Global X coordinate
  int global_y = yy_heads[yy] + local_y; // Global Y coordinate
  int global_z = zz_heads[zz] + local_z; // Global Z coordinate

  if(global_x >= 1 && global_x <= Nx-2 && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2) {
    for(int t=0; t<1; t++) { // we will do 1 time steps in one kernel, just to verify 
      int g_idx = global_x + global_y * Nx + global_z * Nx * Ny; // global idx

      // update H
      Hx[g_idx] = Dax[g_idx] * Hx[g_idx] + Dbx[g_idx] *
                ((Ey[g_idx + Nx * Ny] - Ey[g_idx]) - (Ez[g_idx + Nx] - Ez[g_idx]) - Mx[g_idx] * dx);
      Hy[g_idx] = Day[g_idx] * Hy[g_idx] + Dby[g_idx] *
                ((Ez[g_idx + 1] - Ez[g_idx]) - (Ex[g_idx + Nx * Ny] - Ex[g_idx]) - My[g_idx] * dx);
      Hz[g_idx] = Daz[g_idx] * Hz[g_idx] + Dbz[g_idx] *
                ((Ex[g_idx + Nx] - Ex[g_idx]) - (Ey[g_idx + 1] - Ey[g_idx]) - Mz[g_idx] * dx);
    }
  }
}

__global__ void updateEH_phase_global_mem(float *Ex, float *Ey, float *Ez,
                                          float *Hx, float *Hy, float *Hz,
                                          float *Cax, float *Cbx,
                                          float *Cay, float *Cby,
                                          float *Caz, float *Cbz,
                                          float *Dax, float *Dbx,
                                          float *Day, float *Dby,
                                          float *Daz, float *Dbz,
                                          float *Jx, float *Jy, float *Jz,
                                          float *Mx, float *My, float *Mz,
                                          float dx, 
                                          int Nx, int Ny, int Nz,
                                          int xx_num, int yy_num, int zz_num, 
                                          int *xx_heads, 
                                          int *yy_heads, 
                                          int *zz_heads,
                                          int *xx_tails, 
                                          int *yy_tails, 
                                          int *zz_tails,
                                          int m_or_v_X, int m_or_v_Y, int m_or_v_Z,
                                          size_t block_size,
                                          size_t grid_size) 
{
  // first we map each (xx, yy, zz) to a block
  int xx = blockIdx.x % xx_num;
  int yy = (blockIdx.x % (xx_num * yy_num)) / xx_num;
  int zz = blockIdx.x / (xx_num * yy_num);

  // map each thread in the block to a global index
  int tid = threadIdx.x;
  int local_x = tid % BLX_GPU;                     // X coordinate within the tile
  int local_y = (tid / BLX_GPU) % BLY_GPU;     // Y coordinate within the tile
  int local_z = tid / (BLX_GPU * BLY_GPU);     // Z coordinate within the tile

  __shared__ int indices_X[4];
  __shared__ int indices_Y[4];
  __shared__ int indices_Z[4];

  for(size_t t=0; t<BLT_GPU; t++) {

    int calculate_Ex = 1; // calculate this E tile or not
    int calculate_Hx = 1; // calculate this H tile or not
    int calculate_Ey = 1; 
    int calculate_Hy = 1; 
    int calculate_Ez = 1; 
    int calculate_Hz = 1;

    // first find the range of tile 
    if(tid == 0) { // 1st thread in warp 0 
      get_head_tail(BLX_GPU, BLT_GPU,
                    xx_heads, xx_tails,
                    xx, t,
                    m_or_v_X, // 1 = mountain, 0 = valley
                    Nx,
                    &calculate_Ex, &calculate_Hx,
                    indices_X);
    }
    if(tid == 32) { // 1st thread in warp 1 
      get_head_tail(BLY_GPU, BLT_GPU,
                    yy_heads, yy_tails,
                    yy, t,
                    m_or_v_Y, // 1 = mountain, 0 = valley
                    Ny,
                    &calculate_Ey, &calculate_Hy,
                    indices_Y);
    }
    if(tid == 64) { // 1st thread in warp 2
      get_head_tail(BLZ_GPU, BLT_GPU,
                    zz_heads, zz_tails,
                    zz, t,
                    m_or_v_Z, // 1 = mountain, 0 = valley
                    Nz,
                    &calculate_Ez, &calculate_Hz,
                    indices_Z);
    }
    __syncthreads();

    // update E
    if(calculate_Ex & calculate_Ey & calculate_Ez) {
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

    __syncthreads();

    // update H 
    if(calculate_Hx & calculate_Hy & calculate_Hz) {
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

    __syncthreads();
  }
} 
  
__global__ void updateEH_phase_shmem_EH(float *Ex, float *Ey, float *Ez,
                                        float *Hx, float *Hy, float *Hz,
                                        float *Cax, float *Cbx,
                                        float *Cay, float *Cby,
                                        float *Caz, float *Cbz,
                                        float *Dax, float *Dbx,
                                        float *Day, float *Dby,
                                        float *Daz, float *Dbz,
                                        float *Jx, float *Jy, float *Jz,
                                        float *Mx, float *My, float *Mz,
                                        float dx, 
                                        int Nx, int Ny, int Nz,
                                        int xx_num, int yy_num, int zz_num, 
                                        int *xx_heads, 
                                        int *yy_heads, 
                                        int *zz_heads,
                                        int *xx_tails, 
                                        int *yy_tails, 
                                        int *zz_tails,
                                        int m_or_v_X, int m_or_v_Y, int m_or_v_Z,
                                        size_t block_size,
                                        size_t grid_size) 
{
  // first we map each (xx, yy, zz) to a block
  int xx = blockIdx.x % xx_num;
  int yy = (blockIdx.x % (xx_num * yy_num)) / xx_num;
  int zz = blockIdx.x / (xx_num * yy_num);

  // map each thread in the block to a global index
  int tid = threadIdx.x;
  int local_x = tid % BLX_GPU;                     // X coordinate within the tile
  int local_y = (tid / BLX_GPU) % BLY_GPU;     // Y coordinate within the tile
  int local_z = tid / (BLX_GPU * BLY_GPU);     // Z coordinate within the tile

  // E, H array needs extra HALO space since stencil
  __shared__ float Ex_shmem[BLX_EH * BLY_EH * BLZ_EH];
  __shared__ float Ey_shmem[BLX_EH * BLY_EH * BLZ_EH];
  __shared__ float Ez_shmem[BLX_EH * BLY_EH * BLZ_EH];
  __shared__ float Hx_shmem[BLX_EH * BLY_EH * BLZ_EH];
  __shared__ float Hy_shmem[BLX_EH * BLY_EH * BLZ_EH];
  __shared__ float Hz_shmem[BLX_EH * BLY_EH * BLZ_EH];

  int global_x = xx_heads[xx] + local_x; // Global X coordinate
  int global_y = yy_heads[yy] + local_y; // Global Y coordinate
  int global_z = zz_heads[zz] + local_z; // Global Z coordinate
  int global_idx = global_x + global_y * Nx + global_z * Nx * Ny;

  // load H, stencil pattern x-1, y-1, z-1
  int shared_H_x = local_x + 1;
  int shared_H_y = local_y + 1;
  int shared_H_z = local_z + 1;
  int shared_H_idx = shared_H_x + shared_H_y * BLX_EH + shared_H_z * BLX_EH * BLY_EH;

  if(global_x >= 0 && global_x < Nx && global_y >= 0 && global_y < Ny && global_z >= 0 && global_z < Nz) {
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

  __syncthreads();

  __shared__ int indices_X[4];
  __shared__ int indices_Y[4];
  __shared__ int indices_Z[4];

  for(size_t t=0; t<BLT_GPU; t++) {

    int calculate_Ex = 1; // calculate this E tile or not
    int calculate_Hx = 1; // calculate this H tile or not
    int calculate_Ey = 1; 
    int calculate_Hy = 1; 
    int calculate_Ez = 1; 
    int calculate_Hz = 1;

    // first find the range of tile 
    if(tid == 0) { // 1st thread in warp 0 
      get_head_tail(BLX_GPU, BLT_GPU,
                    xx_heads, xx_tails,
                    xx, t,
                    m_or_v_X, // 1 = mountain, 0 = valley
                    Nx,
                    &calculate_Ex, &calculate_Hx,
                    indices_X);
    }
    if(tid == 32) { // 1st thread in warp 1 
      get_head_tail(BLY_GPU, BLT_GPU,
                    yy_heads, yy_tails,
                    yy, t,
                    m_or_v_Y, // 1 = mountain, 0 = valley
                    Ny,
                    &calculate_Ey, &calculate_Hy,
                    indices_Y);
    }
    if(tid == 64) { // 1st thread in warp 2
      get_head_tail(BLZ_GPU, BLT_GPU,
                    zz_heads, zz_tails,
                    zz, t,
                    m_or_v_Z, // 1 = mountain, 0 = valley
                    Nz,
                    &calculate_Ez, &calculate_Hz,
                    indices_Z);
    }
    __syncthreads();

    // update E
    if(calculate_Ex & calculate_Ey & calculate_Ez) {
      // Ehead is offset
      int g_x = indices_X[0] + local_x; // Global X coordinate
      int g_y = indices_Y[0] + local_y; // Global Y coordinate
      int g_z = indices_Z[0] + local_z; // Global Z coordinate

      if(g_x >= 1 && g_x <= Nx-2 && g_y >= 1 && g_y <= Ny-2 && g_z >= 1 && g_z <= Nz-2 &&
        g_x <= indices_X[1] &&
        g_y <= indices_Y[1] &&
        g_z <= indices_Z[1]) {

        // need to recalculate shared indices
        int Eoffset_X = indices_X[0] - xx_heads[xx];
        int Eoffset_Y = indices_Y[0] - yy_heads[yy];
        int Eoffset_Z = indices_Z[0] - zz_heads[zz];

        int Hoffset_X = indices_X[2] - xx_heads[xx];
        int Hoffset_Y = indices_Y[2] - yy_heads[yy];
        int Hoffset_Z = indices_Z[2] - zz_heads[zz];

        int s_H_x = (m_or_v_X == 0 && xx != 0)? local_x + 1 + Hoffset_X + 1 : local_x + 1 + Hoffset_X;
        int s_H_y = (m_or_v_Y == 0 && yy != 0)? local_y + 1 + Hoffset_Y + 1 : local_y + 1 + Hoffset_Y;
        int s_H_z = (m_or_v_Z == 0 && zz != 0)? local_z + 1 + Hoffset_Z + 1 : local_z + 1 + Hoffset_Z;

        int s_E_x = local_x + Eoffset_X;
        int s_E_y = local_y + Eoffset_Y;
        int s_E_z = local_z + Eoffset_Z;

        int s_H_idx = s_H_x + s_H_y * BLX_EH + s_H_z * BLX_EH * BLY_EH; // s memory idx for H
        int s_E_idx = s_E_x + s_E_y * BLX_EH + s_E_z * BLX_EH * BLY_EH; // s memory idx for E

        int g_idx = g_x + g_y * Nx + g_z * Nx * Ny; // global idx

        // int temp = (g_idx == source_idx)? Jx_source : 0;

        Ex_shmem[s_E_idx] = Cax[g_idx] * Ex_shmem[s_E_idx] + Cbx[g_idx] *
                ((Hz_shmem[s_H_idx] - Hz_shmem[s_H_idx - BLX_EH]) - (Hy_shmem[s_H_idx] - Hy_shmem[s_H_idx - BLX_EH * BLY_EH]) - Jx[g_idx] * dx);
        Ey_shmem[s_E_idx] = Cay[g_idx] * Ey_shmem[s_E_idx] + Cby[g_idx] *
                  ((Hx_shmem[s_H_idx] - Hx_shmem[s_H_idx - BLX_EH * BLY_EH]) - (Hz_shmem[s_H_idx] - Hz_shmem[s_H_idx - 1]) - Jy[g_idx] * dx);
        Ez_shmem[s_E_idx] = Caz[g_idx] * Ez_shmem[s_E_idx] + Cbz[g_idx] *
                  ((Hy_shmem[s_H_idx] - Hy_shmem[s_H_idx - 1]) - (Hx_shmem[s_H_idx] - Hx_shmem[s_H_idx - BLX_EH]) - Jz[g_idx] * dx);

      }
    }

    __syncthreads();

    // update H 
    if(calculate_Hx & calculate_Hy & calculate_Hz) {
      // Hhead is offset
      int g_x = indices_X[2] + local_x; // Global X coordinate
      int g_y = indices_Y[2] + local_y; // Global Y coordinate
      int g_z = indices_Z[2] + local_z; // Global Z coordinate

      if(g_x >= 1 && g_x <= Nx-2 && g_y >= 1 && g_y <= Ny-2 && g_z >= 1 && g_z <= Nz-2 &&
        g_x <= indices_X[3] &&
        g_y <= indices_Y[3] &&
        g_z <= indices_Z[3]) {

        // need to recalculate shared indices
        int Eoffset_X = indices_X[0] - xx_heads[xx];
        int Eoffset_Y = indices_Y[0] - yy_heads[yy];
        int Eoffset_Z = indices_Z[0] - zz_heads[zz];

        int Hoffset_X = indices_X[2] - xx_heads[xx];
        int Hoffset_Y = indices_Y[2] - yy_heads[yy];
        int Hoffset_Z = indices_Z[2] - zz_heads[zz];

        int s_H_x = local_x + 1 + Hoffset_X;
        int s_H_y = local_y + 1 + Hoffset_Y;
        int s_H_z = local_z + 1 + Hoffset_Z;

        int s_E_x = (m_or_v_X == 0 && xx != 0)? local_x + Eoffset_X - 1 : local_x + Eoffset_X;
        int s_E_y = (m_or_v_Y == 0 && yy != 0)? local_y + Eoffset_Y - 1 : local_y + Eoffset_Y;
        int s_E_z = (m_or_v_Z == 0 && zz != 0)? local_z + Eoffset_Z - 1 : local_z + Eoffset_Z;

        int s_H_idx = s_H_x + s_H_y * BLX_EH + s_H_z * BLX_EH * BLY_EH; // s memory idx for H
        int s_E_idx = s_E_x + s_E_y * BLX_EH + s_E_z * BLX_EH * BLY_EH; // s memory idx for E

        int g_idx = g_x + g_y * Nx + g_z * Nx * Ny; // global idx

        Hx_shmem[s_H_idx] = Dax[g_idx] * Hx_shmem[s_H_idx] + Dbx[g_idx] *
                ((Ey_shmem[s_E_idx + BLX_EH * BLY_EH] - Ey_shmem[s_E_idx]) - (Ez_shmem[s_E_idx + BLX_EH] - Ez_shmem[s_E_idx]) - Mx[g_idx] * dx);
        Hy_shmem[s_H_idx] = Day[g_idx] * Hy_shmem[s_H_idx] + Dby[g_idx] *
                  ((Ez_shmem[s_E_idx + 1] - Ez_shmem[s_E_idx]) - (Ex_shmem[s_E_idx + BLX_EH * BLY_EH] - Ex_shmem[s_E_idx]) - My[g_idx] * dx);
        Hz_shmem[s_H_idx] = Daz[g_idx] * Hz_shmem[s_H_idx] + Dbz[g_idx] *
                  ((Ex_shmem[s_E_idx + BLX_EH] - Ex_shmem[s_E_idx]) - (Ey_shmem[s_E_idx + 1] - Ey_shmem[s_E_idx]) - Mz[g_idx] * dx);
      }
    }

    __syncthreads();
  }

  // store E, H to global memory, no HALO needed
  if(global_x >= 1 && global_x <= Nx-2 && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2 &&
     global_x <= xx_tails[xx] &&
     global_y <= yy_tails[yy] &&
     global_z <= zz_tails[zz]) {

     int s_H_x = local_x + 1;
     int s_H_y = local_y + 1;
     int s_H_z = local_z + 1;

     int s_E_x = local_x;
     int s_E_y = local_y;
     int s_E_z = local_z;

     int s_H_idx = s_H_x + s_H_y * BLX_EH + s_H_z * BLX_EH * BLY_EH; // s memory idx for H
     int s_E_idx = s_E_x + s_E_y * BLX_EH + s_E_z * BLX_EH * BLY_EH; // s memory idx for E

     int g_idx = global_x + global_y * Nx + global_z * Nx * Ny; // global idx

     Ex[g_idx] = Ex_shmem[s_E_idx];
     Ey[g_idx] = Ey_shmem[s_E_idx];
     Ez[g_idx] = Ez_shmem[s_E_idx];
     Hx[g_idx] = Hx_shmem[s_H_idx];
     Hy[g_idx] = Hy_shmem[s_H_idx];
     Hz[g_idx] = Hz_shmem[s_H_idx];
  }
} 
  
// use pt on z, dt on xy
__global__ void updateEH_phase_global_mem_2D(float *Ex, float *Ey, float *Ez,
                                  float *Hx, float *Hy, float *Hz,
                                  float *Cax, float *Cbx,
                                  float *Cay, float *Cby,
                                  float *Caz, float *Cbz,
                                  float *Dax, float *Dbx,
                                  float *Day, float *Dby,
                                  float *Daz, float *Dbz,
                                  float *Jx, float *Jy, float *Jz,
                                  float *Mx, float *My, float *Mz,
                                  float dx, 
                                  int Nx, int Ny, int Nz,
                                  int xx_num, int yy_num, int zz_num, 
                                  int *xx_heads, 
                                  int *yy_heads, 
                                  int *xx_tails, 
                                  int *yy_tails, 
                                  int m_or_v_X, int m_or_v_Y, 
                                  size_t block_size,
                                  size_t grid_size) 
{
  // first we map each (xx, yy, zz) to a block
  // int xx = blockIdx.x % xx_num;
  // int yy = (blockIdx.x % (xx_num * yy_num)) / xx_num;
  int xx = blockIdx.x % xx_num;
  int yy = blockIdx.x / xx_num;

  // map each thread in the block to a global index
  int tid = threadIdx.x;
  // int local_x = tid % BLX_GPU_PT;                     // X coordinate within the tile
  // int local_y = (tid / BLX_GPU_PT) % BLY_GPU_PT;     // Y coordinate within the tile
  int local_x = tid % BLX_GPU_PT;                     // X coordinate within the tile
  int local_y = tid / BLX_GPU_PT;     // Y coordinate within the tile

  __shared__ int indices_X[4];
  __shared__ int indices_Y[4];

  for(size_t zz=0; zz<zz_num; zz++) {

    for(size_t t=0; t<BLT_GPU_PT; t++) {

      int calculate_Ex = 1; // calculate this E tile or not
      int calculate_Hx = 1; // calculate this H tile or not
      int calculate_Ey = 1; 
      int calculate_Hy = 1; 

      // first find the range of tile 
      if(tid == 0) { // 1st thread in warp 0 
        get_head_tail(BLX_GPU_PT, BLT_GPU_PT,
                      xx_heads, xx_tails,
                      xx, t,
                      m_or_v_X, // 1 = mountain, 0 = valley
                      Nx,
                      &calculate_Ex, &calculate_Hx,
                      indices_X);
      }
      if(tid == 32) { // 1st thread in warp 1 
        get_head_tail(BLY_GPU_PT, BLT_GPU_PT,
                      yy_heads, yy_tails,
                      yy, t,
                      m_or_v_Y, // 1 = mountain, 0 = valley
                      Ny,
                      &calculate_Ey, &calculate_Hy,
                      indices_Y);
      }

      int global_z_E = get_z_planeE(t, zz, Nz);
      int global_z_H = get_z_planeH(t, zz, Nz); 

      __syncthreads();

      // update E
      if(calculate_Ex & calculate_Ey) {
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

      __syncthreads();

      // update H 
      if(calculate_Hx & calculate_Hy) {
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

      __syncthreads();
    }
  }

}
  
// use pt on z, dt on xy
__global__ void updateEH_phase_shmem_EH_2D(float *Ex, float *Ey, float *Ez,
                                  float *Hx, float *Hy, float *Hz,
                                  float *Cax, float *Cbx,
                                  float *Cay, float *Cby,
                                  float *Caz, float *Cbz,
                                  float *Dax, float *Dbx,
                                  float *Day, float *Dby,
                                  float *Daz, float *Dbz,
                                  float *Jx, float *Jy, float *Jz,
                                  float *Mx, float *My, float *Mz,
                                  float dx, 
                                  int Nx, int Ny, int Nz,
                                  int xx_num, int yy_num, int zz_num, 
                                  int *xx_heads, 
                                  int *yy_heads, 
                                  int *xx_tails, 
                                  int *yy_tails, 
                                  int m_or_v_X, int m_or_v_Y, 
                                  size_t block_size,
                                  size_t grid_size) 
{
  // first we map each (xx, yy, zz) to a block
  // int xx = blockIdx.x % xx_num;
  // int yy = (blockIdx.x % (xx_num * yy_num)) / xx_num;
  int xx = blockIdx.x % xx_num;
  int yy = blockIdx.x / xx_num;

  // map each thread in the block to a global index
  int tid = threadIdx.x;
  // int local_x = tid % BLX_GPU_PT;                     // X coordinate within the tile
  // int local_y = (tid / BLX_GPU_PT) % BLY_GPU_PT;     // Y coordinate within the tile
  int local_x = tid % BLX_GPU_PT;                     // X coordinate within the tile
  int local_y = tid / BLX_GPU_PT;     // Y coordinate within the tile

  // declare shared memory for each block
  __shared__ float Ex_shmem[(BLX_GPU_PT + 1) * (BLY_GPU_PT + 1) * (BLT_GPU_PT + 1)];
  __shared__ float Ey_shmem[(BLX_GPU_PT + 1) * (BLY_GPU_PT + 1) * (BLT_GPU_PT + 1)];
  __shared__ float Ez_shmem[(BLX_GPU_PT + 1) * (BLY_GPU_PT + 1) * (BLT_GPU_PT + 1)];
  __shared__ float Hx_shmem[(BLX_GPU_PT + 1) * (BLY_GPU_PT + 1) * (BLT_GPU_PT + 1)];
  __shared__ float Hy_shmem[(BLX_GPU_PT + 1) * (BLY_GPU_PT + 1) * (BLT_GPU_PT + 1)];
  __shared__ float Hz_shmem[(BLX_GPU_PT + 1) * (BLY_GPU_PT + 1) * (BLT_GPU_PT + 1)];

  __shared__ int indices_X[4];
  __shared__ int indices_Y[4];

  // global index
  int global_x = xx_heads[xx] + local_x; 
  int global_y = yy_heads[yy] + local_y;

  // shared index for H
  int shared_H_x = local_x + 1;
  int shared_H_y = local_y + 1;
  
  // shared index for E
  int shared_E_x = local_x;
  int shared_E_y = local_y;

  for(size_t zz=0; zz<zz_num; zz++) {

    // load shared memory
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

    // do calculation
    int z_start = (zz == 0)? 0 : 4;  
    int z_bound = (zz == 0 || zz == zz_num - 1)? BLT_GPU_PT + 1 : 1;  
    for(size_t t=0; t<BLT_GPU_PT; t++) {
      for(int local_z=z_start; local_z<z_start+z_bound; local_z++) { // each thread iterate Z dimension

        int calculate_Ex = 1; // calculate this E tile or not
        int calculate_Hx = 1; // calculate this H tile or not
        int calculate_Ey = 1; 
        int calculate_Hy = 1; 

        // first find the range of tile 
        if(tid == 0) { // 1st thread in warp 0 
          get_head_tail(BLX_GPU_PT, BLT_GPU_PT,
                        xx_heads, xx_tails,
                        xx, t,
                        m_or_v_X, // 1 = mountain, 0 = valley
                        Nx,
                        &calculate_Ex, &calculate_Hx,
                        indices_X);
        }
        if(tid == 32) { // 1st thread in warp 1 
          get_head_tail(BLY_GPU_PT, BLT_GPU_PT,
                        yy_heads, yy_tails,
                        yy, t,
                        m_or_v_Y, // 1 = mountain, 0 = valley
                        Ny,
                        &calculate_Ey, &calculate_Hy,
                        indices_Y);
        }

        __syncthreads();

        // update E
        if(calculate_Ex & calculate_Ey) {
          // Ehead is offset
          int g_x = indices_X[0] + local_x; // Global X coordinate
          int g_y = indices_Y[0] + local_y; // Global Y coordinate

          int s_E_z = get_z_planeE_shmem(t, local_z, Nz);
          int g_E_z = s_E_z + zz;

          if(g_x >= 1 && g_x <= Nx-2 && g_y >= 1 && g_y <= Ny-2 && g_E_z >= 1 && g_E_z <= Nz-2 &&
            g_x <= indices_X[1] &&
            g_y <= indices_Y[1] &&
            s_E_z != -1) {

            // need to recalculate shared indices
            int Eoffset_X = indices_X[0] - xx_heads[xx];
            int Eoffset_Y = indices_Y[0] - yy_heads[yy];

            int Hoffset_X = indices_X[2] - xx_heads[xx];
            int Hoffset_Y = indices_Y[2] - yy_heads[yy];

            int s_H_x = (m_or_v_X == 0 && xx != 0)? local_x + 1 + Hoffset_X + 1 : local_x + 1 + Hoffset_X;
            int s_H_y = (m_or_v_Y == 0 && yy != 0)? local_y + 1 + Hoffset_Y + 1 : local_y + 1 + Hoffset_Y;

            int s_E_x = local_x + Eoffset_X;
            int s_E_y = local_y + Eoffset_Y;

            // notice that for H in Z dimension, it is using s_E_z
            int s_H_idx = s_H_x + s_H_y * BLX_EH + s_E_z * BLX_EH * BLY_EH; // shared memory idx for H
            int s_E_idx = s_E_x + s_E_y * BLX_EH + s_E_z * BLX_EH * BLY_EH; // shared memory idx for E

            int g_idx = g_x + g_y * Nx + g_E_z * Nx * Ny; // global idx

            Ex_shmem[s_E_idx] = Cax[g_idx] * Ex_shmem[s_E_idx] + Cbx[g_idx] *
                ((Hz_shmem[s_H_idx] - Hz_shmem[s_H_idx - BLX_EH]) - (Hy_shmem[s_H_idx] - Hy_shmem[s_H_idx - BLX_EH * BLY_EH]) - Jx[g_idx] * dx);
            Ey_shmem[s_E_idx] = Cay[g_idx] * Ey_shmem[s_E_idx] + Cby[g_idx] *
                      ((Hx_shmem[s_H_idx] - Hx_shmem[s_H_idx - BLX_EH * BLY_EH]) - (Hz_shmem[s_H_idx] - Hz_shmem[s_H_idx - 1]) - Jy[g_idx] * dx);
            Ez_shmem[s_E_idx] = Caz[g_idx] * Ez_shmem[s_E_idx] + Cbz[g_idx] *
                      ((Hy_shmem[s_H_idx] - Hy_shmem[s_H_idx - 1]) - (Hx_shmem[s_H_idx] - Hx_shmem[s_H_idx - BLX_EH]) - Jz[g_idx] * dx);
          }
        }

        __syncthreads();

        // update H 
        if(calculate_Hx & calculate_Hy) {
          // Hhead is offset
          int g_x = indices_X[2] + local_x; // Global X coordinate
          int g_y = indices_Y[2] + local_y; // Global Y coordinate

          int s_H_z = get_z_planeH_shmem(t, local_z, Nz);
          int g_H_z = s_H_z + zz;

          if(g_x >= 1 && g_x <= Nx-2 && g_y >= 1 && g_y <= Ny-2 && g_H_z >= 1 && g_H_z <= Nz-2 &&
            g_x <= indices_X[3] &&
            g_y <= indices_Y[3] &&
            s_H_z != -1) {

            // need to recalculate shared indices
            int Eoffset_X = indices_X[0] - xx_heads[xx];
            int Eoffset_Y = indices_Y[0] - yy_heads[yy];

            int Hoffset_X = indices_X[2] - xx_heads[xx];
            int Hoffset_Y = indices_Y[2] - yy_heads[yy];

            int s_H_x = local_x + 1 + Hoffset_X;
            int s_H_y = local_y + 1 + Hoffset_Y;

            int s_E_x = (m_or_v_X == 0 && xx != 0)? local_x + Eoffset_X - 1 : local_x + Eoffset_X;
            int s_E_y = (m_or_v_Y == 0 && yy != 0)? local_y + Eoffset_Y - 1 : local_y + Eoffset_Y;

            // notice that for E in Z dimension, it is using s_H_z
            int s_H_idx = s_H_x + s_H_y * BLX_EH + s_H_z * BLX_EH * BLY_EH; // shared memory idx for H
            int s_E_idx = s_E_x + s_E_y * BLX_EH + s_H_z * BLX_EH * BLY_EH; // shared memory idx for E

            int g_idx = g_x + g_y * Nx + g_H_z * Nx * Ny; // global idx

            Hx_shmem[s_H_idx] = Dax[g_idx] * Hx_shmem[s_H_idx] + Dbx[g_idx] *
                ((Ey_shmem[s_E_idx + BLX_EH * BLY_EH] - Ey_shmem[s_E_idx]) - (Ez_shmem[s_E_idx + BLX_EH] - Ez_shmem[s_E_idx]) - Mx[g_idx] * dx);
            Hy_shmem[s_H_idx] = Day[g_idx] * Hy_shmem[s_H_idx] + Dby[g_idx] *
                      ((Ez_shmem[s_E_idx + 1] - Ez_shmem[s_E_idx]) - (Ex_shmem[s_E_idx + BLX_EH * BLY_EH] - Ex_shmem[s_E_idx]) - My[g_idx] * dx);
            Hz_shmem[s_H_idx] = Daz[g_idx] * Hz_shmem[s_H_idx] + Dbz[g_idx] *
                      ((Ex_shmem[s_E_idx + BLX_EH] - Ex_shmem[s_E_idx]) - (Ey_shmem[s_E_idx + 1] - Ey_shmem[s_E_idx]) - Mz[g_idx] * dx);
          }
        }

        __syncthreads();
      }
    }

    // store back to globalmem
    // store E, H to global memory, no HALO needed
    for(int local_z=0; local_z<BLT_GPU_PT+1; local_z++) {
      int global_z = local_z + zz;

      if(global_x >= 1 && global_x <= Nx-2 && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2 &&
        global_x <= xx_tails[xx] &&
        global_y <= yy_tails[yy]) {

        int shared_H_z = local_z;

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

__global__ void updateEH_mil(float *Ex, float *Ey, float *Ez,
                             float *Hx, float *Hy, float *Hz,
                             float *Cax, float *Cbx,
                             float *Cay, float *Cby,
                             float *Caz, float *Cbz,
                             float *Dax, float *Dbx,
                             float *Day, float *Dby,
                             float *Daz, float *Dbz,
                             float *Jx, float *Jy, float *Jz,
                             float *Mx, float *My, float *Mz,
                             float dx, 
                             int Nx, int Ny, int Nz,
                             int xx_num, int yy_num, int zz_num, // number of tiles in each dimensions
                             int *xx_heads, int *yy_heads, int *zz_heads,
                             int *xx_tails, int *yy_tails, int *zz_tails,
                             int *xx_top_heads, int *yy_top_heads, int *zz_top_heads,
                             int *xx_top_tails, int *yy_top_tails, int *zz_top_tails,
                             int *shmem_load_finish,
                             size_t block_size,
                             size_t grid_size) {

  int xx = blockIdx.x % xx_num;
  int yy = (blockIdx.x % (xx_num * yy_num)) / xx_num;
  int zz = blockIdx.x / (xx_num * yy_num); 

  int local_x = threadIdx.x % BLX_MIL;
  int local_y = (threadIdx.x / BLX_MIL) % BLY_MIL;
  int local_z = threadIdx.x / (BLX_MIL * BLY_MIL);

  int global_x = xx_heads[xx] + local_x;
  int global_y = yy_heads[yy] + local_y;
  int global_z = zz_heads[zz] + local_z;
  int global_idx = global_x + global_y * Nx + global_z * Nx * Ny;

  int shared_H_x = local_x + 1;
  int shared_H_y = local_y + 1;
  int shared_H_z = local_z + 1;
  int shared_H_idx = shared_H_x + shared_H_y * BLX_MIL_EH + shared_H_z * BLX_MIL_EH * BLY_MIL_EH;

  int shared_E_x = local_x;
  int shared_E_y = local_y;
  int shared_E_z = local_z;
  int shared_E_idx = shared_E_x + shared_E_y * BLX_MIL_EH + shared_E_z * BLX_MIL_EH * BLY_MIL_EH;

  // declare shared memory
  // 9*9*9*4*6 = 17496btyes
  // block_size = 512
  /*

  block_size = 512, grid_size = 8000

  BLX = 8
  BLX_top = 5
  100/5 = 20
  20*20*20 = 8000

  deviceProp.sharedMemPerBlockOptin = 101376
  deviceProp.sharedMemPerBlock = 48.00 (KB)
  deviceProp.sharedMemPerMultiprocessor = 100.00 (KB)

  deviceProp.multiProcessorCount = 48
  deviceProp.maxBlocksPerMultiProcessor = 16
  deviceProp.maxThreadsPerBlock = 1024
  deviceProp.maxThreadsPerMultiProcessor = 1536

  deviceProp.regsPerBlock = 65536
  deviceProp.regsPerMultiprocessor = 65536

  deviceProp.maxGridSize[0] = 2147483647
  deviceProp.maxGridSize[1] = 65535
  deviceProp.maxGridSize[2] = 65535

  */
  float Ex_shmem[BLX_MIL_EH * BLY_MIL_EH * BLZ_MIL_EH];
  float Ey_shmem[BLX_MIL_EH * BLY_MIL_EH * BLZ_MIL_EH];
  float Ez_shmem[BLX_MIL_EH * BLY_MIL_EH * BLZ_MIL_EH];
  float Hx_shmem[BLX_MIL_EH * BLY_MIL_EH * BLZ_MIL_EH];
  float Hy_shmem[BLX_MIL_EH * BLY_MIL_EH * BLZ_MIL_EH];
  float Hz_shmem[BLX_MIL_EH * BLY_MIL_EH * BLZ_MIL_EH];

  // load shared memory
  if(global_x < Nx && global_y < Ny && global_z < Nz) {

    // load core
    Hx_shmem[shared_H_idx] = Hx[global_idx];
    Hy_shmem[shared_H_idx] = Hy[global_idx];
    Hz_shmem[shared_H_idx] = Hz[global_idx];

    // load HALO region
    if(local_x == 0 && global_x > 0) {
      Hz_shmem[shared_H_x - 1 + shared_H_y * BLX_MIL_EH + shared_H_z * BLX_MIL_EH * BLY_MIL_EH] = Hz[global_x - 1 + global_y * Nx + global_z * Nx * Ny];
      Hy_shmem[shared_H_x - 1 + shared_H_y * BLX_MIL_EH + shared_H_z * BLX_MIL_EH * BLY_MIL_EH] = Hy[global_x - 1 + global_y * Nx + global_z * Nx * Ny];

    }
    if(local_y == 0 && global_y > 0) {
      Hx_shmem[shared_H_x + (shared_H_y - 1) * BLX_MIL_EH + shared_H_z * BLX_MIL_EH * BLY_MIL_EH] = Hx[global_x + (global_y - 1) * Nx + global_z * Nx * Ny];
      Hz_shmem[shared_H_x + (shared_H_y - 1) * BLX_MIL_EH + shared_H_z * BLX_MIL_EH * BLY_MIL_EH] = Hz[global_x + (global_y - 1) * Nx + global_z * Nx * Ny];
    }
    if(local_z == 0 && global_z > 0) {
      Hx_shmem[shared_H_x + shared_H_y * BLX_MIL_EH + (shared_H_z - 1) * BLX_MIL_EH * BLY_MIL_EH] = Hx[global_x + global_y * Nx + (global_z - 1) * Nx * Ny];
      Hy_shmem[shared_H_x + shared_H_y * BLX_MIL_EH + (shared_H_z - 1) * BLX_MIL_EH * BLY_MIL_EH] = Hy[global_x + global_y * Nx + (global_z - 1) * Nx * Ny];
    }
  }

  if(global_x < Nx && global_y < Ny && global_z < Nz) {

    Ex_shmem[shared_E_idx] = Ex[global_idx];
    Ey_shmem[shared_E_idx] = Ey[global_idx];
    Ez_shmem[shared_E_idx] = Ez[global_idx];

    // load HALO region
    if(local_x == BLX_GPU - 1 && global_x < Nx - 1) {
      Ez_shmem[shared_E_x + 1 + shared_E_y * BLX_MIL_EH + shared_E_z * BLX_MIL_EH * BLY_MIL_EH] = Ez[global_x + 1 + global_y * Nx + global_z * Nx * Ny];
      Ey_shmem[shared_E_x + 1 + shared_E_y * BLX_MIL_EH + shared_E_z * BLX_MIL_EH * BLY_MIL_EH] = Ey[global_x + 1 + global_y * Nx + global_z * Nx * Ny];
    }
    if(local_y == BLY_GPU - 1 && global_y < Ny - 1) {
      Ex_shmem[shared_E_x + (shared_E_y + 1) * BLX_MIL_EH + shared_E_z * BLX_MIL_EH * BLY_MIL_EH] = Ex[global_x + (global_y + 1) * Nx + global_z * Nx * Ny];
      Ez_shmem[shared_E_x + (shared_E_y + 1) * BLX_MIL_EH + shared_E_z * BLX_MIL_EH * BLY_MIL_EH] = Ez[global_x + (global_y + 1) * Nx + global_z * Nx * Ny];
    }
    if(local_z == BLZ_GPU - 1 && global_z < Nz - 1) {
      Ex_shmem[shared_E_x + shared_E_y * BLX_MIL_EH + (shared_E_z + 1) * BLX_MIL_EH * BLY_MIL_EH] = Ex[global_x + global_y * Nx + (global_z + 1) * Nx * Ny];
      Ey_shmem[shared_E_x + shared_E_y * BLX_MIL_EH + (shared_E_z + 1) * BLX_MIL_EH * BLY_MIL_EH] = Ey[global_x + global_y * Nx + (global_z + 1) * Nx * Ny];
    }
  }

  __syncthreads();

  // if(threadIdx.x == 0) {
  //   atomicAdd(&shmem_load_finish[xx + yy * xx_num + zz * xx_num * yy_num], 1);
  // }

  // calculation
  for(size_t t=0; t<BLT_MIL; t++) {

    if(global_x >= 1 && global_x <= Nx-2 && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2 &&
       global_x <= xx_tails[xx] &&
       global_y <= yy_tails[yy] &&
       global_z <= zz_tails[zz]) {

      Ex_shmem[shared_E_idx] = Cax[global_idx] * Ex_shmem[shared_E_idx] + Cbx[global_idx] *
                ((Hz_shmem[shared_H_idx] - Hz_shmem[shared_H_idx - BLX_MIL_EH]) - (Hy_shmem[shared_H_idx] - Hy_shmem[shared_H_idx - BLX_MIL_EH * BLY_MIL_EH]) - Jx[global_idx] * dx);
      Ey_shmem[shared_E_idx] = Cay[global_idx] * Ey_shmem[shared_E_idx] + Cby[global_idx] *
                ((Hx_shmem[shared_H_idx] - Hx_shmem[shared_H_idx - BLX_MIL_EH * BLY_MIL_EH]) - (Hz_shmem[shared_H_idx] - Hz_shmem[shared_H_idx - 1]) - Jy[global_idx] * dx);
      Ez_shmem[shared_E_idx] = Caz[global_idx] * Ez_shmem[shared_E_idx] + Cbz[global_idx] *
                ((Hy_shmem[shared_H_idx] - Hy_shmem[shared_H_idx - 1]) - (Hx_shmem[shared_H_idx] - Hx_shmem[shared_H_idx - BLX_MIL_EH]) - Jz[global_idx] * dx);
    }

    __syncthreads();

    if(global_x >= 1 && global_x <= Nx-2 && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2 &&
       global_x <= xx_tails[xx] &&
       global_y <= yy_tails[yy] &&
       global_z <= zz_tails[zz]) {

      Hx_shmem[shared_H_idx] = Dax[global_idx] * Hx_shmem[shared_H_idx] + Dbx[global_idx] *
                ((Ey_shmem[shared_E_idx + BLX_MIL_EH * BLY_MIL_EH] - Ey_shmem[shared_E_idx]) - (Ez_shmem[shared_E_idx + BLX_MIL_EH] - Ez_shmem[shared_E_idx]) - Mx[global_idx] * dx);
      Hy_shmem[shared_H_idx] = Day[global_idx] * Hy_shmem[shared_H_idx] + Dby[global_idx] *
                ((Ez_shmem[shared_E_idx + 1] - Ez_shmem[shared_E_idx]) - (Ex_shmem[shared_E_idx + BLX_MIL_EH * BLY_MIL_EH] - Ex_shmem[shared_E_idx]) - My[global_idx] * dx);
      Hz_shmem[shared_H_idx] = Daz[global_idx] * Hz_shmem[shared_H_idx] + Dbz[global_idx] *
                ((Ex_shmem[shared_E_idx + BLX_MIL_EH] - Ex_shmem[shared_E_idx]) - (Ey_shmem[shared_E_idx + 1] - Ey_shmem[shared_E_idx]) - Mz[global_idx] * dx);
    }

    __syncthreads();
  
  }

  // // before store back to global mem, must check if adjacent block has finish shmem load
  // if(threadIdx.x == 0) { // check xx+1
  //   if(xx+1 <= xx_num-1) {
  //     while(atomicMax(&shmem_load_finish[(xx + 1) + yy * xx_num + zz * xx_num * yy_num], 0) == 0) {}
  //   }
  // } 
  // if(threadIdx.x == 32) { // check xx-1
  //   if(xx-1 >= 0) {
  //     while(atomicMax(&shmem_load_finish[(xx - 1) + yy * xx_num + zz * xx_num * yy_num], 0) == 0) {}
  //   }
  // }
  // if(threadIdx.x == 64) { // check yy+1 
  //   if(yy+1 <= yy_num-1) {
  //     while(atomicMax(&shmem_load_finish[xx + (yy + 1) * xx_num + zz * xx_num * yy_num], 0) == 0) {}
  //   }
  // }
  // if(threadIdx.x == 96) { // check yy-1 
  //   if(yy-1 >= 0) {
  //     while(atomicMax(&shmem_load_finish[xx + (yy - 1) * xx_num + zz * xx_num * yy_num], 0) == 0) {}
  //   }
  // }
  // if(threadIdx.x == 128) { // check zz+1 
  //   if(zz+1 <= zz_num-1) {
  //     while(atomicMax(&shmem_load_finish[xx + yy * xx_num + (zz + 1) * xx_num * yy_num], 0) == 0) {}
  //   }
  // }
  // if(threadIdx.x == 160) { // check zz-1 
  //   if(zz-1 >= 0) {
  //     while(atomicMax(&shmem_load_finish[xx + yy * xx_num + (zz - 1) * xx_num * yy_num], 0) == 0) {}
  //   }
  // }

  // store back to global mem
  if(global_x >= 1 && global_x <= Nx-2 && global_y >= 1 && global_y <= Ny-2 && global_z >= 1 && global_z <= Nz-2 &&
     global_x >= xx_top_heads[xx] && global_x <= xx_top_tails[xx] &&
     global_y >= yy_top_heads[yy] && global_y <= yy_top_tails[yy] &&
     global_z >= zz_top_heads[zz] && global_z <= zz_top_tails[zz]) {
    Ex[global_idx] = Ex_shmem[shared_E_idx];
    Ey[global_idx] = Ey_shmem[shared_E_idx];
    Ez[global_idx] = Ez_shmem[shared_E_idx];
    Hx[global_idx] = Hx_shmem[shared_H_idx];
    Hy[global_idx] = Hy_shmem[shared_H_idx];
    Hz[global_idx] = Hz_shmem[shared_H_idx];
  }

}

//
// -------------------------------------------------- parallelogram tiling --------------------------------------------------
//
__global__ void updateEH_pt(float *Ex, float *Ey, float *Ez,
                            float *Hx, float *Hy, float *Hz,
                            float *Cax, float *Cbx,
                            float *Cay, float *Cby,
                            float *Caz, float *Cbz,
                            float *Dax, float *Dbx,
                            float *Day, float *Dby,
                            float *Daz, float *Dbz,
                            float *Jx, float *Jy, float *Jz,
                            float *Mx, float *My, float *Mz,
                            float dx, 
                            int Nx, int Ny, int Nz,
                            int xx_num, int yy_num, int zz_num, // number of tiles in each dimensions
                            int *xx_heads, 
                            int *yy_heads, 
                            int *zz_heads,
                            gdiamond::Pt_idx *hyperplanes,
                            int hyperplane_head) {

  int block_id = blockIdx.x;  
  int thread_id = threadIdx.x;
  int local_x = thread_id % BLX_PT;
  int local_y = (thread_id / BLX_PT) % BLY_PT;
  int local_z = thread_id / (BLX_PT * BLY_PT);

  // get the index of paralellogram tile of this block
  gdiamond::Pt_idx p = hyperplanes[hyperplane_head + block_id];
  int xx = p.x;
  int yy = p.y;
  int zz = p.z;

  // declare shared memory
  float Ex_shmem[(BLX_PT + BLT_PT) * (BLY_PT + BLT_PT) * (BLZ_PT + BLT_PT)];
  float Ey_shmem[(BLX_PT + BLT_PT) * (BLY_PT + BLT_PT) * (BLZ_PT + BLT_PT)];
  float Ez_shmem[(BLX_PT + BLT_PT) * (BLY_PT + BLT_PT) * (BLZ_PT + BLT_PT)];
  float Hx_shmem[(BLX_PT + BLT_PT) * (BLY_PT + BLT_PT) * (BLZ_PT + BLT_PT)];
  float Hy_shmem[(BLX_PT + BLT_PT) * (BLY_PT + BLT_PT) * (BLZ_PT + BLT_PT)];
  float Hz_shmem[(BLX_PT + BLT_PT) * (BLY_PT + BLT_PT) * (BLZ_PT + BLT_PT)];

  // load shared memory
  for(int l_z = local_z; l_z < (BLZ_PT + BLT_PT); l_z += BLZ_PT) {
    for(int l_y = local_y; l_y < (BLY_PT + BLT_PT); l_y += BLY_PT) {
      for(int l_x = local_x; l_x < (BLX_PT + BLT_PT); l_x += BLX_PT) {
        int g_x = l_x + xx_heads[xx];
        int g_y = l_y + yy_heads[yy];
        int g_z = l_z + zz_heads[zz];

        int g_idx = g_x + g_y * Nx + g_z * Nx * Ny;
        int l_idx = l_x + l_y * (BLX_PT + BLT_PT) + l_z * (BLX_PT + BLT_PT) * (BLY_PT + BLT_PT);

        if(g_x >= 0 && g_x <= Nx-1 &&
           g_y >= 0 && g_y <= Ny-1 &&
           g_z >= 0 && g_z <= Nz-1) {
          Ex_shmem[l_idx] = Ex[g_idx];
          Ey_shmem[l_idx] = Ey[g_idx];
          Ez_shmem[l_idx] = Ez[g_idx];
          Hx_shmem[l_idx] = Hx[g_idx];
          Hy_shmem[l_idx] = Hy[g_idx];
          Hz_shmem[l_idx] = Hz[g_idx];
        }
      }
    }
  }

  __syncthreads();

  // update
  int shared_E_x;
  int shared_E_y;
  int shared_E_z;
  int shared_H_x;
  int shared_H_y;
  int shared_H_z;
  int global_x;
  int global_y;
  int global_z;
  int shared_E_idx;
  int shared_H_idx;
  int global_idx;
  for(int t=0; t<BLT_PT; t++) {
    int offset = BLT_PT - t;

    // update E
    shared_E_x = local_x + offset;
    shared_E_y = local_y + offset;
    shared_E_z = local_z + offset;
    global_x = local_x + offset + xx_heads[xx];
    global_y = local_y + offset + yy_heads[yy];
    global_z = local_z + offset + zz_heads[zz];
    shared_E_idx = shared_E_x + shared_E_y * (BLX_PT + BLT_PT) + shared_E_z * (BLX_PT + BLT_PT) * (BLY_PT + BLT_PT);
    global_idx = global_x + global_y * Nx + global_z * Nx * Ny;

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

    __syncthreads();

    // update H
    shared_H_x = local_x + offset - 1;
    shared_H_y = local_y + offset - 1;
    shared_H_z = local_z + offset - 1;

    global_x = local_x + offset + xx_heads[xx] - 1;
    global_y = local_y + offset + yy_heads[yy] - 1;
    global_z = local_z + offset + zz_heads[zz] - 1;

    shared_H_idx = shared_H_x + shared_H_y * (BLX_PT + BLT_PT) + shared_H_z * (BLX_PT + BLT_PT) * (BLY_PT + BLT_PT);
    global_idx = global_x + global_y * Nx + global_z * Nx * Ny;

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

    __syncthreads();
  }

  // store global memory
  for(int l_z = local_z; l_z < (BLZ_PT + BLT_PT); l_z += BLZ_PT) {
    for(int l_y = local_y; l_y < (BLY_PT + BLT_PT); l_y += BLY_PT) {
      for(int l_x = local_x; l_x < (BLX_PT + BLT_PT); l_x += BLX_PT) {
        int g_x = l_x + xx_heads[xx];
        int g_y = l_y + yy_heads[yy];
        int g_z = l_z + zz_heads[zz];

        int g_idx = g_x + g_y * Nx + g_z * Nx * Ny;
        int l_idx = l_x + l_y * (BLX_PT + BLT_PT) + l_z * (BLX_PT + BLT_PT) * (BLY_PT + BLT_PT);

        if(g_x >= 0 && g_x <= Nx-1 &&
           g_y >= 0 && g_y <= Ny-1 &&
           g_z >= 0 && g_z <= Nz-1) {
          Ex[g_idx] = Ex_shmem[l_idx];
          Ey[g_idx] = Ey_shmem[l_idx];
          Ez[g_idx] = Ez_shmem[l_idx];
          Hx[g_idx] = Hx_shmem[l_idx];
          Hy[g_idx] = Hy_shmem[l_idx];
          Hz[g_idx] = Hz_shmem[l_idx];
        }
      }
    }
  }

}

//
// --------------------------------------------- upper bound speedup check
//

__global__ void updateE_ub_globalmem_only(float *Ex, float *Ey, float *Ez,
                                          float *Hx, float *Hy, float *Hz,
                                          float *Cax, float *Cbx,
                                          float *Cay, float *Cby,
                                          float *Caz, float *Cbz,
                                          float *Dax, float *Dbx,
                                          float *Day, float *Dby,
                                          float *Daz, float *Dbz,
                                          float *Jx, float *Jy, float *Jz,
                                          float *Mx, float *My, float *Mz,
                                          float dx, 
                                          int Nx, int Ny, int Nz,
                                          int xx_num, int yy_num, int zz_num,
                                          int *xx_heads, int *yy_heads, int *zz_heads) 
{
  // first we map each (xx, yy, zz) to a block
  const int xx = blockIdx.x % xx_num;
  const int yy = (blockIdx.x % (xx_num * yy_num)) / xx_num;
  const int zz = blockIdx.x / (xx_num * yy_num);

  // map each thread in the block to a global index
  const int tid = threadIdx.x;
  const int local_x = tid % BLX_UB;                     // X coordinate within the tile
  const int local_y = (tid % (BLX_UB * BLY_UB)) / BLX_UB; 
  const int local_z = tid / (BLX_UB * BLY_UB);     // Z coordinate within the tile
  const int global_x = xx_heads[xx] + local_x; // Global X coordinate
  const int global_y = yy_heads[yy] + local_y; // Global Y coordinate
  const int global_z = zz_heads[zz] + local_z; // Global Z coordinate
  const int global_idx = global_x + global_y * Nx + global_z * Nx * Ny;

  if(global_x >= 1 && global_x <= Nx-2 &&
     global_y >= 1 && global_y <= Ny-2 &&
     global_z >= 1 && global_z <= Nz-2) {

    Ex[global_idx] = Cax[global_idx] * Ex[global_idx] + Cbx[global_idx] *
              ((Hz[global_idx] - Hz[global_idx - Nx]) - (Hy[global_idx] - Hy[global_idx - Nx * Ny]) - Jx[global_idx] * dx);

    Ey[global_idx] = Cay[global_idx] * Ey[global_idx] + Cby[global_idx] *
              ((Hx[global_idx] - Hx[global_idx - Nx * Ny]) - (Hz[global_idx] - Hz[global_idx - 1]) - Jy[global_idx] * dx);

    Ez[global_idx] = Caz[global_idx] * Ez[global_idx] + Cbz[global_idx] *
              ((Hy[global_idx] - Hy[global_idx - 1]) - (Hx[global_idx] - Hx[global_idx - Nx]) - Jz[global_idx] * dx);
  }

}

__global__ void updateH_ub_globalmem_only(float *Ex, float *Ey, float *Ez,
                                          float *Hx, float *Hy, float *Hz,
                                          float *Cax, float *Cbx,
                                          float *Cay, float *Cby,
                                          float *Caz, float *Cbz,
                                          float *Dax, float *Dbx,
                                          float *Day, float *Dby,
                                          float *Daz, float *Dbz,
                                          float *Jx, float *Jy, float *Jz,
                                          float *Mx, float *My, float *Mz,
                                          float dx, 
                                          int Nx, int Ny, int Nz,
                                          int xx_num, int yy_num, int zz_num,
                                          int *xx_heads, int *yy_heads, int *zz_heads) 
{
  // first we map each (xx, yy, zz) to a block
  const int xx = blockIdx.x % xx_num;
  const int yy = (blockIdx.x % (xx_num * yy_num)) / xx_num;
  const int zz = blockIdx.x / (xx_num * yy_num);

  // map each thread in the block to a global index
  const int tid = threadIdx.x;
  const int local_x = tid % BLX_UB;                     // X coordinate within the tile
  const int local_y = (tid / BLX_UB) % BLY_UB;     // Y coordinate within the tile
  const int local_z = tid / (BLX_UB * BLY_UB);     // Z coordinate within the tile
  const int global_x = xx_heads[xx] + local_x; // Global X coordinate
  const int global_y = yy_heads[yy] + local_y; // Global Y coordinate
  const int global_z = zz_heads[zz] + local_z; // Global Z coordinate
  const int global_idx = global_x + global_y * Nx + global_z * Nx * Ny;

  if(global_x >= 1 && global_x <= Nx-2 &&
     global_y >= 1 && global_y <= Ny-2 &&
     global_z >= 1 && global_z <= Nz-2) {

    Hx[global_idx] = Dax[global_idx] * Hx[global_idx] + Dbx[global_idx] *
              ((Ey[global_idx + Nx * Ny] - Ey[global_idx]) - (Ez[global_idx + Nx] - Ez[global_idx]) - Mx[global_idx] * dx);

    Hy[global_idx] = Day[global_idx] * Hy[global_idx] + Dby[global_idx] *
              ((Ez[global_idx + 1] - Ez[global_idx]) - (Ex[global_idx + Nx * Ny] - Ex[global_idx]) - My[global_idx] * dx);

    Hz[global_idx] = Daz[global_idx] * Hz[global_idx] + Dbz[global_idx] *
              ((Ex[global_idx + Nx] - Ex[global_idx]) - (Ey[global_idx + 1] - Ey[global_idx]) - Mz[global_idx] * dx);
  }

}

__global__ void updateEH_ub(float *Ex, float *Ey, float *Ez,
                            float *Hx, float *Hy, float *Hz,
                            float *Cax, float *Cbx,
                            float *Cay, float *Cby,
                            float *Caz, float *Cbz,
                            float *Dax, float *Dbx,
                            float *Day, float *Dby,
                            float *Daz, float *Dbz,
                            float *Jx, float *Jy, float *Jz,
                            float *Mx, float *My, float *Mz,
                            float dx, 
                            int Nx, int Ny, int Nz,
                            int xx_num, int yy_num, int zz_num,
                            int *xx_heads, int *yy_heads, int *zz_heads) 
{


  // // Compute x, y, z in the same order as naive mapping
  // const int thread_id = threadIdx.x;
  // int x = thread_id % Nx;
  // int y = (thread_id % (Nx * Ny)) / Nx;
  // int z = thread_id / (Nx * Ny);

  // // Find which tile (xx, yy, zz) the thread belongs to
  // int xx = x / BLX_UB;
  // int yy = y / BLY_UB;
  // int zz = z / BLZ_UB;

  // // Compute local position inside the tile
  // int local_x = x % BLX_UB;
  // int local_y = y % BLY_UB;
  // int local_z = z % BLZ_UB;

  // // Compute global indices based on tile offsets
  // int global_x = xx_heads[xx] + local_x;
  // int global_y = yy_heads[yy] + local_y;
  // int global_z = zz_heads[zz] + local_z;
 

  // first we map each (xx, yy, zz) to a block
  const int xx = blockIdx.x % xx_num;
  const int yy = (blockIdx.x % (xx_num * yy_num)) / xx_num;
  const int zz = blockIdx.x / (xx_num * yy_num);

  // // map each thread in the block to a global index
  const int tid = threadIdx.x;
  const int local_x = tid % BLX_UB;                     // X coordinate within the tile
  const int local_y = (tid / BLX_UB) % BLY_UB;     // Y coordinate within the tile
  const int local_z = tid / (BLX_UB * BLY_UB);     // Z coordinate within the tile
  const int global_x = xx_heads[xx] + local_x; // Global X coordinate
  const int global_y = yy_heads[yy] + local_y; // Global Y coordinate
  const int global_z = zz_heads[zz] + local_z; // Global Z coordinate
  const int global_idx = global_x + global_y * Nx + global_z * Nx * Ny;

  // only put E, H array into shared memory 
  __shared__ float Ex_shmem[(BLX_UB + 1) * (BLY_UB + 1) * (BLZ_UB + 1)];
  __shared__ float Ey_shmem[(BLX_UB + 1) * (BLY_UB + 1) * (BLZ_UB + 1)];
  __shared__ float Ez_shmem[(BLX_UB + 1) * (BLY_UB + 1) * (BLZ_UB + 1)];
  __shared__ float Hx_shmem[(BLX_UB + 1) * (BLY_UB + 1) * (BLZ_UB + 1)];
  __shared__ float Hy_shmem[(BLX_UB + 1) * (BLY_UB + 1) * (BLZ_UB + 1)];
  __shared__ float Hz_shmem[(BLX_UB + 1) * (BLY_UB + 1) * (BLZ_UB + 1)];

  // shared memory index
  const int shared_H_x = local_x;
  const int shared_H_y = local_y;
  const int shared_H_z = local_z;
  const int shared_H_idx = shared_H_x + shared_H_y * BLX_UB + shared_H_z * BLX_UB * BLY_UB;
  const int shared_E_x = local_x;
  const int shared_E_y = local_y;
  const int shared_E_z = local_z;
  const int shared_E_idx = shared_E_x + shared_E_y * BLX_UB + shared_E_z * BLX_UB * BLY_UB;

  // load H shmem
  if(global_x >= 0 && global_x < Nx && global_y >= 0 && global_y < Ny && global_z >= 0 && global_z < Nz) {
    Hx_shmem[shared_H_idx] = Hx[global_idx];
    Hy_shmem[shared_H_idx] = Hy[global_idx];
    Hz_shmem[shared_H_idx] = Hz[global_idx];
  }

  // load E shmem
  if(global_x >= 0 && global_x < Nx && global_y >= 0 && global_y < Ny && global_z >= 0 && global_z < Nz) {
    Ex_shmem[shared_E_idx] = Ex[global_idx];
    Ey_shmem[shared_E_idx] = Ey[global_idx];
    Ez_shmem[shared_E_idx] = Ez[global_idx];
  }

  __syncthreads();

  for(int t=0; t<BLT_UB; t++) {

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

    __syncthreads();

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

    __syncthreads();
  } 

  // store back to global memory
  if(global_x >= 1 && global_x <= Nx-2 &&
     global_y >= 1 && global_y <= Ny-2 &&
     global_z >= 1 && global_z <= Nz-2) {
  
    Ex[global_idx] = Ex_shmem[shared_E_idx];
    Ey[global_idx] = Ey_shmem[shared_E_idx];
    Ez[global_idx] = Ez_shmem[shared_E_idx];
    Hx[global_idx] = Hx_shmem[shared_H_idx];
    Hy[global_idx] = Hy_shmem[shared_H_idx];
    Hz[global_idx] = Hz_shmem[shared_H_idx];

  }


}


#endif






























