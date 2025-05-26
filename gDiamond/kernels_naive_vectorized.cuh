#ifndef KERNELS_NAIVE_VECTORIZED_CUH
#define KERNELS_NAIVE_VECTORIZED_CUH

#include "gdiamond.hpp"

/*
  this implementation assume X, Y, Z dimension is a multiple of 4
*/

#define FLOAT4(ptr) (reinterpret_cast<float4*>(&(ptr))[0])

__global__ void updateE_3Dmap_fix_vectorized(float * Ex, float * Ey, float * Ez,
                        float * Hx, float * Hy, float * Hz,
                        float * Cax, float * Cbx, float * Cay,
                        float * Cby, float * Caz, float * Cbz,
                        float * Jx, float * Jy, float * Jz,
                        float dx, int Nx, int Ny, int Nz)
{
  unsigned int base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4; 

  // boundary check
  if(base_idx + 3 >= Nx * Ny * Nz) return; 

  // now each base_idx corresponds to the start index of 4 continuous floats
  
  int i = base_idx % Nx;
  int j = (base_idx / Nx) % Ny;
  int k = base_idx / (Nx * Ny);

  // only need to check boundaries for X dimension, 
  if(j < 1 || j >= Ny - 1 || k < 1 || k >= Nz - 1) return; 

  float4 Ex_f4 = FLOAT4(Ex[base_idx]);
  float4 Ey_f4 = FLOAT4(Ey[base_idx]);
  float4 Ez_f4 = FLOAT4(Ez[base_idx]);

  float4 Cax_f4 = FLOAT4(Cax[base_idx]);
  float4 Cay_f4 = FLOAT4(Cay[base_idx]);
  float4 Caz_f4 = FLOAT4(Caz[base_idx]);
  float4 Cbx_f4 = FLOAT4(Cbx[base_idx]);
  float4 Cby_f4 = FLOAT4(Cby[base_idx]);
  float4 Cbz_f4 = FLOAT4(Cbz[base_idx]);

  float4 Jx_f4 = FLOAT4(Jx[base_idx]);
  float4 Jy_f4 = FLOAT4(Jy[base_idx]);
  float4 Jz_f4 = FLOAT4(Jz[base_idx]);

  // center
  float4 Hx_f4_center = FLOAT4(Hx[base_idx]); 
  float4 Hy_f4_center = FLOAT4(Hy[base_idx]);
  float4 Hz_f4_center = FLOAT4(Hz[base_idx]);

  // y - 1 and z - 1 for Hx
  float4 Hx_f4_y = FLOAT4(Hx[base_idx - Nx]); 
  float4 Hx_f4_z = FLOAT4(Hx[base_idx - Nx * Ny]);

  // z - 1 for Hy
  // x - 1 is mis-aligned, thus just a float, but it can be found at Hx_f4_center, so less global memory access here
  float4 Hy_f4_z = FLOAT4(Hy[base_idx - Nx * Ny]);
  float Hy_x = Hy[base_idx - 1];

  // y - 1 for Hz
  // x - 1 is mis-aligned, same thing as above
  float4 Hz_f4_y = FLOAT4(Hz[base_idx - Nx]);
  float Hz_x = Hz[base_idx - 1];

  // handcode calculation for 4 floats
  Ex_f4.x = (i == 0)? 0 : Cax_f4.x * Ex_f4.x + Cbx_f4.x * ((Hz_f4_center.x - Hz_f4_y.x) - (Hy_f4_center.x - Hy_f4_z.x) - Jx_f4.x * dx); 
  Ex_f4.y = Cax_f4.y * Ex_f4.y + Cbx_f4.y * ((Hz_f4_center.y - Hz_f4_y.y) - (Hy_f4_center.y - Hy_f4_z.y) - Jx_f4.y * dx); 
  Ex_f4.z = Cax_f4.z * Ex_f4.z + Cbx_f4.z * ((Hz_f4_center.z - Hz_f4_y.z) - (Hy_f4_center.z - Hy_f4_z.z) - Jx_f4.z * dx); 
  Ex_f4.w = (i + 3 == Nx - 1)? 0 : Cax_f4.w * Ex_f4.w + Cbx_f4.w * ((Hz_f4_center.w - Hz_f4_y.w) - (Hy_f4_center.w - Hy_f4_z.w) - Jx_f4.w * dx); 

  Ey_f4.x = (i == 0)? 0 : Cay_f4.x * Ey_f4.x + Cby_f4.x * ((Hx_f4_center.x - Hx_f4_z.x) - (Hz_f4_center.x - Hz_x) - Jy_f4.x * dx);
  Ey_f4.y = Cay_f4.y * Ey_f4.y + Cby_f4.y * ((Hx_f4_center.y - Hx_f4_z.y) - (Hz_f4_center.y - Hz_f4_center.x) - Jy_f4.y * dx);
  Ey_f4.z = Cay_f4.z * Ey_f4.z + Cby_f4.z * ((Hx_f4_center.z - Hx_f4_z.z) - (Hz_f4_center.z - Hz_f4_center.y) - Jy_f4.z * dx);
  Ey_f4.w = (i + 3 == Nx - 1)? 0 : Cay_f4.w * Ey_f4.w + Cby_f4.w * ((Hx_f4_center.w - Hx_f4_z.w) - (Hz_f4_center.w - Hz_f4_center.z) - Jy_f4.w * dx);

  Ez_f4.x = (i == 0)? 0 : Caz_f4.x * Ez_f4.x + Cbz_f4.x * ((Hy_f4_center.x - Hy_x) - (Hx_f4_center.x - Hx_f4_y.x) - Jz_f4.x * dx);
  Ez_f4.y = Caz_f4.y * Ez_f4.y + Cbz_f4.y * ((Hy_f4_center.y - Hy_f4_center.x) - (Hx_f4_center.y - Hx_f4_y.y) - Jz_f4.y * dx);
  Ez_f4.z = Caz_f4.z * Ez_f4.z + Cbz_f4.z * ((Hy_f4_center.z - Hy_f4_center.y) - (Hx_f4_center.z - Hx_f4_y.z) - Jz_f4.z * dx);
  Ez_f4.w = (i + 3 == Nx - 1)? 0 : Caz_f4.w * Ez_f4.w + Cbz_f4.w * ((Hy_f4_center.w - Hy_f4_center.z) - (Hx_f4_center.w - Hx_f4_y.w) - Jz_f4.w * dx);

  // store back to global memory
  FLOAT4(Ex[base_idx]) = Ex_f4;
  FLOAT4(Ey[base_idx]) = Ey_f4;
  FLOAT4(Ez[base_idx]) = Ez_f4;

}

__global__ void updateH_3Dmap_fix_vectorized(float * Ex, float * Ey, float * Ez,
                        float * Hx, float * Hy, float * Hz,
                        float * Dax, float * Dbx,
                        float * Day, float * Dby,
                        float * Daz, float * Dbz,
                        float * Mx, float * My, float * Mz,
                        float dx, int Nx, int Ny, int Nz)
{
  unsigned int base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4; 

  // boundary check
  if(base_idx + 3 >= Nx * Ny * Nz) return; 

  // now each base_idx corresponds to the start index of 4 continuous floats
  
  int i = base_idx % Nx;
  int j = (base_idx / Nx) % Ny;
  int k = base_idx / (Nx * Ny);

  // only need to check boundaries for X dimension, 
  if(j < 1 || j >= Ny - 1 || k < 1 || k >= Nz - 1) return; 

  float4 Hx_f4 = FLOAT4(Hx[base_idx]);
  float4 Hy_f4 = FLOAT4(Hy[base_idx]);
  float4 Hz_f4 = FLOAT4(Hz[base_idx]);

  float4 Dax_f4 = FLOAT4(Dax[base_idx]);
  float4 Day_f4 = FLOAT4(Day[base_idx]);
  float4 Daz_f4 = FLOAT4(Daz[base_idx]);
  float4 Dbx_f4 = FLOAT4(Dbx[base_idx]);
  float4 Dby_f4 = FLOAT4(Dby[base_idx]);
  float4 Dbz_f4 = FLOAT4(Dbz[base_idx]);

  float4 Mx_f4 = FLOAT4(Mx[base_idx]);
  float4 My_f4 = FLOAT4(My[base_idx]);
  float4 Mz_f4 = FLOAT4(Mz[base_idx]);

  // center
  float4 Ex_f4_center = FLOAT4(Ex[base_idx]); 
  float4 Ey_f4_center = FLOAT4(Ey[base_idx]);
  float4 Ez_f4_center = FLOAT4(Ez[base_idx]);

  // y + 1 and z + 1 for Ex
  float4 Ex_f4_y = FLOAT4(Ex[base_idx + Nx]); 
  float4 Ex_f4_z = FLOAT4(Ex[base_idx + Nx * Ny]);

  // z + 1 for Ey
  // x + 1 is mis-aligned, thus just a float, but it can be found at Ex_f4_center, so less global memory access here
  float4 Ey_f4_z = FLOAT4(Ey[base_idx + Nx * Ny]);
  float Ey_x = Ey[base_idx + 4];

  // y + 1 for Ez
  // x + 1 is mis-aligned, same thing as above
  float4 Ez_f4_y = FLOAT4(Ez[base_idx + Nx]);
  float Ez_x = Ez[base_idx + 4];

  // handcode calculation for 4 floats
  Hx_f4.x = (i == 0)? 0 : Dax_f4.x * Hx_f4.x + Dbx_f4.x * ((Ey_f4_z.x - Ey_f4_center.x) - (Ez_f4_y.x - Ez_f4_center.x) - Mx_f4.x * dx);
  Hx_f4.y = Dax_f4.y * Hx_f4.y + Dbx_f4.y * ((Ey_f4_z.y - Ey_f4_center.y) - (Ez_f4_y.y - Ez_f4_center.y) - Mx_f4.y * dx);
  Hx_f4.z = Dax_f4.z * Hx_f4.z + Dbx_f4.z * ((Ey_f4_z.z - Ey_f4_center.z) - (Ez_f4_y.z - Ez_f4_center.z) - Mx_f4.z * dx);
  Hx_f4.w = (i + 3 == Nx - 1)? 0 : Dax_f4.w * Hx_f4.w + Dbx_f4.w * ((Ey_f4_z.w - Ey_f4_center.w) - (Ez_f4_y.w - Ez_f4_center.w) - Mx_f4.w * dx);

  Hy_f4.x = (i == 0)? 0 : Day_f4.x * Hy_f4.x + Dby_f4.x * ((Ez_f4_center.y - Ez_f4_center.x) - (Ex_f4_z.x - Ex_f4_center.x) - My_f4.x * dx);
  Hy_f4.y = Day_f4.y * Hy_f4.y + Dby_f4.y * ((Ez_f4_center.z - Ez_f4_center.y) - (Ex_f4_z.y - Ex_f4_center.y) - My_f4.y * dx);
  Hy_f4.z = Day_f4.z * Hy_f4.z + Dby_f4.z * ((Ez_f4_center.w - Ez_f4_center.z) - (Ex_f4_z.z - Ex_f4_center.z) - My_f4.z * dx);
  Hy_f4.w = (i + 3 == Nx - 1)? 0 : Day_f4.w * Hy_f4.w + Dby_f4.w * ((Ez_x - Ez_f4_center.w) - (Ex_f4_z.w - Ex_f4_center.w) - My_f4.w * dx);

  Hz_f4.x = (i == 0)? 0 : Daz_f4.x * Hz_f4.x + Dbz_f4.x * ((Ex_f4_y.x - Ex_f4_center.x) - (Ey_f4_center.y - Ey_f4_center.x) - Mz_f4.x * dx); 
  Hz_f4.y = Daz_f4.y * Hz_f4.y + Dbz_f4.y * ((Ex_f4_y.y - Ex_f4_center.y) - (Ey_f4_center.z - Ey_f4_center.y) - Mz_f4.y * dx);
  Hz_f4.z = Daz_f4.z * Hz_f4.z + Dbz_f4.z * ((Ex_f4_y.z - Ex_f4_center.z) - (Ey_f4_center.w - Ey_f4_center.z) - Mz_f4.z * dx);
  Hz_f4.w = Daz_f4.w * Hz_f4.w + Dbz_f4.w * ((Ex_f4_y.w - Ex_f4_center.w) - (Ey_x - Ey_f4_center.w) - Mz_f4.w * dx);

  // store back to global memory
  FLOAT4(Hx[base_idx]) = Hx_f4;
  FLOAT4(Hy[base_idx]) = Hy_f4;
  FLOAT4(Hz[base_idx]) = Hz_f4;

}

#endif
























