#ifndef KERNELS_NAIVE_VECTORIZED_CUH
#define KERNELS_NAIVE_VECTORIZED_CUH

#include "gdiamond.hpp"

// pad X, Y, Z dimension by 4 to simplify kernel control float
#define LEFT_PAD 4
#define RIGHT_PAD 4

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
  
  int i, j, k;
  // the first float
  i = base_idx % Nx;
  j = (base_idx / Nx) % Ny;
  k = base_idx / (Nx * Ny);

  // 
  
}

__global__ void updateH_3Dmap_fix_vectorized(float * Ex, float * Ey, float * Ez,
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

#endif


