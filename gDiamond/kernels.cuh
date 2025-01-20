#ifndef KERNELS_CUH
#define KERNELS_CUH

__global__ void updateE_2Dmap(float * __restrict__ Ex, float * __restrict__ Ey, float * __restrict__ Ez,
                        float * __restrict__ Hx, float * __restrict__ Hy, float * __restrict__ Hz,
                        float * __restrict__ Cax, float * __restrict__ Cbx, float * __restrict__ Cay,
                        float * __restrict__ Cby, float * __restrict__ Caz, float * __restrict__ Cbz,
                        float * __restrict__ Jx, float * __restrict__ Jy, float * __restrict__ Jz,
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

__global__ void updateH_2Dmap(float * __restrict__ Ex, float * __restrict__ Ey, float * __restrict__ Ez,
                        float * __restrict__ Hx, float * __restrict__ Hy, float * __restrict__ Hz,
                        float * __restrict__ Dax, float * __restrict__ Dbx,
                        float * __restrict__ Day, float * __restrict__ Dby,
                        float * __restrict__ Daz, float * __restrict__ Dbz,
                        float * __restrict__ Mx, float * __restrict__ My, float * __restrict__ Mz,
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

#endif
