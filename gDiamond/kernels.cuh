#ifndef KERNELS_CUH
#define KERNELS_CUH

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

#endif






























