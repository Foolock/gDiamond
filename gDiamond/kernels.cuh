#ifndef KERNELS_CUH
#define KERNELS_CUH

#define BLX_GPU 8
#define BLY_GPU 8
#define BLZ_GPU 8
#define BLT_GPU 4 

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
                               int *xx_heads, int *xx_tails,
                               int *yy_heads, int *yy_tails,
                               int *zz_heads, int *zz_tails
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
  __shared__ float Jx_shmem[BLX_GPU * BLY_GPU * BLZ_GPU];
  __shared__ float Jy_shmem[BLX_GPU * BLY_GPU * BLZ_GPU];
  __shared__ float Jz_shmem[BLX_GPU * BLY_GPU * BLZ_GPU];
  __shared__ float Mx_shmem[BLX_GPU * BLY_GPU * BLZ_GPU];
  __shared__ float My_shmem[BLX_GPU * BLY_GPU * BLZ_GPU];
  __shared__ float Mz_shmem[BLX_GPU * BLY_GPU * BLZ_GPU];


  // leave E, H, first, stencil leads to complicated shared memory load
  // __shared__ float Ex_shmem[(BLX_GPU + 2) * (BLY_GPU + 2) * (BLZ_GPU + 2)];
  // __shared__ float Ey_shmem[(BLX_GPU + 2) * (BLY_GPU + 2) * (BLZ_GPU + 2)];
  // __shared__ float Ez_shmem[(BLX_GPU + 2) * (BLY_GPU + 2) * (BLZ_GPU + 2)];
  // __shared__ float Hx_shmem[(BLX_GPU + 2) * (BLY_GPU + 2) * (BLZ_GPU + 2)];
  // __shared__ float Hy_shmem[(BLX_GPU + 2) * (BLY_GPU + 2) * (BLZ_GPU + 2)];
  // __shared__ float Hz_shmem[(BLX_GPU + 2) * (BLY_GPU + 2) * (BLZ_GPU + 2)];

  // map each thread in the block to a global index
  int tile_x_size = xx_tails[xx] - xx_heads[xx] + 1;
  int tile_y_size = yy_tails[yy] - yy_heads[yy] + 1;
  int tile_z_size = zz_tails[zz] - zz_heads[zz] + 1;

  int local_x = threadIdx.x % tile_x_size;                     // X coordinate within the tile
  int local_y = (threadIdx.x / tile_x_size) % tile_y_size;     // Y coordinate within the tile
  int local_z = threadIdx.x / (tile_x_size * tile_y_size);     // Z coordinate within the tile

  int global_x = xx_heads[xx] + local_x; // Global X coordinate
  int global_y = yy_heads[yy] + local_y; // Global Y coordinate
  int global_z = zz_heads[zz] + local_z; // Global Z coordinate

  int global_idx = (global_z * Ny + global_y) * Nx + global_x;
  int local_idx = (local_z * tile_y_size + local_y) * tile_x_size + local_x;

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
  Jx_shmem[local_idx] = Jx[global_idx];  
  Jy_shmem[local_idx] = Jy[global_idx];  
  Jz_shmem[local_idx] = Jz[global_idx];  
  Mx_shmem[local_idx] = Mx[global_idx];  
  My_shmem[local_idx] = My[global_idx];  
  Mz_shmem[local_idx] = Mz[global_idx];  

  __syncthreads();

  for(int t=0; t<BLT; t++) {

  }


}

#endif






























