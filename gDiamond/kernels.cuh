#ifndef KERNELS_CUH
#define KERNELS_CUH

#define BLX_GPU 8
#define BLY_GPU 8
#define BLZ_GPU 8
#define BLT_GPU 4 
#define BLX_EH (BLX_GPU + 1)
#define BLY_EH (BLY_GPU + 1)
#define BLZ_EH (BLZ_GPU + 1)

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
                               int *zz_heads 
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
  if(global_x >= 0 && global_x < Nx && global_y >= 0 && global_y < Ny && global_z >= 0 && global_z < Nz) {
     Ex[global_idx] = Ex_shmem[shared_E_idx]; 
     Ey[global_idx] = Ey_shmem[shared_E_idx];
     Ez[global_idx] = Ez_shmem[shared_E_idx];
     Hx[global_idx] = Hx_shmem[shared_H_idx]; 
     Hy[global_idx] = Hy_shmem[shared_H_idx];
     Hz[global_idx] = Hz_shmem[shared_H_idx];
  }
}



#endif






























