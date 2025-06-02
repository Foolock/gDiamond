// mix mapping version 2, one-to-one mapping on X and Y and Z
#ifndef KERNELS_MM_VER3_CUH
#define KERNELS_MM_VER3_CUH

#include "gdiamond.hpp"

// mix mapping
#define BLT_MM_V3 2 

// one-to-one mapping in X dimension
#define NTX_MM_V3 16
#define MOUNTAIN_X_V3 16 
// valley is actually mountain top 
#define VALLEY_X_V3 (MOUNTAIN_X_V3 - 2 * (BLT_MM_V3 - 1) - 1) 

// one-to-one mapping in Y dimension
#define NTY_MM_V3 8 
#define MOUNTAIN_Y_V3 8 
#define VALLEY_Y_V3 (MOUNTAIN_Y_V3 - 2 * (BLT_MM_V3 - 1) - 1)

// one-to-many mapping in Z dimension
#define NTZ_MM_V3 8 
#define MOUNTAIN_Z_V3 8 
#define VALLEY_Z_V3 (MOUNTAIN_Z_V3 - 2 * (BLT_MM_V3 - 1) - 1)

// padding
#define LEFT_PAD_MM_V3 BLT_MM_V3
#define RIGHT_PAD_MM_V3 BLT_MM_V3

// tile size
#define BLX_MM_V3 MOUNTAIN_X_V3 
#define BLY_MM_V3 MOUNTAIN_Y_V3 
#define BLZ_MM_V3 MOUNTAIN_Z_V3 

// shared memory size
#define H_SHX_V3 (BLX_MM_V3 + 1)
#define H_SHY_V3 (BLY_MM_V3 + 1)
#define H_SHZ_V3 (BLZ_MM_V3 + 1)
#define E_SHX_V3 BLX_MM_V3
#define E_SHY_V3 BLY_MM_V3
#define E_SHZ_V3 BLZ_MM_V3

__global__ void updateEH_mix_mapping_kernel_ver3(float* Ex_pad_src, float* Ey_pad_src, float* Ez_pad_src,
                                                 float* Hx_pad_src, float* Hy_pad_src, float* Hz_pad_src,
                                                 float* Ex_pad_dst, float* Ey_pad_dst, float* Ez_pad_dst,
                                                 float* Hx_pad_dst, float* Hy_pad_dst, float* Hz_pad_dst,
                                                 float* Cax, float* Cbx,
                                                 float* Cay, float* Cby,
                                                 float* Caz, float* Cbz,
                                                 float* Dax, float* Dbx,
                                                 float* Day, float* Dby,
                                                 float* Daz, float* Dbz,
                                                 float* Jx, float* Jy, float* Jz,
                                                 float* Mx, float* My, float* Mz,
                                                 float dx,
                                                 int Nx, int Ny, int Nz,
                                                 int Nx_pad, int Ny_pad, int Nz_pad,
                                                 int xx_num, int yy_num, int zz_num,
                                                 int* xx_heads,
                                                 int* yy_heads,
                                                 int* zz_heads) {

  const unsigned int block_id = blockIdx.x;
  const unsigned int thread_id = threadIdx.x;

  const int xx = block_id % xx_num;
  const int yy = (block_id / xx_num) % yy_num;
  const int zz = block_id / (xx_num * yy_num);

  const int local_x = thread_id % NTX_MM_V3;
  const int local_y = (thread_id / NTX_MM_V3) % NTY_MM_V3;
  const int local_z = thread_id / (NTX_MM_V3 * NTY_MM_V3);

  const int global_x = xx_heads[xx] + local_x;
  const int global_y = yy_heads[yy] + local_y;
  const int global_z = zz_heads[zz] + local_z;

  const int H_shared_x = local_x + 1;
  const int H_shared_y = local_y + 1;
  const int H_shared_z = local_z + 1;

  const int E_shared_x = local_x;
  const int E_shared_y = local_y;
  const int E_shared_z = local_z;

  int global_idx;
  int H_shared_idx;
  int E_shared_idx;

  __shared__ float Hx_shmem[H_SHX_V3 * H_SHY_V3 * H_SHZ_V3];
  __shared__ float Hy_shmem[H_SHX_V3 * H_SHY_V3 * H_SHZ_V3];
  __shared__ float Hz_shmem[H_SHX_V3 * H_SHY_V3 * H_SHZ_V3];
  __shared__ float Ex_shmem[E_SHX_V3 * E_SHY_V3 * E_SHZ_V3];
  __shared__ float Ey_shmem[E_SHX_V3 * E_SHY_V3 * E_SHZ_V3];
  __shared__ float Ez_shmem[E_SHX_V3 * E_SHY_V3 * E_SHZ_V3];

  // load shared memory

  // load core ---------------------------------------------
  global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
  H_shared_idx = H_shared_x + H_shared_y * H_SHX_V3 + H_shared_z * H_SHX_V3 * H_SHY_V3;
  E_shared_idx = E_shared_x + E_shared_y * E_SHX_V3 + E_shared_z * E_SHX_V3 * E_SHY_V3;
  Hx_shmem[H_shared_idx] = Hx_pad_src[global_idx];
  Hy_shmem[H_shared_idx] = Hy_pad_src[global_idx];
  Hz_shmem[H_shared_idx] = Hz_pad_src[global_idx];
  Ex_shmem[E_shared_idx] = Ex_pad_src[global_idx];
  Ey_shmem[E_shared_idx] = Ey_pad_src[global_idx];
  Ez_shmem[E_shared_idx] = Ez_pad_src[global_idx];

  /*
    load HALO takes a lot of time, could be remove if using more threads
    when Nx = 565, Ny = 229, Nz = 229, Tx = Ty = Tz = 20, timesteps = 90
    gpu runtime (naive): 1.06061s
    gpu performance: 2514.23Mcells/s
    remove HALO directly here, producing wrong results:
    gpu runtime (mix mapping ver2): 0.949055s
    gpu performance: 2809.77Mcells/s
    if keep HALO:
    gpu runtime (mix mapping ver2): 1.00708s
    gpu performance: 2647.89Mcells/s
  */

  // load HALO ---------------------------------------------
  // H HALO
  if (local_x == 0) {
    int halo_x = 0;
    int global_x_halo = xx_heads[xx] + halo_x - 1;

    global_idx = global_x_halo + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
    H_shared_idx = halo_x + H_shared_y * H_SHX_V3 + H_shared_z * H_SHX_V3 * H_SHY_V3;

    Hx_shmem[H_shared_idx] = Hx_pad_src[global_idx];
    Hy_shmem[H_shared_idx] = Hy_pad_src[global_idx];
    Hz_shmem[H_shared_idx] = Hz_pad_src[global_idx];
  }
  if (local_y == 0) {
    int halo_y = 0;
    int global_y_halo = yy_heads[yy] + halo_y - 1;

    global_idx = global_x + global_y_halo * Nx_pad + global_z * Nx_pad * Ny_pad;
    H_shared_idx = H_shared_x + halo_y * H_SHX_V3 + H_shared_z * H_SHX_V3 * H_SHY_V3;

    Hx_shmem[H_shared_idx] = Hx_pad_src[global_idx];
    Hy_shmem[H_shared_idx] = Hy_pad_src[global_idx];
    Hz_shmem[H_shared_idx] = Hz_pad_src[global_idx];
  }
  if (local_z == 0) {
    int halo_z = 0;
    int global_z_halo = zz_heads[zz] + halo_z - 1;

    global_idx = global_x + global_y * Nx_pad + global_z_halo * Nx_pad * Ny_pad;
    H_shared_idx = H_shared_x + H_shared_y * H_SHX_V3 + halo_z * H_SHX_V3 * H_SHY_V3;

    Hx_shmem[H_shared_idx] = Hx_pad_src[global_idx];
    Hy_shmem[H_shared_idx] = Hy_pad_src[global_idx];
    Hz_shmem[H_shared_idx] = Hz_pad_src[global_idx];
  }

  // E HALO is not needed since there is no valley tile in mix mapping ver 3

  __syncthreads();

  // calculation

  // we pad all the dimension, so need to substract LEFT_PAD here to correctly access constant arrays
  global_idx = (global_x - LEFT_PAD_MM_V3) + (global_y - LEFT_PAD_MM_V3) * Nx + (global_z - LEFT_PAD_MM_V3) * Nx * Ny;
  E_shared_idx = E_shared_x + E_shared_y * E_SHX_V3 + E_shared_z * E_SHX_V3 * E_SHY_V3;
  H_shared_idx = H_shared_x + H_shared_y * H_SHX_V3 + H_shared_z * H_SHX_V3 * H_SHY_V3;

  // X head and tail is refer to padded global_x
  // same thing applys to Y and Z
  int calE_head_X, calE_tail_X;
  int calH_head_X, calH_tail_X;
  int calE_head_Y, calE_tail_Y;
  int calH_head_Y, calH_tail_Y;
  int calE_head_Z, calE_tail_Z;
  int calH_head_Z, calH_tail_Z;

  for(int t = 0; t < BLT_MM_V3; t++) {

    calE_head_X = xx_heads[xx] + t;
    calE_tail_X = xx_heads[xx] + BLX_MM_V3 - 1 - t;
    calH_head_X = calE_head_X;
    calH_tail_X = calE_tail_X - 1;

    calE_head_Y = yy_heads[yy] + t;
    calE_tail_Y = yy_heads[yy] + BLY_MM_V3 - 1 - t;
    calH_head_Y = calE_head_Y;
    calH_tail_Y = calE_tail_Y - 1;

    calE_head_Z = zz_heads[zz] + t;
    calE_tail_Z = zz_heads[zz] + BLZ_MM_V3 - 1 - t;
    calH_head_Z = calE_head_Z;
    calH_tail_Z = calE_tail_Z - 1;

    // update E
    if(global_x >= 1 + LEFT_PAD_MM_V3 && global_x <= Nx - 2 + LEFT_PAD_MM_V3 &&
       global_y >= 1 + LEFT_PAD_MM_V3 && global_y <= Ny - 2 + LEFT_PAD_MM_V3 &&
       global_z >= 1 + LEFT_PAD_MM_V3 && global_z <= Nz - 2 + LEFT_PAD_MM_V3 &&
       global_x >= calE_head_X && global_x <= calE_tail_X &&
       global_y >= calE_head_Y && global_y <= calE_tail_Y &&
       global_z >= calE_head_Z && global_z <= calE_tail_Z) {

      Ex_shmem[E_shared_idx] = Cax[global_idx] * Ex_shmem[E_shared_idx] + Cbx[global_idx] *
                ((Hz_shmem[H_shared_idx] - Hz_shmem[H_shared_idx - H_SHX_V3]) - (Hy_shmem[H_shared_idx] - Hy_shmem[H_shared_idx - H_SHX_V3 * H_SHY_V3]) - Jx[global_idx] * dx);

      Ey_shmem[E_shared_idx] = Cay[global_idx] * Ey_shmem[E_shared_idx] + Cby[global_idx] *
                ((Hx_shmem[H_shared_idx] - Hx_shmem[H_shared_idx - H_SHX_V3 * H_SHY_V3]) - (Hz_shmem[H_shared_idx] - Hz_shmem[H_shared_idx - 1]) - Jy[global_idx] * dx);

      Ez_shmem[E_shared_idx] = Caz[global_idx] * Ez_shmem[E_shared_idx] + Cbz[global_idx] *
                ((Hy_shmem[H_shared_idx] - Hy_shmem[H_shared_idx - 1]) - (Hx_shmem[H_shared_idx] - Hx_shmem[H_shared_idx - H_SHX_V3]) - Jz[global_idx] * dx);
    }

    __syncthreads();
  __syncthreads();
  __syncthreads();

    // update H 
    if(global_x >= 1 + LEFT_PAD_MM_V3 && global_x <= Nx - 2 + LEFT_PAD_MM_V3 &&
       global_y >= 1 + LEFT_PAD_MM_V3 && global_y <= Ny - 2 + LEFT_PAD_MM_V3 &&
       global_z >= 1 + LEFT_PAD_MM_V3 && global_z <= Nz - 2 + LEFT_PAD_MM_V3 &&
       global_x >= calH_head_X && global_x <= calH_tail_X &&
       global_y >= calH_head_Y && global_y <= calH_tail_Y &&
       global_z >= calH_head_Z && global_z <= calH_tail_Z) {

      Hx_shmem[H_shared_idx] = Dax[global_idx] * Hx_shmem[H_shared_idx] + Dbx[global_idx] *
                ((Ey_shmem[E_shared_idx + E_SHX_V3 * E_SHY_V3] - Ey_shmem[E_shared_idx]) - (Ez_shmem[E_shared_idx + E_SHX_V3] - Ez_shmem[E_shared_idx]) - Mx[global_idx] * dx);

      Hy_shmem[H_shared_idx] = Day[global_idx] * Hy_shmem[H_shared_idx] + Dby[global_idx] *
                ((Ez_shmem[E_shared_idx + 1] - Ez_shmem[E_shared_idx]) - (Ex_shmem[E_shared_idx + E_SHX_V3 * E_SHY_V3] - Ex_shmem[E_shared_idx]) - My[global_idx] * dx);

      Hz_shmem[H_shared_idx] = Daz[global_idx] * Hz_shmem[H_shared_idx] + Dbz[global_idx] *
                ((Ex_shmem[E_shared_idx + E_SHX_V3] - Ex_shmem[E_shared_idx]) - (Ey_shmem[E_shared_idx + 1] - Ey_shmem[E_shared_idx]) - Mz[global_idx] * dx);
    }

    __syncthreads();
  }

  // store back to global memory

  // no need to recalculate H_shared_idx, E_shared_idx
  global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;

  // X head and tail is refer to padded global_x
  // same thing applys to Y and Z
  const int storeE_head_X = xx_heads[xx] + LEFT_PAD_MM_V3 - 1;
  const int storeE_tail_X = storeE_head_X + VALLEY_X_V3;
  const int storeH_head_X = storeE_head_X;
  const int storeH_tail_X = storeE_tail_X - 1;
  const int storeE_head_Y = yy_heads[yy] + LEFT_PAD_MM_V3 - 1;
  const int storeE_tail_Y = storeE_head_Y + VALLEY_Y_V3;
  const int storeH_head_Y = storeE_head_Y;
  const int storeH_tail_Y = storeE_tail_Y - 1;
  const int storeE_head_Z = zz_heads[zz] + LEFT_PAD_MM_V3 - 1;
  const int storeE_tail_Z = storeE_head_Z + VALLEY_Z_V3;
  const int storeH_head_Z = storeE_head_Z;
  const int storeH_tail_Z = storeE_tail_Z - 1;

  if(global_x >= 1 + LEFT_PAD_MM_V3 && global_x <= Nx - 2 + LEFT_PAD_MM_V3 &&
     global_y >= 1 + LEFT_PAD_MM_V3 && global_y <= Ny - 2 + LEFT_PAD_MM_V3 &&
     global_z >= 1 + LEFT_PAD_MM_V3 && global_z <= Nz - 2 + LEFT_PAD_MM_V3 &&
     global_x >= storeH_head_X && global_x <= storeH_tail_X &&
     global_y >= storeH_head_Y && global_y <= storeH_tail_Y &&
     global_z >= storeH_head_Z && global_z <= storeH_tail_Z) {

    Hx_pad_dst[global_idx] = Hx_shmem[H_shared_idx];
    Hy_pad_dst[global_idx] = Hy_shmem[H_shared_idx];
    Hz_pad_dst[global_idx] = Hz_shmem[H_shared_idx];
  }

  if(global_x >= 1 + LEFT_PAD_MM_V3 && global_x <= Nx - 2 + LEFT_PAD_MM_V3 &&
     global_y >= 1 + LEFT_PAD_MM_V3 && global_y <= Ny - 2 + LEFT_PAD_MM_V3 &&
     global_z >= 1 + LEFT_PAD_MM_V3 && global_z <= Nz - 2 + LEFT_PAD_MM_V3 &&
     global_x >= storeE_head_X && global_x <= storeE_tail_X &&
     global_y >= storeE_head_Y && global_y <= storeE_tail_Y &&
     global_z >= storeE_head_Z && global_z <= storeE_tail_Z) {

    Ex_pad_dst[global_idx] = Ex_shmem[E_shared_idx];
    Ey_pad_dst[global_idx] = Ey_shmem[E_shared_idx];
    Ez_pad_dst[global_idx] = Ez_shmem[E_shared_idx];
  }
}

template<int t>
__device__ void timestep_body(
    const int global_idx, const int H_shared_idx, const int E_shared_idx,
    const int global_x, const int global_y, const int global_z,
    const int Nx, const int Ny, const int Nz,
    const int* xx_heads, const int* yy_heads, const int* zz_heads,
    const int xx, const int yy, const int zz,
    float dx,
    const float* Cax, const float* Cbx,
    const float* Cay, const float* Cby,
    const float* Caz, const float* Cbz,
    const float* Dax, const float* Dbx,
    const float* Day, const float* Dby,
    const float* Daz, const float* Dbz,
    const float* Jx, const float* Jy, const float* Jz,
    const float* Mx, const float* My, const float* Mz,
    float* Ex_shmem, float* Ey_shmem, float* Ez_shmem,
    float* Hx_shmem, float* Hy_shmem, float* Hz_shmem
) {

  constexpr int calE_head_X = t;
  constexpr int calE_tail_X = BLX_MM_V3 - 1 - t;
  constexpr int calH_head_X = calE_head_X;
  constexpr int calH_tail_X = calE_tail_X - 1;

  constexpr int calE_head_Y = t;
  constexpr int calE_tail_Y = BLY_MM_V3 - 1 - t;
  constexpr int calH_head_Y = calE_head_Y;
  constexpr int calH_tail_Y = calE_tail_Y - 1;

  constexpr int calE_head_Z = t;
  constexpr int calE_tail_Z = BLZ_MM_V3 - 1 - t;
  constexpr int calH_head_Z = calE_head_Z;
  constexpr int calH_tail_Z = calE_tail_Z - 1;

  // update E
  if(global_x >= 1 + LEFT_PAD_MM_V3 && global_x <= Nx - 2 + LEFT_PAD_MM_V3 &&
     global_y >= 1 + LEFT_PAD_MM_V3 && global_y <= Ny - 2 + LEFT_PAD_MM_V3 &&
     global_z >= 1 + LEFT_PAD_MM_V3 && global_z <= Nz - 2 + LEFT_PAD_MM_V3 &&
     global_x >= xx_heads[xx] + calE_head_X && global_x <= xx_heads[xx] + calE_tail_X &&
     global_y >= yy_heads[yy] + calE_head_Y && global_y <= yy_heads[yy] + calE_tail_Y &&
     global_z >= zz_heads[zz] + calE_head_Z && global_z <= zz_heads[zz] + calE_tail_Z) {

    Ex_shmem[E_shared_idx] = Cax[global_idx] * Ex_shmem[E_shared_idx] + Cbx[global_idx] *
              ((Hz_shmem[H_shared_idx] - Hz_shmem[H_shared_idx - H_SHX_V3]) - (Hy_shmem[H_shared_idx] - Hy_shmem[H_shared_idx - H_SHX_V3 * H_SHY_V3]) - Jx[global_idx] * dx);

    Ey_shmem[E_shared_idx] = Cay[global_idx] * Ey_shmem[E_shared_idx] + Cby[global_idx] *
              ((Hx_shmem[H_shared_idx] - Hx_shmem[H_shared_idx - H_SHX_V3 * H_SHY_V3]) - (Hz_shmem[H_shared_idx] - Hz_shmem[H_shared_idx - 1]) - Jy[global_idx] * dx);

    Ez_shmem[E_shared_idx] = Caz[global_idx] * Ez_shmem[E_shared_idx] + Cbz[global_idx] *
              ((Hy_shmem[H_shared_idx] - Hy_shmem[H_shared_idx - 1]) - (Hx_shmem[H_shared_idx] - Hx_shmem[H_shared_idx - H_SHX_V3]) - Jz[global_idx] * dx);
  }

  __syncthreads();

  // update H
  if(global_x >= 1 + LEFT_PAD_MM_V3 && global_x <= Nx - 2 + LEFT_PAD_MM_V3 &&
     global_y >= 1 + LEFT_PAD_MM_V3 && global_y <= Ny - 2 + LEFT_PAD_MM_V3 &&
     global_z >= 1 + LEFT_PAD_MM_V3 && global_z <= Nz - 2 + LEFT_PAD_MM_V3 &&
     global_x >= xx_heads[xx] + calH_head_X && global_x <= xx_heads[xx] + calH_tail_X &&
     global_y >= yy_heads[yy] + calH_head_Y && global_y <= yy_heads[yy] + calH_tail_Y &&
     global_z >= zz_heads[zz] + calH_head_Z && global_z <= zz_heads[zz] + calH_tail_Z) {

    Hx_shmem[H_shared_idx] = Dax[global_idx] * Hx_shmem[H_shared_idx] + Dbx[global_idx] *
              ((Ey_shmem[E_shared_idx + E_SHX_V3 * E_SHY_V3] - Ey_shmem[E_shared_idx]) - (Ez_shmem[E_shared_idx + E_SHX_V3] - Ez_shmem[E_shared_idx]) - Mx[global_idx] * dx);

    Hy_shmem[H_shared_idx] = Day[global_idx] * Hy_shmem[H_shared_idx] + Dby[global_idx] *
              ((Ez_shmem[E_shared_idx + 1] - Ez_shmem[E_shared_idx]) - (Ex_shmem[E_shared_idx + E_SHX_V3 * E_SHY_V3] - Ex_shmem[E_shared_idx]) - My[global_idx] * dx);

    Hz_shmem[H_shared_idx] = Daz[global_idx] * Hz_shmem[H_shared_idx] + Dbz[global_idx] *
              ((Ex_shmem[E_shared_idx + E_SHX_V3] - Ex_shmem[E_shared_idx]) - (Ey_shmem[E_shared_idx + 1] - Ey_shmem[E_shared_idx]) - Mz[global_idx] * dx);
  }

  __syncthreads();
}

template<int t>
__device__ void unroll_timesteps(
    const int global_idx, const int H_shared_idx, const int E_shared_idx,
    const int global_x, const int global_y, const int global_z,
    const int Nx, const int Ny, const int Nz,
    const int* xx_heads, const int* yy_heads, const int* zz_heads,
    const int xx, const int yy, const int zz,
    float dx,
    const float* Cax, const float* Cbx,
    const float* Cay, const float* Cby,
    const float* Caz, const float* Cbz,
    const float* Dax, const float* Dbx,
    const float* Day, const float* Dby,
    const float* Daz, const float* Dbz,
    const float* Jx, const float* Jy, const float* Jz,
    const float* Mx, const float* My, const float* Mz,
    float* Ex_shmem, float* Ey_shmem, float* Ez_shmem,
    float* Hx_shmem, float* Hy_shmem, float* Hz_shmem
) {
    timestep_body<t>(
    global_idx, H_shared_idx, E_shared_idx,
    global_x, global_y, global_z,
    Nx, Ny, Nz,
    xx_heads, yy_heads, zz_heads,
    xx, yy, zz,
    dx,
    Cax, Cbx,
    Cay, Cby,
    Caz, Cbz,
    Dax, Dbx,
    Day, Dby,
    Daz, Dbz,
    Jx, Jy, Jz,
    Mx, My, Mz,
    Ex_shmem, Ey_shmem, Ez_shmem,
    Hx_shmem, Hy_shmem, Hz_shmem
    );
    if constexpr (t + 1 < BLT_MM_V3) {
        unroll_timesteps<t + 1>(
        global_idx, H_shared_idx, E_shared_idx,
        global_x, global_y, global_z,
        Nx, Ny, Nz,
        xx_heads, yy_heads, zz_heads,
        xx, yy, zz,
        dx,
        Cax, Cbx,
        Cay, Cby,
        Caz, Cbz,
        Dax, Dbx,
        Day, Dby,
        Daz, Dbz,
        Jx, Jy, Jz,
        Mx, My, Mz,
        Ex_shmem, Ey_shmem, Ez_shmem,
        Hx_shmem, Hy_shmem, Hz_shmem
        );
    }
}

__global__ void updateEH_mix_mapping_kernel_ver3_unroll(float* Ex_pad_src, float* Ey_pad_src, float* Ez_pad_src,
                                                        float* Hx_pad_src, float* Hy_pad_src, float* Hz_pad_src,
                                                        float* Ex_pad_dst, float* Ey_pad_dst, float* Ez_pad_dst,
                                                        float* Hx_pad_dst, float* Hy_pad_dst, float* Hz_pad_dst,
                                                        float* Cax, float* Cbx,
                                                        float* Cay, float* Cby,
                                                        float* Caz, float* Cbz,
                                                        float* Dax, float* Dbx,
                                                        float* Day, float* Dby,
                                                        float* Daz, float* Dbz,
                                                        float* Jx, float* Jy, float* Jz,
                                                        float* Mx, float* My, float* Mz,
                                                        float dx,
                                                        int Nx, int Ny, int Nz,
                                                        int Nx_pad, int Ny_pad, int Nz_pad,
                                                        int xx_num, int yy_num, int zz_num,
                                                        int* xx_heads,
                                                        int* yy_heads,
                                                        int* zz_heads) {

  const unsigned int block_id = blockIdx.x;
  const unsigned int thread_id = threadIdx.x;

  const int xx = block_id % xx_num;
  const int yy = (block_id / xx_num) % yy_num;
  const int zz = block_id / (xx_num * yy_num);

  const int local_x = thread_id % NTX_MM_V3;
  const int local_y = (thread_id / NTX_MM_V3) % NTY_MM_V3;
  const int local_z = thread_id / (NTX_MM_V3 * NTY_MM_V3);

  const int global_x = xx_heads[xx] + local_x;
  const int global_y = yy_heads[yy] + local_y;
  const int global_z = zz_heads[zz] + local_z;

  const int H_shared_x = local_x + 1;
  const int H_shared_y = local_y + 1;
  const int H_shared_z = local_z + 1;

  const int E_shared_x = local_x;
  const int E_shared_y = local_y;
  const int E_shared_z = local_z;

  int global_idx;
  int H_shared_idx;
  int E_shared_idx;

  __shared__ float Hx_shmem[H_SHX_V3 * H_SHY_V3 * H_SHZ_V3];
  __shared__ float Hy_shmem[H_SHX_V3 * H_SHY_V3 * H_SHZ_V3];
  __shared__ float Hz_shmem[H_SHX_V3 * H_SHY_V3 * H_SHZ_V3];
  __shared__ float Ex_shmem[E_SHX_V3 * E_SHY_V3 * E_SHZ_V3];
  __shared__ float Ey_shmem[E_SHX_V3 * E_SHY_V3 * E_SHZ_V3];
  __shared__ float Ez_shmem[E_SHX_V3 * E_SHY_V3 * E_SHZ_V3];

  // load shared memory

  // load core ---------------------------------------------
  global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
  H_shared_idx = H_shared_x + H_shared_y * H_SHX_V3 + H_shared_z * H_SHX_V3 * H_SHY_V3;
  E_shared_idx = E_shared_x + E_shared_y * E_SHX_V3 + E_shared_z * E_SHX_V3 * E_SHY_V3;
  Hx_shmem[H_shared_idx] = Hx_pad_src[global_idx];
  Hy_shmem[H_shared_idx] = Hy_pad_src[global_idx];
  Hz_shmem[H_shared_idx] = Hz_pad_src[global_idx];
  Ex_shmem[E_shared_idx] = Ex_pad_src[global_idx];
  Ey_shmem[E_shared_idx] = Ey_pad_src[global_idx];
  Ez_shmem[E_shared_idx] = Ez_pad_src[global_idx];

  /*
    load HALO takes a lot of time, could be remove if using more threads
    when Nx = 565, Ny = 229, Nz = 229, Tx = Ty = Tz = 20, timesteps = 90
    gpu runtime (naive): 1.06061s
    gpu performance: 2514.23Mcells/s
    remove HALO directly here, producing wrong results:
    gpu runtime (mix mapping ver2): 0.949055s
    gpu performance: 2809.77Mcells/s
    if keep HALO:
    gpu runtime (mix mapping ver2): 1.00708s
    gpu performance: 2647.89Mcells/s
  */

  // load HALO ---------------------------------------------
  // H HALO
  if (local_x == 0) {
    int halo_x = 0;
    int global_x_halo = xx_heads[xx] + halo_x - 1;

    global_idx = global_x_halo + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
    H_shared_idx = halo_x + H_shared_y * H_SHX_V3 + H_shared_z * H_SHX_V3 * H_SHY_V3;

    Hx_shmem[H_shared_idx] = Hx_pad_src[global_idx];
    Hy_shmem[H_shared_idx] = Hy_pad_src[global_idx];
    Hz_shmem[H_shared_idx] = Hz_pad_src[global_idx];
  }
  if (local_y == 0) {
    int halo_y = 0;
    int global_y_halo = yy_heads[yy] + halo_y - 1;

    global_idx = global_x + global_y_halo * Nx_pad + global_z * Nx_pad * Ny_pad;
    H_shared_idx = H_shared_x + halo_y * H_SHX_V3 + H_shared_z * H_SHX_V3 * H_SHY_V3;

    Hx_shmem[H_shared_idx] = Hx_pad_src[global_idx];
    Hy_shmem[H_shared_idx] = Hy_pad_src[global_idx];
    Hz_shmem[H_shared_idx] = Hz_pad_src[global_idx];
  }
  if (local_z == 0) {
    int halo_z = 0;
    int global_z_halo = zz_heads[zz] + halo_z - 1;

    global_idx = global_x + global_y * Nx_pad + global_z_halo * Nx_pad * Ny_pad;
    H_shared_idx = H_shared_x + H_shared_y * H_SHX_V3 + halo_z * H_SHX_V3 * H_SHY_V3;

    Hx_shmem[H_shared_idx] = Hx_pad_src[global_idx];
    Hy_shmem[H_shared_idx] = Hy_pad_src[global_idx];
    Hz_shmem[H_shared_idx] = Hz_pad_src[global_idx];
  }

  // E HALO is not needed since there is no valley tile in mix mapping ver 3

  __syncthreads();

  // calculation

  // we pad all the dimension, so need to substract LEFT_PAD here to correctly access constant arrays
  global_idx = (global_x - LEFT_PAD_MM_V3) + (global_y - LEFT_PAD_MM_V3) * Nx + (global_z - LEFT_PAD_MM_V3) * Nx * Ny;
  E_shared_idx = E_shared_x + E_shared_y * E_SHX_V3 + E_shared_z * E_SHX_V3 * E_SHY_V3;
  H_shared_idx = H_shared_x + H_shared_y * H_SHX_V3 + H_shared_z * H_SHX_V3 * H_SHY_V3;
  unroll_timesteps<0>(
  global_idx, H_shared_idx, E_shared_idx,
  global_x, global_y, global_z,
  Nx, Ny, Nz,
  xx_heads, yy_heads, zz_heads,
  xx, yy, zz,
  dx,
  Cax, Cbx,
  Cay, Cby,
  Caz, Cbz,
  Dax, Dbx,
  Day, Dby,
  Daz, Dbz,
  Jx, Jy, Jz,
  Mx, My, Mz,
  Ex_shmem, Ey_shmem, Ez_shmem,
  Hx_shmem, Hy_shmem, Hz_shmem
  );

  // store back to global memory

  // no need to recalculate H_shared_idx, E_shared_idx
  global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;

  // X head and tail is refer to unpadded global_x
  // same thing applys to Y and Z
  const int storeE_head_X = xx_heads[xx] + LEFT_PAD_MM_V3 - 1;
  const int storeE_tail_X = storeE_head_X + VALLEY_X_V3;
  const int storeH_head_X = storeE_head_X;
  const int storeH_tail_X = storeE_tail_X - 1;
  const int storeE_head_Y = yy_heads[yy] + LEFT_PAD_MM_V3 - 1;
  const int storeE_tail_Y = storeE_head_Y + VALLEY_Y_V3;
  const int storeH_head_Y = storeE_head_Y;
  const int storeH_tail_Y = storeE_tail_Y - 1;
  const int storeE_head_Z = zz_heads[zz] + LEFT_PAD_MM_V3 - 1;
  const int storeE_tail_Z = storeE_head_Z + VALLEY_Z_V3;
  const int storeH_head_Z = storeE_head_Z;
  const int storeH_tail_Z = storeE_tail_Z - 1;

  if(global_x >= 1 + LEFT_PAD_MM_V3 && global_x <= Nx - 2 + LEFT_PAD_MM_V3 &&
     global_y >= 1 + LEFT_PAD_MM_V3 && global_y <= Ny - 2 + LEFT_PAD_MM_V3 &&
     global_z >= 1 + LEFT_PAD_MM_V3 && global_z <= Nz - 2 + LEFT_PAD_MM_V3 &&
     global_x >= storeH_head_X && global_x <= storeH_tail_X &&
     global_y >= storeH_head_Y && global_y <= storeH_tail_Y &&
     global_z >= storeH_head_Z && global_z <= storeH_tail_Z) {

    Hx_pad_dst[global_idx] = Hx_shmem[H_shared_idx];
    Hy_pad_dst[global_idx] = Hy_shmem[H_shared_idx];
    Hz_pad_dst[global_idx] = Hz_shmem[H_shared_idx];
  }

  if(global_x >= 1 + LEFT_PAD_MM_V3 && global_x <= Nx - 2 + LEFT_PAD_MM_V3 &&
     global_y >= 1 + LEFT_PAD_MM_V3 && global_y <= Ny - 2 + LEFT_PAD_MM_V3 &&
     global_z >= 1 + LEFT_PAD_MM_V3 && global_z <= Nz - 2 + LEFT_PAD_MM_V3 &&
     global_x >= storeE_head_X && global_x <= storeE_tail_X &&
     global_y >= storeE_head_Y && global_y <= storeE_tail_Y &&
     global_z >= storeE_head_Z && global_z <= storeE_tail_Z) {

    Ex_pad_dst[global_idx] = Ex_shmem[E_shared_idx];
    Ey_pad_dst[global_idx] = Ey_shmem[E_shared_idx];
    Ez_pad_dst[global_idx] = Ez_shmem[E_shared_idx];
  }

}



#endif


































