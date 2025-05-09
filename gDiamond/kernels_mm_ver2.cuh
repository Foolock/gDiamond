// mix mapping version 2, one-to-one mapping on X and Y and Z
#ifndef KERNELS_MM_VER2_CUH
#define KERNELS_MM_VER2_CUH

#include "gdiamond.hpp"

// mix mapping
#define BLT_MM_V2 3 

// one-to-one mapping in X dimension
#define NTX_MM_V2 16
#define MOUNTAIN_X_V2 16 
#define VALLEY_X_V2 (MOUNTAIN_X_V2 - 2 * (BLT_MM_V2 - 1) - 1) 

// one-to-one mapping in Y dimension
#define NTY_MM_V2 8 
#define MOUNTAIN_Y_V2 8 
#define VALLEY_Y_V2 (MOUNTAIN_Y_V2 - 2 * (BLT_MM_V2 - 1) - 1)

// one-to-many mapping in Z dimension
#define NTZ_MM_V2 8 
#define MOUNTAIN_Z_V2 8 
#define VALLEY_Z_V2 (MOUNTAIN_Z_V2 - 2 * (BLT_MM_V2 - 1) - 1)

// padding
#define LEFT_PAD_MM_V2 BLT_MM_V2
#define RIGHT_PAD_MM_V2 BLT_MM_V2

// tile size
#define BLX_MM_V2 MOUNTAIN_X_V2 
#define BLY_MM_V2 MOUNTAIN_Y_V2 
#define BLZ_MM_V2 MOUNTAIN_Z_V2 

template <bool X_is_mountain, bool Y_is_mountain, bool Z_is_mountain>
__global__ void updateEH_mix_mapping_kernel_ver2(float* Ex_pad, float* Ey_pad, float* Ez_pad,
                                                 float* Hx_pad, float* Hy_pad, float* Hz_pad,
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

  const int local_x = thread_id % NTX_MM_V2;
  const int local_y = (thread_id / NTX_MM_V2) % NTY_MM_V2;
  const int local_z = thread_id / (NTX_MM_V2 * NTY_MM_V2);

  const int global_x = xx_heads[xx] + local_x;
  const int global_y = yy_heads[yy] + local_y;
  const int global_z = zz_heads[zz] + local_z;

  const int H_shared_x = X_is_mountain ? local_x + 1 : local_x;
  const int H_shared_y = Y_is_mountain ? local_y + 1 : local_y;
  const int H_shared_z = Z_is_mountain ? local_z + 1 : local_z;

  const int E_shared_x = local_x;
  const int E_shared_y = local_y;
  const int E_shared_z = local_z;

  int global_idx;
  // global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
  // if(global_x == 5 && global_y == 5 && global_z == 5) {
  //   printf("checking kernel, Hx_pad[global_idx] = %f\n", Hx_pad[global_idx]);
  // }
  int H_shared_idx;
  int E_shared_idx;

  // declare shared memory
  constexpr int H_SHX = (X_is_mountain)? BLX_MM_V2 + 1 : BLX_MM_V2;
  constexpr int H_SHY = (Y_is_mountain)? BLY_MM_V2 + 1 : BLY_MM_V2;
  constexpr int H_SHZ = (Z_is_mountain)? BLZ_MM_V2 + 1 : BLZ_MM_V2;
  constexpr int E_SHX = (X_is_mountain)? BLX_MM_V2 : BLX_MM_V2 + 1;
  constexpr int E_SHY = (Y_is_mountain)? BLY_MM_V2 : BLY_MM_V2 + 1;
  constexpr int E_SHZ = (Z_is_mountain)? BLZ_MM_V2 : BLZ_MM_V2 + 1;
  __shared__ float Hx_shmem[H_SHX * H_SHY * H_SHZ];
  __shared__ float Hy_shmem[H_SHX * H_SHY * H_SHZ];
  __shared__ float Hz_shmem[H_SHX * H_SHY * H_SHZ];
  __shared__ float Ex_shmem[E_SHX * E_SHY * E_SHZ];
  __shared__ float Ey_shmem[E_SHX * E_SHY * E_SHZ];
  __shared__ float Ez_shmem[E_SHX * E_SHY * E_SHZ];

  // load shared memory

  // load core ---------------------------------------------
  global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
  H_shared_idx = H_shared_x + H_shared_y * H_SHX + H_shared_z * H_SHX * H_SHY;
  E_shared_idx = E_shared_x + E_shared_y * E_SHX + E_shared_z * E_SHX * E_SHY;
  Hx_shmem[H_shared_idx] = Hx_pad[global_idx];
  Hy_shmem[H_shared_idx] = Hy_pad[global_idx];
  Hz_shmem[H_shared_idx] = Hz_pad[global_idx];
  Ex_shmem[E_shared_idx] = Ex_pad[global_idx];
  Ey_shmem[E_shared_idx] = Ey_pad[global_idx];
  Ez_shmem[E_shared_idx] = Ez_pad[global_idx];

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
  constexpr bool loadH_HALO_needed_x = X_is_mountain;
  constexpr bool loadH_HALO_needed_y = Y_is_mountain;
  constexpr bool loadH_HALO_needed_z = Z_is_mountain;
  constexpr bool loadE_HALO_needed_x = !X_is_mountain;
  constexpr bool loadE_HALO_needed_y = !Y_is_mountain;
  constexpr bool loadE_HALO_needed_z = !Z_is_mountain;

  // H HALO
  if (loadH_HALO_needed_x && local_x == 0) {
    int halo_x = 0;
    int global_x_halo = xx_heads[xx] + halo_x - 1;

    global_idx = global_x_halo + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
    H_shared_idx = halo_x + H_shared_y * H_SHX + H_shared_z * H_SHX * H_SHY;

    Hx_shmem[H_shared_idx] = Hx_pad[global_idx];
    Hy_shmem[H_shared_idx] = Hy_pad[global_idx];
    Hz_shmem[H_shared_idx] = Hz_pad[global_idx];
  }
  if (loadH_HALO_needed_y && local_y == 0) {
    int halo_y = 0;
    int global_y_halo = yy_heads[yy] + halo_y - 1;

    global_idx = global_x + global_y_halo * Nx_pad + global_z * Nx_pad * Ny_pad;
    H_shared_idx = H_shared_x + halo_y * H_SHX + H_shared_z * H_SHX * H_SHY;

    Hx_shmem[H_shared_idx] = Hx_pad[global_idx];
    Hy_shmem[H_shared_idx] = Hy_pad[global_idx];
    Hz_shmem[H_shared_idx] = Hz_pad[global_idx];
  }
  if (loadH_HALO_needed_z && local_z == 0) {
    int halo_z = 0;
    int global_z_halo = zz_heads[zz] + halo_z - 1;

    global_idx = global_x + global_y * Nx_pad + global_z_halo * Nx_pad * Ny_pad;
    H_shared_idx = H_shared_x + H_shared_y * H_SHX + halo_z * H_SHX * H_SHY;

    Hx_shmem[H_shared_idx] = Hx_pad[global_idx];
    Hy_shmem[H_shared_idx] = Hy_pad[global_idx];
    Hz_shmem[H_shared_idx] = Hz_pad[global_idx];
  }

  // E HALO
  if (loadE_HALO_needed_x && local_x == NTX_MM_V2 - 1) {
    int halo_x = local_x + 1;
    int global_x_halo = xx_heads[xx] + halo_x;

    global_idx = global_x_halo + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
    E_shared_idx = halo_x + E_shared_y * E_SHX + E_shared_z * E_SHX * E_SHY;

    Ex_shmem[E_shared_idx] = Ex_pad[global_idx];
    Ey_shmem[E_shared_idx] = Ey_pad[global_idx];
    Ez_shmem[E_shared_idx] = Ez_pad[global_idx];
  }
  if (loadE_HALO_needed_y && local_y == NTY_MM_V2 - 1) {
    int halo_y = local_y + 1;
    int global_y_halo = yy_heads[yy] + halo_y;

    global_idx = global_x + global_y_halo * Nx_pad + global_z * Nx_pad * Ny_pad;
    E_shared_idx = E_shared_x + halo_y * E_SHX + E_shared_z * E_SHX * E_SHY;

    Ex_shmem[E_shared_idx] = Ex_pad[global_idx];
    Ey_shmem[E_shared_idx] = Ey_pad[global_idx];
    Ez_shmem[E_shared_idx] = Ez_pad[global_idx];
  }
  if (loadE_HALO_needed_z && local_z == NTZ_MM_V2 - 1) {
    int halo_z = local_z + 1;
    int global_z_halo = zz_heads[zz] + halo_z;

    global_idx = global_x + global_y * Nx_pad + global_z_halo * Nx_pad * Ny_pad;
    E_shared_idx = E_shared_x + E_shared_y * E_SHX + halo_z * E_SHX * E_SHY;

    Ex_shmem[E_shared_idx] = Ex_pad[global_idx];
    Ey_shmem[E_shared_idx] = Ey_pad[global_idx];
    Ez_shmem[E_shared_idx] = Ez_pad[global_idx];
  }

  __syncthreads();

  // calculation

  // we pad all the dimension, so need to substract LEFT_PAD here to correctly access constant arrays
  global_idx = (global_x - LEFT_PAD_MM_V2) + (global_y - LEFT_PAD_MM_V2) * Nx + (global_z - LEFT_PAD_MM_V2) * Nx * Ny;
  E_shared_idx = E_shared_x + E_shared_y * E_SHX + E_shared_z * E_SHX * E_SHY;
  H_shared_idx = H_shared_x + H_shared_y * H_SHX + H_shared_z * H_SHX * H_SHY;

  // X head and tail is refer to unpadded global_x
  // same thing applys to Y and Z
  int calE_head_X, calE_tail_X;
  int calH_head_X, calH_tail_X;
  int calE_head_Y, calE_tail_Y;
  int calH_head_Y, calH_tail_Y;
  int calE_head_Z, calE_tail_Z;
  int calH_head_Z, calH_tail_Z;

  for(int t = 0; t < BLT_MM_V2; t++) {

    if constexpr (X_is_mountain) {
      calE_head_X = xx_heads[xx] + t;
      calE_tail_X = xx_heads[xx] + BLX_MM_V2 - 1 - t;
      calH_head_X = calE_head_X;
      calH_tail_X = calE_tail_X - 1;
    }
    else {
      calE_head_X = xx_heads[xx] + BLT_MM_V2 - t;
      calE_tail_X = xx_heads[xx] + BLX_MM_V2 - 1 - (BLT_MM_V2 - t -1);
      calH_head_X = calE_head_X - 1;
      calH_tail_X = calE_tail_X;
    }

    if constexpr (Y_is_mountain) {
      calE_head_Y = yy_heads[yy] + t;
      calE_tail_Y = yy_heads[yy] + BLY_MM_V2 - 1 - t;
      calH_head_Y = calE_head_Y;
      calH_tail_Y = calE_tail_Y - 1;
    }
    else {
      calE_head_Y = yy_heads[yy] + BLT_MM_V2 - t;
      calE_tail_Y = yy_heads[yy] + BLY_MM_V2 - 1 - (BLT_MM_V2 - t -1);
      calH_head_Y = calE_head_Y - 1;
      calH_tail_Y = calE_tail_Y;
    }

    if constexpr (Z_is_mountain) {
      calE_head_Z = zz_heads[zz] + t;
      calE_tail_Z = zz_heads[zz] + BLZ_MM_V2 - 1 - t;
      calH_head_Z = calE_head_Z;
      calH_tail_Z = calE_tail_Z - 1;
    }
    else {
      calE_head_Z = zz_heads[zz] + BLT_MM_V2 - t;
      calE_tail_Z = zz_heads[zz] + BLZ_MM_V2 - 1 - (BLT_MM_V2 - t -1);
      calH_head_Z = calE_head_Z - 1;
      calH_tail_Z = calE_tail_Z;
    }

    // update E
    if(global_x >= 1 + LEFT_PAD_MM_V2 && global_x <= Nx - 2 + LEFT_PAD_MM_V2 &&
       global_y >= 1 + LEFT_PAD_MM_V2 && global_y <= Ny - 2 + LEFT_PAD_MM_V2 &&
       global_z >= 1 + LEFT_PAD_MM_V2 && global_z <= Nz - 2 + LEFT_PAD_MM_V2 &&
       global_x >= calE_head_X && global_x <= calE_tail_X &&
       global_y >= calE_head_Y && global_y <= calE_tail_Y &&
       global_z >= calE_head_Z && global_z <= calE_tail_Z) {

      Ex_shmem[E_shared_idx] = Cax[global_idx] * Ex_shmem[E_shared_idx] + Cbx[global_idx] *
                ((Hz_shmem[H_shared_idx] - Hz_shmem[H_shared_idx - H_SHX]) - (Hy_shmem[H_shared_idx] - Hy_shmem[H_shared_idx - H_SHX * H_SHY]) - Jx[global_idx] * dx);

      Ey_shmem[E_shared_idx] = Cay[global_idx] * Ey_shmem[E_shared_idx] + Cby[global_idx] *
                ((Hx_shmem[H_shared_idx] - Hx_shmem[H_shared_idx - H_SHX * H_SHY]) - (Hz_shmem[H_shared_idx] - Hz_shmem[H_shared_idx - 1]) - Jy[global_idx] * dx);

      Ez_shmem[E_shared_idx] = Caz[global_idx] * Ez_shmem[E_shared_idx] + Cbz[global_idx] *
                ((Hy_shmem[H_shared_idx] - Hy_shmem[H_shared_idx - 1]) - (Hx_shmem[H_shared_idx] - Hx_shmem[H_shared_idx - H_SHX]) - Jz[global_idx] * dx);
    }

    __syncthreads();

    // update H 
    if(global_x >= 1 + LEFT_PAD_MM_V2 && global_x <= Nx - 2 + LEFT_PAD_MM_V2 &&
       global_y >= 1 + LEFT_PAD_MM_V2 && global_y <= Ny - 2 + LEFT_PAD_MM_V2 &&
       global_z >= 1 + LEFT_PAD_MM_V2 && global_z <= Nz - 2 + LEFT_PAD_MM_V2 &&
       global_x >= calH_head_X && global_x <= calH_tail_X &&
       global_y >= calH_head_Y && global_y <= calH_tail_Y &&
       global_z >= calH_head_Z && global_z <= calH_tail_Z) {

      Hx_shmem[H_shared_idx] = Dax[global_idx] * Hx_shmem[H_shared_idx] + Dbx[global_idx] *
                ((Ey_shmem[E_shared_idx + E_SHX * E_SHY] - Ey_shmem[E_shared_idx]) - (Ez_shmem[E_shared_idx + E_SHX] - Ez_shmem[E_shared_idx]) - Mx[global_idx] * dx);

      Hy_shmem[H_shared_idx] = Day[global_idx] * Hy_shmem[H_shared_idx] + Dby[global_idx] *
                ((Ez_shmem[E_shared_idx + 1] - Ez_shmem[E_shared_idx]) - (Ex_shmem[E_shared_idx + E_SHX * E_SHY] - Ex_shmem[E_shared_idx]) - My[global_idx] * dx);

      Hz_shmem[H_shared_idx] = Daz[global_idx] * Hz_shmem[H_shared_idx] + Dbz[global_idx] *
                ((Ex_shmem[E_shared_idx + E_SHX] - Ex_shmem[E_shared_idx]) - (Ey_shmem[E_shared_idx + 1] - Ey_shmem[E_shared_idx]) - Mz[global_idx] * dx);
    }

    __syncthreads();
  }

  // store back to global memory

  // no need to recalculate H_shared_idx, E_shared_idx
  global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;

  // X head and tail is refer to unpadded global_x
  // same thing applys to Y and Z
  const int storeE_head_X = X_is_mountain ? xx_heads[xx] : xx_heads[xx] + 1;
  const int storeE_tail_X = X_is_mountain ? (storeE_head_X + BLX_MM_V2 - 1) : (xx_heads[xx] + BLX_MM_V2 - 1);
  const int storeH_head_X = X_is_mountain ? storeE_head_X : xx_heads[xx];
  const int storeH_tail_X = X_is_mountain ? (storeE_tail_X - 1) : (storeH_head_X + BLX_MM_V2 - 1);
  const int storeE_head_Y = Y_is_mountain ? yy_heads[yy] : yy_heads[yy] + 1;
  const int storeE_tail_Y = Y_is_mountain ? (storeE_head_Y + BLY_MM_V2 - 1) : (yy_heads[yy] + BLY_MM_V2 - 1);
  const int storeH_head_Y = Y_is_mountain ? storeE_head_Y : yy_heads[yy];
  const int storeH_tail_Y = Y_is_mountain ? (storeE_tail_Y - 1) : (storeH_head_Y + BLY_MM_V2 - 1);
  const int storeE_head_Z = Z_is_mountain ? zz_heads[zz] : zz_heads[zz] + 1;
  const int storeE_tail_Z = Z_is_mountain ? (storeE_head_Z + BLZ_MM_V2 - 1) : (zz_heads[zz] + BLZ_MM_V2 - 1);
  const int storeH_head_Z = Z_is_mountain ? storeE_head_Z : zz_heads[zz];
  const int storeH_tail_Z = Z_is_mountain ? (storeE_tail_Z - 1) : (storeH_head_Z + BLZ_MM_V2 - 1); 

  if(global_x >= 1 + LEFT_PAD_MM_V2 && global_x <= Nx - 2 + LEFT_PAD_MM_V2 &&
     global_y >= 1 + LEFT_PAD_MM_V2 && global_y <= Ny - 2 + LEFT_PAD_MM_V2 &&
     global_z >= 1 + LEFT_PAD_MM_V2 && global_z <= Nz - 2 + LEFT_PAD_MM_V2 &&
     global_x >= storeH_head_X && global_x <= storeH_tail_X &&
     global_y >= storeH_head_Y && global_y <= storeH_tail_Y &&
     global_z >= storeH_head_Z && global_z <= storeH_tail_Z) {

    Hx_pad[global_idx] = Hx_shmem[H_shared_idx];
    Hy_pad[global_idx] = Hy_shmem[H_shared_idx];
    Hz_pad[global_idx] = Hz_shmem[H_shared_idx];
  }

  if(global_x >= 1 + LEFT_PAD_MM_V2 && global_x <= Nx - 2 + LEFT_PAD_MM_V2 &&
     global_y >= 1 + LEFT_PAD_MM_V2 && global_y <= Ny - 2 + LEFT_PAD_MM_V2 &&
     global_z >= 1 + LEFT_PAD_MM_V2 && global_z <= Nz - 2 + LEFT_PAD_MM_V2 &&
     global_x >= storeE_head_X && global_x <= storeE_tail_X &&
     global_y >= storeE_head_Y && global_y <= storeE_tail_Y &&
     global_z >= storeE_head_Z && global_z <= storeE_tail_Z) {

    Ex_pad[global_idx] = Ex_shmem[E_shared_idx];
    Ey_pad[global_idx] = Ey_shmem[E_shared_idx];
    Ez_pad[global_idx] = Ez_shmem[E_shared_idx];
  }
}

template<int t, bool X_is_mountain, bool Y_is_mountain, bool Z_is_mountain>
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

  constexpr int H_SHX = (X_is_mountain)? BLX_MM_V2 + 1 : BLX_MM_V2;
  constexpr int H_SHY = (Y_is_mountain)? BLY_MM_V2 + 1 : BLY_MM_V2;
  constexpr int E_SHX = (X_is_mountain)? BLX_MM_V2 : BLX_MM_V2 + 1;
  constexpr int E_SHY = (Y_is_mountain)? BLY_MM_V2 : BLY_MM_V2 + 1;

  // === Compute calE/calH bounds using t (known at compile-time) ===
  constexpr int calE_head_X = X_is_mountain ? 0 + t : BLT_MM_V2 - t;
  constexpr int calE_tail_X = X_is_mountain ? BLX_MM_V2 - 1 - t : BLX_MM_V2 - 1 - (BLT_MM_V2 - t - 1);
  constexpr int calH_head_X = X_is_mountain ? calE_head_X : calE_head_X - 1;
  constexpr int calH_tail_X = X_is_mountain ? calE_tail_X - 1 : calE_tail_X;

  constexpr int calE_head_Y = Y_is_mountain ? 0 + t : BLT_MM_V2 - t;
  constexpr int calE_tail_Y = Y_is_mountain ? BLY_MM_V2 - 1 - t : BLY_MM_V2 - 1 - (BLT_MM_V2 - t - 1);
  constexpr int calH_head_Y = Y_is_mountain ? calE_head_Y : calE_head_Y - 1;
  constexpr int calH_tail_Y = Y_is_mountain ? calE_tail_Y - 1 : calE_tail_Y;

  constexpr int calE_head_Z = Z_is_mountain ? 0 + t : BLT_MM_V2 - t;
  constexpr int calE_tail_Z = Z_is_mountain ? BLZ_MM_V2 - 1 - t : BLZ_MM_V2 - 1 - (BLT_MM_V2 - t - 1);
  constexpr int calH_head_Z = Z_is_mountain ? calE_head_Z : calE_head_Z - 1;
  constexpr int calH_tail_Z = Z_is_mountain ? calE_tail_Z - 1 : calE_tail_Z;

  // === Update E ===
  if(global_x >= 1 + LEFT_PAD_MM_V2 && global_x <= Nx - 2 + LEFT_PAD_MM_V2 &&
     global_y >= 1 + LEFT_PAD_MM_V2 && global_y <= Ny - 2 + LEFT_PAD_MM_V2 &&
     global_z >= 1 + LEFT_PAD_MM_V2 && global_z <= Nz - 2 + LEFT_PAD_MM_V2 &&
     global_x >= xx_heads[xx] + calE_head_X && global_x <= xx_heads[xx] + calE_tail_X &&
     global_y >= yy_heads[yy] + calE_head_Y && global_y <= yy_heads[yy] + calE_tail_Y &&
     global_z >= zz_heads[zz] + calE_head_Z && global_z <= zz_heads[zz] + calE_tail_Z) {

    Ex_shmem[E_shared_idx] = Cax[global_idx] * Ex_shmem[E_shared_idx] + Cbx[global_idx] *
              ((Hz_shmem[H_shared_idx] - Hz_shmem[H_shared_idx - H_SHX]) - (Hy_shmem[H_shared_idx] - Hy_shmem[H_shared_idx - H_SHX * H_SHY]) - Jx[global_idx] * dx);

    Ey_shmem[E_shared_idx] = Cay[global_idx] * Ey_shmem[E_shared_idx] + Cby[global_idx] *
              ((Hx_shmem[H_shared_idx] - Hx_shmem[H_shared_idx - H_SHX * H_SHY]) - (Hz_shmem[H_shared_idx] - Hz_shmem[H_shared_idx - 1]) - Jy[global_idx] * dx);

    Ez_shmem[E_shared_idx] = Caz[global_idx] * Ez_shmem[E_shared_idx] + Cbz[global_idx] *
              ((Hy_shmem[H_shared_idx] - Hy_shmem[H_shared_idx - 1]) - (Hx_shmem[H_shared_idx] - Hx_shmem[H_shared_idx - H_SHX]) - Jz[global_idx] * dx);
  }

  __syncthreads();

  // === Update H ===
  if(global_x >= 1 + LEFT_PAD_MM_V2 && global_x <= Nx - 2 + LEFT_PAD_MM_V2 &&
     global_y >= 1 + LEFT_PAD_MM_V2 && global_y <= Ny - 2 + LEFT_PAD_MM_V2 &&
     global_z >= 1 + LEFT_PAD_MM_V2 && global_z <= Nz - 2 + LEFT_PAD_MM_V2 &&
     global_x >= xx_heads[xx] + calH_head_X && global_x <= xx_heads[xx] + calH_tail_X &&
     global_y >= yy_heads[yy] + calH_head_Y && global_y <= yy_heads[yy] + calH_tail_Y &&
     global_z >= zz_heads[zz] + calH_head_Z && global_z <= zz_heads[zz] + calH_tail_Z) {

    Hx_shmem[H_shared_idx] = Dax[global_idx] * Hx_shmem[H_shared_idx] + Dbx[global_idx] *
              ((Ey_shmem[E_shared_idx + E_SHX * E_SHY] - Ey_shmem[E_shared_idx]) - (Ez_shmem[E_shared_idx + E_SHX] - Ez_shmem[E_shared_idx]) - Mx[global_idx] * dx);

    Hy_shmem[H_shared_idx] = Day[global_idx] * Hy_shmem[H_shared_idx] + Dby[global_idx] *
              ((Ez_shmem[E_shared_idx + 1] - Ez_shmem[E_shared_idx]) - (Ex_shmem[E_shared_idx + E_SHX * E_SHY] - Ex_shmem[E_shared_idx]) - My[global_idx] * dx);

    Hz_shmem[H_shared_idx] = Daz[global_idx] * Hz_shmem[H_shared_idx] + Dbz[global_idx] *
              ((Ex_shmem[E_shared_idx + E_SHX] - Ex_shmem[E_shared_idx]) - (Ey_shmem[E_shared_idx + 1] - Ey_shmem[E_shared_idx]) - Mz[global_idx] * dx);
  }

  __syncthreads();
}

template<int t, bool X_is_mountain, bool Y_is_mountain, bool Z_is_mountain>
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
    timestep_body<t, X_is_mountain, Y_is_mountain, Z_is_mountain>(
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
    if constexpr (t + 1 < BLT_MM_V2) {
        unroll_timesteps<t + 1, X_is_mountain, Y_is_mountain, Z_is_mountain>(
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

template <bool X_is_mountain, bool Y_is_mountain, bool Z_is_mountain>
__global__ void updateEH_mix_mapping_kernel_ver2_unroll(float* Ex_pad, float* Ey_pad, float* Ez_pad,
                                                 float* Hx_pad, float* Hy_pad, float* Hz_pad,
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

  const int local_x = thread_id % NTX_MM_V2;
  const int local_y = (thread_id / NTX_MM_V2) % NTY_MM_V2;
  const int local_z = thread_id / (NTX_MM_V2 * NTY_MM_V2);

  const int global_x = xx_heads[xx] + local_x;
  const int global_y = yy_heads[yy] + local_y;
  const int global_z = zz_heads[zz] + local_z;

  const int H_shared_x = X_is_mountain ? local_x + 1 : local_x;
  const int H_shared_y = Y_is_mountain ? local_y + 1 : local_y;
  const int H_shared_z = Z_is_mountain ? local_z + 1 : local_z;

  const int E_shared_x = local_x;
  const int E_shared_y = local_y;
  const int E_shared_z = local_z;

  int global_idx;
  int H_shared_idx;
  int E_shared_idx;

  // declare shared memory
  constexpr int H_SHX = (X_is_mountain)? BLX_MM_V2 + 1 : BLX_MM_V2;
  constexpr int H_SHY = (Y_is_mountain)? BLY_MM_V2 + 1 : BLY_MM_V2;
  constexpr int H_SHZ = (Z_is_mountain)? BLZ_MM_V2 + 1 : BLZ_MM_V2;
  constexpr int E_SHX = (X_is_mountain)? BLX_MM_V2 : BLX_MM_V2 + 1;
  constexpr int E_SHY = (Y_is_mountain)? BLY_MM_V2 : BLY_MM_V2 + 1;
  constexpr int E_SHZ = (Z_is_mountain)? BLZ_MM_V2 : BLZ_MM_V2 + 1;
  __shared__ float Hx_shmem[H_SHX * H_SHY * H_SHZ];
  __shared__ float Hy_shmem[H_SHX * H_SHY * H_SHZ];
  __shared__ float Hz_shmem[H_SHX * H_SHY * H_SHZ];
  __shared__ float Ex_shmem[E_SHX * E_SHY * E_SHZ];
  __shared__ float Ey_shmem[E_SHX * E_SHY * E_SHZ];
  __shared__ float Ez_shmem[E_SHX * E_SHY * E_SHZ];

  // load shared memory

  // load core ---------------------------------------------
  global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
  H_shared_idx = H_shared_x + H_shared_y * H_SHX + H_shared_z * H_SHX * H_SHY;
  E_shared_idx = E_shared_x + E_shared_y * E_SHX + E_shared_z * E_SHX * E_SHY;
  Hx_shmem[H_shared_idx] = Hx_pad[global_idx];
  Hy_shmem[H_shared_idx] = Hy_pad[global_idx];
  Hz_shmem[H_shared_idx] = Hz_pad[global_idx];
  Ex_shmem[E_shared_idx] = Ex_pad[global_idx];
  Ey_shmem[E_shared_idx] = Ey_pad[global_idx];
  Ez_shmem[E_shared_idx] = Ez_pad[global_idx];

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
  constexpr bool loadH_HALO_needed_x = X_is_mountain;
  constexpr bool loadH_HALO_needed_y = Y_is_mountain;
  constexpr bool loadH_HALO_needed_z = Z_is_mountain;
  constexpr bool loadE_HALO_needed_x = !X_is_mountain;
  constexpr bool loadE_HALO_needed_y = !Y_is_mountain;
  constexpr bool loadE_HALO_needed_z = !Z_is_mountain;

  // H HALO
  if (loadH_HALO_needed_x && local_x == 0) {
    int halo_x = 0;
    int global_x_halo = xx_heads[xx] + halo_x - 1;

    global_idx = global_x_halo + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
    H_shared_idx = halo_x + H_shared_y * H_SHX + H_shared_z * H_SHX * H_SHY;

    Hx_shmem[H_shared_idx] = Hx_pad[global_idx];
    Hy_shmem[H_shared_idx] = Hy_pad[global_idx];
    Hz_shmem[H_shared_idx] = Hz_pad[global_idx];
  }
  if (loadH_HALO_needed_y && local_y == 0) {
    int halo_y = 0;
    int global_y_halo = yy_heads[yy] + halo_y - 1;

    global_idx = global_x + global_y_halo * Nx_pad + global_z * Nx_pad * Ny_pad;
    H_shared_idx = H_shared_x + halo_y * H_SHX + H_shared_z * H_SHX * H_SHY;

    Hx_shmem[H_shared_idx] = Hx_pad[global_idx];
    Hy_shmem[H_shared_idx] = Hy_pad[global_idx];
    Hz_shmem[H_shared_idx] = Hz_pad[global_idx];
  }
  if (loadH_HALO_needed_z && local_z == 0) {
    int halo_z = 0;
    int global_z_halo = zz_heads[zz] + halo_z - 1;

    global_idx = global_x + global_y * Nx_pad + global_z_halo * Nx_pad * Ny_pad;
    H_shared_idx = H_shared_x + H_shared_y * H_SHX + halo_z * H_SHX * H_SHY;

    Hx_shmem[H_shared_idx] = Hx_pad[global_idx];
    Hy_shmem[H_shared_idx] = Hy_pad[global_idx];
    Hz_shmem[H_shared_idx] = Hz_pad[global_idx];
  }

  // E HALO
  if (loadE_HALO_needed_x && local_x == NTX_MM_V2 - 1) {
    int halo_x = local_x + 1;
    int global_x_halo = xx_heads[xx] + halo_x;

    global_idx = global_x_halo + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
    E_shared_idx = halo_x + E_shared_y * E_SHX + E_shared_z * E_SHX * E_SHY;

    Ex_shmem[E_shared_idx] = Ex_pad[global_idx];
    Ey_shmem[E_shared_idx] = Ey_pad[global_idx];
    Ez_shmem[E_shared_idx] = Ez_pad[global_idx];
  }
  if (loadE_HALO_needed_y && local_y == NTY_MM_V2 - 1) {
    int halo_y = local_y + 1;
    int global_y_halo = yy_heads[yy] + halo_y;

    global_idx = global_x + global_y_halo * Nx_pad + global_z * Nx_pad * Ny_pad;
    E_shared_idx = E_shared_x + halo_y * E_SHX + E_shared_z * E_SHX * E_SHY;

    Ex_shmem[E_shared_idx] = Ex_pad[global_idx];
    Ey_shmem[E_shared_idx] = Ey_pad[global_idx];
    Ez_shmem[E_shared_idx] = Ez_pad[global_idx];
  }
  if (loadE_HALO_needed_z && local_z == NTZ_MM_V2 - 1) {
    int halo_z = local_z + 1;
    int global_z_halo = zz_heads[zz] + halo_z;

    global_idx = global_x + global_y * Nx_pad + global_z_halo * Nx_pad * Ny_pad;
    E_shared_idx = E_shared_x + E_shared_y * E_SHX + halo_z * E_SHX * E_SHY;

    Ex_shmem[E_shared_idx] = Ex_pad[global_idx];
    Ey_shmem[E_shared_idx] = Ey_pad[global_idx];
    Ez_shmem[E_shared_idx] = Ez_pad[global_idx];
  }

  __syncthreads();

  // calculation

  // we pad all the dimension, so need to substract LEFT_PAD here to correctly access constant arrays
  global_idx = (global_x - LEFT_PAD_MM_V2) + (global_y - LEFT_PAD_MM_V2) * Nx + (global_z - LEFT_PAD_MM_V2) * Nx * Ny;
  E_shared_idx = E_shared_x + E_shared_y * E_SHX + E_shared_z * E_SHX * E_SHY;
  H_shared_idx = H_shared_x + H_shared_y * H_SHX + H_shared_z * H_SHX * H_SHY;
  unroll_timesteps<0, X_is_mountain, Y_is_mountain, Z_is_mountain>(
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
  const int storeE_head_X = X_is_mountain ? xx_heads[xx] : xx_heads[xx] + 1;
  const int storeE_tail_X = X_is_mountain ? (storeE_head_X + BLX_MM_V2 - 1) : (xx_heads[xx] + BLX_MM_V2 - 1);
  const int storeH_head_X = X_is_mountain ? storeE_head_X : xx_heads[xx];
  const int storeH_tail_X = X_is_mountain ? (storeE_tail_X - 1) : (storeH_head_X + BLX_MM_V2 - 1);
  const int storeE_head_Y = Y_is_mountain ? yy_heads[yy] : yy_heads[yy] + 1;
  const int storeE_tail_Y = Y_is_mountain ? (storeE_head_Y + BLY_MM_V2 - 1) : (yy_heads[yy] + BLY_MM_V2 - 1);
  const int storeH_head_Y = Y_is_mountain ? storeE_head_Y : yy_heads[yy];
  const int storeH_tail_Y = Y_is_mountain ? (storeE_tail_Y - 1) : (storeH_head_Y + BLY_MM_V2 - 1);
  const int storeE_head_Z = Z_is_mountain ? zz_heads[zz] : zz_heads[zz] + 1;
  const int storeE_tail_Z = Z_is_mountain ? (storeE_head_Z + BLZ_MM_V2 - 1) : (zz_heads[zz] + BLZ_MM_V2 - 1);
  const int storeH_head_Z = Z_is_mountain ? storeE_head_Z : zz_heads[zz];
  const int storeH_tail_Z = Z_is_mountain ? (storeE_tail_Z - 1) : (storeH_head_Z + BLZ_MM_V2 - 1); 

  if(global_x >= 1 + LEFT_PAD_MM_V2 && global_x <= Nx - 2 + LEFT_PAD_MM_V2 &&
     global_y >= 1 + LEFT_PAD_MM_V2 && global_y <= Ny - 2 + LEFT_PAD_MM_V2 &&
     global_z >= 1 + LEFT_PAD_MM_V2 && global_z <= Nz - 2 + LEFT_PAD_MM_V2 &&
     global_x >= storeH_head_X && global_x <= storeH_tail_X &&
     global_y >= storeH_head_Y && global_y <= storeH_tail_Y &&
     global_z >= storeH_head_Z && global_z <= storeH_tail_Z) {

    Hx_pad[global_idx] = Hx_shmem[H_shared_idx];
    Hy_pad[global_idx] = Hy_shmem[H_shared_idx];
    Hz_pad[global_idx] = Hz_shmem[H_shared_idx];
  }

  if(global_x >= 1 + LEFT_PAD_MM_V2 && global_x <= Nx - 2 + LEFT_PAD_MM_V2 &&
     global_y >= 1 + LEFT_PAD_MM_V2 && global_y <= Ny - 2 + LEFT_PAD_MM_V2 &&
     global_z >= 1 + LEFT_PAD_MM_V2 && global_z <= Nz - 2 + LEFT_PAD_MM_V2 &&
     global_x >= storeE_head_X && global_x <= storeE_tail_X &&
     global_y >= storeE_head_Y && global_y <= storeE_tail_Y &&
     global_z >= storeE_head_Z && global_z <= storeE_tail_Z) {

    Ex_pad[global_idx] = Ex_shmem[E_shared_idx];
    Ey_pad[global_idx] = Ey_shmem[E_shared_idx];
    Ez_pad[global_idx] = Ez_shmem[E_shared_idx];
  }
}


#endif


































