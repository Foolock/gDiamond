// mix mapping version 4, one-to-one mapping on X and Y and Z
// break big mountain tile into one small mountain and multiple parallelograms
#ifndef KERNELS_MM_VER4_CUH
#define KERNELS_MM_VER4_CUH

#include "gdiamond.hpp"

// cannot change the BLT here!!!!!! I did not parameterize BLT
#define BLT_MM_V4 4 

/*
 * X dimension parameters
 */

// number of threads used in X dimension
#define NTX_MM_V4 16
// width of mountain bottom of the small mountain (replication part) in X 
#define BLX_R 8
// width of parallelogram in X 
#define BLX_P NTX_MM_V4
// number of parallelgrams in each big mountain in X
#define NUM_P_X 1
// width of mountain bottom of the big mountain in X
#define MOUNTAIN_X_V4 (BLX_R + NUM_P_X * BLX_P)
#define VALLEY_X_V4 (MOUNTAIN_X_V4 - 2 * (BLT_MM_V4 - 1) - 1) 

/*
 * Y dimension parameters
 */

// number of threads used in Y dimension
#define NTY_MM_V4 8 
// width of mountain bottom of the small mountain (replication part) in Y 
#define BLY_R 8
// width of parallelogram in Y 
#define BLY_P NTY_MM_V4
// number of parallelgrams in each big mountain in Y
#define NUM_P_Y 1 
// width of mountain bottom of the big mountain in Y
#define MOUNTAIN_Y_V4 (BLY_R + NUM_P_Y * BLY_P)
#define VALLEY_Y_V4 (MOUNTAIN_Y_V4 - 2 * (BLT_MM_V4 - 1) - 1) 

/*
 * Z dimension parameters
 */

// number of threads used in Z dimension
#define NTZ_MM_V4 8 
// width of mountain bottom of the small mountain (replication part) in Z 
#define BLZ_R 8
// width of parallelogram in Z 
#define BLZ_P NTZ_MM_V4
// number of parallelgrams in each big mountain in Z
#define NUM_P_Z 1 
// width of mountain bottom of the big mountain in Z
#define MOUNTAIN_Z_V4 (BLZ_R + NUM_P_Z * BLZ_P)
#define VALLEY_Z_V4 (MOUNTAIN_Z_V4 - 2 * (BLT_MM_V4 - 1) - 1) 

// padding
#define LEFT_PAD_MM_V4 BLT_MM_V4
#define RIGHT_PAD_MM_V4 BLT_MM_V4

// shared memory size
// based on NTX = 16, NTY = 8, NTZ = 8,
// shared memory can at most include 2 time steps
// (16 + 2) * (8 + 2) * (8 + 2) * 4 * 6 ~= 43K
#define H_SHX_V4 (BLX_P + 2) 
#define H_SHY_V4 (BLY_P + 2) 
#define H_SHZ_V4 (BLZ_P + 2) 
#define E_SHX_V4 (BLX_P + 2)
#define E_SHY_V4 (BLY_P + 2)
#define E_SHZ_V4 (BLZ_P + 2)

#define REINDEX_H_X(H_shared_x) (((H_shared_x) == 0 || (H_shared_x) == 1) ? ((H_shared_x) + NTX_MM_V4) : ((H_shared_x) - 2))
#define REINDEX_H_Y(H_shared_y) (((H_shared_y) == 0 || (H_shared_y) == 1) ? ((H_shared_y) + NTY_MM_V4) : ((H_shared_y) - 2))
#define REINDEX_H_Z(H_shared_z) (((H_shared_z) == 0 || (H_shared_z) == 1) ? ((H_shared_z) + NTZ_MM_V4) : ((H_shared_z) - 2))

#define REINDEX_E_X(E_shared_x) (((E_shared_x) == 0 || (E_shared_x) == 1) ? ((E_shared_x) + NTX_MM_V4) : ((E_shared_x) - 2))
#define REINDEX_E_Y(E_shared_y) (((E_shared_y) == 0 || (E_shared_y) == 1) ? ((E_shared_y) + NTY_MM_V4) : ((E_shared_y) - 2))
#define REINDEX_E_Z(E_shared_z) (((E_shared_z) == 0 || (E_shared_z) == 1) ? ((E_shared_z) + NTZ_MM_V4) : ((E_shared_z) - 2))

__global__ void updateEH_mix_mapping_kernel_ver4(float* Ex_pad_src, float* Ey_pad_src, float* Ez_pad_src,
                                                 float* Hx_pad_src, float* Hy_pad_src, float* Hz_pad_src,
                                                 float* Ex_pad_rep, float* Ey_pad_rep, float* Ez_pad_rep,
                                                 float* Hx_pad_rep, float* Hy_pad_rep, float* Hz_pad_rep,
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

  const int local_x = thread_id % NTX_MM_V4;
  const int local_y = (thread_id / NTX_MM_V4) % NTY_MM_V4;
  const int local_z = thread_id / (NTX_MM_V4 * NTY_MM_V4);

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

  // declare shared memory
  // parallelogram calculation used more shared memory than replication calculation
  __shared__ float Hx_shmem[H_SHX_V4 * H_SHY_V4 * H_SHZ_V4];
  __shared__ float Hy_shmem[H_SHX_V4 * H_SHY_V4 * H_SHZ_V4];
  __shared__ float Hz_shmem[H_SHX_V4 * H_SHY_V4 * H_SHZ_V4];
  __shared__ float Ex_shmem[E_SHX_V4 * E_SHY_V4 * E_SHZ_V4];
  __shared__ float Ey_shmem[E_SHX_V4 * E_SHY_V4 * E_SHZ_V4];
  __shared__ float Ez_shmem[E_SHX_V4 * E_SHY_V4 * E_SHZ_V4];

  // load shared memory (replication)

  // load core ---------------------------------------------
  const int load_head_X = xx_heads[xx]; 
  const int load_tail_X = xx_heads[xx] + BLX_R - 1;
  H_shared_idx = H_shared_x + H_shared_y * H_SHX_V4 + H_shared_z * H_SHX_V4 * H_SHY_V4;
  E_shared_idx = E_shared_x + E_shared_y * E_SHX_V4 + E_shared_z * E_SHX_V4 * E_SHY_V4;
  global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
  if(global_x >= load_head_X && global_x <= load_tail_X) {
    Hx_shmem[H_shared_idx] = Hx_pad_src[global_idx];
    Hy_shmem[H_shared_idx] = Hy_pad_src[global_idx];
    Hz_shmem[H_shared_idx] = Hz_pad_src[global_idx];
    Ex_shmem[E_shared_idx] = Ex_pad_src[global_idx];
    Ey_shmem[E_shared_idx] = Ey_pad_src[global_idx];
    Ez_shmem[E_shared_idx] = Ez_pad_src[global_idx];
  }

  // H HALO
  // E HALO is not needed since there is no valley tile in mix mapping ver4 
  if (local_x == 0) {
    int halo_x = 0;
    int global_x_halo = xx_heads[xx] + halo_x - 1;

    global_idx = global_x_halo + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
    H_shared_idx = halo_x + H_shared_y * H_SHX_V4 + H_shared_z * H_SHX_V4 * H_SHY_V4;

    Hx_shmem[H_shared_idx] = Hx_pad_src[global_idx];
    Hy_shmem[H_shared_idx] = Hy_pad_src[global_idx];
    Hz_shmem[H_shared_idx] = Hz_pad_src[global_idx];
  }
  if (local_y == 0) {
    int halo_y = 0;
    int global_y_halo = yy_heads[yy] + halo_y - 1;

    global_idx = global_x + global_y_halo * Nx_pad + global_z * Nx_pad * Ny_pad;
    H_shared_idx = H_shared_x + halo_y * H_SHX_V4 + H_shared_z * H_SHX_V4 * H_SHY_V4;

    Hx_shmem[H_shared_idx] = Hx_pad_src[global_idx];
    Hy_shmem[H_shared_idx] = Hy_pad_src[global_idx];
    Hz_shmem[H_shared_idx] = Hz_pad_src[global_idx];
  }
  if (local_z == 0) {
    int halo_z = 0;
    int global_z_halo = zz_heads[zz] + halo_z - 1;

    global_idx = global_x + global_y * Nx_pad + global_z_halo * Nx_pad * Ny_pad;
    H_shared_idx = H_shared_x + H_shared_y * H_SHX_V4 + halo_z * H_SHX_V4 * H_SHY_V4;

    Hx_shmem[H_shared_idx] = Hx_pad_src[global_idx];
    Hy_shmem[H_shared_idx] = Hy_pad_src[global_idx];
    Hz_shmem[H_shared_idx] = Hz_pad_src[global_idx];
  }

  __syncthreads();

  // calculation (replication)

  // we pad all the dimension, so need to substract LEFT_PAD here to correctly access constant arrays
  global_idx = (global_x - LEFT_PAD_MM_V4) + (global_y - LEFT_PAD_MM_V4) * Nx + (global_z - LEFT_PAD_MM_V4) * Nx * Ny;
  E_shared_idx = E_shared_x + E_shared_y * E_SHX_V4 + E_shared_z * E_SHX_V4 * E_SHY_V4;
  H_shared_idx = H_shared_x + H_shared_y * H_SHX_V4 + H_shared_z * H_SHX_V4 * H_SHY_V4;

  // X head and tail is refer to padded global_x
  // same thing applys to Y and Z
  int calE_head_X, calE_tail_X;
  int calH_head_X, calH_tail_X;
  int calE_head_Y, calE_tail_Y;
  int calH_head_Y, calH_tail_Y;
  int calE_head_Z, calE_tail_Z;
  int calH_head_Z, calH_tail_Z;

  for(int t = 0; t < BLT_MM_V4; t++) {

    calE_head_X = xx_heads[xx] + t;
    calE_tail_X = xx_heads[xx] + BLX_R - 1 - t;
    calH_head_X = calE_head_X;
    calH_tail_X = calE_tail_X - 1;

    calE_head_Y = yy_heads[yy] + t;
    calE_tail_Y = yy_heads[yy] + BLY_R - 1 - t;
    calH_head_Y = calE_head_Y;
    calH_tail_Y = calE_tail_Y - 1;

    calE_head_Z = zz_heads[zz] + t;
    calE_tail_Z = zz_heads[zz] + BLZ_R - 1 - t;
    calH_head_Z = calE_head_Z;
    calH_tail_Z = calE_tail_Z - 1;

    // update E
    if(global_x >= 1 + LEFT_PAD_MM_V4 && global_x <= Nx - 2 + LEFT_PAD_MM_V4 &&
       global_y >= 1 + LEFT_PAD_MM_V4 && global_y <= Ny - 2 + LEFT_PAD_MM_V4 &&
       global_z >= 1 + LEFT_PAD_MM_V4 && global_z <= Nz - 2 + LEFT_PAD_MM_V4 &&
       global_x >= calE_head_X && global_x <= calE_tail_X &&
       global_y >= calE_head_Y && global_y <= calE_tail_Y &&
       global_z >= calE_head_Z && global_z <= calE_tail_Z) {

      Ex_shmem[E_shared_idx] = Cax[global_idx] * Ex_shmem[E_shared_idx] + Cbx[global_idx] *
                ((Hz_shmem[H_shared_idx] - Hz_shmem[H_shared_idx - H_SHX_V4]) - (Hy_shmem[H_shared_idx] - Hy_shmem[H_shared_idx - H_SHX_V4 * H_SHY_V4]) - Jx[global_idx] * dx);

      Ey_shmem[E_shared_idx] = Cay[global_idx] * Ey_shmem[E_shared_idx] + Cby[global_idx] *
                ((Hx_shmem[H_shared_idx] - Hx_shmem[H_shared_idx - H_SHX_V4 * H_SHY_V4]) - (Hz_shmem[H_shared_idx] - Hz_shmem[H_shared_idx - 1]) - Jy[global_idx] * dx);

      Ez_shmem[E_shared_idx] = Caz[global_idx] * Ez_shmem[E_shared_idx] + Cbz[global_idx] *
                ((Hy_shmem[H_shared_idx] - Hy_shmem[H_shared_idx - 1]) - (Hx_shmem[H_shared_idx] - Hx_shmem[H_shared_idx - H_SHX_V4]) - Jz[global_idx] * dx);
    }

    __syncthreads();

    // update H
    if(global_x >= 1 + LEFT_PAD_MM_V4 && global_x <= Nx - 2 + LEFT_PAD_MM_V4 &&
       global_y >= 1 + LEFT_PAD_MM_V4 && global_y <= Ny - 2 + LEFT_PAD_MM_V4 &&
       global_z >= 1 + LEFT_PAD_MM_V4 && global_z <= Nz - 2 + LEFT_PAD_MM_V4 &&
       global_x >= calH_head_X && global_x <= calH_tail_X &&
       global_y >= calH_head_Y && global_y <= calH_tail_Y &&
       global_z >= calH_head_Z && global_z <= calH_tail_Z) {

      Hx_shmem[H_shared_idx] = Dax[global_idx] * Hx_shmem[H_shared_idx] + Dbx[global_idx] *
                ((Ey_shmem[E_shared_idx + E_SHX_V4 * E_SHY_V4] - Ey_shmem[E_shared_idx]) - (Ez_shmem[E_shared_idx + E_SHX_V4] - Ez_shmem[E_shared_idx]) - Mx[global_idx] * dx);

      Hy_shmem[H_shared_idx] = Day[global_idx] * Hy_shmem[H_shared_idx] + Dby[global_idx] *
                ((Ez_shmem[E_shared_idx + 1] - Ez_shmem[E_shared_idx]) - (Ex_shmem[E_shared_idx + E_SHX_V4 * E_SHY_V4] - Ex_shmem[E_shared_idx]) - My[global_idx] * dx);

      Hz_shmem[H_shared_idx] = Daz[global_idx] * Hz_shmem[H_shared_idx] + Dbz[global_idx] *
                ((Ex_shmem[E_shared_idx + E_SHX_V4] - Ex_shmem[E_shared_idx]) - (Ey_shmem[E_shared_idx + 1] - Ey_shmem[E_shared_idx]) - Mz[global_idx] * dx);
    }

    __syncthreads();
  }

  // store back to global memory (replication)

  // no need to recalculate H_shared_idx, E_shared_idx
  global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;

  // X head and tail is refer to padded global_x
  // same thing applys to Y and Z
  const int storeE_head_X = xx_heads[xx] + BLX_R - BLT_MM_V4;
  const int storeE_tail_X = xx_heads[xx] + BLX_R - 1;
  const int storeH_head_X = storeE_head_X;
  // one extra store in H to make it simple
  // for parallelogram calculation
  const int storeH_tail_X = storeE_tail_X;

  const int storeE_head_Y = yy_heads[yy] + BLY_R - BLT_MM_V4;
  const int storeE_tail_Y = yy_heads[yy] + BLY_R - 1;
  const int storeH_head_Y = storeE_head_Y;
  const int storeH_tail_Y = storeE_tail_Y;

  const int storeE_head_Z = zz_heads[zz] + BLZ_R - BLT_MM_V4;
  const int storeE_tail_Z = zz_heads[zz] + BLZ_R - 1;
  const int storeH_head_Z = storeE_head_Z;
  const int storeH_tail_Z = storeE_tail_Z;

  // store H ---------------------------------------------
  if(global_x >= 1 + LEFT_PAD_MM_V4 && global_x <= Nx - 2 + LEFT_PAD_MM_V4 &&
     global_y >= 1 + LEFT_PAD_MM_V4 && global_y <= Ny - 2 + LEFT_PAD_MM_V4 &&
     global_z >= 1 + LEFT_PAD_MM_V4 && global_z <= Nz - 2 + LEFT_PAD_MM_V4 &&
     global_x >= storeH_head_X && global_x <= storeH_tail_X &&
     global_y >= storeH_head_Y && global_y <= storeH_tail_Y &&
     global_z >= storeH_head_Z && global_z <= storeH_tail_Z) {

    Hx_pad_rep[global_idx] = Hx_shmem[H_shared_idx];
    Hy_pad_rep[global_idx] = Hy_shmem[H_shared_idx];
    Hz_pad_rep[global_idx] = Hz_shmem[H_shared_idx];
  }

  // store E ---------------------------------------------
  if(global_x >= 1 + LEFT_PAD_MM_V4 && global_x <= Nx - 2 + LEFT_PAD_MM_V4 &&
     global_y >= 1 + LEFT_PAD_MM_V4 && global_y <= Ny - 2 + LEFT_PAD_MM_V4 &&
     global_z >= 1 + LEFT_PAD_MM_V4 && global_z <= Nz - 2 + LEFT_PAD_MM_V4 &&
     global_x >= storeE_head_X && global_x <= storeE_tail_X &&
     global_y >= storeE_head_Y && global_y <= storeE_tail_Y &&
     global_z >= storeE_head_Z && global_z <= storeE_tail_Z) {

    Ex_pad_rep[global_idx] = Ex_shmem[E_shared_idx];
    Ey_pad_rep[global_idx] = Ey_shmem[E_shared_idx];
    Ez_pad_rep[global_idx] = Ez_shmem[E_shared_idx];
  }

}

template<int t>
__device__ void cal_rep_loop_body(
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
  constexpr int calE_tail_X = BLX_R - 1 - t;
  constexpr int calH_head_X = calE_head_X;
  constexpr int calH_tail_X = calE_tail_X - 1;

  constexpr int calE_head_Y = t;
  constexpr int calE_tail_Y = BLY_R - 1 - t;
  constexpr int calH_head_Y = calE_head_Y;
  constexpr int calH_tail_Y = calE_tail_Y - 1;

  constexpr int calE_head_Z = t;
  constexpr int calE_tail_Z = BLZ_R - 1 - t;
  constexpr int calH_head_Z = calE_head_Z;
  constexpr int calH_tail_Z = calE_tail_Z - 1;

  // update E
  if(global_x >= 1 + LEFT_PAD_MM_V4 && global_x <= Nx - 2 + LEFT_PAD_MM_V4 &&
     global_y >= 1 + LEFT_PAD_MM_V4 && global_y <= Ny - 2 + LEFT_PAD_MM_V4 &&
     global_z >= 1 + LEFT_PAD_MM_V4 && global_z <= Nz - 2 + LEFT_PAD_MM_V4 &&
     global_x >= xx_heads[xx] + calE_head_X && global_x <= xx_heads[xx] + calE_tail_X &&
     global_y >= yy_heads[yy] + calE_head_Y && global_y <= yy_heads[yy] + calE_tail_Y &&
     global_z >= zz_heads[zz] + calE_head_Z && global_z <= zz_heads[zz] + calE_tail_Z) {

    Ex_shmem[E_shared_idx] = Cax[global_idx] * Ex_shmem[E_shared_idx] + Cbx[global_idx] *
              ((Hz_shmem[H_shared_idx] - Hz_shmem[H_shared_idx - H_SHX_V4]) - (Hy_shmem[H_shared_idx] - Hy_shmem[H_shared_idx - H_SHX_V4 * H_SHY_V4]) - Jx[global_idx] * dx);

    Ey_shmem[E_shared_idx] = Cay[global_idx] * Ey_shmem[E_shared_idx] + Cby[global_idx] *
              ((Hx_shmem[H_shared_idx] - Hx_shmem[H_shared_idx - H_SHX_V4 * H_SHY_V4]) - (Hz_shmem[H_shared_idx] - Hz_shmem[H_shared_idx - 1]) - Jy[global_idx] * dx);

    Ez_shmem[E_shared_idx] = Caz[global_idx] * Ez_shmem[E_shared_idx] + Cbz[global_idx] *
              ((Hy_shmem[H_shared_idx] - Hy_shmem[H_shared_idx - 1]) - (Hx_shmem[H_shared_idx] - Hx_shmem[H_shared_idx - H_SHX_V4]) - Jz[global_idx] * dx);
  }

  __syncthreads();

  // update H
  if(global_x >= 1 + LEFT_PAD_MM_V4 && global_x <= Nx - 2 + LEFT_PAD_MM_V4 &&
     global_y >= 1 + LEFT_PAD_MM_V4 && global_y <= Ny - 2 + LEFT_PAD_MM_V4 &&
     global_z >= 1 + LEFT_PAD_MM_V4 && global_z <= Nz - 2 + LEFT_PAD_MM_V4 &&
     global_x >= xx_heads[xx] + calH_head_X && global_x <= xx_heads[xx] + calH_tail_X &&
     global_y >= yy_heads[yy] + calH_head_Y && global_y <= yy_heads[yy] + calH_tail_Y &&
     global_z >= zz_heads[zz] + calH_head_Z && global_z <= zz_heads[zz] + calH_tail_Z) {

    Hx_shmem[H_shared_idx] = Dax[global_idx] * Hx_shmem[H_shared_idx] + Dbx[global_idx] *
              ((Ey_shmem[E_shared_idx + E_SHX_V4 * E_SHY_V4] - Ey_shmem[E_shared_idx]) - (Ez_shmem[E_shared_idx + E_SHX_V4] - Ez_shmem[E_shared_idx]) - Mx[global_idx] * dx);

    Hy_shmem[H_shared_idx] = Day[global_idx] * Hy_shmem[H_shared_idx] + Dby[global_idx] *
              ((Ez_shmem[E_shared_idx + 1] - Ez_shmem[E_shared_idx]) - (Ex_shmem[E_shared_idx + E_SHX_V4 * E_SHY_V4] - Ex_shmem[E_shared_idx]) - My[global_idx] * dx);

    Hz_shmem[H_shared_idx] = Daz[global_idx] * Hz_shmem[H_shared_idx] + Dbz[global_idx] *
              ((Ex_shmem[E_shared_idx + E_SHX_V4] - Ex_shmem[E_shared_idx]) - (Ey_shmem[E_shared_idx + 1] - Ey_shmem[E_shared_idx]) - Mz[global_idx] * dx);
  }

  __syncthreads();
}

template<int t>
__device__ void unroll_cal_rep_loop(
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
    cal_rep_loop_body<t>(
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
    if constexpr (t + 1 < BLT_MM_V4) {
      unroll_cal_rep_loop<t + 1>(
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

template<int t>
__device__ void cal_p_tile_first_2_steps_loop_body(
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

  constexpr int calE_head_X = BLX_R - t;
  constexpr int calE_tail_X = calE_head_X + BLX_P - 1;
  constexpr int calH_head_X = calE_head_X - 1;
  constexpr int calH_tail_X = calE_tail_X - 1;

  constexpr int calE_head_Y = BLY_R - t;
  constexpr int calE_tail_Y = calE_head_Y + BLY_P - 1;
  constexpr int calH_head_Y = calE_head_Y - 1;
  constexpr int calH_tail_Y = calE_tail_Y - 1;

  constexpr int calE_head_Z = BLZ_R - t;
  constexpr int calE_tail_Z = calE_head_Z + BLZ_P - 1;
  constexpr int calH_head_Z = calE_head_Z - 1;
  constexpr int calH_tail_Z = calE_tail_Z - 1;

  // update E
  if(global_x >= 1 + LEFT_PAD_MM_V4 && global_x <= Nx - 2 + LEFT_PAD_MM_V4 &&
     global_y >= 1 + LEFT_PAD_MM_V4 && global_y <= Ny - 2 + LEFT_PAD_MM_V4 &&
     global_z >= 1 + LEFT_PAD_MM_V4 && global_z <= Nz - 2 + LEFT_PAD_MM_V4 &&
     global_x >= xx_heads[xx] + calE_head_X && global_x <= xx_heads[xx] + calE_tail_X &&
     global_y >= yy_heads[yy] + calE_head_Y && global_y <= yy_heads[yy] + calE_tail_Y &&
     global_z >= zz_heads[zz] + calE_head_Z && global_z <= zz_heads[zz] + calE_tail_Z) {

    Ex_shmem[E_shared_idx] = Cax[global_idx] * Ex_shmem[E_shared_idx] + Cbx[global_idx] *
              ((Hz_shmem[H_shared_idx] - Hz_shmem[H_shared_idx - H_SHX_V4]) - (Hy_shmem[H_shared_idx] - Hy_shmem[H_shared_idx - H_SHX_V4 * H_SHY_V4]) - Jx[global_idx] * dx);

    Ey_shmem[E_shared_idx] = Cay[global_idx] * Ey_shmem[E_shared_idx] + Cby[global_idx] *
              ((Hx_shmem[H_shared_idx] - Hx_shmem[H_shared_idx - H_SHX_V4 * H_SHY_V4]) - (Hz_shmem[H_shared_idx] - Hz_shmem[H_shared_idx - 1]) - Jy[global_idx] * dx);

    Ez_shmem[E_shared_idx] = Caz[global_idx] * Ez_shmem[E_shared_idx] + Cbz[global_idx] *
              ((Hy_shmem[H_shared_idx] - Hy_shmem[H_shared_idx - 1]) - (Hx_shmem[H_shared_idx] - Hx_shmem[H_shared_idx - H_SHX_V4]) - Jz[global_idx] * dx);
  }

  __syncthreads();

  // update H
  if(global_x >= 1 + LEFT_PAD_MM_V4 && global_x <= Nx - 2 + LEFT_PAD_MM_V4 &&
     global_y >= 1 + LEFT_PAD_MM_V4 && global_y <= Ny - 2 + LEFT_PAD_MM_V4 &&
     global_z >= 1 + LEFT_PAD_MM_V4 && global_z <= Nz - 2 + LEFT_PAD_MM_V4 &&
     global_x >= xx_heads[xx] + calH_head_X && global_x <= xx_heads[xx] + calH_tail_X &&
     global_y >= yy_heads[yy] + calH_head_Y && global_y <= yy_heads[yy] + calH_tail_Y &&
     global_z >= zz_heads[zz] + calH_head_Z && global_z <= zz_heads[zz] + calH_tail_Z) {

    Hx_shmem[H_shared_idx] = Dax[global_idx] * Hx_shmem[H_shared_idx] + Dbx[global_idx] *
              ((Ey_shmem[E_shared_idx + E_SHX_V4 * E_SHY_V4] - Ey_shmem[E_shared_idx]) - (Ez_shmem[E_shared_idx + E_SHX_V4] - Ez_shmem[E_shared_idx]) - Mx[global_idx] * dx);

    Hy_shmem[H_shared_idx] = Day[global_idx] * Hy_shmem[H_shared_idx] + Dby[global_idx] *
              ((Ez_shmem[E_shared_idx + 1] - Ez_shmem[E_shared_idx]) - (Ex_shmem[E_shared_idx + E_SHX_V4 * E_SHY_V4] - Ex_shmem[E_shared_idx]) - My[global_idx] * dx);

    Hz_shmem[H_shared_idx] = Daz[global_idx] * Hz_shmem[H_shared_idx] + Dbz[global_idx] *
              ((Ex_shmem[E_shared_idx + E_SHX_V4] - Ex_shmem[E_shared_idx]) - (Ey_shmem[E_shared_idx + 1] - Ey_shmem[E_shared_idx]) - Mz[global_idx] * dx);
  }

  __syncthreads();
}

template<int t>
__device__ void unroll_cal_p_tile_first_2_steps_loop(
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
    cal_p_tile_first_2_steps_loop_body<t>(
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
    if constexpr (t + 1 < BLT_MM_V4 / 2) {
      unroll_cal_p_tile_first_2_steps_loop<t + 1>(
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

template<int t>
__device__ void cal_p_tile_later_2_steps_loop_body(
    const int global_idx, const int H_shared_idx, const int E_shared_idx,
    const int global_x, const int global_y, const int global_z,
    const int E_shared_idx_x_stencil, const int E_shared_idx_y_stencil, const int E_shared_idx_z_stencil,
    const int H_shared_idx_x_stencil, const int H_shared_idx_y_stencil, const int H_shared_idx_z_stencil,
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

  constexpr int calE_head_X = BLX_R - t - 2;
  constexpr int calE_tail_X = calE_head_X + BLX_P - 1;
  constexpr int calH_head_X = calE_head_X - 1;
  constexpr int calH_tail_X = calE_tail_X - 1;

  constexpr int calE_head_Y = BLY_R - t - 2;
  constexpr int calE_tail_Y = calE_head_Y + BLY_P - 1;
  constexpr int calH_head_Y = calE_head_Y - 1;
  constexpr int calH_tail_Y = calE_tail_Y - 1;

  constexpr int calE_head_Z = BLZ_R - t - 2;
  constexpr int calE_tail_Z = calE_head_Z + BLZ_P - 1;
  constexpr int calH_head_Z = calE_head_Z - 1;
  constexpr int calH_tail_Z = calE_tail_Z - 1;

  // update E
  if(global_x >= 1 + LEFT_PAD_MM_V4 && global_x <= Nx - 2 + LEFT_PAD_MM_V4 &&
     global_y >= 1 + LEFT_PAD_MM_V4 && global_y <= Ny - 2 + LEFT_PAD_MM_V4 &&
     global_z >= 1 + LEFT_PAD_MM_V4 && global_z <= Nz - 2 + LEFT_PAD_MM_V4 &&
     global_x >= xx_heads[xx] + calE_head_X && global_x <= xx_heads[xx] + calE_tail_X &&
     global_y >= yy_heads[yy] + calE_head_Y && global_y <= yy_heads[yy] + calE_tail_Y &&
     global_z >= zz_heads[zz] + calE_head_Z && global_z <= zz_heads[zz] + calE_tail_Z) {

    Ex_shmem[E_shared_idx] = Cax[global_idx] * Ex_shmem[E_shared_idx] + Cbx[global_idx] *
              ((Hz_shmem[H_shared_idx] - Hz_shmem[H_shared_idx_y_stencil]) - (Hy_shmem[H_shared_idx] - Hy_shmem[H_shared_idx_z_stencil]) - Jx[global_idx] * dx);

    Ey_shmem[E_shared_idx] = Cay[global_idx] * Ey_shmem[E_shared_idx] + Cby[global_idx] *
              ((Hx_shmem[H_shared_idx] - Hx_shmem[H_shared_idx_z_stencil]) - (Hz_shmem[H_shared_idx] - Hz_shmem[H_shared_idx_x_stencil]) - Jy[global_idx] * dx);

    Ez_shmem[E_shared_idx] = Caz[global_idx] * Ez_shmem[E_shared_idx] + Cbz[global_idx] *
              ((Hy_shmem[H_shared_idx] - Hy_shmem[H_shared_idx_x_stencil]) - (Hx_shmem[H_shared_idx] - Hx_shmem[H_shared_idx_y_stencil]) - Jz[global_idx] * dx);
  }

  __syncthreads();

  // update H
  if(global_x >= 1 + LEFT_PAD_MM_V4 && global_x <= Nx - 2 + LEFT_PAD_MM_V4 &&
     global_y >= 1 + LEFT_PAD_MM_V4 && global_y <= Ny - 2 + LEFT_PAD_MM_V4 &&
     global_z >= 1 + LEFT_PAD_MM_V4 && global_z <= Nz - 2 + LEFT_PAD_MM_V4 &&
     global_x >= xx_heads[xx] + calH_head_X && global_x <= xx_heads[xx] + calH_tail_X &&
     global_y >= yy_heads[yy] + calH_head_Y && global_y <= yy_heads[yy] + calH_tail_Y &&
     global_z >= zz_heads[zz] + calH_head_Z && global_z <= zz_heads[zz] + calH_tail_Z) {

    Hx_shmem[H_shared_idx] = Dax[global_idx] * Hx_shmem[H_shared_idx] + Dbx[global_idx] *
              ((Ey_shmem[E_shared_idx_z_stencil] - Ey_shmem[E_shared_idx]) - (Ez_shmem[E_shared_idx_y_stencil] - Ez_shmem[E_shared_idx]) - Mx[global_idx] * dx);

    Hy_shmem[H_shared_idx] = Day[global_idx] * Hy_shmem[H_shared_idx] + Dby[global_idx] *
              ((Ez_shmem[E_shared_idx_x_stencil] - Ez_shmem[E_shared_idx]) - (Ex_shmem[E_shared_idx_z_stencil] - Ex_shmem[E_shared_idx]) - My[global_idx] * dx);

    Hz_shmem[H_shared_idx] = Daz[global_idx] * Hz_shmem[H_shared_idx] + Dbz[global_idx] *
              ((Ex_shmem[E_shared_idx_y_stencil] - Ex_shmem[E_shared_idx]) - (Ey_shmem[E_shared_idx_x_stencil] - Ey_shmem[E_shared_idx]) - Mz[global_idx] * dx);
  }

  __syncthreads();
}

template<int t>
__device__ void unroll_cal_p_tile_later_2_steps_loop(
    const int global_idx, const int H_shared_idx, const int E_shared_idx,
    const int global_x, const int global_y, const int global_z,
    const int E_shared_idx_x_stencil, const int E_shared_idx_y_stencil, const int E_shared_idx_z_stencil,
    const int H_shared_idx_x_stencil, const int H_shared_idx_y_stencil, const int H_shared_idx_z_stencil,
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
    cal_p_tile_later_2_steps_loop_body<t>(
    global_idx, H_shared_idx, E_shared_idx,
    global_x, global_y, global_z,
    E_shared_idx_x_stencil, E_shared_idx_y_stencil, E_shared_idx_z_stencil,
    H_shared_idx_x_stencil, H_shared_idx_y_stencil, H_shared_idx_z_stencil,
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
    if constexpr (t + 1 < BLT_MM_V4 / 2) {
      unroll_cal_p_tile_later_2_steps_loop<t + 1>(
      global_idx, H_shared_idx, E_shared_idx,
      global_x, global_y, global_z,
      E_shared_idx_x_stencil, E_shared_idx_y_stencil, E_shared_idx_z_stencil,
      H_shared_idx_x_stencil, H_shared_idx_y_stencil, H_shared_idx_z_stencil,
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

__global__ void updateEH_mix_mapping_kernel_ver4_unroll(float* Ex_pad_src, float* Ey_pad_src, float* Ez_pad_src,
                                                 float* Hx_pad_src, float* Hy_pad_src, float* Hz_pad_src,
                                                 float* Ex_pad_rep, float* Ey_pad_rep, float* Ez_pad_rep,
                                                 float* Hx_pad_rep, float* Hy_pad_rep, float* Hz_pad_rep,
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

  const int local_x = thread_id % NTX_MM_V4;
  const int local_y = (thread_id / NTX_MM_V4) % NTY_MM_V4;
  const int local_z = thread_id / (NTX_MM_V4 * NTY_MM_V4);

  int global_x = xx_heads[xx] + local_x;
  int global_y = yy_heads[yy] + local_y;
  int global_z = zz_heads[zz] + local_z;

  int H_shared_x = local_x + 1;
  int H_shared_y = local_y + 1;
  int H_shared_z = local_z + 1;

  int E_shared_x = local_x;
  int E_shared_y = local_y;
  int E_shared_z = local_z;

  int global_idx;
  int H_shared_idx;
  int E_shared_idx;

  // declare shared memory
  // parallelogram calculation used more shared memory than replication calculation
  __shared__ float Hx_shmem[H_SHX_V4 * H_SHY_V4 * H_SHZ_V4];
  __shared__ float Hy_shmem[H_SHX_V4 * H_SHY_V4 * H_SHZ_V4];
  __shared__ float Hz_shmem[H_SHX_V4 * H_SHY_V4 * H_SHZ_V4];
  __shared__ float Ex_shmem[E_SHX_V4 * E_SHY_V4 * E_SHZ_V4];
  __shared__ float Ey_shmem[E_SHX_V4 * E_SHY_V4 * E_SHZ_V4];
  __shared__ float Ez_shmem[E_SHX_V4 * E_SHY_V4 * E_SHZ_V4];

  /*
   * starting for replication 
   */

  // load shared memory (replication)

  // load core ---------------------------------------------
  const int load_head_X = xx_heads[xx]; 
  const int load_tail_X = xx_heads[xx] + BLX_R - 1;
  H_shared_idx = H_shared_x + H_shared_y * H_SHX_V4 + H_shared_z * H_SHX_V4 * H_SHY_V4;
  E_shared_idx = E_shared_x + E_shared_y * E_SHX_V4 + E_shared_z * E_SHX_V4 * E_SHY_V4;
  global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
  if(global_x >= load_head_X && global_x <= load_tail_X) {
    Hx_shmem[H_shared_idx] = Hx_pad_src[global_idx];
    Hy_shmem[H_shared_idx] = Hy_pad_src[global_idx];
    Hz_shmem[H_shared_idx] = Hz_pad_src[global_idx];
    Ex_shmem[E_shared_idx] = Ex_pad_src[global_idx];
    Ey_shmem[E_shared_idx] = Ey_pad_src[global_idx];
    Ez_shmem[E_shared_idx] = Ez_pad_src[global_idx];
  }

  // H HALO
  // E HALO is not needed since there is no valley tile in mix mapping ver4 
  if (local_x == 0) {
    int halo_x = 0;
    int global_x_halo = xx_heads[xx] + halo_x - 1;

    global_idx = global_x_halo + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
    H_shared_idx = halo_x + H_shared_y * H_SHX_V4 + H_shared_z * H_SHX_V4 * H_SHY_V4;

    Hx_shmem[H_shared_idx] = Hx_pad_src[global_idx];
    Hy_shmem[H_shared_idx] = Hy_pad_src[global_idx];
    Hz_shmem[H_shared_idx] = Hz_pad_src[global_idx];
  }
  if (local_y == 0) {
    int halo_y = 0;
    int global_y_halo = yy_heads[yy] + halo_y - 1;

    global_idx = global_x + global_y_halo * Nx_pad + global_z * Nx_pad * Ny_pad;
    H_shared_idx = H_shared_x + halo_y * H_SHX_V4 + H_shared_z * H_SHX_V4 * H_SHY_V4;

    Hx_shmem[H_shared_idx] = Hx_pad_src[global_idx];
    Hy_shmem[H_shared_idx] = Hy_pad_src[global_idx];
    Hz_shmem[H_shared_idx] = Hz_pad_src[global_idx];
  }
  if (local_z == 0) {
    int halo_z = 0;
    int global_z_halo = zz_heads[zz] + halo_z - 1;

    global_idx = global_x + global_y * Nx_pad + global_z_halo * Nx_pad * Ny_pad;
    H_shared_idx = H_shared_x + H_shared_y * H_SHX_V4 + halo_z * H_SHX_V4 * H_SHY_V4;

    Hx_shmem[H_shared_idx] = Hx_pad_src[global_idx];
    Hy_shmem[H_shared_idx] = Hy_pad_src[global_idx];
    Hz_shmem[H_shared_idx] = Hz_pad_src[global_idx];
  }

  __syncthreads();

  // calculation (replication)

  // we pad all the dimension, so need to substract LEFT_PAD here to correctly access constant arrays
  global_idx = (global_x - LEFT_PAD_MM_V4) + (global_y - LEFT_PAD_MM_V4) * Nx + (global_z - LEFT_PAD_MM_V4) * Nx * Ny;
  E_shared_idx = E_shared_x + E_shared_y * E_SHX_V4 + E_shared_z * E_SHX_V4 * E_SHY_V4;
  H_shared_idx = H_shared_x + H_shared_y * H_SHX_V4 + H_shared_z * H_SHX_V4 * H_SHY_V4;
  unroll_cal_rep_loop<0>(
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

  // store back to global memory (replication)

  // no need to recalculate H_shared_idx, E_shared_idx
  global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;

  // X head and tail is refer to padded global_x
  // same thing applys to Y and Z
  int storeE_head_X = xx_heads[xx] + BLX_R - BLT_MM_V4;
  int storeE_tail_X = xx_heads[xx] + BLX_R - 1;
  int storeH_head_X = storeE_head_X;
  int storeH_tail_X = storeE_tail_X - 1;

  int storeE_head_Y = yy_heads[yy] + BLY_R - BLT_MM_V4;
  int storeE_tail_Y = yy_heads[yy] + BLY_R - 1;
  int storeH_head_Y = storeE_head_Y;
  int storeH_tail_Y = storeE_tail_Y - 1;

  int storeE_head_Z = zz_heads[zz] + BLZ_R - BLT_MM_V4;
  int storeE_tail_Z = zz_heads[zz] + BLZ_R - 1;
  int storeH_head_Z = storeE_head_Z;
  int storeH_tail_Z = storeE_tail_Z - 1;

  // store H ---------------------------------------------
  if(global_x >= 1 + LEFT_PAD_MM_V4 && global_x <= Nx - 2 + LEFT_PAD_MM_V4 &&
     global_y >= 1 + LEFT_PAD_MM_V4 && global_y <= Ny - 2 + LEFT_PAD_MM_V4 &&
     global_z >= 1 + LEFT_PAD_MM_V4 && global_z <= Nz - 2 + LEFT_PAD_MM_V4 &&
     global_x >= storeH_head_X && global_x <= storeH_tail_X &&
     global_y >= storeH_head_Y && global_y <= storeH_tail_Y &&
     global_z >= storeH_head_Z && global_z <= storeH_tail_Z) {

    Hx_pad_rep[global_idx] = Hx_shmem[H_shared_idx];
    Hy_pad_rep[global_idx] = Hy_shmem[H_shared_idx];
    Hz_pad_rep[global_idx] = Hz_shmem[H_shared_idx];
  }

  // store E ---------------------------------------------
  if(global_x >= 1 + LEFT_PAD_MM_V4 && global_x <= Nx - 2 + LEFT_PAD_MM_V4 &&
     global_y >= 1 + LEFT_PAD_MM_V4 && global_y <= Ny - 2 + LEFT_PAD_MM_V4 &&
     global_z >= 1 + LEFT_PAD_MM_V4 && global_z <= Nz - 2 + LEFT_PAD_MM_V4 &&
     global_x >= storeE_head_X && global_x <= storeE_tail_X &&
     global_y >= storeE_head_Y && global_y <= storeE_tail_Y &&
     global_z >= storeE_head_Z && global_z <= storeE_tail_Z) {

    Ex_pad_rep[global_idx] = Ex_shmem[E_shared_idx];
    Ey_pad_rep[global_idx] = Ey_shmem[E_shared_idx];
    Ez_pad_rep[global_idx] = Ez_shmem[E_shared_idx];
  }

  __syncthreads();

  /*
   * starting for 1st parallelogram tile
   */

  H_shared_x = local_x + 2;
  H_shared_y = local_y + 2;
  H_shared_z = local_z + 2;
  E_shared_x = local_x + 2;
  E_shared_y = local_y + 2;
  E_shared_z = local_z + 2;
  global_x = xx_heads[xx] + local_x + BLX_R;
  global_y = yy_heads[yy] + local_y + BLY_R;
  global_z = zz_heads[zz] + local_z + BLZ_R;

  // load core ---------------------------------------------
  H_shared_idx = H_shared_x + H_shared_y * H_SHX_V4 + H_shared_z * H_SHX_V4 * H_SHY_V4;
  E_shared_idx = E_shared_x + E_shared_y * E_SHX_V4 + E_shared_z * E_SHX_V4 * E_SHY_V4;
  global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
  Hx_shmem[H_shared_idx] = Hx_pad_src[global_idx];
  Hy_shmem[H_shared_idx] = Hy_pad_src[global_idx];
  Hz_shmem[H_shared_idx] = Hz_pad_src[global_idx];
  Ex_shmem[E_shared_idx] = Ex_pad_src[global_idx];
  Ey_shmem[E_shared_idx] = Ey_pad_src[global_idx];
  Ez_shmem[E_shared_idx] = Ez_pad_src[global_idx];

  // H HALO
  if(local_x < 2) { // 2 HALOs in X
    int halo_x = local_x;
    int global_x_halo = xx_heads[xx] + BLX_R + halo_x - 2;

    global_idx = global_x_halo + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
    H_shared_idx = halo_x + H_shared_y * H_SHX_V4 + H_shared_z * H_SHX_V4 * H_SHY_V4;

    Hx_shmem[H_shared_idx] = Hx_pad_rep[global_idx];
    Hy_shmem[H_shared_idx] = Hy_pad_rep[global_idx];
    Hz_shmem[H_shared_idx] = Hz_pad_rep[global_idx];
  }
  if(local_y < 2) { // 2 HALOs in Y
    int halo_y = local_y;
    int global_y_halo = yy_heads[yy] + BLY_R + halo_y - 2;

    global_idx = global_x + global_y_halo * Nx_pad + global_z * Nx_pad * Ny_pad;
    H_shared_idx = H_shared_x + halo_y * H_SHX_V4 + H_shared_z * H_SHX_V4 * H_SHY_V4;

    Hx_shmem[H_shared_idx] = Hx_pad_rep[global_idx];
    Hy_shmem[H_shared_idx] = Hy_pad_rep[global_idx];
    Hz_shmem[H_shared_idx] = Hz_pad_rep[global_idx];
  }
  if(local_z < 2) { // 2 HALOs in Z
    int halo_z = local_z;
    int global_z_halo = zz_heads[zz] + BLZ_R + halo_z - 2;

    global_idx = global_x + global_y * Nx_pad + global_z_halo * Nx_pad * Ny_pad;
    H_shared_idx = H_shared_x + H_shared_y * H_SHX_V4 + halo_z * H_SHX_V4 * H_SHY_V4;

    Hx_shmem[H_shared_idx] = Hx_pad_rep[global_idx];
    Hy_shmem[H_shared_idx] = Hy_pad_rep[global_idx];
    Hz_shmem[H_shared_idx] = Hz_pad_rep[global_idx];
  }

  // E HALO
  // I want to use some other threads than those who working on H HALO to load E HALO
  if(local_x >= NTX_MM_V4 - 2) {
    int halo_x = local_x - NTX_MM_V4 + 2;
    int global_x_halo = xx_heads[xx] + BLX_R + halo_x - 2;

    global_idx = global_x_halo + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
    E_shared_idx = halo_x + E_shared_y * E_SHX_V4 + E_shared_z * E_SHX_V4 * E_SHY_V4;

    Ex_shmem[E_shared_idx] = Ex_pad_rep[global_idx];
    Ey_shmem[E_shared_idx] = Ey_pad_rep[global_idx];
    Ez_shmem[E_shared_idx] = Ez_pad_rep[global_idx];
  }
  if(local_y >= NTY_MM_V4 - 2) {
    int halo_y = local_y - NTY_MM_V4 + 2;
    int global_y_halo = yy_heads[yy] + BLY_R + halo_y - 2;

    global_idx = global_x + global_y_halo * Nx_pad + global_z * Nx_pad * Ny_pad;
    E_shared_idx = E_shared_x + halo_y * E_SHX_V4 + E_shared_z * E_SHX_V4 * E_SHY_V4;

    Ex_shmem[E_shared_idx] = Ex_pad_rep[global_idx];
    Ey_shmem[E_shared_idx] = Ey_pad_rep[global_idx];
    Ez_shmem[E_shared_idx] = Ez_pad_rep[global_idx];
  }
  if(local_z >= NTZ_MM_V4 - 2) {
    int halo_z = local_z - NTZ_MM_V4 + 2;
    int global_z_halo = zz_heads[zz] + BLZ_R + halo_z - 2;

    global_idx = global_x + global_y * Nx_pad + global_z_halo * Nx_pad * Ny_pad;
    E_shared_idx = E_shared_x + E_shared_y * E_SHX_V4 + halo_z * E_SHX_V4 * E_SHY_V4;

    Ex_shmem[E_shared_idx] = Ex_pad_rep[global_idx];
    Ey_shmem[E_shared_idx] = Ey_pad_rep[global_idx];
    Ez_shmem[E_shared_idx] = Ez_pad_rep[global_idx];
  }

  __syncthreads();

  // calculation (1st parallelogram, i.e., Py = 0, Pz = 0)
  // first two steps
  global_idx = (global_x - LEFT_PAD_MM_V4) + (global_y - LEFT_PAD_MM_V4) * Nx + (global_z - LEFT_PAD_MM_V4) * Nx * Ny;
  E_shared_idx = E_shared_x + E_shared_y * E_SHX_V4 + E_shared_z * E_SHX_V4 * E_SHY_V4;
  H_shared_idx = H_shared_x + H_shared_y * H_SHX_V4 + H_shared_z * H_SHX_V4 * H_SHY_V4;
  unroll_cal_p_tile_first_2_steps_loop<0>(
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

  // after the first 2 timesteps, need to evict old data to global memory
  // and load new data to shared memory

  // evict H Halo
  if(local_x < 2) {
    int halo_x = local_x + NTX_MM_V4;
    int global_x_halo = xx_heads[xx] + BLX_R + halo_x - 2;

    global_idx = global_x_halo + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
    H_shared_idx = halo_x + H_shared_y * H_SHX_V4 + H_shared_z * H_SHX_V4 * H_SHY_V4;

    Hx_pad_dst[global_idx] = Hx_shmem[H_shared_idx];
    Hy_pad_dst[global_idx] = Hy_shmem[H_shared_idx];
    Hz_pad_dst[global_idx] = Hz_shmem[H_shared_idx];
  }
  if(local_y < 2) {
    int halo_y = local_y + NTY_MM_V4;
    int global_y_halo = yy_heads[yy] + BLY_R + halo_y - 2;

    global_idx = global_x + global_y_halo * Nx_pad + global_z * Nx_pad * Ny_pad;
    H_shared_idx = H_shared_x + halo_y * H_SHX_V4 + H_shared_z * H_SHX_V4 * H_SHY_V4;

    Hx_pad_dst[global_idx] = Hx_shmem[H_shared_idx];
    Hy_pad_dst[global_idx] = Hy_shmem[H_shared_idx];
    Hz_pad_dst[global_idx] = Hz_shmem[H_shared_idx];
  }
  if(local_z < 2) {
    int halo_z = local_z + NTZ_MM_V4;
    int global_z_halo = zz_heads[zz] + BLZ_R + halo_z - 2;

    global_idx = global_x + global_y * Nx_pad + global_z_halo * Nx_pad * Ny_pad;
    H_shared_idx = H_shared_x + H_shared_y * H_SHX_V4 + halo_z * H_SHX_V4 * H_SHY_V4;

    Hx_pad_dst[global_idx] = Hx_shmem[H_shared_idx];
    Hy_pad_dst[global_idx] = Hy_shmem[H_shared_idx];
    Hz_pad_dst[global_idx] = Hz_shmem[H_shared_idx];
  }

  // evict E HALO
  if(local_x >= NTX_MM_V4 - 2) {
    int halo_x = local_x + 2;
    int global_x_halo = xx_heads[xx] + BLX_R + local_x;

    global_idx = global_x_halo + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
    E_shared_idx = halo_x + E_shared_y * E_SHX_V4 + E_shared_z * E_SHX_V4 * E_SHY_V4;

    Ex_pad_dst[global_idx] = Ex_shmem[E_shared_idx];
    Ey_pad_dst[global_idx] = Ey_shmem[E_shared_idx];
    Ez_pad_dst[global_idx] = Ez_shmem[E_shared_idx];
  }
  if(local_y >= NTY_MM_V4 - 2) {
    int halo_y = local_y + 2;
    int global_y_halo = yy_heads[yy] + BLY_R + local_y;

    global_idx = global_x + global_y_halo * Nx_pad + global_z * Nx_pad * Ny_pad;
    E_shared_idx = E_shared_x + halo_y * E_SHX_V4 + E_shared_z * E_SHX_V4 * E_SHY_V4;

    Ex_pad_dst[global_idx] = Ex_shmem[E_shared_idx];
    Ey_pad_dst[global_idx] = Ey_shmem[E_shared_idx];
    Ez_pad_dst[global_idx] = Ez_shmem[E_shared_idx];
  }
  if(local_z >= NTZ_MM_V4 - 2) {
    int halo_z = local_z + 2;
    int global_z_halo = zz_heads[zz] + BLZ_R + local_z;

    global_idx = global_x + global_y * Nx_pad + global_z_halo * Nx_pad * Ny_pad;
    E_shared_idx = E_shared_x + E_shared_y * E_SHX_V4 + halo_z * E_SHX_V4 * E_SHY_V4;

    Ex_pad_dst[global_idx] = Ex_shmem[E_shared_idx];
    Ey_pad_dst[global_idx] = Ey_shmem[E_shared_idx];
    Ez_pad_dst[global_idx] = Ez_shmem[E_shared_idx];
  }

  __syncthreads();

  // load H Halo
  if(local_x < 2) {
    int halo_x = local_x + NTX_MM_V4;
    int global_x_halo = xx_heads[xx] + BLX_R + local_x - 4;

    global_idx = global_x_halo + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
    H_shared_idx = halo_x + H_shared_y * H_SHX_V4 + H_shared_z * H_SHX_V4 * H_SHY_V4;

    Hx_shmem[H_shared_idx] = Hx_pad_rep[global_idx];
    Hy_shmem[H_shared_idx] = Hy_pad_rep[global_idx];
    Hz_shmem[H_shared_idx] = Hz_pad_rep[global_idx];
  }
  if(local_y < 2) {
    int halo_y = local_y + NTY_MM_V4;
    int global_y_halo = yy_heads[yy] + BLY_R + local_y - 4;

    global_idx = global_x + global_y_halo * Nx_pad + global_z * Nx_pad * Ny_pad;
    H_shared_idx = H_shared_x + halo_y * H_SHX_V4 + H_shared_z * H_SHX_V4 * H_SHY_V4;

    Hx_shmem[H_shared_idx] = Hx_pad_rep[global_idx];
    Hy_shmem[H_shared_idx] = Hy_pad_rep[global_idx];
    Hz_shmem[H_shared_idx] = Hz_pad_rep[global_idx];
  }
  if(local_z < 2) {
    int halo_z = local_z + NTZ_MM_V4;
    int global_z_halo = zz_heads[zz] + BLZ_R + local_z - 4;

    global_idx = global_x + global_y * Nx_pad + global_z_halo * Nx_pad * Ny_pad;
    H_shared_idx = H_shared_x + H_shared_y * H_SHX_V4 + halo_z * H_SHX_V4 * H_SHY_V4;

    Hx_shmem[H_shared_idx] = Hx_pad_rep[global_idx];
    Hy_shmem[H_shared_idx] = Hy_pad_rep[global_idx];
    Hz_shmem[H_shared_idx] = Hz_pad_rep[global_idx];
  }

  // load E Halo
  if(local_x >= NTX_MM_V4 - 2) {
    int halo_x = local_x + 2;
    int global_x_halo = xx_heads[xx] + BLX_R + local_x - (NTX_MM_V4 + 2);

    global_idx = global_x_halo + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
    E_shared_idx = halo_x + E_shared_y * E_SHX_V4 + E_shared_z * E_SHX_V4 * E_SHY_V4;

    Ex_shmem[E_shared_idx] = Ex_pad_dst[global_idx];
    Ey_shmem[E_shared_idx] = Ey_pad_dst[global_idx];
    Ez_shmem[E_shared_idx] = Ez_pad_dst[global_idx];
  }
  if(local_y >= NTY_MM_V4 - 2) {
    int halo_y = local_y + 2;
    int global_y_halo = yy_heads[yy] + BLY_R + local_y - (NTY_MM_V4 + 2);

    global_idx = global_x + global_y_halo * Nx_pad + global_z * Nx_pad * Ny_pad;
    E_shared_idx = E_shared_x + halo_y * E_SHX_V4 + E_shared_z * E_SHX_V4 * E_SHY_V4;

    Ex_shmem[E_shared_idx] = Ex_pad_dst[global_idx];
    Ey_shmem[E_shared_idx] = Ey_pad_dst[global_idx];
    Ez_shmem[E_shared_idx] = Ez_pad_dst[global_idx];
  }
  if(local_z >= NTZ_MM_V4 - 2) {
    int halo_z = local_z + 2;
    int global_z_halo = zz_heads[zz] + BLZ_R + local_z - (NTZ_MM_V4 + 2);

    global_idx = global_x + global_y * Nx_pad + global_z_halo * Nx_pad * Ny_pad;
    E_shared_idx = E_shared_x + E_shared_y * E_SHX_V4 + halo_z * E_SHX_V4 * E_SHY_V4;

    Ex_shmem[E_shared_idx] = Ex_pad_dst[global_idx];
    Ey_shmem[E_shared_idx] = Ey_pad_dst[global_idx];
    Ez_shmem[E_shared_idx] = Ez_pad_dst[global_idx];
  }

  __syncthreads();

  // calculation (1st parallelogram, i.e., Py = 0, Pz = 0)
  // later two steps
  // we pad all the dimension, so need to substract LEFT_PAD here to correctly access constant arrays
  global_idx = (global_x - LEFT_PAD_MM_V4) + (global_y - LEFT_PAD_MM_V4) * Nx + (global_z - LEFT_PAD_MM_V4) * Nx * Ny;
  // before calculate E_shared_idx and H_shared_idx, we need to perform re-indexing
  E_shared_idx = REINDEX_E_X(E_shared_x) + REINDEX_E_Y(E_shared_y) * E_SHX_V4 + REINDEX_E_Z(E_shared_z) * E_SHX_V4 * E_SHY_V4;
  H_shared_idx = REINDEX_H_X(H_shared_x) + REINDEX_H_Y(H_shared_y) * H_SHX_V4 + REINDEX_H_Z(H_shared_z) * H_SHX_V4 * H_SHY_V4;
  int H_shared_idx_x_stencil = REINDEX_H_X(H_shared_x - 1) + REINDEX_H_Y(H_shared_y) * H_SHX_V4 + REINDEX_H_Z(H_shared_z) * H_SHX_V4 * H_SHY_V4;
  int H_shared_idx_y_stencil = REINDEX_H_X(H_shared_x) + REINDEX_H_Y(H_shared_y - 1) * H_SHX_V4 + REINDEX_H_Z(H_shared_z) * H_SHX_V4 * H_SHY_V4;
  int H_shared_idx_z_stencil = REINDEX_H_X(H_shared_x) + REINDEX_H_Y(H_shared_y) * H_SHX_V4 + REINDEX_H_Z(H_shared_z - 1) * H_SHX_V4 * H_SHY_V4;
  int E_shared_idx_x_stencil = REINDEX_E_X(E_shared_x + 1) + REINDEX_E_Y(E_shared_y) * E_SHX_V4 + REINDEX_E_Z(E_shared_z) * E_SHX_V4 * E_SHY_V4;
  int E_shared_idx_y_stencil = REINDEX_E_X(E_shared_x) + REINDEX_E_Y(E_shared_y + 1) * E_SHX_V4 + REINDEX_E_Z(E_shared_z) * E_SHX_V4 * E_SHY_V4;
  int E_shared_idx_z_stencil = REINDEX_E_X(E_shared_x) + REINDEX_E_Y(E_shared_y) * E_SHX_V4 + REINDEX_E_Z(E_shared_z + 1) * E_SHX_V4 * E_SHY_V4;
  unroll_cal_p_tile_later_2_steps_loop<0>(
  global_idx, H_shared_idx, E_shared_idx,
  global_x, global_y, global_z,
  E_shared_idx_x_stencil, E_shared_idx_y_stencil, E_shared_idx_z_stencil,
  H_shared_idx_x_stencil, H_shared_idx_y_stencil, H_shared_idx_z_stencil,
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

  // store back to global memory (1st parallelogram, Py = 0, Pz = 0)
  storeE_head_X = xx_heads[xx] - BLT_MM_V4 + 1;
  storeE_tail_X = storeE_head_X + BLX_P - 1; 
  storeH_head_X = storeE_head_X - 1;
  storeH_tail_X = storeE_tail_X - 1; 

  storeE_head_Y = yy_heads[yy] - BLT_MM_V4 + 1;
  storeE_tail_Y = storeE_head_Y + BLY_P - 1; 
  storeH_head_Y = storeE_head_Y - 1;
  storeH_tail_Y = storeE_tail_Y - 1; 

  storeE_head_Z = zz_heads[zz] - BLT_MM_V4 + 1;
  storeE_tail_Z = storeE_head_Z + BLZ_P - 1; 
  storeH_head_Z = storeE_head_Z - 1;
  storeH_tail_Z = storeE_tail_Z - 1; 

  H_shared_x = local_x;
  H_shared_y = local_y;
  H_shared_z = local_z;
  E_shared_x = local_x;
  E_shared_y = local_y;
  E_shared_z = local_z;
  global_x = xx_heads[xx] + BLX_R - 4 + local_x;
  global_y = yy_heads[yy] + BLY_R - 4 + local_y;
  global_z = zz_heads[zz] + BLZ_R - 4 + local_z;

  global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
  H_shared_idx = REINDEX_H_X(H_shared_x) + REINDEX_H_Y(H_shared_y) * H_SHX_V4 + REINDEX_H_Z(H_shared_z) * H_SHX_V4 * H_SHY_V4;
  E_shared_idx = REINDEX_E_X(E_shared_x) + REINDEX_E_Y(E_shared_y) * E_SHX_V4 + REINDEX_E_Z(E_shared_z) * E_SHX_V4 * E_SHY_V4;

  // store H core ---------------------------------------------
  if(global_x >= 1 + LEFT_PAD_MM_V4 && global_x <= Nx - 2 + LEFT_PAD_MM_V4 &&
     global_y >= 1 + LEFT_PAD_MM_V4 && global_y <= Ny - 2 + LEFT_PAD_MM_V4 &&
     global_z >= 1 + LEFT_PAD_MM_V4 && global_z <= Nz - 2 + LEFT_PAD_MM_V4 &&
     global_x >= storeH_head_X && global_x <= storeH_tail_X &&
     global_y >= storeH_head_Y && global_y <= storeH_tail_Y &&
     global_z >= storeH_head_Z && global_z <= storeH_tail_Z) {

    Hx_pad_dst[global_idx] = Hx_shmem[H_shared_idx];
    Hy_pad_dst[global_idx] = Hy_shmem[H_shared_idx];
    Hz_pad_dst[global_idx] = Hz_shmem[H_shared_idx];
  }

  // store E core ---------------------------------------------
  if(global_x >= 1 + LEFT_PAD_MM_V4 && global_x <= Nx - 2 + LEFT_PAD_MM_V4 &&
     global_y >= 1 + LEFT_PAD_MM_V4 && global_y <= Ny - 2 + LEFT_PAD_MM_V4 &&
     global_z >= 1 + LEFT_PAD_MM_V4 && global_z <= Nz - 2 + LEFT_PAD_MM_V4 &&
     global_x >= storeE_head_X && global_x <= storeE_tail_X &&
     global_y >= storeE_head_Y && global_y <= storeE_tail_Y &&
     global_z >= storeE_head_Z && global_z <= storeE_tail_Z) {

    Ex_pad_dst[global_idx] = Ex_shmem[E_shared_idx];
    Ey_pad_dst[global_idx] = Ey_shmem[E_shared_idx];
    Ez_pad_dst[global_idx] = Ez_shmem[E_shared_idx];
  }

  // store H Halo ---------------------------------------------
  if(local_x < 2) {
    int halo_x = local_x + NTX_MM_V4;
    int global_x_halo = xx_heads[xx] + BLX_R + halo_x - 4;

    global_idx = global_x_halo + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
    H_shared_idx = REINDEX_H_X(halo_x) + REINDEX_H_Y(H_shared_y) * H_SHX_V4 + REINDEX_H_Z(H_shared_z) * H_SHX_V4 * H_SHY_V4; 

    Hx_pad_dst[global_idx] = Hx_shmem[H_shared_idx];
    Hy_pad_dst[global_idx] = Hy_shmem[H_shared_idx];
    Hz_pad_dst[global_idx] = Hz_shmem[H_shared_idx];
  }
  if(local_y < 2) {
    int halo_y = local_y + NTY_MM_V4;
    int global_y_halo = yy_heads[yy] + BLY_R + halo_y - 4;

    global_idx = global_x + global_y_halo * Nx_pad + global_z * Nx_pad * Ny_pad;
    H_shared_idx = REINDEX_H_X(H_shared_x) + REINDEX_H_Y(halo_y) * H_SHX_V4 + REINDEX_H_Z(H_shared_z) * H_SHX_V4 * H_SHY_V4;

    Hx_pad_dst[global_idx] = Hx_shmem[H_shared_idx];
    Hy_pad_dst[global_idx] = Hy_shmem[H_shared_idx];
    Hz_pad_dst[global_idx] = Hz_shmem[H_shared_idx];
  }
  if(local_z < 2) {
    int halo_z = local_z + NTZ_MM_V4;
    int global_z_halo = zz_heads[zz] + BLZ_R + halo_z - 4;

    global_idx = global_x + global_y * Nx_pad + global_z_halo * Nx_pad * Ny_pad;
    H_shared_idx = REINDEX_H_X(H_shared_x) + REINDEX_H_Y(H_shared_y) * H_SHX_V4 + REINDEX_H_Z(halo_z) * H_SHX_V4 * H_SHY_V4;

    Hx_pad_dst[global_idx] = Hx_shmem[H_shared_idx];
    Hy_pad_dst[global_idx] = Hy_shmem[H_shared_idx];
    Hz_pad_dst[global_idx] = Hz_shmem[H_shared_idx];
  }
  
  // store E Halo ---------------------------------------------
  if(local_x >= NTX_MM_V4 - 2) {
    int halo_x = local_x + 2;
    int global_x_halo = xx_heads[xx] + BLX_R + halo_x - 4;

    global_idx = global_x_halo + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
    E_shared_idx = REINDEX_E_X(halo_x) + REINDEX_E_Y(E_shared_y) * E_SHX_V4 + REINDEX_E_Z(E_shared_z) * E_SHX_V4 * E_SHY_V4;

    Ex_pad_dst[global_idx] = Ex_shmem[E_shared_idx];
    Ey_pad_dst[global_idx] = Ey_shmem[E_shared_idx];
    Ez_pad_dst[global_idx] = Ez_shmem[E_shared_idx];
  }
  if(local_y >= NTY_MM_V4 - 2) {
    int halo_y = local_y + 2;
    int global_y_halo = yy_heads[yy] + BLY_R + halo_y - 4;

    global_idx = global_x + global_y_halo * Nx_pad + global_z * Nx_pad * Ny_pad;
    E_shared_idx = REINDEX_E_X(E_shared_x) + REINDEX_E_Y(halo_y) * E_SHX_V4 + REINDEX_E_Z(E_shared_z) * E_SHX_V4 * E_SHY_V4;

    Ex_pad_dst[global_idx] = Ex_shmem[E_shared_idx];
    Ey_pad_dst[global_idx] = Ey_shmem[E_shared_idx];
    Ez_pad_dst[global_idx] = Ez_shmem[E_shared_idx];
  }
  if(local_z >= NTZ_MM_V4 - 2) {
    int halo_z = local_z + 2;
    int global_z_halo = zz_heads[zz] + BLZ_R + halo_z - 4;

    global_idx = global_x + global_y * Nx_pad + global_z_halo * Nx_pad * Ny_pad;
    E_shared_idx = REINDEX_E_X(E_shared_x) + REINDEX_E_Y(E_shared_y) * E_SHX_V4 + REINDEX_E_Z(halo_z) * E_SHX_V4 * E_SHY_V4;

    Ex_pad_dst[global_idx] = Ex_shmem[E_shared_idx];
    Ey_pad_dst[global_idx] = Ey_shmem[E_shared_idx];
    Ez_pad_dst[global_idx] = Ez_shmem[E_shared_idx];
  }

}


#endif


































