#ifndef KERNELS_MM_CUH
#define KERNELS_MM_CUH

#include "gdiamond.hpp"

// mix mapping
#define BLT_MM 4 

// one-to-one mapping in X dimension
#define NTX_MM 16
#define MOUNTAIN_X 16 
#define VALLEY_X (MOUNTAIN_X - 2 * (BLT_MM - 1) - 1) 

// one-to-many mapping in Y dimension
#define NTY_MM 4
#define MOUNTAIN_Y 10
#define VALLEY_Y (MOUNTAIN_Y - 2 * (BLT_MM - 1) - 1)

// one-to-many mapping in Z dimension
#define NTZ_MM 4
#define MOUNTAIN_Z 10
#define VALLEY_Z (MOUNTAIN_Z - 2 * (BLT_MM - 1) - 1)

// padding
#define LEFT_PAD_MM BLT_MM
#define RIGHT_PAD_MM BLT_MM

// tile size
#define BLX_MM MOUNTAIN_X 
#define BLY_MM MOUNTAIN_Y 
#define BLZ_MM MOUNTAIN_Z 

template <bool X_is_mountain, bool Y_is_mountain, bool Z_is_mountain>
__global__ void updateEH_mix_mapping_kernel(float* Ex_pad, float* Ey_pad, float* Ez_pad,
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

  const unsigned int xx = block_id % xx_num;
  const unsigned int yy = (block_id / xx_num) % yy_num;
  const unsigned int zz = block_id / (xx_num * yy_num);

  const int local_x = thread_id % NTX_MM;
  const int local_y = (thread_id / NTX_MM) % NTY_MM;
  const int local_z = thread_id / (NTX_MM * NTY_MM);

  int global_x, global_y, global_z;
  int H_shared_x, H_shared_y, H_shared_z;
  int E_shared_x, E_shared_y, E_shared_z;

  // declare shared memory
  constexpr int H_SHX = (X_is_mountain)? 17 : 16;
  constexpr int H_SHY = (Y_is_mountain)? 11 : 10;
  constexpr int H_SHZ = (Z_is_mountain)? 11 : 10;
  constexpr int E_SHX = (X_is_mountain)? 16 : 17;
  constexpr int E_SHY = (Y_is_mountain)? 10 : 11;
  constexpr int E_SHZ = (Z_is_mountain)? 10 : 11;
  __shared__ float Hx_shmem[H_SHX * H_SHY * H_SHZ];
  __shared__ float Hy_shmem[H_SHX * H_SHY * H_SHZ];
  __shared__ float Hz_shmem[H_SHX * H_SHY * H_SHZ];
  __shared__ float Ex_shmem[E_SHX * E_SHY * E_SHZ];
  __shared__ float Ey_shmem[E_SHX * E_SHY * E_SHZ];
  __shared__ float Ez_shmem[E_SHX * E_SHY * E_SHZ];

  // load shared memory
  // load H ---------------------------------------------
  if constexpr (X_is_mountain) {
    H_shared_x = local_x + 1;
  }
  else {
    H_shared_x = local_x;
  }
  global_x = xx_heads[xx] + local_x;

  for(H_shared_y = local_y; H_shared_y <= H_SHY - 1; H_shared_y += NTY_MM) {

    if constexpr (Y_is_mountain) { global_y = yy_heads[yy] + H_shared_y - 1; }
    else { global_y = yy_heads[yy] + H_shared_y; }

    for(H_shared_z = local_z; H_shared_z <= H_SHZ - 1; H_shared_z += NTZ_MM) {

      if constexpr (Z_is_mountain) { global_z = zz_heads[zz] + H_shared_z - 1; }
      else { global_z = zz_heads[zz] + H_shared_z; }

      int global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
      int H_shared_idx = H_shared_x + H_shared_y * H_SHX + H_shared_z * H_SHX * H_SHY;
      Hx_shmem[H_shared_idx] = Hx_pad[global_idx];
      Hy_shmem[H_shared_idx] = Hy_pad[global_idx];
      Hz_shmem[H_shared_idx] = Hz_pad[global_idx];

    }
  }

  bool loadH_HALO_needed;
  if constexpr (X_is_mountain) { loadH_HALO_needed = true; }
  else { loadH_HALO_needed = false; }
  // if mountain, tid = 0 load one extra H at xx_heads[xx] - 1
  if(loadH_HALO_needed && local_x == 0) {
    H_shared_x = 0;
    global_x = xx_heads[xx] + H_shared_x - 1;
    for(H_shared_y = local_y; H_shared_y <= H_SHY - 1; H_shared_y += NTY_MM) {

      if constexpr (Y_is_mountain) { global_y = yy_heads[yy] + H_shared_y - 1; }
      else { global_y = yy_heads[yy] + H_shared_y; }

      for(H_shared_z = local_z; H_shared_z <= H_SHZ - 1; H_shared_z += NTZ_MM) {

        if constexpr (Z_is_mountain) { global_z = zz_heads[zz] + H_shared_z - 1; }
        else { global_z = zz_heads[zz] + H_shared_z; }

        int global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
        int H_shared_idx = H_shared_x + H_shared_y * H_SHX + H_shared_z * H_SHX * H_SHY;
        Hx_shmem[H_shared_idx] = Hx_pad[global_idx];
        Hy_shmem[H_shared_idx] = Hy_pad[global_idx];
        Hz_shmem[H_shared_idx] = Hz_pad[global_idx];

      }
    }
  }

  // load E ---------------------------------------------
  E_shared_x = local_x;
  global_x = xx_heads[xx] + E_shared_x;
  for(E_shared_y = local_y; E_shared_y <= E_SHY - 1; E_shared_y += NTY_MM) {
    global_y = yy_heads[yy] + E_shared_y;
    for(E_shared_z = local_z; E_shared_z <= E_SHZ - 1; E_shared_z += NTY_MM) {
      global_z = zz_heads[zz] + E_shared_z;
      int global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
      int E_shared_idx = E_shared_x + E_shared_y * E_SHX + E_shared_z * E_SHX * E_SHY;
      Ex_shmem[E_shared_idx] = Ex_pad[global_idx];
      Ey_shmem[E_shared_idx] = Ey_pad[global_idx];
      Ez_shmem[E_shared_idx] = Ez_pad[global_idx];
    }
  }

  bool loadE_HALO_needed;
  if constexpr (!X_is_mountain) { loadE_HALO_needed = true; }
  else { loadE_HALO_needed = false; }
  // if valley, tid = NTX_MM - 1 load one extra E at xx_heads[xx] + NTX_MM 
  if(loadE_HALO_needed && local_x == NTX_MM - 1) {
    E_shared_x = local_x + 1;
    global_x = xx_heads[xx] + E_shared_x;
    for(E_shared_y = local_y; E_shared_y <= E_SHY - 1; E_shared_y += NTY_MM) {
      global_y = yy_heads[yy] + E_shared_y;
      for(E_shared_z = local_z; E_shared_z <= E_SHZ - 1; E_shared_z += NTY_MM) {
        global_z = zz_heads[zz] + E_shared_z;
        int global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad; 
        int E_shared_idx = E_shared_x + E_shared_y * E_SHX + E_shared_z * E_SHX * E_SHY;
        Ex_shmem[E_shared_idx] = Ex_pad[global_idx];
        Ey_shmem[E_shared_idx] = Ey_pad[global_idx];
        Ez_shmem[E_shared_idx] = Ez_pad[global_idx];
      }
    }
  }

  __syncthreads();

  // calculation

  // X head and tail is refer to unpadded global_x
  int calE_head_X, calE_tail_X;
  int calH_head_X, calH_tail_X;
  // Y head and tail is refer to shared_y
  int calE_head_Y, calE_tail_Y;
  int calH_head_Y, calH_tail_Y;
  // Z head and tail is refer to shared_z
  int calE_head_Z, calE_tail_Z;
  int calH_head_Z, calH_tail_Z;
  for(int t = 0; t < BLT_MM; t++) {
    if constexpr (X_is_mountain) {
      calE_head_X = xx_heads[xx] + t;
      calE_tail_X = xx_heads[xx] + BLX_MM - 1 - t;
      calH_head_X = calE_head_X;
      calH_tail_X = calE_tail_X - 1;
    }
    else {
      calE_head_X = xx_heads[xx] + BLT_MM - t;
      calE_tail_X = xx_heads[xx] + BLX_MM - 1 - (BLT_MM - t -1);
      calH_head_X = calE_head_X - 1;
      calH_tail_X = calE_tail_X;
    }
    if constexpr (Y_is_mountain) {
      // notice the starting point of shared H is one pixel left than that of shared E
      calE_head_Y = t;
      calH_head_Y = calE_head_Y + 1;
      calE_tail_Y = E_SHY - t - 1; 
      calH_tail_Y = calE_tail_Y;
    }
    else {
      calE_head_Y = BLT_MM - t;
      calH_head_Y = calE_head_Y - 1;
      calE_tail_Y = E_SHY - (BLT_MM - t) - 1;
      calH_tail_Y = calE_tail_Y;
    }
    if constexpr (Z_is_mountain) {
      // notice the starting point of shared H is one pixel left than that of shared E
      calE_head_Z = t;
      calH_head_Z = calE_head_Z + 1;
      calE_tail_Z = E_SHZ - t - 1; 
      calH_tail_Z = calE_tail_Z;
    }
    else {
      calE_head_Z = BLT_MM - t;
      calH_head_Z = calE_head_Z - 1;
      calE_tail_Z = E_SHZ - (BLT_MM - t) - 1;
      calH_tail_Z = calE_tail_Z;
    }

    // update E 
    E_shared_x = local_x;
    if constexpr (X_is_mountain) { H_shared_x = E_shared_x + 1; }
    else { H_shared_x = E_shared_x; }
    global_x = xx_heads[xx] + local_x;
    for(E_shared_y = calE_head_Y + local_y; E_shared_y <= calE_tail_Y; E_shared_y += NTY_MM) {

      if constexpr (Y_is_mountain) { H_shared_y = E_shared_y + 1; }
      else { H_shared_y = E_shared_y; }
      global_y = yy_heads[yy] + E_shared_y;

      for(E_shared_z = calE_head_Z + local_z; E_shared_z <= calE_tail_Z; E_shared_z += NTZ_MM) {

        if constexpr (Z_is_mountain) { H_shared_z = E_shared_z + 1; }
        else { H_shared_z = E_shared_z; }
        global_z = zz_heads[zz] + E_shared_z;

        // we pad all the dimension, so need to substract LEFT_PAD here to correctly access constant arrays
        int global_idx = (global_x - LEFT_PAD_MM) + (global_y - LEFT_PAD_MM) * Nx + (global_z - LEFT_PAD_MM) * Nx * Ny;
        int E_shared_idx = E_shared_x + E_shared_y * E_SHX + E_shared_z * E_SHX * E_SHY;
        int H_shared_idx = H_shared_x + H_shared_y * H_SHX + H_shared_z * H_SHX * H_SHY;

        if(global_x >= 1 + LEFT_PAD_MM && global_x <= Nx - 2 + LEFT_PAD_MM &&
           global_y >= 1 + LEFT_PAD_MM && global_y <= Ny - 2 + LEFT_PAD_MM &&
           global_z >= 1 + LEFT_PAD_MM && global_z <= Nz - 2 + LEFT_PAD_MM &&
           global_x >= calE_head_X && global_x <= calE_tail_X) {

          Ex_shmem[E_shared_idx] = Cax[global_idx] * Ex_shmem[E_shared_idx] + Cbx[global_idx] *
                    ((Hz_shmem[H_shared_idx] - Hz_shmem[H_shared_idx - H_SHX]) - (Hy_shmem[H_shared_idx] - Hy_shmem[H_shared_idx - H_SHX * H_SHY]) - Jx[global_idx] * dx);

          Ey_shmem[E_shared_idx] = Cay[global_idx] * Ey_shmem[E_shared_idx] + Cby[global_idx] *
                    ((Hx_shmem[H_shared_idx] - Hx_shmem[H_shared_idx - H_SHX * H_SHY]) - (Hz_shmem[H_shared_idx] - Hz_shmem[H_shared_idx - 1]) - Jy[global_idx] * dx);

          Ez_shmem[E_shared_idx] = Caz[global_idx] * Ez_shmem[E_shared_idx] + Cbz[global_idx] *
                    ((Hy_shmem[H_shared_idx] - Hy_shmem[H_shared_idx - 1]) - (Hx_shmem[H_shared_idx] - Hx_shmem[H_shared_idx - H_SHX]) - Jz[global_idx] * dx);

        }
      }
    }

    __syncthreads();

    // update H 
    E_shared_x = local_x; 
    if constexpr (X_is_mountain) { H_shared_x = E_shared_x + 1; }
    else { H_shared_x = E_shared_x; }
    global_x = xx_heads[xx] + local_x;

    for(H_shared_y = calH_head_Y + local_y; H_shared_y <= calH_tail_Y; H_shared_y += NTY_MM) {

      if constexpr (Y_is_mountain) { E_shared_y = H_shared_y - 1; }
      else { E_shared_y = H_shared_y; }
      global_y = yy_heads[yy] + E_shared_y;

      for(H_shared_z = calH_head_Z + local_z; H_shared_z <= calH_tail_Z; H_shared_z += NTZ_MM) {

        if constexpr (Z_is_mountain) { E_shared_z = H_shared_z - 1; }
        else { E_shared_z = H_shared_z; }
        global_z = zz_heads[zz] + E_shared_z;

        int global_idx = (global_x - LEFT_PAD_MM) + (global_y - LEFT_PAD_MM) * Nx + (global_z - LEFT_PAD_MM) * Nx * Ny;
        int E_shared_idx = E_shared_x + E_shared_y * E_SHX + E_shared_z * E_SHX * E_SHY;
        int H_shared_idx = H_shared_x + H_shared_y * H_SHX + H_shared_z * H_SHX * H_SHY;

        if(global_x >= 1 + LEFT_PAD_MM && global_x <= Nx - 2 + LEFT_PAD_MM && 
           global_y >= 1 + LEFT_PAD_MM && global_y <= Ny - 2 + LEFT_PAD_MM && 
           global_z >= 1 + LEFT_PAD_MM && global_z <= Nz - 2 + LEFT_PAD_MM &&
           global_x >= calH_head_X && global_x <= calH_tail_X) {

          Hx_shmem[H_shared_idx] = Dax[global_idx] * Hx_shmem[H_shared_idx] + Dbx[global_idx] *
                    ((Ey_shmem[E_shared_idx + E_SHX * E_SHY] - Ey_shmem[E_shared_idx]) - (Ez_shmem[E_shared_idx + E_SHX] - Ez_shmem[E_shared_idx]) - Mx[global_idx] * dx);

          Hy_shmem[H_shared_idx] = Day[global_idx] * Hy_shmem[H_shared_idx] + Dby[global_idx] *
                    ((Ez_shmem[E_shared_idx + 1] - Ez_shmem[E_shared_idx]) - (Ex_shmem[E_shared_idx + E_SHX * E_SHY] - Ex_shmem[E_shared_idx]) - My[global_idx] * dx);

          Hz_shmem[H_shared_idx] = Daz[global_idx] * Hz_shmem[H_shared_idx] + Dbz[global_idx] *
                    ((Ex_shmem[E_shared_idx + E_SHX] - Ex_shmem[E_shared_idx]) - (Ey_shmem[E_shared_idx + 1] - Ey_shmem[E_shared_idx]) - Mz[global_idx] * dx);

        }
      }
    }

    __syncthreads();
  }

  // store back to global memory

  // X head and tail is refer to unpadded global_x
  int storeE_head_X, storeE_tail_X;
  int storeH_head_X, storeH_tail_X;

  if constexpr (X_is_mountain) {
    storeE_head_X = xx_heads[xx];
    storeE_tail_X = storeE_head_X + BLX_MM - 1;
    storeH_head_X = storeE_head_X;
    storeH_tail_X = storeE_tail_X - 1;
  }
  else {
    storeH_head_X = xx_heads[xx];
    storeH_tail_X = storeH_head_X + BLX_MM - 1;
    storeE_head_X = storeH_head_X + 1;
    storeE_tail_X = storeH_tail_X;
  }

  // Y head and tail is refer to shared_y
  // same for Z
  int storeE_head_Y, storeE_tail_Y;
  int storeH_head_Y, storeH_tail_Y;

  if constexpr (Y_is_mountain) {
    storeE_head_Y = 0;
    storeE_tail_Y = E_SHY - 1;
    storeH_head_Y = 1;
    storeH_tail_Y = H_SHY - 2;
  }
  else {
    storeE_head_Y = 1;
    storeE_tail_Y = E_SHY - 2;
    storeH_head_Y = 0;
    storeH_tail_Y = H_SHY - 1;
  }

  int storeE_head_Z, storeE_tail_Z;
  int storeH_head_Z, storeH_tail_Z;

  if constexpr (Z_is_mountain) {
    storeE_head_Z = 0;
    storeE_tail_Z = E_SHZ - 1;
    storeH_head_Z = 1;
    storeH_tail_Z = H_SHZ - 2;
  }
  else {
    storeE_head_Z = 1;
    storeE_tail_Z = E_SHZ - 2;
    storeH_head_Z = 0;
    storeH_tail_Z = H_SHZ - 1;
  }

  // store H ---------------------------------------------
  if constexpr (X_is_mountain) {
    H_shared_x = local_x + 1;
  }
  else {
    H_shared_x = local_x;
  }
  global_x = xx_heads[xx] + local_x;

  for(H_shared_y = storeH_head_Y + local_y; H_shared_y <= storeH_tail_Y; H_shared_y += NTY_MM) {

    if constexpr (Y_is_mountain) { global_y = yy_heads[yy] + H_shared_y - 1; }
    else { global_y = yy_heads[yy] + H_shared_y; }

    for(H_shared_z = storeH_head_Z + local_z; H_shared_z <= storeH_tail_Z; H_shared_z += NTZ_MM) {

      if constexpr (Z_is_mountain) { global_z = zz_heads[zz] + H_shared_z - 1; }
      else { global_z = zz_heads[zz] + H_shared_z; }

      if(global_x >= 1 + LEFT_PAD_MM && global_x <= Nx - 2 + LEFT_PAD_MM &&
         global_y >= 1 + LEFT_PAD_MM && global_y <= Ny - 2 + LEFT_PAD_MM &&
         global_z >= 1 + LEFT_PAD_MM && global_z <= Nz - 2 + LEFT_PAD_MM &&
         global_x >= storeH_head_X && global_x <= storeH_tail_X) {

        int global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
        int H_shared_idx = H_shared_x + H_shared_y * H_SHX + H_shared_z * H_SHX * H_SHY;
        Hx_pad[global_idx] = Hx_shmem[H_shared_idx];
        Hy_pad[global_idx] = Hy_shmem[H_shared_idx];
        Hz_pad[global_idx] = Hz_shmem[H_shared_idx];

      }
    }
  }

  // store E ---------------------------------------------
  E_shared_x = local_x;
  global_x = xx_heads[xx] + E_shared_x;

  for(E_shared_y = storeE_head_Y + local_y; E_shared_y <= storeE_tail_Y; E_shared_y += NTY_MM) {
    global_y = yy_heads[yy] + E_shared_y;
    for(E_shared_z = storeE_head_Z + local_z; E_shared_z <= storeE_tail_Z; E_shared_z += NTZ_MM) {
      global_z = zz_heads[zz] + E_shared_z;
      
      if(global_x >= 1 + LEFT_PAD_MM && global_x <= Nx - 2 + LEFT_PAD_MM && 
         global_y >= 1 + LEFT_PAD_MM && global_y <= Ny - 2 + LEFT_PAD_MM && 
         global_z >= 1 + LEFT_PAD_MM && global_z <= Nz - 2 + LEFT_PAD_MM &&
         global_x >= storeE_head_X && global_x <= storeE_tail_X) {

        int global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad; 
        int E_shared_idx = E_shared_x + E_shared_y * E_SHX + E_shared_z * E_SHX * E_SHY;  
        Ex_pad[global_idx] = Ex_shmem[E_shared_idx];
        Ey_pad[global_idx] = Ey_shmem[E_shared_idx];
        Ez_pad[global_idx] = Ez_shmem[E_shared_idx];

      }
    }
  }


}

template <int Iter, int MaxIter, typename Func>
__device__ inline void static_unroll(Func&& f) {
  if constexpr (Iter < MaxIter) {
    f(Iter);
    static_unroll<Iter + 1, MaxIter>(f);
  }
}

template<int t, bool X_is_mountain, bool Y_is_mountain, bool Z_is_mountain>
__device__ void timestep_body(
    int local_x, int local_y, int local_z,
    int xx, int yy, int zz,
    int Nx, int Ny, int Nz,
    float dx,
    const int* xx_heads, const int* yy_heads, const int* zz_heads,
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

    constexpr int H_SHX = (X_is_mountain)? BLX_MM + 1 : BLX_MM; 
    constexpr int H_SHY = (Y_is_mountain)? BLY_MM + 1 : BLY_MM;
    constexpr int E_SHX = (X_is_mountain)? BLX_MM : BLX_MM + 1; 
    constexpr int E_SHY = (Y_is_mountain)? BLY_MM : BLY_MM + 1;
    constexpr int E_SHZ = (Z_is_mountain)? BLZ_MM : BLZ_MM + 1;
    constexpr int BLT = BLT_MM;
    constexpr int BLX = BLX_MM;

    // === Compute calE/calH bounds using t (known at compile-time) ===
    constexpr int calE_head_X = X_is_mountain ? 0 + t : BLT - t;
    constexpr int calE_tail_X = X_is_mountain ? BLX - 1 - t : BLX - 1 - (BLT - t - 1);
    constexpr int calH_head_X = X_is_mountain ? calE_head_X : calE_head_X - 1;
    constexpr int calH_tail_X = X_is_mountain ? calE_tail_X - 1 : calE_tail_X;

    constexpr int calE_head_Y = Y_is_mountain ? t : BLT - t;
    constexpr int calE_tail_Y = Y_is_mountain ? E_SHY - t - 1 : E_SHY - (BLT - t) - 1;
    constexpr int calH_head_Y = Y_is_mountain ? calE_head_Y + 1 : calE_head_Y - 1;
    constexpr int calH_tail_Y = calE_tail_Y;

    constexpr int calE_head_Z = Z_is_mountain ? t : BLT - t;
    constexpr int calE_tail_Z = Z_is_mountain ? E_SHZ - t - 1 : E_SHZ - (BLT - t) - 1;
    constexpr int calH_head_Z = Z_is_mountain ? calE_head_Z + 1 : calE_head_Z - 1;
    constexpr int calH_tail_Z = calE_tail_Z;

    // === Update E ===
    int E_shared_x = local_x;
    int H_shared_x = X_is_mountain ? E_shared_x + 1 : E_shared_x;
    int global_x = xx_heads[xx] + local_x;

    constexpr int E_num_y_iters_cal = (calE_tail_Y - calE_head_Y + 1 + NTY_MM - 1) / NTY_MM;  
    constexpr int E_num_z_iters_cal = (calE_tail_Z - calE_head_Z + 1 + NTZ_MM - 1) / NTZ_MM;  

    #pragma unroll
    for(int E_y_iter = 0; E_y_iter < E_num_y_iters_cal; E_y_iter++) {
      int E_shared_y = calE_head_Y + local_y + E_y_iter * NTY_MM;
      if(E_shared_y <= calE_tail_Y) {

        #pragma unroll
        for(int E_z_iter = 0; E_z_iter < E_num_z_iters_cal; E_z_iter++) {
          int E_shared_z = calE_head_Z + local_z + E_z_iter * NTZ_MM;
          if(E_shared_z <= calE_tail_Z) {
            
            int H_shared_y = Y_is_mountain ? E_shared_y + 1 : E_shared_y;
            int global_y = yy_heads[yy] + E_shared_y;
            int H_shared_z = Z_is_mountain ? E_shared_z + 1 : E_shared_z;
            int global_z = zz_heads[zz] + E_shared_z;

            int global_idx = (global_x - LEFT_PAD_MM) + (global_y - LEFT_PAD_MM) * Nx + (global_z - LEFT_PAD_MM) * Nx * Ny;
            int E_shared_idx = E_shared_x + E_shared_y * E_SHX + E_shared_z * E_SHX * E_SHY;
            int H_shared_idx = H_shared_x + H_shared_y * H_SHX + H_shared_z * H_SHX * H_SHY;

            if (global_x >= 1 + LEFT_PAD_MM && global_x <= Nx - 2 + LEFT_PAD_MM &&
                global_y >= 1 + LEFT_PAD_MM && global_y <= Ny - 2 + LEFT_PAD_MM &&
                global_z >= 1 + LEFT_PAD_MM && global_z <= Nz - 2 + LEFT_PAD_MM &&
                global_x >= xx_heads[xx] + calE_head_X && global_x <= xx_heads[xx] + calE_tail_X) {

                // Update E
                Ex_shmem[E_shared_idx] = Cax[global_idx] * Ex_shmem[E_shared_idx] + Cbx[global_idx] *
                    ((Hz_shmem[H_shared_idx] - Hz_shmem[H_shared_idx - H_SHX]) -
                     (Hy_shmem[H_shared_idx] - Hy_shmem[H_shared_idx - H_SHX * H_SHY]) -
                     Jx[global_idx] * dx);

                Ey_shmem[E_shared_idx] = Cay[global_idx] * Ey_shmem[E_shared_idx] + Cby[global_idx] *
                    ((Hx_shmem[H_shared_idx] - Hx_shmem[H_shared_idx - H_SHX * H_SHY]) -
                     (Hz_shmem[H_shared_idx] - Hz_shmem[H_shared_idx - 1]) -
                     Jy[global_idx] * dx);

                Ez_shmem[E_shared_idx] = Caz[global_idx] * Ez_shmem[E_shared_idx] + Cbz[global_idx] *
                    ((Hy_shmem[H_shared_idx] - Hy_shmem[H_shared_idx - 1]) -
                     (Hx_shmem[H_shared_idx] - Hx_shmem[H_shared_idx - H_SHX]) -
                     Jz[global_idx] * dx);
            }
          }
        }
      }
    }

    __syncthreads();

    // === Update H ===
    constexpr int H_num_y_iters_cal = (calH_tail_Y - calH_head_Y + 1 + NTY_MM - 1) / NTY_MM;  
    constexpr int H_num_z_iters_cal = (calH_tail_Z - calH_head_Z + 1 + NTZ_MM - 1) / NTZ_MM;  

    #pragma unroll
    for(int H_y_iter = 0; H_y_iter < H_num_y_iters_cal; H_y_iter++) {
      int H_shared_y = calH_head_Y + local_y + H_y_iter * NTY_MM;
      if(H_shared_y <= calH_tail_Y) {

        #pragma unroll
        for(int H_z_iter = 0; H_z_iter < H_num_z_iters_cal; H_z_iter++) {
          int H_shared_z = calH_head_Z + local_z + H_z_iter * NTZ_MM;
          if(H_shared_z <= calH_tail_Z) {

            int E_shared_y = Y_is_mountain ? H_shared_y - 1 : H_shared_y;
            int global_y = yy_heads[yy] + E_shared_y;
            int E_shared_z = Z_is_mountain ? H_shared_z - 1 : H_shared_z;
            int global_z = zz_heads[zz] + E_shared_z;

            int global_idx = (global_x - LEFT_PAD_MM) + (global_y - LEFT_PAD_MM) * Nx + (global_z - LEFT_PAD_MM) * Nx * Ny;
            int E_shared_idx = E_shared_x + E_shared_y * E_SHX + E_shared_z * E_SHX * E_SHY;
            int H_shared_idx = H_shared_x + H_shared_y * H_SHX + H_shared_z * H_SHX * H_SHY;
            if (global_x >= 1 + LEFT_PAD_MM && global_x <= Nx - 2 + LEFT_PAD_MM &&
                global_y >= 1 + LEFT_PAD_MM && global_y <= Ny - 2 + LEFT_PAD_MM &&
                global_z >= 1 + LEFT_PAD_MM && global_z <= Nz - 2 + LEFT_PAD_MM &&
                global_x >= xx_heads[xx] + calH_head_X && global_x <= xx_heads[xx] + calH_tail_X) {

                // Update H
                Hx_shmem[H_shared_idx] = Dax[global_idx] * Hx_shmem[H_shared_idx] + Dbx[global_idx] *
                    ((Ey_shmem[E_shared_idx + E_SHX * E_SHY] - Ey_shmem[E_shared_idx]) -
                     (Ez_shmem[E_shared_idx + E_SHX] - Ez_shmem[E_shared_idx]) -
                     Mx[global_idx] * dx);

                Hy_shmem[H_shared_idx] = Day[global_idx] * Hy_shmem[H_shared_idx] + Dby[global_idx] *
                    ((Ez_shmem[E_shared_idx + 1] - Ez_shmem[E_shared_idx]) -
                     (Ex_shmem[E_shared_idx + E_SHX * E_SHY] - Ex_shmem[E_shared_idx]) -
                     My[global_idx] * dx);

                Hz_shmem[H_shared_idx] = Daz[global_idx] * Hz_shmem[H_shared_idx] + Dbz[global_idx] *
                    ((Ex_shmem[E_shared_idx + E_SHX] - Ex_shmem[E_shared_idx]) -
                     (Ey_shmem[E_shared_idx + 1] - Ey_shmem[E_shared_idx]) -
                     Mz[global_idx] * dx);
            }
          }
        }
      }
    }
    
    __syncthreads();
}

template<int t, bool X_is_mountain, bool Y_is_mountain, bool Z_is_mountain>
__device__ void unroll_timesteps(
    int local_x, int local_y, int local_z,
    int xx, int yy, int zz,
    int Nx, int Ny, int Nz,
    float dx,
    const int* xx_heads, const int* yy_heads, const int* zz_heads,
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
    local_x, local_y, local_z,
    xx, yy, zz,
    Nx, Ny, Nz,
    dx,
    xx_heads, yy_heads, zz_heads,
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
    if constexpr (t + 1 < BLT_MM) {
        unroll_timesteps<t + 1, X_is_mountain, Y_is_mountain, Z_is_mountain>(
        local_x, local_y, local_z,
        xx, yy, zz,
        Nx, Ny, Nz,
        dx,
        xx_heads, yy_heads, zz_heads,
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
__global__ void updateEH_mix_mapping_kernel_unroll(float* Ex_pad, float* Ey_pad, float* Ez_pad,
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

  const unsigned int xx = block_id % xx_num;
  const unsigned int yy = (block_id / xx_num) % yy_num;
  const unsigned int zz = block_id / (xx_num * yy_num);

  const int local_x = thread_id % NTX_MM;
  const int local_y = (thread_id / NTX_MM) % NTY_MM;
  const int local_z = thread_id / (NTX_MM * NTY_MM);

  int global_x, global_y, global_z;
  int H_shared_x, H_shared_y, H_shared_z;
  int E_shared_x, E_shared_y, E_shared_z;

  // declare shared memory
  constexpr int H_SHX = (X_is_mountain)? BLX_MM + 1 : BLX_MM; 
  constexpr int H_SHY = (Y_is_mountain)? BLY_MM + 1 : BLY_MM;
  constexpr int H_SHZ = (Z_is_mountain)? BLZ_MM + 1 : BLZ_MM;
  constexpr int E_SHX = (X_is_mountain)? BLX_MM : BLX_MM + 1; 
  constexpr int E_SHY = (Y_is_mountain)? BLY_MM : BLY_MM + 1;
  constexpr int E_SHZ = (Z_is_mountain)? BLZ_MM : BLZ_MM + 1;
  __shared__ float Hx_shmem[H_SHX * H_SHY * H_SHZ];
  __shared__ float Hy_shmem[H_SHX * H_SHY * H_SHZ];
  __shared__ float Hz_shmem[H_SHX * H_SHY * H_SHZ];
  __shared__ float Ex_shmem[E_SHX * E_SHY * E_SHZ];
  __shared__ float Ey_shmem[E_SHX * E_SHY * E_SHZ];
  __shared__ float Ez_shmem[E_SHX * E_SHY * E_SHZ];

  // load shared memory
  // load H ---------------------------------------------
  if constexpr (X_is_mountain) {
    H_shared_x = local_x + 1;
  }
  else {
    H_shared_x = local_x;
  }
  global_x = xx_heads[xx] + local_x;

  // one-to-many mapping in Y and Z
  constexpr int H_num_y_iters_load = (H_SHY + NTY_MM - 1) / NTY_MM;
  constexpr int H_num_z_iters_load = (H_SHZ + NTZ_MM - 1) / NTZ_MM;

  #pragma unroll
  for(int H_y_iter = 0; H_y_iter < H_num_y_iters_load; H_y_iter++) {
    H_shared_y = local_y + H_y_iter * NTY_MM;
    if(H_shared_y <= H_SHY - 1) {

      #pragma unroll
      for(int H_z_iter = 0; H_z_iter < H_num_z_iters_load; H_z_iter++) {
        H_shared_z = local_z + H_z_iter * NTZ_MM; 
        if(H_shared_z <= H_SHZ - 1) {

          if constexpr (Y_is_mountain) { global_y = yy_heads[yy] + H_shared_y - 1; }
          else { global_y = yy_heads[yy] + H_shared_y; }
          if constexpr (Z_is_mountain) { global_z = zz_heads[zz] + H_shared_z - 1; }
          else { global_z = zz_heads[zz] + H_shared_z; }
     
          int global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
          int H_shared_idx = H_shared_x + H_shared_y * H_SHX + H_shared_z * H_SHX * H_SHY;
          Hx_shmem[H_shared_idx] = Hx_pad[global_idx];
          Hy_shmem[H_shared_idx] = Hy_pad[global_idx];
          Hz_shmem[H_shared_idx] = Hz_pad[global_idx];
        }
      }
    }
  }
  
  bool loadH_HALO_needed;
  if constexpr (X_is_mountain) { loadH_HALO_needed = true; }
  else { loadH_HALO_needed = false; }
  // if mountain, tid = 0 load one extra H at xx_heads[xx] - 1
  if(loadH_HALO_needed && local_x == 0) {
    H_shared_x = 0;
    global_x = xx_heads[xx] + H_shared_x - 1;

    #pragma unroll
    for(int H_y_iter = 0; H_y_iter < H_num_y_iters_load; H_y_iter++) {
      H_shared_y = local_y + H_y_iter * NTY_MM;
      if(H_shared_y <= H_SHY - 1) {

        #pragma unroll
        for(int H_z_iter = 0; H_z_iter < H_num_z_iters_load; H_z_iter++) {
          H_shared_z = local_z + H_z_iter * NTZ_MM; 
          if(H_shared_z <= H_SHZ - 1) {

            if constexpr (Y_is_mountain) { global_y = yy_heads[yy] + H_shared_y - 1; }
            else { global_y = yy_heads[yy] + H_shared_y; }
            if constexpr (Z_is_mountain) { global_z = zz_heads[zz] + H_shared_z - 1; }
            else { global_z = zz_heads[zz] + H_shared_z; }
       
            int global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
            int H_shared_idx = H_shared_x + H_shared_y * H_SHX + H_shared_z * H_SHX * H_SHY;
            Hx_shmem[H_shared_idx] = Hx_pad[global_idx];
            Hy_shmem[H_shared_idx] = Hy_pad[global_idx];
            Hz_shmem[H_shared_idx] = Hz_pad[global_idx];
          }
        }
      }
    }
  }

  // load E ---------------------------------------------
  E_shared_x = local_x;
  global_x = xx_heads[xx] + E_shared_x;

  constexpr int E_num_y_iters_load = (E_SHY + NTY_MM - 1) / NTY_MM;
  constexpr int E_num_z_iters_load = (E_SHZ + NTZ_MM - 1) / NTZ_MM;

  #pragma unroll
  for(int E_y_iter = 0; E_y_iter < E_num_y_iters_load; E_y_iter++) {
    E_shared_y = local_y + E_y_iter * NTY_MM;
    if(E_shared_y <= E_SHY - 1) {

      #pragma unroll
      for(int E_z_iter = 0; E_z_iter < E_num_z_iters_load; E_z_iter++) {
        E_shared_z = local_z + E_z_iter * NTZ_MM;
        if(E_shared_z <= E_SHZ - 1) {

          global_y = yy_heads[yy] + E_shared_y;
          global_z = zz_heads[zz] + E_shared_z;
          int global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
          int E_shared_idx = E_shared_x + E_shared_y * E_SHX + E_shared_z * E_SHX * E_SHY;
          Ex_shmem[E_shared_idx] = Ex_pad[global_idx];
          Ey_shmem[E_shared_idx] = Ey_pad[global_idx];
          Ez_shmem[E_shared_idx] = Ez_pad[global_idx];
        }
      }
    }
  }

  bool loadE_HALO_needed;
  if constexpr (!X_is_mountain) { loadE_HALO_needed = true; }
  else { loadE_HALO_needed = false; }
  // if valley, tid = NTX_MM - 1 load one extra E at xx_heads[xx] + NTX_MM 
  if(loadE_HALO_needed && local_x == NTX_MM - 1) {
    E_shared_x = local_x + 1;
    global_x = xx_heads[xx] + E_shared_x;

    #pragma unroll
    for(int E_y_iter = 0; E_y_iter < E_num_y_iters_load; E_y_iter++) {
      E_shared_y = local_y + E_y_iter * NTY_MM;
      if(E_shared_y <= E_SHY - 1) {

        #pragma unroll
        for(int E_z_iter = 0; E_z_iter < E_num_z_iters_load; E_z_iter++) {
          E_shared_z = local_z + E_z_iter * NTZ_MM;
          if(E_shared_z <= E_SHZ - 1) {

            global_y = yy_heads[yy] + E_shared_y;
            global_z = zz_heads[zz] + E_shared_z;
            int global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
            int E_shared_idx = E_shared_x + E_shared_y * E_SHX + E_shared_z * E_SHX * E_SHY;
            Ex_shmem[E_shared_idx] = Ex_pad[global_idx];
            Ey_shmem[E_shared_idx] = Ey_pad[global_idx];
            Ez_shmem[E_shared_idx] = Ez_pad[global_idx];
          }
        }
      }
    }
  }

  __syncthreads();

  // calculation

  unroll_timesteps<0, X_is_mountain, Y_is_mountain, Z_is_mountain>(
  local_x, local_y, local_z,
  xx, yy, zz,
  Nx, Ny, Nz,
  dx,
  xx_heads, yy_heads, zz_heads,
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

  // X head and tail is refer to unpadded global_x
  int storeE_head_X, storeE_tail_X;
  int storeH_head_X, storeH_tail_X;

  if constexpr (X_is_mountain) {
    storeE_head_X = xx_heads[xx];
    storeE_tail_X = storeE_head_X + BLX_MM - 1;
    storeH_head_X = storeE_head_X;
    storeH_tail_X = storeE_tail_X - 1;
  }
  else {
    storeH_head_X = xx_heads[xx];
    storeH_tail_X = storeH_head_X + BLX_MM - 1;
    storeE_head_X = storeH_head_X + 1;
    storeE_tail_X = storeH_tail_X;
  }

  // Y head and tail is refer to shared_y
  // same for Z
  int storeE_head_Y, storeE_tail_Y;
  int storeH_head_Y, storeH_tail_Y;

  if constexpr (Y_is_mountain) {
    storeE_head_Y = 0;
    storeE_tail_Y = E_SHY - 1;
    storeH_head_Y = 1;
    storeH_tail_Y = H_SHY - 2;
  }
  else {
    storeE_head_Y = 1;
    storeE_tail_Y = E_SHY - 2;
    storeH_head_Y = 0;
    storeH_tail_Y = H_SHY - 1;
  }

  int storeE_head_Z, storeE_tail_Z;
  int storeH_head_Z, storeH_tail_Z;

  if constexpr (Z_is_mountain) {
    storeE_head_Z = 0;
    storeE_tail_Z = E_SHZ - 1;
    storeH_head_Z = 1;
    storeH_tail_Z = H_SHZ - 2;
  }
  else {
    storeE_head_Z = 1;
    storeE_tail_Z = E_SHZ - 2;
    storeH_head_Z = 0;
    storeH_tail_Z = H_SHZ - 1;
  }

  // store H ---------------------------------------------
  if constexpr (X_is_mountain) {
    H_shared_x = local_x + 1;
  }
  else {
    H_shared_x = local_x;
  }
  global_x = xx_heads[xx] + local_x;

  // got H_SHY(Z) elements to store in Y(Z) dimension
  constexpr int H_num_y_iters_store = (H_SHY + NTY_MM - 1) / NTY_MM; 
  constexpr int H_num_z_iters_store = (H_SHZ + NTZ_MM - 1) / NTZ_MM; 

  #pragma unroll
  for(int H_y_iter = 0; H_y_iter < H_num_y_iters_store; H_y_iter++) {
    H_shared_y = storeH_head_Y + local_y + H_y_iter * NTY_MM;
    if(H_shared_y <= storeH_tail_Y) {

      #pragma unroll
      for(int H_z_iter = 0; H_z_iter < H_num_z_iters_store; H_z_iter++) {
        H_shared_z = storeH_head_Z + local_z + H_z_iter * NTZ_MM;
        if(H_shared_z <= storeH_tail_Z) {
          
          if constexpr (Y_is_mountain) { global_y = yy_heads[yy] + H_shared_y - 1; }
          else { global_y = yy_heads[yy] + H_shared_y; }
          if constexpr (Z_is_mountain) { global_z = zz_heads[zz] + H_shared_z - 1; }
          else { global_z = zz_heads[zz] + H_shared_z; }

          if(global_x >= 1 + LEFT_PAD_MM && global_x <= Nx - 2 + LEFT_PAD_MM &&
             global_y >= 1 + LEFT_PAD_MM && global_y <= Ny - 2 + LEFT_PAD_MM &&
             global_z >= 1 + LEFT_PAD_MM && global_z <= Nz - 2 + LEFT_PAD_MM &&
             global_x >= storeH_head_X && global_x <= storeH_tail_X) {

            int global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
            int H_shared_idx = H_shared_x + H_shared_y * H_SHX + H_shared_z * H_SHX * H_SHY;
            Hx_pad[global_idx] = Hx_shmem[H_shared_idx];
            Hy_pad[global_idx] = Hy_shmem[H_shared_idx];
            Hz_pad[global_idx] = Hz_shmem[H_shared_idx];
          }
        }
      }
    } 
  }

  // store E ---------------------------------------------
  E_shared_x = local_x;
  global_x = xx_heads[xx] + E_shared_x;

  constexpr int E_num_y_iters_store = (E_SHY + NTY_MM - 1) / NTY_MM; 
  constexpr int E_num_z_iters_store = (E_SHZ + NTZ_MM - 1) / NTZ_MM; 

  #pragma unroll
  for(int E_y_iter = 0; E_y_iter < E_num_y_iters_store; E_y_iter++) {
    E_shared_y = storeE_head_Y + local_y + E_y_iter * NTY_MM;
    if(E_shared_y <= storeE_tail_Y) {

      #pragma unroll
      for(int E_z_iter = 0; E_z_iter < E_num_z_iters_store; E_z_iter++) {
        E_shared_z = storeE_head_Z + local_z + E_z_iter * NTZ_MM;
        if(E_shared_z <= storeE_tail_Z) {

          global_y = yy_heads[yy] + E_shared_y;
          global_z = zz_heads[zz] + E_shared_z;

          if(global_x >= 1 + LEFT_PAD_MM && global_x <= Nx - 2 + LEFT_PAD_MM && 
             global_y >= 1 + LEFT_PAD_MM && global_y <= Ny - 2 + LEFT_PAD_MM && 
             global_z >= 1 + LEFT_PAD_MM && global_z <= Nz - 2 + LEFT_PAD_MM &&
             global_x >= storeE_head_X && global_x <= storeE_tail_X) {

            int global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad; 
            int E_shared_idx = E_shared_x + E_shared_y * E_SHX + E_shared_z * E_SHX * E_SHY;  
            Ex_pad[global_idx] = Ex_shmem[E_shared_idx];
            Ey_pad[global_idx] = Ey_shmem[E_shared_idx];
            Ez_pad[global_idx] = Ez_shmem[E_shared_idx];

          }
        }
      }
    }
  }
  
}



#endif


































