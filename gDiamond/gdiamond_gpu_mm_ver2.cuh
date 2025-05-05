#ifndef GDIAMOND_GPU_MM_VER2_CUH
#define GDIAMOND_GPU_MM_VER2_CUH

#include "gdiamond.hpp"
#include "kernels_mm_ver2.cuh"
#include <cuda_runtime.h>

// handle errors in CUDA call
#define CUDACHECK(call)                                                        \
{                                                                          \
   const cudaError_t error = call;                                         \
   if (error != cudaSuccess)                                               \
   {                                                                       \
      printf("Error: %s:%d, ", __FILE__, __LINE__);                        \
      printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));   \
      exit(1);                                                             \
   }                                                                       \
} (void)0  // Ensures a semicolon is required after the macro call.

namespace gdiamond {

template <bool X_is_mountain, bool Y_is_mountain, bool Z_is_mountain>
void gDiamond::_updateEH_mix_mapping_ver2(std::vector<float>& Ex_pad, std::vector<float>& Ey_pad, std::vector<float>& Ez_pad,
                                          std::vector<float>& Hx_pad, std::vector<float>& Hy_pad, std::vector<float>& Hz_pad,
                                          const std::vector<float>& Cax, const std::vector<float>& Cbx,
                                          const std::vector<float>& Cay, const std::vector<float>& Cby,
                                          const std::vector<float>& Caz, const std::vector<float>& Cbz,
                                          const std::vector<float>& Dax, const std::vector<float>& Dbx,
                                          const std::vector<float>& Day, const std::vector<float>& Dby,
                                          const std::vector<float>& Daz, const std::vector<float>& Dbz,
                                          const std::vector<float>& Jx, const std::vector<float>& Jy, const std::vector<float>& Jz,
                                          const std::vector<float>& Mx, const std::vector<float>& My, const std::vector<float>& Mz,
                                          float dx,
                                          int Nx, int Ny, int Nz,
                                          int Nx_pad, int Ny_pad, int Nz_pad,
                                          int xx_num, int yy_num, int zz_num,
                                          const std::vector<int>& xx_heads,
                                          const std::vector<int>& yy_heads,
                                          const std::vector<int>& zz_heads,
                                          size_t block_size,
                                          size_t grid_size) {

  for(size_t block_id = 0; block_id < grid_size; block_id++) {

    const int xx = block_id % xx_num;
    const int yy = (block_id / xx_num) % yy_num;
    const int zz = block_id / (xx_num * yy_num);
    int local_x, local_y, local_z;
    int global_x, global_y, global_z;
    int H_shared_x, H_shared_y, H_shared_z;
    int E_shared_x, E_shared_y, E_shared_z;
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
    float Hx_shmem[H_SHX * H_SHY * H_SHZ];
    float Hy_shmem[H_SHX * H_SHY * H_SHZ];
    float Hz_shmem[H_SHX * H_SHY * H_SHZ];
    float Ex_shmem[E_SHX * E_SHY * E_SHZ];
    float Ey_shmem[E_SHX * E_SHY * E_SHZ];
    float Ez_shmem[E_SHX * E_SHY * E_SHZ];

    // if(xx == 0 && yy == 0 && zz == 0) {
    //   std::cout << "H_SHX = " << H_SHX << ", H_SHY = " << H_SHY << ", H_SHZ = " << H_SHZ << "\n";
    //   std::cout << "E_SHX = " << E_SHX << ", E_SHY = " << E_SHY << ", E_SHZ = " << E_SHZ << "\n";
    // }

    // load shared memory
    for(size_t thread_id = 0; thread_id < block_size; thread_id++) {
      
      // X dimension has 1 extra HALO load, one thread load one element, 
      // if mountain, tid = 0 load one extra H at xx_heads[xx] - 1
      // if valley, tid = NTX_MM_V2 - 1 load one extra E at xx_heads[xx] + NTX_MM_V2 
      // same thing applys to Y, Z dimension

      local_x = thread_id % NTX_MM_V2;
      local_y = (thread_id / NTX_MM_V2) % NTY_MM_V2;
      local_z = thread_id / (NTX_MM_V2 * NTY_MM_V2);

      // load H ---------------------------------------------
      if constexpr (X_is_mountain) { H_shared_x = local_x + 1; }
      else { H_shared_x = local_x; }
      if constexpr (Y_is_mountain) { H_shared_y = local_y + 1; }
      else { H_shared_y = local_y; }
      if constexpr (Z_is_mountain) { H_shared_z = local_z + 1; }
      else { H_shared_z = local_z; }
      global_x = xx_heads[xx] + local_x;
      global_y = yy_heads[yy] + local_y;
      global_z = zz_heads[zz] + local_z;

      global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
      H_shared_idx = H_shared_x + H_shared_y * H_SHX + H_shared_z * H_SHX * H_SHY;
      Hx_shmem[H_shared_idx] = Hx_pad[global_idx];
      Hy_shmem[H_shared_idx] = Hy_pad[global_idx];
      Hz_shmem[H_shared_idx] = Hz_pad[global_idx];

      bool loadH_HALO_needed_x;
      bool loadH_HALO_needed_y;
      bool loadH_HALO_needed_z;
      if constexpr (X_is_mountain) { loadH_HALO_needed_x = true; }
      else { loadH_HALO_needed_x = false; }
      if constexpr (Y_is_mountain) { loadH_HALO_needed_y = true; }
      else { loadH_HALO_needed_y = false; }
      if constexpr (Z_is_mountain) { loadH_HALO_needed_z = true; }
      else { loadH_HALO_needed_z = false; }

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

      
      // load E ---------------------------------------------
      // if(xx == 0 && yy == 0 && zz == 0) {
      //   std::cout << "-------------------------------------------------\n";
      //   std::cout << "thread_id = " << thread_id << "\n";
      //   std::cout << "local_x = " << local_x << ", local_y = " << local_y << ", local_z = " << local_z << "\n";
      // }
      E_shared_x = local_x;
      E_shared_y = local_y;
      E_shared_z = local_z;
      global_x = xx_heads[xx] + E_shared_x;
      global_y = yy_heads[yy] + E_shared_y;
      global_z = zz_heads[zz] + E_shared_z;

      global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad; 
      E_shared_idx = E_shared_x + E_shared_y * E_SHX + E_shared_z * E_SHX * E_SHY;
      Ex_shmem[E_shared_idx] = Ex_pad[global_idx];
      Ey_shmem[E_shared_idx] = Ey_pad[global_idx];
      Ez_shmem[E_shared_idx] = Ez_pad[global_idx];

      bool loadE_HALO_needed_x;
      bool loadE_HALO_needed_y;
      bool loadE_HALO_needed_z;
      if constexpr (!X_is_mountain) { loadE_HALO_needed_x = true; }
      else { loadE_HALO_needed_x = false; }
      if constexpr (!Y_is_mountain) { loadE_HALO_needed_y = true; }
      else { loadE_HALO_needed_y = false; }
      if constexpr (!Z_is_mountain) { loadE_HALO_needed_z = true; }
      else { loadE_HALO_needed_z = false; }

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
      
    }

    // calculation
    for(int t = 0; t < BLT_MM_V2; t++) {
      
      // X head and tail is refer to unpadded global_x
      // same thing applys to Y and Z 
      int calE_head_X, calE_tail_X;
      int calH_head_X, calH_tail_X;
      int calE_head_Y, calE_tail_Y;
      int calH_head_Y, calH_tail_Y;
      int calE_head_Z, calE_tail_Z;
      int calH_head_Z, calH_tail_Z;

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

      // if(xx == 0 && yy == 0 && zz == 0) {
      //   std::cout << "t = " << t << "\n";
      //   std::cout << "calE_head_X = " << calE_head_X << ", calE_tail_X = " << calE_tail_X
      //             << ", calH_head_X = " << calH_head_X << ", calH_tail_X = " << calH_tail_X << "\n";
      //   std::cout << "calE_head_Y = " << calE_head_Y << ", calE_tail_Y = " << calE_tail_Y
      //             << ", calH_head_Y = " << calH_head_Y << ", calH_tail_Y = " << calH_tail_Y << "\n";
      //   std::cout << "calE_head_Z = " << calE_head_Z << ", calE_tail_Z = " << calE_tail_Z
      //             << ", calH_head_Z = " << calH_head_Z << ", calH_tail_Z = " << calH_tail_Z << "\n";
      // }

      // update E
      for(size_t thread_id = 0; thread_id < block_size; thread_id++) {
        local_x = thread_id % NTX_MM_V2;
        local_y = (thread_id / NTX_MM_V2) % NTY_MM_V2;
        local_z = thread_id / (NTX_MM_V2 * NTY_MM_V2);

        E_shared_x = local_x;
        E_shared_y = local_y;
        E_shared_z = local_z;
        if constexpr (X_is_mountain) { H_shared_x = E_shared_x + 1; }
        else { H_shared_x = E_shared_x; }
        if constexpr (Y_is_mountain) { H_shared_y = E_shared_y + 1; }
        else { H_shared_y = E_shared_y; }
        if constexpr (Z_is_mountain) { H_shared_z = E_shared_z + 1; }
        else { H_shared_z = E_shared_z; }
        global_x = xx_heads[xx] + local_x;
        global_y = yy_heads[yy] + local_y;
        global_z = zz_heads[zz] + local_z;

        // we pad all the dimension, so need to substract LEFT_PAD here to correctly access constant arrays
        global_idx = (global_x - LEFT_PAD_MM_V2) + (global_y - LEFT_PAD_MM_V2) * Nx + (global_z - LEFT_PAD_MM_V2) * Nx * Ny;
        E_shared_idx = E_shared_x + E_shared_y * E_SHX + E_shared_z * E_SHX * E_SHY;
        H_shared_idx = H_shared_x + H_shared_y * H_SHX + H_shared_z * H_SHX * H_SHY;

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

          // if(xx == 0 && yy == 0 && zz == 0) {
          //   std::cout << "------------------------------------------------------\n";
          //   std::cout << "t = " << t << "\n";
          //   std::cout << "local_x = " << local_x << ", local_y = " << local_y << ", local_z = " << local_z << "\n";
          //   std::cout << "E_shared_x = " << E_shared_x << ", E_shared_y = " << E_shared_y << ", E_shared_z = " << E_shared_z << "\n";
          //   std::cout << "H_shared_x = " << H_shared_x << ", H_shared_y = " << H_shared_y << ", H_shared_z = " << H_shared_z << "\n";
          //   std::cout << "global_x = " << global_x << ", global_y = " << global_y << ", global_z = " << global_z << "\n";
          // }
        }
      }

      // update H 
      for(size_t thread_id = 0; thread_id < block_size; thread_id++) {
        local_x = thread_id % NTX_MM_V2;
        local_y = (thread_id / NTX_MM_V2) % NTY_MM_V2;
        local_z = thread_id / (NTX_MM_V2 * NTY_MM_V2);

        E_shared_x = local_x;
        E_shared_y = local_y;
        E_shared_z = local_z;
        if constexpr (X_is_mountain) { H_shared_x = E_shared_x + 1; }
        else { H_shared_x = E_shared_x; }
        if constexpr (Y_is_mountain) { H_shared_y = E_shared_y + 1; }
        else { H_shared_y = E_shared_y; }
        if constexpr (Z_is_mountain) { H_shared_z = E_shared_z + 1; }
        else { H_shared_z = E_shared_z; }
        global_x = xx_heads[xx] + local_x;
        global_y = yy_heads[yy] + local_y;
        global_z = zz_heads[zz] + local_z;

        global_idx = (global_x - LEFT_PAD_MM_V2) + (global_y - LEFT_PAD_MM_V2) * Nx + (global_z - LEFT_PAD_MM_V2) * Nx * Ny;
        E_shared_idx = E_shared_x + E_shared_y * E_SHX + E_shared_z * E_SHX * E_SHY;
        H_shared_idx = H_shared_x + H_shared_y * H_SHX + H_shared_z * H_SHX * H_SHY;

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

          // if(xx == 1 && yy == 1 && zz == 1) {
          //   std::cout << "------------------------------------------------------\n";
          //   std::cout << "t = " << t << "\n";
          //   std::cout << "local_x = " << local_x << ", local_y = " << local_y << ", local_z = " << local_z << "\n";
          //   std::cout << "E_shared_x = " << E_shared_x << ", E_shared_y = " << E_shared_y << ", E_shared_z = " << E_shared_z << "\n";
          //   std::cout << "H_shared_x = " << H_shared_x << ", H_shared_y = " << H_shared_y << ", H_shared_z = " << H_shared_z << "\n";
          //   std::cout << "global_x = " << global_x << ", global_y = " << global_y << ", global_z = " << global_z << "\n";
          // }

        }
      }
    }

    // store back to global memory

    // X head and tail is refer to unpadded global_x
    // same thing applys to Y and Z
    int storeE_head_X, storeE_tail_X;
    int storeH_head_X, storeH_tail_X;
    int storeE_head_Y, storeE_tail_Y;
    int storeH_head_Y, storeH_tail_Y;
    int storeE_head_Z, storeE_tail_Z;
    int storeH_head_Z, storeH_tail_Z;

    if constexpr (X_is_mountain) {
      storeE_head_X = xx_heads[xx];
      storeE_tail_X = storeE_head_X + BLX_MM_V2 - 1;
      storeH_head_X = storeE_head_X;
      storeH_tail_X = storeE_tail_X - 1;
    }
    else {
      storeH_head_X = xx_heads[xx];
      storeH_tail_X = storeH_head_X + BLX_MM_V2 - 1;
      storeE_head_X = storeH_head_X + 1;
      storeE_tail_X = storeH_tail_X;
    }  

    if constexpr (Y_is_mountain) {
      storeE_head_Y = yy_heads[yy];
      storeE_tail_Y = storeE_head_Y + BLY_MM_V2 - 1;
      storeH_head_Y = storeE_head_Y;
      storeH_tail_Y = storeE_tail_Y - 1;
    }
    else {
      storeH_head_Y = yy_heads[yy];
      storeH_tail_Y = storeH_head_Y + BLY_MM_V2 - 1;
      storeE_head_Y = storeH_head_Y + 1;
      storeE_tail_Y = storeH_tail_Y;
    }

    if constexpr (Z_is_mountain) {
      storeE_head_Z = zz_heads[zz];
      storeE_tail_Z = storeE_head_Z + BLZ_MM_V2 - 1;
      storeH_head_Z = storeE_head_Z;
      storeH_tail_Z = storeE_tail_Z - 1;
    }
    else {
      storeH_head_Z = zz_heads[zz];
      storeH_tail_Z = storeH_head_Z + BLZ_MM_V2 - 1;
      storeE_head_Z = storeH_head_Z + 1;
      storeE_tail_Z = storeH_tail_Z;
    }

    // if(xx == 0 && yy == 0 && zz == 0) {
    //   std::cout << "storeE_head_X = " << storeE_head_X << ", storeE_tail_X = " << storeE_tail_X
    //             << ", storeH_head_X = " << storeH_head_X << ", storeH_tail_X = " << storeH_tail_X << "\n";
    //   std::cout << "storeE_head_Y = " << storeE_head_Y << ", storeE_tail_Y = " << storeE_tail_Y
    //             << ", storeH_head_Y = " << storeH_head_Y << ", storeH_tail_Y = " << storeH_tail_Y << "\n";
    //   std::cout << "storeE_head_Z = " << storeE_head_Z << ", storeE_tail_Z = " << storeE_tail_Z
    //             << ", storeH_head_Z = " << storeH_head_Z << ", storeH_tail_Z = " << storeH_tail_Z << "\n";
    // }

    for(size_t thread_id = 0; thread_id < block_size; thread_id++) {
      local_x = thread_id % NTX_MM_V2;
      local_y = (thread_id / NTX_MM_V2) % NTY_MM_V2;
      local_z = thread_id / (NTX_MM_V2 * NTY_MM_V2);

      // store H ---------------------------------------------
      if constexpr (X_is_mountain) { H_shared_x = local_x + 1; }
      else { H_shared_x = local_x; }
      if constexpr (Y_is_mountain) { H_shared_y = local_y + 1; }
      else { H_shared_y = local_y; }
      if constexpr (Z_is_mountain) { H_shared_z = local_z + 1; }
      else { H_shared_z = local_z; }
      global_x = xx_heads[xx] + local_x;
      global_y = yy_heads[yy] + local_y;
      global_z = zz_heads[zz] + local_z;

      if(global_x >= 1 + LEFT_PAD_MM_V2 && global_x <= Nx - 2 + LEFT_PAD_MM_V2 &&
         global_y >= 1 + LEFT_PAD_MM_V2 && global_y <= Ny - 2 + LEFT_PAD_MM_V2 &&
         global_z >= 1 + LEFT_PAD_MM_V2 && global_z <= Nz - 2 + LEFT_PAD_MM_V2 &&
         global_x >= storeH_head_X && global_x <= storeH_tail_X && 
         global_y >= storeH_head_Y && global_y <= storeH_tail_Y && 
         global_z >= storeH_head_Z && global_z <= storeH_tail_Z) {

        global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
        H_shared_idx = H_shared_x + H_shared_y * H_SHX + H_shared_z * H_SHX * H_SHY;
        Hx_pad[global_idx] = Hx_shmem[H_shared_idx];
        Hy_pad[global_idx] = Hy_shmem[H_shared_idx];
        Hz_pad[global_idx] = Hz_shmem[H_shared_idx];

        // if(xx == 0 && yy == 0 && zz == 0) {
        //   std::cout << "------------------------------------------------------\n";
        //   std::cout << "local_x = " << local_x << ", local_y = " << local_y << ", local_z = " << local_z << "\n";
        //   std::cout << "H_shared_x = " << H_shared_x << ", H_shared_y = " << H_shared_y << ", H_shared_z = " << H_shared_z << "\n";
        //   std::cout << "global_x = " << global_x << ", global_y = " << global_y << ", global_z = " << global_z << "\n";
        // }

      }

      // store E ---------------------------------------------
      E_shared_x = local_x;
      E_shared_y = local_y;
      E_shared_z = local_z;
      global_x = xx_heads[xx] + E_shared_x;
      global_y = yy_heads[yy] + E_shared_y;
      global_z = zz_heads[zz] + E_shared_z;

      if(global_x >= 1 + LEFT_PAD_MM_V2 && global_x <= Nx - 2 + LEFT_PAD_MM_V2 &&
         global_y >= 1 + LEFT_PAD_MM_V2 && global_y <= Ny - 2 + LEFT_PAD_MM_V2 &&
         global_z >= 1 + LEFT_PAD_MM_V2 && global_z <= Nz - 2 + LEFT_PAD_MM_V2 &&
         global_x >= storeE_head_X && global_x <= storeE_tail_X && 
         global_y >= storeE_head_Y && global_y <= storeE_tail_Y && 
         global_z >= storeE_head_Z && global_z <= storeE_tail_Z) {

        global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad; 
        E_shared_idx = E_shared_x + E_shared_y * E_SHX + E_shared_z * E_SHX * E_SHY;  
        Ex_pad[global_idx] = Ex_shmem[E_shared_idx];
        Ey_pad[global_idx] = Ey_shmem[E_shared_idx];
        Ez_pad[global_idx] = Ez_shmem[E_shared_idx];

        // if(xx == 0 && yy == 0 && zz == 0) {
        //   std::cout << "------------------------------------------------------\n";
        //   std::cout << "local_x = " << local_x << ", local_y = " << local_y << ", local_z = " << local_z << "\n";
        //   std::cout << "E_shared_x = " << E_shared_x << ", E_shared_y = " << E_shared_y << ", E_shared_z = " << E_shared_z << "\n";
        //   std::cout << "global_x = " << global_x << ", global_y = " << global_y << ", global_z = " << global_z << "\n";
        // }

      }
    }
  }

}

void gDiamond::update_FDTD_mix_mapping_sequential_ver2(size_t num_timesteps, size_t Tx, size_t Ty, size_t Tz) {

  // clear source Mz for experiments
  _Mz.clear();

  // transfer source
  for(size_t t=0; t<num_timesteps; t++) {
    float Mz_value = M_source_amp * std::sin(SOURCE_OMEGA * t * dt);
    _Mz[_source_idx] = Mz_value;
  }

  // pad E, H array
  const size_t Nx_pad = _Nx + LEFT_PAD_MM_V2 + RIGHT_PAD_MM_V2; 
  const size_t Ny_pad = _Ny + LEFT_PAD_MM_V2 + RIGHT_PAD_MM_V2; 
  const size_t Nz_pad = _Nz + LEFT_PAD_MM_V2 + RIGHT_PAD_MM_V2; 
  const size_t padded_length = Nx_pad * Ny_pad * Nz_pad;

  std::vector<float> Ex_pad(padded_length, 0);
  std::vector<float> Ey_pad(padded_length, 0);
  std::vector<float> Ez_pad(padded_length, 0);
  std::vector<float> Hx_pad(padded_length, 0);
  std::vector<float> Hy_pad(padded_length, 0);
  std::vector<float> Hz_pad(padded_length, 0);

  // transfer data to padded arrays
  for(size_t z = 0; z < _Nz; z++) {
    for(size_t y = 0; y < _Ny; y++) {
      for(size_t x = 0; x < _Nx; x++) {
        size_t x_pad = x + LEFT_PAD_MM_V2;
        size_t y_pad = y + LEFT_PAD_MM_V2;
        size_t z_pad = z + LEFT_PAD_MM_V2;
        size_t unpadded_index = x + y * _Nx + z * _Nx * _Ny;      
        size_t padded_index = x_pad + y_pad * Nx_pad + z_pad * Nx_pad * Ny_pad;
        Ex_pad[padded_index] = _Ex_simu[unpadded_index];
        Ey_pad[padded_index] = _Ey_simu[unpadded_index];
        Ez_pad[padded_index] = _Ez_simu[unpadded_index];
        Hx_pad[padded_index] = _Hx_simu[unpadded_index];
        Hy_pad[padded_index] = _Hy_simu[unpadded_index];
        Hz_pad[padded_index] = _Hz_simu[unpadded_index];
      }
    }
  }

  // tiling parameters
  size_t xx_num_m = Tx + 1;
  size_t xx_num_v = xx_num_m;
  size_t yy_num_m = Ty + 1;
  size_t yy_num_v = yy_num_m;
  size_t zz_num_m = Tz + 1;
  size_t zz_num_v = yy_num_m;
  std::vector<int> xx_heads_m(xx_num_m, 0); // head indices of mountains
  std::vector<int> xx_heads_v(xx_num_v, 0); // head indices of valleys
  std::vector<int> yy_heads_m(yy_num_m, 0);
  std::vector<int> yy_heads_v(yy_num_v, 0);
  std::vector<int> zz_heads_m(zz_num_m, 0);
  std::vector<int> zz_heads_v(zz_num_v, 0);

  for(size_t index=0; index<xx_num_m; index++) {
    xx_heads_m[index] = (index == 0)? 1 :
                             xx_heads_m[index-1] + (MOUNTAIN_X_V2 + VALLEY_X_V2);
  }
  for(size_t index=0; index<xx_num_v; index++) {
    xx_heads_v[index] = (index == 0)? LEFT_PAD_MM_V2 + VALLEY_X_V2 :
                             xx_heads_v[index-1] + (MOUNTAIN_X_V2 + VALLEY_X_V2);
  }
  for(size_t index=0; index<yy_num_m; index++) {
    yy_heads_m[index] = (index == 0)? 1 :
                             yy_heads_m[index-1] + (MOUNTAIN_Y_V2 + VALLEY_Y_V2);
  }
  for(size_t index=0; index<yy_num_v; index++) {
    yy_heads_v[index] = (index == 0)? LEFT_PAD_MM_V2 + VALLEY_Y_V2 :
                             yy_heads_v[index-1] + (MOUNTAIN_Y_V2 + VALLEY_Y_V2);
  }
  for(size_t index=0; index<zz_num_m; index++) {
    zz_heads_m[index] = (index == 0)? 1 :
                             zz_heads_m[index-1] + (MOUNTAIN_Z_V2 + VALLEY_Z_V2);
  }
  for(size_t index=0; index<zz_num_v; index++) {
    zz_heads_v[index] = (index == 0)? LEFT_PAD_MM_V2 + VALLEY_Z_V2 :
                             zz_heads_v[index-1] + (MOUNTAIN_Z_V2 + VALLEY_Z_V2);
  } 

  // std::cout << "xx_heads_m = ";
  // for(const auto& data : xx_heads_m) {
  //   std::cout << data << " ";
  // }
  // std::cout << "\n";
  // std::cout << "xx_heads_v = ";
  // for(const auto& data : xx_heads_v) {
  //   std::cout << data << " ";
  // }
  // std::cout << "\n";
  // std::cout << "yy_heads_m = ";
  // for(const auto& data : yy_heads_m) {
  //   std::cout << data << " ";
  // }
  // std::cout << "\n";
  // std::cout << "yy_heads_v = ";
  // for(const auto& data : yy_heads_v) {
  //   std::cout << data << " ";
  // }
  // std::cout << "\n";
  // std::cout << "zz_heads_m = ";
  // for(const auto& data : zz_heads_m) {
  //   std::cout << data << " ";
  // }
  // std::cout << "\n";
  // std::cout << "zz_heads_v = ";
  // for(const auto& data : zz_heads_v) {
  //   std::cout << data << " ";
  // }
  // std::cout << "\n";

  size_t block_size = NTX_MM_V2 * NTY_MM_V2 * NTZ_MM_V2;
  std::cout << "block_size = " << block_size << "\n";
  size_t grid_size;

  for(size_t tt = 0; tt < num_timesteps / BLT_MM_V2; tt++) {
    // phase 1. m, m, m
    grid_size = xx_num_m * yy_num_m * zz_num_m;
    _updateEH_mix_mapping_ver2<true, true, true>(Ex_pad, Ey_pad, Ez_pad,
                                                 Hx_pad, Hy_pad, Hz_pad,
                                                 _Cax, _Cbx,
                                                 _Cay, _Cby,
                                                 _Caz, _Cbz,
                                                 _Dax, _Dbx,
                                                 _Day, _Dby,
                                                 _Daz, _Dbz,
                                                 _Jx, _Jy, _Jz,
                                                 _Mx, _My, _Mz,
                                                 _dx,
                                                 _Nx, _Ny, _Nz,
                                                 Nx_pad, Ny_pad, Nz_pad,
                                                 xx_num_m, yy_num_m, zz_num_m,
                                                 xx_heads_m,
                                                 yy_heads_m,
                                                 zz_heads_m,
                                                 block_size,
                                                 grid_size);

    // phase 2. v, m, m
    grid_size = xx_num_v * yy_num_m * zz_num_m;
    _updateEH_mix_mapping_ver2<false, true, true>(Ex_pad, Ey_pad, Ez_pad,
                                            Hx_pad, Hy_pad, Hz_pad,
                                            _Cax, _Cbx,
                                            _Cay, _Cby,
                                            _Caz, _Cbz,
                                            _Dax, _Dbx,
                                            _Day, _Dby,
                                            _Daz, _Dbz,
                                            _Jx, _Jy, _Jz,
                                            _Mx, _My, _Mz,
                                            _dx,
                                            _Nx, _Ny, _Nz,
                                            Nx_pad, Ny_pad, Nz_pad,
                                            xx_num_v, yy_num_m, zz_num_m,
                                            xx_heads_v,
                                            yy_heads_m,
                                            zz_heads_m,
                                            block_size,
                                            grid_size);

    // phase 3. m, v, m
    grid_size = xx_num_m * yy_num_v * zz_num_m;
    _updateEH_mix_mapping_ver2<true, false, true>(Ex_pad, Ey_pad, Ez_pad,
                                            Hx_pad, Hy_pad, Hz_pad,
                                            _Cax, _Cbx,
                                            _Cay, _Cby,
                                            _Caz, _Cbz,
                                            _Dax, _Dbx,
                                            _Day, _Dby,
                                            _Daz, _Dbz,
                                            _Jx, _Jy, _Jz,
                                            _Mx, _My, _Mz,
                                            _dx,
                                            _Nx, _Ny, _Nz,
                                            Nx_pad, Ny_pad, Nz_pad,
                                            xx_num_m, yy_num_v, zz_num_m,
                                            xx_heads_m,
                                            yy_heads_v,
                                            zz_heads_m,
                                            block_size,
                                            grid_size);

    // phase 4. m, m, v
    grid_size = xx_num_m * yy_num_m * zz_num_v;
    _updateEH_mix_mapping_ver2<true, true, false>(Ex_pad, Ey_pad, Ez_pad,
                                            Hx_pad, Hy_pad, Hz_pad,
                                            _Cax, _Cbx,
                                            _Cay, _Cby,
                                            _Caz, _Cbz,
                                            _Dax, _Dbx,
                                            _Day, _Dby,
                                            _Daz, _Dbz,
                                            _Jx, _Jy, _Jz,
                                            _Mx, _My, _Mz,
                                            _dx,
                                            _Nx, _Ny, _Nz,
                                            Nx_pad, Ny_pad, Nz_pad,
                                            xx_num_m, yy_num_m, zz_num_v,
                                            xx_heads_m,
                                            yy_heads_m,
                                            zz_heads_v,
                                            block_size,
                                            grid_size);

    // phase 5. v, v, m
    grid_size = xx_num_v * yy_num_v * zz_num_m;
    _updateEH_mix_mapping_ver2<false, false, true>(Ex_pad, Ey_pad, Ez_pad,
                                            Hx_pad, Hy_pad, Hz_pad,
                                            _Cax, _Cbx,
                                            _Cay, _Cby,
                                            _Caz, _Cbz,
                                            _Dax, _Dbx,
                                            _Day, _Dby,
                                            _Daz, _Dbz,
                                            _Jx, _Jy, _Jz,
                                            _Mx, _My, _Mz,
                                            _dx,
                                            _Nx, _Ny, _Nz,
                                            Nx_pad, Ny_pad, Nz_pad,
                                            xx_num_v, yy_num_v, zz_num_m,
                                            xx_heads_v,
                                            yy_heads_v,
                                            zz_heads_m,
                                            block_size,
                                            grid_size);

    // phase 6. v, m, v
    grid_size = xx_num_v * yy_num_m * zz_num_v;
    _updateEH_mix_mapping_ver2<false, true, false>(Ex_pad, Ey_pad, Ez_pad,
                                            Hx_pad, Hy_pad, Hz_pad,
                                            _Cax, _Cbx,
                                            _Cay, _Cby,
                                            _Caz, _Cbz,
                                            _Dax, _Dbx,
                                            _Day, _Dby,
                                            _Daz, _Dbz,
                                            _Jx, _Jy, _Jz,
                                            _Mx, _My, _Mz,
                                            _dx,
                                            _Nx, _Ny, _Nz,
                                            Nx_pad, Ny_pad, Nz_pad,
                                            xx_num_v, yy_num_m, zz_num_v,
                                            xx_heads_v,
                                            yy_heads_m,
                                            zz_heads_v,
                                            block_size,
                                            grid_size);

    // phase 7. m, v, v
    grid_size = xx_num_m * yy_num_v * zz_num_v;
    _updateEH_mix_mapping_ver2<true, false, false>(Ex_pad, Ey_pad, Ez_pad,
                                            Hx_pad, Hy_pad, Hz_pad,
                                            _Cax, _Cbx,
                                            _Cay, _Cby,
                                            _Caz, _Cbz,
                                            _Dax, _Dbx,
                                            _Day, _Dby,
                                            _Daz, _Dbz,
                                            _Jx, _Jy, _Jz,
                                            _Mx, _My, _Mz,
                                            _dx,
                                            _Nx, _Ny, _Nz,
                                            Nx_pad, Ny_pad, Nz_pad,
                                            xx_num_m, yy_num_v, zz_num_v,
                                            xx_heads_m,
                                            yy_heads_v,
                                            zz_heads_v,
                                            block_size,
                                            grid_size);

    // phase 8. v, v, v
    grid_size = xx_num_v * yy_num_v * zz_num_v;
    _updateEH_mix_mapping_ver2<false, false, false>(Ex_pad, Ey_pad, Ez_pad,
                                                    Hx_pad, Hy_pad, Hz_pad,
                                                    _Cax, _Cbx,
                                                    _Cay, _Cby,
                                                    _Caz, _Cbz,
                                                    _Dax, _Dbx,
                                                    _Day, _Dby,
                                                    _Daz, _Dbz,
                                                    _Jx, _Jy, _Jz,
                                                    _Mx, _My, _Mz,
                                                    _dx,
                                                    _Nx, _Ny, _Nz,
                                                    Nx_pad, Ny_pad, Nz_pad,
                                                    xx_num_v, yy_num_v, zz_num_v,
                                                    xx_heads_v,
                                                    yy_heads_v,
                                                    zz_heads_v,
                                                    block_size,
                                                    grid_size);

  }

  // transfer data back to unpadded arrays
  for(size_t z = 0; z < _Nz; z++) {
    for(size_t y = 0; y < _Ny; y++) {
      for(size_t x = 0; x < _Nx; x++) {
        size_t x_pad = x + LEFT_PAD_MM_V2;
        size_t y_pad = y + LEFT_PAD_MM_V2;
        size_t z_pad = z + LEFT_PAD_MM_V2;
        size_t unpadded_index = x + y * _Nx + z * _Nx * _Ny;      
        size_t padded_index = x_pad + y_pad * Nx_pad + z_pad * Nx_pad * Ny_pad;
        _Ex_simu[unpadded_index] = Ex_pad[padded_index];
        _Ey_simu[unpadded_index] = Ey_pad[padded_index];
        _Ez_simu[unpadded_index] = Ez_pad[padded_index];
        _Hx_simu[unpadded_index] = Hx_pad[padded_index];
        _Hy_simu[unpadded_index] = Hy_pad[padded_index];
        _Hz_simu[unpadded_index] = Hz_pad[padded_index];
      }
    }
  }

}   

void gDiamond::update_FDTD_mix_mapping_gpu_ver2(size_t num_timesteps, size_t Tx, size_t Ty, size_t Tz) {

  // pad E, H array
  const size_t Nx_pad = _Nx + LEFT_PAD_MM_V2 + RIGHT_PAD_MM_V2; 
  const size_t Ny_pad = _Ny + LEFT_PAD_MM_V2 + RIGHT_PAD_MM_V2; 
  const size_t Nz_pad = _Nz + LEFT_PAD_MM_V2 + RIGHT_PAD_MM_V2; 
  const size_t padded_length = Nx_pad * Ny_pad * Nz_pad;

  std::vector<float> Ex_pad(padded_length, 0);
  std::vector<float> Ey_pad(padded_length, 0);
  std::vector<float> Ez_pad(padded_length, 0);
  std::vector<float> Hx_pad(padded_length, 0);
  std::vector<float> Hy_pad(padded_length, 0);
  std::vector<float> Hz_pad(padded_length, 0);

  // tiling parameters
  size_t xx_num_m = Tx + 1;
  size_t xx_num_v = xx_num_m;
  size_t yy_num_m = Ty + 1;
  size_t yy_num_v = yy_num_m;
  size_t zz_num_m = Tz + 1;
  size_t zz_num_v = yy_num_m;
  std::vector<int> xx_heads_m(xx_num_m, 0); // head indices of mountains
  std::vector<int> xx_heads_v(xx_num_v, 0); // head indices of valleys
  std::vector<int> yy_heads_m(yy_num_m, 0);
  std::vector<int> yy_heads_v(yy_num_v, 0);
  std::vector<int> zz_heads_m(zz_num_m, 0);
  std::vector<int> zz_heads_v(zz_num_v, 0);

  for(size_t index=0; index<xx_num_m; index++) {
    xx_heads_m[index] = (index == 0)? 1 :
                             xx_heads_m[index-1] + (MOUNTAIN_X_V2 + VALLEY_X_V2);
  }
  for(size_t index=0; index<xx_num_v; index++) {
    xx_heads_v[index] = (index == 0)? LEFT_PAD_MM_V2 + VALLEY_X_V2 :
                             xx_heads_v[index-1] + (MOUNTAIN_X_V2 + VALLEY_X_V2);
  }
  for(size_t index=0; index<yy_num_m; index++) {
    yy_heads_m[index] = (index == 0)? 1 :
                             yy_heads_m[index-1] + (MOUNTAIN_Y_V2 + VALLEY_Y_V2);
  }
  for(size_t index=0; index<yy_num_v; index++) {
    yy_heads_v[index] = (index == 0)? LEFT_PAD_MM_V2 + VALLEY_Y_V2 :
                             yy_heads_v[index-1] + (MOUNTAIN_Y_V2 + VALLEY_Y_V2);
  }
  for(size_t index=0; index<zz_num_m; index++) {
    zz_heads_m[index] = (index == 0)? 1 :
                             zz_heads_m[index-1] + (MOUNTAIN_Z_V2 + VALLEY_Z_V2);
  }
  for(size_t index=0; index<zz_num_v; index++) {
    zz_heads_v[index] = (index == 0)? LEFT_PAD_MM_V2 + VALLEY_Z_V2 :
                             zz_heads_v[index-1] + (MOUNTAIN_Z_V2 + VALLEY_Z_V2);
  } 

  // std::cout << "xx_heads_m = ";
  // for(const auto& data : xx_heads_m) {
  //   std::cout << data << " ";
  // }
  // std::cout << "\n";
  // std::cout << "xx_heads_v = ";
  // for(const auto& data : xx_heads_v) {
  //   std::cout << data << " ";
  // }
  // std::cout << "\n";
  // std::cout << "yy_heads_m = ";
  // for(const auto& data : yy_heads_m) {
  //   std::cout << data << " ";
  // }
  // std::cout << "\n";
  // std::cout << "yy_heads_v = ";
  // for(const auto& data : yy_heads_v) {
  //   std::cout << data << " ";
  // }
  // std::cout << "\n";
  // std::cout << "zz_heads_m = ";
  // for(const auto& data : zz_heads_m) {
  //   std::cout << data << " ";
  // }
  // std::cout << "\n";
  // std::cout << "zz_heads_v = ";
  // for(const auto& data : zz_heads_v) {
  //   std::cout << data << " ";
  // }
  // std::cout << "\n";

  float *d_Ex_pad, *d_Ey_pad, *d_Ez_pad;
  float *d_Hx_pad, *d_Hy_pad, *d_Hz_pad;

  float *Jx, *Jy, *Jz;
  float *Mx, *My, *Mz;
  float *Cax, *Cay, *Caz, *Cbx, *Cby, *Cbz;
  float *Dax, *Day, *Daz, *Dbx, *Dby, *Dbz;

  int *d_xx_heads_m, *d_xx_heads_v;
  int *d_yy_heads_m, *d_yy_heads_v;
  int *d_zz_heads_m, *d_zz_heads_v;

  size_t unpadded_length = _Nx * _Ny * _Nz;

  CUDACHECK(cudaMalloc(&d_Ex_pad, sizeof(float) * padded_length));
  CUDACHECK(cudaMalloc(&d_Ey_pad, sizeof(float) * padded_length));
  CUDACHECK(cudaMalloc(&d_Ez_pad, sizeof(float) * padded_length));
  CUDACHECK(cudaMalloc(&d_Hx_pad, sizeof(float) * padded_length));
  CUDACHECK(cudaMalloc(&d_Hy_pad, sizeof(float) * padded_length));
  CUDACHECK(cudaMalloc(&d_Hz_pad, sizeof(float) * padded_length));

  CUDACHECK(cudaMalloc(&Jx, sizeof(float) * unpadded_length));
  CUDACHECK(cudaMalloc(&Jy, sizeof(float) * unpadded_length));
  CUDACHECK(cudaMalloc(&Jz, sizeof(float) * unpadded_length));
  CUDACHECK(cudaMalloc(&Mx, sizeof(float) * unpadded_length));
  CUDACHECK(cudaMalloc(&My, sizeof(float) * unpadded_length));
  CUDACHECK(cudaMalloc(&Mz, sizeof(float) * unpadded_length));
  CUDACHECK(cudaMalloc(&Cax, sizeof(float) * unpadded_length));
  CUDACHECK(cudaMalloc(&Cbx, sizeof(float) * unpadded_length));
  CUDACHECK(cudaMalloc(&Cay, sizeof(float) * unpadded_length));
  CUDACHECK(cudaMalloc(&Cby, sizeof(float) * unpadded_length));
  CUDACHECK(cudaMalloc(&Caz, sizeof(float) * unpadded_length));
  CUDACHECK(cudaMalloc(&Cbz, sizeof(float) * unpadded_length));
  CUDACHECK(cudaMalloc(&Dax, sizeof(float) * unpadded_length));
  CUDACHECK(cudaMalloc(&Dbx, sizeof(float) * unpadded_length));
  CUDACHECK(cudaMalloc(&Day, sizeof(float) * unpadded_length));
  CUDACHECK(cudaMalloc(&Dby, sizeof(float) * unpadded_length));
  CUDACHECK(cudaMalloc(&Daz, sizeof(float) * unpadded_length));
  CUDACHECK(cudaMalloc(&Dbz, sizeof(float) * unpadded_length));

  CUDACHECK(cudaMalloc(&d_xx_heads_m, sizeof(int) * xx_num_m));
  CUDACHECK(cudaMalloc(&d_xx_heads_v, sizeof(int) * xx_num_v));
  CUDACHECK(cudaMalloc(&d_yy_heads_m, sizeof(int) * yy_num_m));
  CUDACHECK(cudaMalloc(&d_yy_heads_v, sizeof(int) * yy_num_v));
  CUDACHECK(cudaMalloc(&d_zz_heads_m, sizeof(int) * zz_num_m));
  CUDACHECK(cudaMalloc(&d_zz_heads_v, sizeof(int) * zz_num_v));

  // initialize E, H as 0
  CUDACHECK(cudaMemset(d_Ex_pad, 0, sizeof(float) * padded_length));
  CUDACHECK(cudaMemset(d_Ey_pad, 0, sizeof(float) * padded_length));
  CUDACHECK(cudaMemset(d_Ez_pad, 0, sizeof(float) * padded_length));
  CUDACHECK(cudaMemset(d_Hx_pad, 0, sizeof(float) * padded_length));
  CUDACHECK(cudaMemset(d_Hy_pad, 0, sizeof(float) * padded_length));
  CUDACHECK(cudaMemset(d_Hz_pad, 0, sizeof(float) * padded_length));

  // initialize J, M, Ca, Cb, Da, Db as 0
  CUDACHECK(cudaMemset(Jx, 0, sizeof(float) * unpadded_length));
  CUDACHECK(cudaMemset(Jy, 0, sizeof(float) * unpadded_length));
  CUDACHECK(cudaMemset(Jz, 0, sizeof(float) * unpadded_length));
  CUDACHECK(cudaMemset(Mx, 0, sizeof(float) * unpadded_length));
  CUDACHECK(cudaMemset(My, 0, sizeof(float) * unpadded_length));
  CUDACHECK(cudaMemset(Mz, 0, sizeof(float) * unpadded_length));
  CUDACHECK(cudaMemset(Cax, 0, sizeof(float) * unpadded_length));
  CUDACHECK(cudaMemset(Cbx, 0, sizeof(float) * unpadded_length));
  CUDACHECK(cudaMemset(Cay, 0, sizeof(float) * unpadded_length));
  CUDACHECK(cudaMemset(Cby, 0, sizeof(float) * unpadded_length));
  CUDACHECK(cudaMemset(Caz, 0, sizeof(float) * unpadded_length));
  CUDACHECK(cudaMemset(Cbz, 0, sizeof(float) * unpadded_length));
  CUDACHECK(cudaMemset(Dax, 0, sizeof(float) * unpadded_length));
  CUDACHECK(cudaMemset(Dbx, 0, sizeof(float) * unpadded_length));
  CUDACHECK(cudaMemset(Day, 0, sizeof(float) * unpadded_length));
  CUDACHECK(cudaMemset(Dby, 0, sizeof(float) * unpadded_length));
  CUDACHECK(cudaMemset(Daz, 0, sizeof(float) * unpadded_length));
  CUDACHECK(cudaMemset(Dbz, 0, sizeof(float) * unpadded_length));

  // transfer source
  for(size_t t=0; t<num_timesteps; t++) {
    float Mz_value = M_source_amp * std::sin(SOURCE_OMEGA * t * dt);
    CUDACHECK(cudaMemcpy(Mz + _source_idx, &Mz_value, sizeof(float), cudaMemcpyHostToDevice));
  }

  // copy Ca, Cb, Da, Db
  CUDACHECK(cudaMemcpyAsync(Cax, _Cax.data(), sizeof(float) * unpadded_length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Cay, _Cay.data(), sizeof(float) * unpadded_length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Caz, _Caz.data(), sizeof(float) * unpadded_length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Cbx, _Cbx.data(), sizeof(float) * unpadded_length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Cby, _Cby.data(), sizeof(float) * unpadded_length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Cbz, _Cbz.data(), sizeof(float) * unpadded_length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Dax, _Dax.data(), sizeof(float) * unpadded_length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Day, _Day.data(), sizeof(float) * unpadded_length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Daz, _Daz.data(), sizeof(float) * unpadded_length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Dbx, _Dbx.data(), sizeof(float) * unpadded_length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Dby, _Dby.data(), sizeof(float) * unpadded_length, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(Dbz, _Dbz.data(), sizeof(float) * unpadded_length, cudaMemcpyHostToDevice));

  // copy tiling parameters
  CUDACHECK(cudaMemcpyAsync(d_xx_heads_m, xx_heads_m.data(), sizeof(int) * xx_num_m, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(d_xx_heads_v, xx_heads_v.data(), sizeof(int) * xx_num_v, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(d_yy_heads_m, yy_heads_m.data(), sizeof(int) * yy_num_m, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(d_yy_heads_v, yy_heads_v.data(), sizeof(int) * yy_num_v, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(d_zz_heads_m, zz_heads_m.data(), sizeof(int) * zz_num_m, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(d_zz_heads_v, zz_heads_v.data(), sizeof(int) * zz_num_v, cudaMemcpyHostToDevice));

  size_t block_size = NTX_MM_V2 * NTY_MM_V2 * NTZ_MM_V2;
  std::cout << "block_size = " << block_size << "\n";
  size_t grid_size;

  for(size_t tt = 0; tt < num_timesteps / BLT_MM_V2; tt++) {

  }


  CUDACHECK(cudaFree(d_Ex_pad));
  CUDACHECK(cudaFree(d_Ey_pad));
  CUDACHECK(cudaFree(d_Ez_pad));
  CUDACHECK(cudaFree(d_Hx_pad));
  CUDACHECK(cudaFree(d_Hy_pad));
  CUDACHECK(cudaFree(d_Hz_pad));

  CUDACHECK(cudaFree(Jx));
  CUDACHECK(cudaFree(Jy));
  CUDACHECK(cudaFree(Jz));
  CUDACHECK(cudaFree(Mx));
  CUDACHECK(cudaFree(My));
  CUDACHECK(cudaFree(Mz));
  CUDACHECK(cudaFree(Cax));
  CUDACHECK(cudaFree(Cbx));
  CUDACHECK(cudaFree(Cay));
  CUDACHECK(cudaFree(Cby));
  CUDACHECK(cudaFree(Caz));
  CUDACHECK(cudaFree(Cbz));
  CUDACHECK(cudaFree(Dax));
  CUDACHECK(cudaFree(Dbx));
  CUDACHECK(cudaFree(Day));
  CUDACHECK(cudaFree(Dby));
  CUDACHECK(cudaFree(Daz));
  CUDACHECK(cudaFree(Dbz));

  CUDACHECK(cudaFree(d_xx_heads_m));
  CUDACHECK(cudaFree(d_xx_heads_v));
  CUDACHECK(cudaFree(d_yy_heads_m));
  CUDACHECK(cudaFree(d_yy_heads_v));
  CUDACHECK(cudaFree(d_zz_heads_m));
  CUDACHECK(cudaFree(d_zz_heads_v));

}   

} // end of namespace gdiamond

#endif




























