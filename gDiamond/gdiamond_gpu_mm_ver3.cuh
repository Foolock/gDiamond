#ifndef GDIAMOND_GPU_MM_VER3_CUH
#define GDIAMOND_GPU_MM_VER3_CUH

#include "gdiamond.hpp"
#include "kernels_mm_ver3.cuh"
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

#define SWAP_PTR(a, b) do { auto _tmp = (a); (a) = (b); (b) = _tmp; } while (0)

namespace gdiamond {

// notice that for overlapped mix mapping ver3, we need to have a src and dst copy of E, H data
void gDiamond::_updateEH_mix_mapping_ver3(std::vector<float>& Ex_pad_src, std::vector<float>& Ey_pad_src, std::vector<float>& Ez_pad_src,
                                          std::vector<float>& Hx_pad_src, std::vector<float>& Hy_pad_src, std::vector<float>& Hz_pad_src,
                                          std::vector<float>& Ex_pad_dst, std::vector<float>& Ey_pad_dst, std::vector<float>& Ez_pad_dst,
                                          std::vector<float>& Hx_pad_dst, std::vector<float>& Hy_pad_dst, std::vector<float>& Hz_pad_dst,
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
    float Hx_shmem[H_SHX_V3 * H_SHY_V3 * H_SHZ_V3];
    float Hy_shmem[H_SHX_V3 * H_SHY_V3 * H_SHZ_V3];
    float Hz_shmem[H_SHX_V3 * H_SHY_V3 * H_SHZ_V3];
    float Ex_shmem[E_SHX_V3 * E_SHY_V3 * E_SHZ_V3];
    float Ey_shmem[E_SHX_V3 * E_SHY_V3 * E_SHZ_V3];
    float Ez_shmem[E_SHX_V3 * E_SHY_V3 * E_SHZ_V3];

    // load shared memory
    for(size_t thread_id = 0; thread_id < block_size; thread_id++) {

      // X dimension has 1 extra HALO load, one thread load one element,
      // since every tile is mountain, tid = 0 load one extra H at xx_heads[xx] - 1
      // same thing applys to Y, Z dimension

      local_x = thread_id % NTX_MM_V3;
      local_y = (thread_id / NTX_MM_V3) % NTY_MM_V3;
      local_z = thread_id / (NTX_MM_V3 * NTY_MM_V3);
      H_shared_x = local_x + 1;
      H_shared_y = local_y + 1;
      H_shared_z = local_z + 1;
      E_shared_x = local_x;
      E_shared_y = local_y;
      E_shared_z = local_z;
      global_x = xx_heads[xx] + local_x;
      global_y = yy_heads[yy] + local_y;
      global_z = zz_heads[zz] + local_z;

      // load core ---------------------------------------------
      H_shared_idx = H_shared_x + H_shared_y * H_SHX_V3 + H_shared_z * H_SHX_V3 * H_SHY_V3;
      E_shared_idx = E_shared_x + E_shared_y * E_SHX_V3 + E_shared_z * E_SHX_V3 * E_SHY_V3;
      global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
      Hx_shmem[H_shared_idx] = Hx_pad_src[global_idx];
      Hy_shmem[H_shared_idx] = Hy_pad_src[global_idx];
      Hz_shmem[H_shared_idx] = Hz_pad_src[global_idx];
      Ex_shmem[E_shared_idx] = Ex_pad_src[global_idx];
      Ey_shmem[E_shared_idx] = Ey_pad_src[global_idx];
      Ez_shmem[E_shared_idx] = Ez_pad_src[global_idx];

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

      /*
      // E HALO
      // E HALO is not needed since there is no valley tile in mix mapping ver 3
      if (local_x == NTX_MM_V3 - 1) {
        int halo_x = local_x + 1;
        int global_x_halo = xx_heads[xx] + halo_x;

        global_idx = global_x_halo + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
        E_shared_idx = halo_x + E_shared_y * E_SHX_V3 + E_shared_z * E_SHX_V3 * E_SHY_V3;

        Ex_shmem[E_shared_idx] = Ex_pad_src[global_idx];
        Ey_shmem[E_shared_idx] = Ey_pad_src[global_idx];
        Ez_shmem[E_shared_idx] = Ez_pad_src[global_idx];
      }
      if (local_y == NTY_MM_V3 - 1) {
        int halo_y = local_y + 1;
        int global_y_halo = yy_heads[yy] + halo_y;

        global_idx = global_x + global_y_halo * Nx_pad + global_z * Nx_pad * Ny_pad;
        E_shared_idx = E_shared_x + halo_y * E_SHX_V3 + E_shared_z * E_SHX_V3 * E_SHY_V3;

        Ex_shmem[E_shared_idx] = Ex_pad_src[global_idx];
        Ey_shmem[E_shared_idx] = Ey_pad_src[global_idx];
        Ez_shmem[E_shared_idx] = Ez_pad_src[global_idx];
      }
      if (local_z == NTZ_MM_V3 - 1) {
        int halo_z = local_z + 1;
        int global_z_halo = zz_heads[zz] + halo_z;

        global_idx = global_x + global_y * Nx_pad + global_z_halo * Nx_pad * Ny_pad;
        E_shared_idx = E_shared_x + E_shared_y * E_SHX_V3 + halo_z * E_SHX_V3 * E_SHY_V3;

        Ex_shmem[E_shared_idx] = Ex_pad_src[global_idx];
        Ey_shmem[E_shared_idx] = Ey_pad_src[global_idx];
        Ez_shmem[E_shared_idx] = Ez_pad_src[global_idx];
      }
      */
    }

    // calculation
    for(int t = 0; t < BLT_MM_V3; t++) {

      // X head and tail is refer to padded global_x
      // same thing applys to Y and Z
      int calE_head_X = xx_heads[xx] + t;
      int calE_tail_X = xx_heads[xx] + BLX_MM_V3 - 1 - t;
      int calH_head_X = calE_head_X;
      int calH_tail_X = calE_tail_X - 1;
      
      int calE_head_Y = yy_heads[yy] + t;
      int calE_tail_Y = yy_heads[yy] + BLY_MM_V3 - 1 - t;
      int calH_head_Y = calE_head_Y;
      int calH_tail_Y = calE_tail_Y - 1;

      int calE_head_Z = zz_heads[zz] + t;
      int calE_tail_Z = zz_heads[zz] + BLZ_MM_V3 - 1 - t;
      int calH_head_Z = calE_head_Z;
      int calH_tail_Z = calE_tail_Z - 1;

      // update E
      for(size_t thread_id = 0; thread_id < block_size; thread_id++) {

        local_x = thread_id % NTX_MM_V3;
        local_y = (thread_id / NTX_MM_V3) % NTY_MM_V3;
        local_z = thread_id / (NTX_MM_V3 * NTY_MM_V3);
        H_shared_x = local_x + 1;
        H_shared_y = local_y + 1;
        H_shared_z = local_z + 1;
        E_shared_x = local_x;
        E_shared_y = local_y;
        E_shared_z = local_z;
        global_x = xx_heads[xx] + local_x;
        global_y = yy_heads[yy] + local_y;
        global_z = zz_heads[zz] + local_z;

        // we pad all the dimension, so need to substract LEFT_PAD here to correctly access constant arrays
        global_idx = (global_x - LEFT_PAD_MM_V3) + (global_y - LEFT_PAD_MM_V3) * Nx + (global_z - LEFT_PAD_MM_V3) * Nx * Ny;
        E_shared_idx = E_shared_x + E_shared_y * E_SHX_V3 + E_shared_z * E_SHX_V3 * E_SHY_V3;
        H_shared_idx = H_shared_x + H_shared_y * H_SHX_V3 + H_shared_z * H_SHX_V3 * H_SHY_V3;

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
      }

      // update H
      for(size_t thread_id = 0; thread_id < block_size; thread_id++) {

        local_x = thread_id % NTX_MM_V3;
        local_y = (thread_id / NTX_MM_V3) % NTY_MM_V3;
        local_z = thread_id / (NTX_MM_V3 * NTY_MM_V3);
        H_shared_x = local_x + 1;
        H_shared_y = local_y + 1;
        H_shared_z = local_z + 1;
        E_shared_x = local_x;
        E_shared_y = local_y;
        E_shared_z = local_z;
        global_x = xx_heads[xx] + local_x;
        global_y = yy_heads[yy] + local_y;
        global_z = zz_heads[zz] + local_z;

        global_idx = (global_x - LEFT_PAD_MM_V3) + (global_y - LEFT_PAD_MM_V3) * Nx + (global_z - LEFT_PAD_MM_V3) * Nx * Ny;
        E_shared_idx = E_shared_x + E_shared_y * E_SHX_V3 + E_shared_z * E_SHX_V3 * E_SHY_V3;
        H_shared_idx = H_shared_x + H_shared_y * H_SHX_V3 + H_shared_z * H_SHX_V3 * H_SHY_V3;

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
      }
    }

    // store back to global memory

    // X head and tail is refer to padded global_x
    // same thing applys to Y and Z
    const int storeE_head_X = xx_heads[xx] + LEFT_PAD_MM_V3 - 1;
    const int storeE_tail_X = storeE_head_X + VALLEY_X_V3 - 1; 
    const int storeH_head_X = storeE_head_X;
    const int storeH_tail_X = storeE_tail_X; 
    const int storeE_head_Y = yy_heads[yy] + LEFT_PAD_MM_V3 - 1;
    const int storeE_tail_Y = storeE_head_Y + VALLEY_Y_V3 - 1; 
    const int storeH_head_Y = storeE_head_Y;
    const int storeH_tail_Y = storeE_tail_Y; 
    const int storeE_head_Z = zz_heads[zz] + LEFT_PAD_MM_V3 - 1;
    const int storeE_tail_Z = storeE_head_Z + VALLEY_Z_V3 - 1; 
    const int storeH_head_Z = storeE_head_Z;
    const int storeH_tail_Z = storeE_tail_Z; 

    // std::cout << "xx = " << xx << ", yy = " << yy << ", zz = " << zz << "\n";
    // std::cout << "storeE_head_X = " << storeE_head_X << ", storeE_tail_X = " << storeE_tail_X
    //           << ", storeH_head_X = " << storeH_head_X << ", storeH_tail_X = " << storeH_tail_X << "\n";
    // std::cout << "storeE_head_Y = " << storeE_head_Y << ", storeE_tail_Y = " << storeE_tail_Y
    //           << ", storeH_head_Y = " << storeH_head_Y << ", storeH_tail_Y = " << storeH_tail_Y << "\n";
    // std::cout << "storeE_head_Z = " << storeE_head_Z << ", storeE_tail_Z = " << storeE_tail_Z
    //           << ", storeH_head_Z = " << storeH_head_Z << ", storeH_tail_Z = " << storeH_tail_Z << "\n";
    // std::cout << "\n";

    for(size_t thread_id = 0; thread_id < block_size; thread_id++) {
      
      local_x = thread_id % NTX_MM_V3;
      local_y = (thread_id / NTX_MM_V3) % NTY_MM_V3;
      local_z = thread_id / (NTX_MM_V3 * NTY_MM_V3);
      H_shared_x = local_x + 1;
      H_shared_y = local_y + 1;
      H_shared_z = local_z + 1;
      E_shared_x = local_x;
      E_shared_y = local_y;
      E_shared_z = local_z;
      global_x = xx_heads[xx] + E_shared_x;
      global_y = yy_heads[yy] + E_shared_y;
      global_z = zz_heads[zz] + E_shared_z;

      // store H ---------------------------------------------
      if(global_x >= 1 + LEFT_PAD_MM_V3 && global_x <= Nx - 2 + LEFT_PAD_MM_V3 &&
         global_y >= 1 + LEFT_PAD_MM_V3 && global_y <= Ny - 2 + LEFT_PAD_MM_V3 &&
         global_z >= 1 + LEFT_PAD_MM_V3 && global_z <= Nz - 2 + LEFT_PAD_MM_V3 &&
         global_x >= storeH_head_X && global_x <= storeH_tail_X &&
         global_y >= storeH_head_Y && global_y <= storeH_tail_Y &&
         global_z >= storeH_head_Z && global_z <= storeH_tail_Z) {

        global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad;
        H_shared_idx = H_shared_x + H_shared_y * H_SHX_V3 + H_shared_z * H_SHX_V3 * H_SHY_V3;
        Hx_pad_dst[global_idx] = Hx_shmem[H_shared_idx];
        Hy_pad_dst[global_idx] = Hy_shmem[H_shared_idx];
        Hz_pad_dst[global_idx] = Hz_shmem[H_shared_idx];
      }

      // store E ---------------------------------------------
      if(global_x >= 1 + LEFT_PAD_MM_V3 && global_x <= Nx - 2 + LEFT_PAD_MM_V3 &&
         global_y >= 1 + LEFT_PAD_MM_V3 && global_y <= Ny - 2 + LEFT_PAD_MM_V3 &&
         global_z >= 1 + LEFT_PAD_MM_V3 && global_z <= Nz - 2 + LEFT_PAD_MM_V3 &&
         global_x >= storeE_head_X && global_x <= storeE_tail_X && 
         global_y >= storeE_head_Y && global_y <= storeE_tail_Y && 
         global_z >= storeE_head_Z && global_z <= storeE_tail_Z) {

        global_idx = global_x + global_y * Nx_pad + global_z * Nx_pad * Ny_pad; 
        E_shared_idx = E_shared_x + E_shared_y * E_SHX_V3 + E_shared_z * E_SHX_V3 * E_SHY_V3;  
        Ex_pad_dst[global_idx] = Ex_shmem[E_shared_idx];
        Ey_pad_dst[global_idx] = Ey_shmem[E_shared_idx];
        Ez_pad_dst[global_idx] = Ez_shmem[E_shared_idx];
      }
    }

  }
}

void gDiamond::update_FDTD_mix_mapping_sequential_ver3(size_t num_timesteps, size_t Tx, size_t Ty, size_t Tz) {

  std::cout << "running update_FDTD_mix_mapping_sequential_ver3\n";

  // clear source Mz for experiments
  _Mz.clear();

  // transfer source
  for(size_t t=0; t<num_timesteps; t++) {
    float Mz_value = M_source_amp * std::sin(SOURCE_OMEGA * t * dt);
    _Mz[_source_idx] = Mz_value;
  }

  // pad E, H array
  const size_t Nx_pad = _Nx + LEFT_PAD_MM_V3 + RIGHT_PAD_MM_V3;
  const size_t Ny_pad = _Ny + LEFT_PAD_MM_V3 + RIGHT_PAD_MM_V3;
  const size_t Nz_pad = _Nz + LEFT_PAD_MM_V3 + RIGHT_PAD_MM_V3;
  const size_t padded_length = Nx_pad * Ny_pad * Nz_pad;

  std::cerr << "Nx_pad = " << Nx_pad << ", Ny_pad = " << Ny_pad << ", Nz_pad = " << Nz_pad << "\n";

  std::vector<float> Ex_pad_src(padded_length, 0);
  std::vector<float> Ey_pad_src(padded_length, 0);
  std::vector<float> Ez_pad_src(padded_length, 0);
  std::vector<float> Hx_pad_src(padded_length, 0);
  std::vector<float> Hy_pad_src(padded_length, 0);
  std::vector<float> Hz_pad_src(padded_length, 0);

  std::vector<float> Ex_pad_dst(padded_length, 0);
  std::vector<float> Ey_pad_dst(padded_length, 0);
  std::vector<float> Ez_pad_dst(padded_length, 0);
  std::vector<float> Hx_pad_dst(padded_length, 0);
  std::vector<float> Hy_pad_dst(padded_length, 0);
  std::vector<float> Hz_pad_dst(padded_length, 0);

  // tiling parameters
  // for mix mapping ver3, all tiles are mountains
  size_t xx_num = Tx;
  size_t yy_num = Ty;
  size_t zz_num = Tz;
  std::vector<int> xx_heads(xx_num, 0); // head indices of mountains
  std::vector<int> yy_heads(yy_num, 0);
  std::vector<int> zz_heads(zz_num, 0);

  for(size_t index=0; index<xx_num; index++) {
    xx_heads[index] = (index == 0)? 1 :
                             xx_heads[index-1] + (VALLEY_X_V3);
  }

  for(size_t index=0; index<yy_num; index++) {
    yy_heads[index] = (index == 0)? 1 :
                             yy_heads[index-1] + (VALLEY_Y_V3);
  }

  for(size_t index=0; index<zz_num; index++) {
    zz_heads[index] = (index == 0)? 1 :
                             zz_heads[index-1] + (VALLEY_Z_V3);
  }

  // std::cout << "xx_heads = ";
  // for(const auto& data : xx_heads) {
  //   std::cout << data << " ";
  // }
  // std::cout << "\n";
  // std::cout << "yy_heads = ";
  // for(const auto& data : yy_heads) {
  //   std::cout << data << " ";
  // }
  // std::cout << "\n";
  // std::cout << "zz_heads = ";
  // for(const auto& data : zz_heads) {
  //   std::cout << data << " ";
  // }
  // std::cout << "\n";

  size_t block_size = NTX_MM_V3 * NTY_MM_V3 * NTZ_MM_V3;
  std::cout << "block_size = " << block_size << "\n";
  size_t grid_size = xx_num * yy_num * zz_num;

  for(size_t tt = 0; tt < num_timesteps / BLT_MM_V3; tt++) {
    
    _updateEH_mix_mapping_ver3(Ex_pad_src, Ey_pad_src, Ez_pad_src,
                               Hx_pad_src, Hy_pad_src, Hz_pad_src,
                               Ex_pad_dst, Ey_pad_dst, Ez_pad_dst,
                               Hx_pad_dst, Hy_pad_dst, Hz_pad_dst,
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
                               xx_num, yy_num, zz_num,
                               xx_heads,
                               yy_heads,
                               zz_heads,
                               block_size,
                               grid_size);

    // swap src and dst
    std::swap(Ex_pad_src, Ex_pad_dst);
    std::swap(Ey_pad_src, Ey_pad_dst);
    std::swap(Ez_pad_src, Ez_pad_dst);
    std::swap(Hx_pad_src, Hx_pad_dst);
    std::swap(Hy_pad_src, Hy_pad_dst);
    std::swap(Hz_pad_src, Hz_pad_dst);
  }

  // transfer data back to unpadded arrays
  for(size_t z = 0; z < _Nz; z++) {
    for(size_t y = 0; y < _Ny; y++) {
      for(size_t x = 0; x < _Nx; x++) {
        size_t x_pad = x + LEFT_PAD_MM_V3;
        size_t y_pad = y + LEFT_PAD_MM_V3;
        size_t z_pad = z + LEFT_PAD_MM_V3;
        size_t unpadded_index = x + y * _Nx + z * _Nx * _Ny;
        size_t padded_index = x_pad + y_pad * Nx_pad + z_pad * Nx_pad * Ny_pad;
        _Ex_simu[unpadded_index] = Ex_pad_src[padded_index];
        _Ey_simu[unpadded_index] = Ey_pad_src[padded_index];
        _Ez_simu[unpadded_index] = Ez_pad_src[padded_index];
        _Hx_simu[unpadded_index] = Hx_pad_src[padded_index];
        _Hy_simu[unpadded_index] = Hy_pad_src[padded_index];
        _Hz_simu[unpadded_index] = Hz_pad_src[padded_index];
      }
    }
  }

}   

void gDiamond::update_FDTD_mix_mapping_gpu_ver3(size_t num_timesteps, size_t Tx, size_t Ty, size_t Tz) {

  std::cerr << "running update_FDTD_mix_mapping_gpu_ver3\n";

  // pad E, H array
  const size_t Nx_pad = _Nx + LEFT_PAD_MM_V3 + RIGHT_PAD_MM_V3;
  const size_t Ny_pad = _Ny + LEFT_PAD_MM_V3 + RIGHT_PAD_MM_V3;
  const size_t Nz_pad = _Nz + LEFT_PAD_MM_V3 + RIGHT_PAD_MM_V3;
  const size_t padded_length = Nx_pad * Ny_pad * Nz_pad;

  std::vector<float> Ex_pad_src(padded_length, 0);
  std::vector<float> Ey_pad_src(padded_length, 0);
  std::vector<float> Ez_pad_src(padded_length, 0);
  std::vector<float> Hx_pad_src(padded_length, 0);
  std::vector<float> Hy_pad_src(padded_length, 0);
  std::vector<float> Hz_pad_src(padded_length, 0);

  std::vector<float> Ex_pad_dst(padded_length, 0);
  std::vector<float> Ey_pad_dst(padded_length, 0);
  std::vector<float> Ez_pad_dst(padded_length, 0);
  std::vector<float> Hx_pad_dst(padded_length, 0);
  std::vector<float> Hy_pad_dst(padded_length, 0);
  std::vector<float> Hz_pad_dst(padded_length, 0);

  // tiling parameters
  // for mix mapping ver3, all tiles are mountains
  size_t xx_num = Tx;
  size_t yy_num = Ty;
  size_t zz_num = Tz;
  std::vector<int> xx_heads(xx_num, 0); // head indices of mountains
  std::vector<int> yy_heads(yy_num, 0);
  std::vector<int> zz_heads(zz_num, 0);

  for(size_t index=0; index<xx_num; index++) {
    xx_heads[index] = (index == 0)? 1 :
                             xx_heads[index-1] + (VALLEY_X_V3);
  }

  for(size_t index=0; index<yy_num; index++) {
    yy_heads[index] = (index == 0)? 1 :
                             yy_heads[index-1] + (VALLEY_Y_V3);
  }

  for(size_t index=0; index<zz_num; index++) {
    zz_heads[index] = (index == 0)? 1 :
                             zz_heads[index-1] + (VALLEY_Z_V3);
  }

  float *d_Ex_pad_src, *d_Ey_pad_src, *d_Ez_pad_src;
  float *d_Hx_pad_src, *d_Hy_pad_src, *d_Hz_pad_src;
  float *d_Ex_pad_dst, *d_Ey_pad_dst, *d_Ez_pad_dst;
  float *d_Hx_pad_dst, *d_Hy_pad_dst, *d_Hz_pad_dst;

  float *Jx, *Jy, *Jz;
  float *Mx, *My, *Mz;
  float *Cax, *Cay, *Caz, *Cbx, *Cby, *Cbz;
  float *Dax, *Day, *Daz, *Dbx, *Dby, *Dbz;

  int *d_xx_heads;
  int *d_yy_heads;
  int *d_zz_heads;

  size_t unpadded_length = _Nx * _Ny * _Nz;

  CUDACHECK(cudaMalloc(&d_Ex_pad_src, sizeof(float) * padded_length));
  CUDACHECK(cudaMalloc(&d_Ey_pad_src, sizeof(float) * padded_length));
  CUDACHECK(cudaMalloc(&d_Ez_pad_src, sizeof(float) * padded_length));
  CUDACHECK(cudaMalloc(&d_Hx_pad_src, sizeof(float) * padded_length));
  CUDACHECK(cudaMalloc(&d_Hy_pad_src, sizeof(float) * padded_length));
  CUDACHECK(cudaMalloc(&d_Hz_pad_src, sizeof(float) * padded_length));

  CUDACHECK(cudaMalloc(&d_Ex_pad_dst, sizeof(float) * padded_length));
  CUDACHECK(cudaMalloc(&d_Ey_pad_dst, sizeof(float) * padded_length));
  CUDACHECK(cudaMalloc(&d_Ez_pad_dst, sizeof(float) * padded_length));
  CUDACHECK(cudaMalloc(&d_Hx_pad_dst, sizeof(float) * padded_length));
  CUDACHECK(cudaMalloc(&d_Hy_pad_dst, sizeof(float) * padded_length));
  CUDACHECK(cudaMalloc(&d_Hz_pad_dst, sizeof(float) * padded_length));

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

  CUDACHECK(cudaMalloc(&d_xx_heads, sizeof(int) * xx_num));
  CUDACHECK(cudaMalloc(&d_yy_heads, sizeof(int) * yy_num));
  CUDACHECK(cudaMalloc(&d_zz_heads, sizeof(int) * zz_num));

  // initialize E, H as 0
  CUDACHECK(cudaMemset(d_Ex_pad_src, 0, sizeof(float) * padded_length));
  CUDACHECK(cudaMemset(d_Ey_pad_src, 0, sizeof(float) * padded_length));
  CUDACHECK(cudaMemset(d_Ez_pad_src, 0, sizeof(float) * padded_length));
  CUDACHECK(cudaMemset(d_Hx_pad_src, 0, sizeof(float) * padded_length));
  CUDACHECK(cudaMemset(d_Hy_pad_src, 0, sizeof(float) * padded_length));
  CUDACHECK(cudaMemset(d_Hz_pad_src, 0, sizeof(float) * padded_length));
  CUDACHECK(cudaMemset(d_Ex_pad_dst, 0, sizeof(float) * padded_length));
  CUDACHECK(cudaMemset(d_Ey_pad_dst, 0, sizeof(float) * padded_length));
  CUDACHECK(cudaMemset(d_Ez_pad_dst, 0, sizeof(float) * padded_length));
  CUDACHECK(cudaMemset(d_Hx_pad_dst, 0, sizeof(float) * padded_length));
  CUDACHECK(cudaMemset(d_Hy_pad_dst, 0, sizeof(float) * padded_length));
  CUDACHECK(cudaMemset(d_Hz_pad_dst, 0, sizeof(float) * padded_length));

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
  CUDACHECK(cudaMemcpyAsync(d_xx_heads, xx_heads.data(), sizeof(int) * xx_num, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(d_yy_heads, yy_heads.data(), sizeof(int) * yy_num, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpyAsync(d_zz_heads, zz_heads.data(), sizeof(int) * zz_num, cudaMemcpyHostToDevice));

  size_t block_size = NTX_MM_V3 * NTY_MM_V3 * NTZ_MM_V3;
  std::cout << "block_size = " << block_size << "\n";
  size_t grid_size = xx_num * yy_num * zz_num;

  // Allow large shared memory size
  int total_shmem_btyes = (H_SHX_V3 * H_SHY_V3 * H_SHZ_V3 * 3 + E_SHX_V3 * E_SHY_V3 * E_SHZ_V3 * 3) * sizeof(float); 
  cudaFuncSetAttribute(
    updateEH_mix_mapping_kernel_ver3_unroll,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    (int)total_shmem_btyes
  );
  std::cout << "total_shmem_btyes = " << total_shmem_btyes << "\n";
  std::cout << "grid_size = " << grid_size << "\n";

  auto start = std::chrono::high_resolution_clock::now();
  for(size_t tt = 0; tt < num_timesteps / BLT_MM_V3; tt++) {
    updateEH_mix_mapping_kernel_ver3_unroll<<<grid_size, block_size, total_shmem_btyes>>>(d_Ex_pad_src, d_Ey_pad_src, d_Ez_pad_src,
                                                                d_Hx_pad_src, d_Hy_pad_src, d_Hz_pad_src,
                                                                d_Ex_pad_dst, d_Ey_pad_dst, d_Ez_pad_dst,
                                                                d_Hx_pad_dst, d_Hy_pad_dst, d_Hz_pad_dst,
                                                                Cax, Cbx,
                                                                Cay, Cby,
                                                                Caz, Cbz,
                                                                Dax, Dbx,
                                                                Day, Dby,
                                                                Daz, Dbz,
                                                                Jx, Jy, Jz,
                                                                Mx, My, Mz,
                                                                _dx,
                                                                _Nx, _Ny, _Nz,
                                                                Nx_pad, Ny_pad, Nz_pad,
                                                                xx_num, yy_num, zz_num,
                                                                d_xx_heads,
                                                                d_yy_heads,
                                                                d_zz_heads);

    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //   printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    // }

    SWAP_PTR(d_Ex_pad_src, d_Ex_pad_dst);
    SWAP_PTR(d_Ey_pad_src, d_Ey_pad_dst);
    SWAP_PTR(d_Ez_pad_src, d_Ez_pad_dst);
    SWAP_PTR(d_Hx_pad_src, d_Hx_pad_dst);
    SWAP_PTR(d_Hy_pad_src, d_Hy_pad_dst);
    SWAP_PTR(d_Hz_pad_src, d_Hz_pad_dst);
    // std::swap(d_Ex_pad_src, d_Ex_pad_dst);
    // std::swap(d_Ey_pad_src, d_Ey_pad_dst);
    // std::swap(d_Ez_pad_src, d_Ez_pad_dst);
    // std::swap(d_Hx_pad_src, d_Hx_pad_dst);
    // std::swap(d_Hy_pad_src, d_Hy_pad_dst);
    // std::swap(d_Hz_pad_src, d_Hz_pad_dst);
  }
  cudaDeviceSynchronize();

  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "gpu runtime (mix mapping ver3): " << std::chrono::duration<double>(end-start).count() << "s\n"; 
  std::cout << "gpu performance: " << (_Nx * _Ny * _Nz / 1.0e6 * num_timesteps) / std::chrono::duration<double>(end-start).count() << "Mcells/s\n";

  // copy E, H back to host
  CUDACHECK(cudaMemcpyAsync(Ex_pad_src.data(), d_Ex_pad_src, sizeof(float) * padded_length, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpyAsync(Ey_pad_src.data(), d_Ey_pad_src, sizeof(float) * padded_length, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpyAsync(Ez_pad_src.data(), d_Ez_pad_src, sizeof(float) * padded_length, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpyAsync(Hx_pad_src.data(), d_Hx_pad_src, sizeof(float) * padded_length, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpyAsync(Hy_pad_src.data(), d_Hy_pad_src, sizeof(float) * padded_length, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpyAsync(Hz_pad_src.data(), d_Hz_pad_src, sizeof(float) * padded_length, cudaMemcpyDeviceToHost));

  // check padded arrays
  // for(size_t z = 0; z < Nz_pad; z++) {
  //   for(size_t y = 0; y < Ny_pad; y++) {
  //     for(size_t x = 0; x < Nx_pad; x++) {
  //       size_t padded_index = x + y * Nx_pad + z * Nx_pad * Ny_pad;
  //       if(Ex_pad_src[padded_index] != 0) {
  //         std::cout << "(x, y, z) = " << x << ", " << y << ", " << z << ", ";
  //         std::cout << "Ex_pad_src[padded_index] = " << Ex_pad_src[padded_index] << "\n";
  //       }
  //     }
  //   }
  // }

  // transfer data back to unpadded arrays
  for(size_t z = 0; z < _Nz; z++) {
    for(size_t y = 0; y < _Ny; y++) {
      for(size_t x = 0; x < _Nx; x++) {
        size_t x_pad = x + LEFT_PAD_MM_V3;
        size_t y_pad = y + LEFT_PAD_MM_V3;
        size_t z_pad = z + LEFT_PAD_MM_V3;
        size_t unpadded_index = x + y * _Nx + z * _Nx * _Ny;
        size_t padded_index = x_pad + y_pad * Nx_pad + z_pad * Nx_pad * Ny_pad;
        _Ex_gpu[unpadded_index] = Ex_pad_src[padded_index];
        _Ey_gpu[unpadded_index] = Ey_pad_src[padded_index];
        _Ez_gpu[unpadded_index] = Ez_pad_src[padded_index];
        _Hx_gpu[unpadded_index] = Hx_pad_src[padded_index];
        _Hy_gpu[unpadded_index] = Hy_pad_src[padded_index];
        _Hz_gpu[unpadded_index] = Hz_pad_src[padded_index];
      }
    }
  }

  CUDACHECK(cudaFree(d_Ex_pad_src));
  CUDACHECK(cudaFree(d_Ey_pad_src));
  CUDACHECK(cudaFree(d_Ez_pad_src));
  CUDACHECK(cudaFree(d_Hx_pad_src));
  CUDACHECK(cudaFree(d_Hy_pad_src));
  CUDACHECK(cudaFree(d_Hz_pad_src));

  CUDACHECK(cudaFree(d_Ex_pad_dst));
  CUDACHECK(cudaFree(d_Ey_pad_dst));
  CUDACHECK(cudaFree(d_Ez_pad_dst));
  CUDACHECK(cudaFree(d_Hx_pad_dst));
  CUDACHECK(cudaFree(d_Hy_pad_dst));
  CUDACHECK(cudaFree(d_Hz_pad_dst));

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

  CUDACHECK(cudaFree(d_xx_heads));
  CUDACHECK(cudaFree(d_yy_heads));
  CUDACHECK(cudaFree(d_zz_heads));
}   

} // end of namespace gdiamond

#endif









































