#ifndef GDIAMOND_GPU_MM_VER4_CUH
#define GDIAMOND_GPU_MM_VER4_CUH

#include "gdiamond.hpp"
#include "kernels_mm_ver4.cuh"
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

void gDiamond::_updateEH_mix_mapping_ver4(std::vector<float>& Ex_pad_src, std::vector<float>& Ey_pad_src, std::vector<float>& Ez_pad_src,
                                          std::vector<float>& Hx_pad_src, std::vector<float>& Hy_pad_src, std::vector<float>& Hz_pad_src,
                                          std::vector<float>& Ex_pad_rep, std::vector<float>& Ey_pad_rep, std::vector<float>& Ez_pad_rep,
                                          std::vector<float>& Hx_pad_rep, std::vector<float>& Hy_pad_rep, std::vector<float>& Hz_pad_rep,
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
    // parallelogram calculation used more shared memory than replication calculation
    float Hx_shmem[H_SHX_V4 * H_SHY_V4 * H_SHZ_V4];
    float Hy_shmem[H_SHX_V4 * H_SHY_V4 * H_SHZ_V4];
    float Hz_shmem[H_SHX_V4 * H_SHY_V4 * H_SHZ_V4];
    float Ex_shmem[E_SHX_V4 * E_SHY_V4 * E_SHZ_V4];
    float Ey_shmem[E_SHX_V4 * E_SHY_V4 * E_SHZ_V4];
    float Ez_shmem[E_SHX_V4 * E_SHY_V4 * E_SHZ_V4];

    // load shared memory (replication part)
    // since in X dimension, BLX_R = 8 and we are using 16 threads
    // we don't need to load that much of elements
    // these bounds are for loading core
    // so there is no difference between loadE and loadH 
    // there is no explicit bound for loading in Y, Z dimension
    // bounds are refer to padded global_x
    const int load_head_X = xx_heads[xx]; 
    const int load_tail_X = xx_heads[xx] + BLX_R - 1;
    for(size_t thread_id = 0; thread_id < block_size; thread_id++) {
      local_x = thread_id % NTX_MM_V4;
      local_y = (thread_id / NTX_MM_V4) % NTY_MM_V4;
      local_z = thread_id / (NTX_MM_V4 * NTY_MM_V4);
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

    }

  }

}

void gDiamond::update_FDTD_mix_mapping_sequential_ver4(size_t num_timesteps, size_t Tx, size_t Ty, size_t Tz) {

  std::cout << "running update_FDTD_mix_mapping_sequential_ver4...\n";

  // pad E, H array
  const size_t Nx_pad = _Nx + LEFT_PAD_MM_V4 + RIGHT_PAD_MM_V4;
  const size_t Ny_pad = _Ny + LEFT_PAD_MM_V4 + RIGHT_PAD_MM_V4;
  const size_t Nz_pad = _Nz + LEFT_PAD_MM_V4 + RIGHT_PAD_MM_V4;
  const size_t padded_length = Nx_pad * Ny_pad * Nz_pad;

  /*
   * replication cal read from src, write to rep
   * parallelogram cal read from src and rep, write to dst (with no gap, entirely overwite)
   * then after each kernel, swap src and dst, src stores the latest results
   */
  std::vector<float> Ex_pad_src(padded_length, 0);
  std::vector<float> Ey_pad_src(padded_length, 0);
  std::vector<float> Ez_pad_src(padded_length, 0);
  std::vector<float> Hx_pad_src(padded_length, 0);
  std::vector<float> Hy_pad_src(padded_length, 0);
  std::vector<float> Hz_pad_src(padded_length, 0);

  std::vector<float> Ex_pad_rep(padded_length, 0);
  std::vector<float> Ey_pad_rep(padded_length, 0);
  std::vector<float> Ez_pad_rep(padded_length, 0);
  std::vector<float> Hx_pad_rep(padded_length, 0);
  std::vector<float> Hy_pad_rep(padded_length, 0);
  std::vector<float> Hz_pad_rep(padded_length, 0);

  std::vector<float> Ex_pad_dst(padded_length, 0);
  std::vector<float> Ey_pad_dst(padded_length, 0);
  std::vector<float> Ez_pad_dst(padded_length, 0);
  std::vector<float> Hx_pad_dst(padded_length, 0);
  std::vector<float> Hy_pad_dst(padded_length, 0);
  std::vector<float> Hz_pad_dst(padded_length, 0);

  // tiling parameters
  // for mix mapping ver4, all tiles are mountains
  size_t xx_num = Tx;
  size_t yy_num = Ty;
  size_t zz_num = Tz;
  std::vector<int> xx_heads(xx_num, 0); // head indices of big mountains
  std::vector<int> yy_heads(yy_num, 0);
  std::vector<int> zz_heads(zz_num, 0);

  for(size_t index=0; index<xx_num; index++) {
    xx_heads[index] = (index == 0)? 1 :
                             xx_heads[index-1] + NUM_P_X * BLX_P;
  }

  for(size_t index=0; index<yy_num; index++) {
    yy_heads[index] = (index == 0)? 1 :
                             yy_heads[index-1] + NUM_P_Y * BLY_P;
  }

  for(size_t index=0; index<zz_num; index++) {
    zz_heads[index] = (index == 0)? 1 :
                             zz_heads[index-1] + NUM_P_Z * BLZ_P;
  }

  std::cout << "xx_heads = ";
  for(const auto& data : xx_heads) {
    std::cout << data << " ";
  }
  std::cout << "\n";
  std::cout << "yy_heads = ";
  for(const auto& data : yy_heads) {
    std::cout << data << " ";
  }
  std::cout << "\n";
  std::cout << "zz_heads = ";
  for(const auto& data : zz_heads) {
    std::cout << data << " ";
  }
  std::cout << "\n";

  size_t block_size = NTX_MM_V4 * NTY_MM_V4 * NTZ_MM_V4;
  std::cout << "block_size = " << block_size << "\n";
  size_t grid_size = xx_num * yy_num * zz_num;

  for(size_t tt = 0; tt < num_timesteps / BLT_MM_V4; tt++) {

  }

}  


} // end of namespace gdiamond

#endif









































