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









































