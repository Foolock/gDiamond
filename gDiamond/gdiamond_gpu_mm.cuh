#ifndef GDIAMOND_GPU_MM_CUH
#define GDIAMOND_GPU_MM_CUH

#include "gdiamond.hpp"
#include "kernels_mm.cuh"
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
void gDiamond::_updateEH_mix_mapping(std::vector<float>& Ex_pad, std::vector<float>& Ey_pad, std::vector<float>& Ez_pad,
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
    
    // declare shared memory
    constexpr int H_SHX = (X_is_mountain)? 17 : 16;
    constexpr int H_SHY = (Y_is_mountain)? 11 : 10;
    constexpr int H_SHZ = (Z_is_mountain)? 11 : 10;
    constexpr int E_SHX = (X_is_mountain)? 16 : 17;
    constexpr int E_SHY = (Y_is_mountain)? 10 : 11;
    constexpr int E_SHZ = (Z_is_mountain)? 10 : 11;
    float Hx_shmem[H_SHX * H_SHY * H_SHZ];
    float Hy_shmem[H_SHX * H_SHY * H_SHZ];
    float Hz_shmem[H_SHX * H_SHY * H_SHZ];
    float Ex_shmem[E_SHX * E_SHY * E_SHZ];
    float Ey_shmem[E_SHX * E_SHY * E_SHZ];
    float Ez_shmem[E_SHX * E_SHY * E_SHZ];

    if(xx == 0 && yy == 0 && zz == 0) {
      std::cout << "H_SHX = " << H_SHX << ", H_SHY = " << H_SHY << ", H_SHZ = " << H_SHZ << "\n";
      std::cout << "E_SHX = " << E_SHX << ", E_SHY = " << E_SHY << ", E_SHZ = " << E_SHZ << "\n";
    }

    // load shared memory
    for(size_t thread_id = 0; thread_id < block_size; thread_id++) {

      local_x = thread_id % BLX_MM;
      local_y = (thread_id / BLX_MM) % BLY_MM;
      local_z = thread_id / (BLX_MM * BLY_MM);

      // load H
      // X dimension has 1 extra HALO load, one thread load one element, 
      // if mountain, tid = 0 load one extra H at xx_heads[xx] - 1
      // if valley, tid = NTX_MM - 1 load one extra E at xx_heads[xx] + NTX_MM 
      // Y, Z dimension will do iterative load in the range [loadH_head, loadH_tail]
      int loadH_head_y;
      int loadH_head_z;
      int loadH_tail_y;
      int loadH_tail_z;

      if constexpr (X_is_mountain) { 
        H_shared_x = local_x + 1; 
        global_x = xx_heads[xx] + H_shared_x - 1;
      }
      else { 
        H_shared_x = local_x; 
        global_x = xx_heads[xx] + H_shared_x;
      }

      if constexpr (Y_is_mountain) { loadH_head_y = yy_heads[yy] - 1; }
      else { loadH_head_y = yy_heads[yy]; }

      if constexpr (Z_is_mountain) { loadH_head_z = zz_heads[zz] - 1; } 
      else { loadH_head_z = zz_heads[zz]; }

      loadH_tail_y = loadH_head_y + H_SHY - 1;
      loadH_tail_z = loadH_head_z + H_SHZ - 1;

      if(xx == 0 && yy == 0 && zz == 0) {
        std::cout << "-------------------------------------------------\n";
        std::cout << "thread_id = " << thread_id << "\n";
        std::cout << "local_x = " << local_x << ", local_y = " << local_y << ", local_z = " << local_z << "\n";
        std::cout << "loadH_head_y = " << loadH_head_y << ", loadH_tail_y = " << loadH_tail_y << "\n";
        std::cout << "loadH_head_z = " << loadH_head_z << ", loadH_tail_z = " << loadH_tail_z << "\n";
      }
      for(H_shared_y = local_y; H_shared_y <= loadH_tail_y; H_shared_y += NTY_MM) {

        if constexpr (Y_is_mountain) { global_y = yy_heads[yy] + H_shared_y - 1; }
        else { global_y = yy_heads[yy] + H_shared_y; }

        for(H_shared_z = local_z; H_shared_z <= loadH_tail_z; H_shared_z += NTZ_MM) {
          
          if constexpr (Z_is_mountain) { global_z = zz_heads[zz] + H_shared_z - 1; }
          else { global_z = zz_heads[zz] + H_shared_z; }

          if(xx == 0 && yy == 0 && zz == 0) {
            std::cout << "H_shared_x = " << H_shared_x << ", H_shared_y = " << H_shared_y << ", H_shared_z = " << H_shared_z << "\n";
            std::cout << "global_x = " << global_x << ", global_y = " << global_y << ", global_z = " << global_z << "\n";
          }

        }
      }

      // load HALO for H_shared_x = 0

    }

  }  
}

void gDiamond::update_FDTD_mix_mapping_sequential(size_t num_timesteps, size_t Tx, size_t Ty, size_t Tz) { // simulate GPU workflow  

  // pad E, H array
  const size_t Nx_pad = _Nx + LEFT_PAD_MM + RIGHT_PAD_MM; 
  const size_t Ny_pad = _Ny + LEFT_PAD_MM + RIGHT_PAD_MM; 
  const size_t Nz_pad = _Nz + LEFT_PAD_MM + RIGHT_PAD_MM; 
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
        size_t x_pad = x + LEFT_PAD_MM;
        size_t y_pad = y + LEFT_PAD_MM;
        size_t z_pad = z + LEFT_PAD_MM;
        size_t unpadded_index = x + y * _Nx + z * _Nx * _Ny;      
        size_t padded_index = x_pad + y_pad * Nx_pad + z_pad * Nx_pad * Ny_pad;
        Ex_pad[padded_index] = _Ex[unpadded_index];
        Ey_pad[padded_index] = _Ey[unpadded_index];
        Ez_pad[padded_index] = _Ez[unpadded_index];
        Hx_pad[padded_index] = _Hx[unpadded_index];
        Hy_pad[padded_index] = _Hy[unpadded_index];
        Hz_pad[padded_index] = _Hz[unpadded_index];
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
                             xx_heads_m[index-1] + (MOUNTAIN_X + VALLEY_X);
  }
  for(size_t index=0; index<xx_num_v; index++) {
    xx_heads_v[index] = (index == 0)? LEFT_PAD_MM + VALLEY_X :
                             xx_heads_v[index-1] + (MOUNTAIN_X + VALLEY_X);
  }
  for(size_t index=0; index<yy_num_m; index++) {
    yy_heads_m[index] = (index == 0)? 1 :
                             yy_heads_m[index-1] + (MOUNTAIN_Y + VALLEY_Y);
  }
  for(size_t index=0; index<yy_num_v; index++) {
    yy_heads_v[index] = (index == 0)? LEFT_PAD_MM + VALLEY_Y :
                             yy_heads_v[index-1] + (MOUNTAIN_Y + VALLEY_Y);
  }
  for(size_t index=0; index<zz_num_m; index++) {
    zz_heads_m[index] = (index == 0)? 1 :
                             zz_heads_m[index-1] + (MOUNTAIN_Z + VALLEY_Z);
  }
  for(size_t index=0; index<zz_num_v; index++) {
    zz_heads_v[index] = (index == 0)? LEFT_PAD_MM + VALLEY_Z :
                             zz_heads_v[index-1] + (MOUNTAIN_Z + VALLEY_Z);
  }

  std::cout << "xx_heads_m = ";
  for(const auto& data : xx_heads_m) {
    std::cout << data << " ";
  }
  std::cout << "\n";
  std::cout << "xx_heads_v = ";
  for(const auto& data : xx_heads_v) {
    std::cout << data << " ";
  }
  std::cout << "\n";
  std::cout << "yy_heads_m = ";
  for(const auto& data : yy_heads_m) {
    std::cout << data << " ";
  }
  std::cout << "\n";
  std::cout << "yy_heads_v = ";
  for(const auto& data : yy_heads_v) {
    std::cout << data << " ";
  }
  std::cout << "\n";
  std::cout << "zz_heads_m = ";
  for(const auto& data : zz_heads_m) {
    std::cout << data << " ";
  }
  std::cout << "\n";
  std::cout << "zz_heads_v = ";
  for(const auto& data : zz_heads_v) {
    std::cout << data << " ";
  }
  std::cout << "\n";

  size_t block_size = NTX_MM * NTY_MM * NTZ_MM;
  std::cout << "block_size = " << block_size << "\n";
  size_t grid_size;

  for(size_t tt = 0; tt < num_timesteps / BLT_MM; tt++) {

    // phase 1. m, m, m
    grid_size = xx_num_m * yy_num_m * zz_num_m;
    _updateEH_mix_mapping<true, true, true>(Ex_pad, Ey_pad, Ez_pad,
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

    // phase 8. v, v, v 
    // grid_size = xx_num_v * yy_num_v * zz_num_v;
    // _updateEH_mix_mapping<false, false, false>(Ex_pad, Ey_pad, Ez_pad,
    //                                         Hx_pad, Hy_pad, Hz_pad,
    //                                         _Cax, _Cbx,
    //                                         _Cay, _Cby,
    //                                         _Caz, _Cbz,
    //                                         _Dax, _Dbx,
    //                                         _Day, _Dby,
    //                                         _Daz, _Dbz,
    //                                         _Jx, _Jy, _Jz,
    //                                         _Mx, _My, _Mz,
    //                                         _dx, 
    //                                         _Nx, _Ny, _Nz,
    //                                         Nx_pad, Ny_pad, Nz_pad, 
    //                                         xx_num_v, yy_num_v, zz_num_v, 
    //                                         xx_heads_v, 
    //                                         yy_heads_v,
    //                                         zz_heads_v,
    //                                         block_size,
    //                                         grid_size);


  }

}

} // end of namespace gdiamond

#endif




























