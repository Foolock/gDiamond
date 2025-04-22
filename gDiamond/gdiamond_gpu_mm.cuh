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
    
    // declare shared memory
    constexpr int SHX_H = (X_is_mountain)? 17 : 16;
    constexpr int SHY_H = (Y_is_mountain)? 11 : 10;
    constexpr int SHZ_H = (Z_is_mountain)? 11 : 10;
    constexpr int SHX_E = (X_is_mountain)? 16 : 17;
    constexpr int SHY_E = (Y_is_mountain)? 10 : 11;
    constexpr int SHZ_E = (Z_is_mountain)? 10 : 11;
    float Hx_shmem[SHX_H * SHY_H * SHZ_H];
    float Hy_shmem[SHX_H * SHY_H * SHZ_H];
    float Hz_shmem[SHX_H * SHY_H * SHZ_H];
    float Ex_shmem[SHX_E * SHY_E * SHZ_E];
    float Ey_shmem[SHX_E * SHY_E * SHZ_E];
    float Ez_shmem[SHX_E * SHY_E * SHZ_E];

    // load shared memory

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

  for(size_t tt = 0; tt < num_timesteps; tt++) {
   
  }

}

} // end of namespace gdiamond

#endif




























