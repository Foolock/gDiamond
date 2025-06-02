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

}  


} // end of namespace gdiamond

#endif









































