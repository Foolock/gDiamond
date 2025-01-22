#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include "gdiamond_gpu.cuh"

// --------------------------------------------------------
// Testcase: check correctness of gpu compared to single thread implementation 
// --------------------------------------------------------
TEST_CASE("check correctness of gpu" * doctest::timeout(300)) {

  size_t Nx = 100;
  size_t Ny = 100;
  size_t Nz = 100;
  size_t num_timesteps = 100;
  gdiamond::gDiamond exp(Nx, Ny, Nz); 

  exp.update_FDTD_seq(num_timesteps);
  exp.update_FDTD_gpu_3D_warp_underutilization_fix(num_timesteps);

  REQUIRE(exp.check_correctness_gpu() == true);

}


