#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include "gdiamond_omp.hpp"

// --------------------------------------------------------
// Testcase: check correctness of openmp compared to single thread implementation 
// --------------------------------------------------------
TEST_CASE("check correctness of openmp, BLX=BLY=BLZ=16, BLT=5" * doctest::timeout(300)) {

  size_t Nx = 100;
  size_t Ny = 100;
  size_t Nz = 100;
  size_t num_timesteps = 100;
  gdiamond::gDiamond exp(Nx, Ny, Nz); 

  exp.update_FDTD_seq(num_timesteps);
  exp.update_FDTD_omp_dt(16, 16, 16, 5, num_timesteps);

  REQUIRE(exp.check_correctness_omp_dt() == true);

}






