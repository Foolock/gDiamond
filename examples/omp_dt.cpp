#include "gdiamond_omp.hpp"

int main() {

  size_t Nx = 100;
  size_t Ny = 100;
  size_t Nz = 100;
  size_t num_timesteps = 5;
  gdiamond::gDiamond exp(Nx, Ny, Nz); 

  exp.update_FDTD_seq(num_timesteps);
  size_t BLX = 7;
  size_t BLY = 7;
  size_t BLZ = 7;
  size_t BLT = 5;
  exp.update_FDTD_omp_dt(BLX, BLY, BLZ, BLT, num_timesteps);

  if(!exp.check_correctness_omp_dt()) {
    std::cerr << "error: omp_dt result wrong\n";
    std::exit(EXIT_FAILURE);
  }
  return 0;
}
