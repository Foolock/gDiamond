#include "gdiamond_omp.hpp"

int main() {

  size_t Nx = 100;
  size_t Ny = 100;
  size_t Nz = 100;
  size_t num_timesteps = 100;
  gdiamond::gDiamond exp(Nx, Ny, Nz); 

  exp.update_FDTD_seq(num_timesteps);
  exp.update_FDTD_omp_dt(31, 31, 31, 10, 100);

  if(!exp.check_correctness_omp_dt()) {
    std::cerr << "error: omp_dt result wrong\n";
    std::exit(EXIT_FAILURE);
  }
  return 0;
}
