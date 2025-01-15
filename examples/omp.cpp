#include "gdiamond_omp.hpp"

int main() {

  size_t Nx = 100;
  size_t Ny = 100;
  size_t Nz = 100;
  size_t num_timesteps = 100;
  gdiamond::gDiamond exp(Nx, Ny, Nz); 

  exp.update_FDTD_seq(num_timesteps);
  exp.update_FDTD_omp(num_timesteps);

  if(!exp.check_correctness_omp()) {
    std::cerr << "error: results not match\n";
    std::exit(EXIT_FAILURE);
  }
  return 0;
}
