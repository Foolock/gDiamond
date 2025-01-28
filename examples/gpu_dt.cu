#include "gdiamond_gpu.cuh"

int main() {

  size_t Nx = 100;
  size_t Ny = 100;
  size_t Nz = 100;
  size_t num_timesteps = 100;
  gdiamond::gDiamond exp(Nx, Ny, Nz); 

  exp.update_FDTD_seq_check_result(num_timesteps);
  exp.update_FDTD_gpu_fuse_kernel(num_timesteps);

  if(!exp.check_correctness_gpu()) {
    std::cerr << "error: results not match\n";
    std::exit(EXIT_FAILURE);
  }

  return 0;
}
