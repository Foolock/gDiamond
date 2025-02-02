#include "gdiamond_gpu.cuh"

int main() {

  size_t Nx = 19;
  size_t Ny = 19;
  size_t Nz = 19;
  size_t num_timesteps = 100;
  gdiamond::gDiamond exp(Nx, Ny, Nz); 

  exp.update_FDTD_seq_check_result(num_timesteps);
  exp.update_FDTD_gpu_simulation_1_D(num_timesteps);

  // exp.print_results();

  if(!exp.check_correctness_simu()) {
    std::cerr << "error: results not match\n";
    std::exit(EXIT_FAILURE);
  }

  return 0;
}
