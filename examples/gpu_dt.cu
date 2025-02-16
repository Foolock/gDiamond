#include "gdiamond_gpu.cuh"

int main(int argc, char* argv[]) {

  if(argc != 5) {
    std::cerr << "usage: ./example/gpu_dt Nx Ny Nz num_timesteps\n";
    std::exit(EXIT_FAILURE);
  }

  size_t Nx = std::atoi(argv[1]);
  size_t Ny = std::atoi(argv[2]);
  size_t Nz = std::atoi(argv[3]);
  size_t num_timesteps = std::atoi(argv[4]);
  gdiamond::gDiamond exp(Nx, Ny, Nz); 

  exp.update_FDTD_seq_check_result(num_timesteps);
  exp.update_FDTD_gpu_simulation_1_D_pt_shmem(num_timesteps);

  // exp.print_results();

  if(!exp.check_correctness_gpu_2D()) {
    std::cerr << "error: results not match\n";
    std::exit(EXIT_FAILURE);
  }

  return 0;
}
