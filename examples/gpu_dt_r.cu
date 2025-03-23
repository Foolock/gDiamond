#include "gdiamond_gpu.cuh"

int main(int argc, char* argv[]) {

  if(argc != 5) {
    std::cerr << "usage: ./example/gpu_dt Tx Ty Tz num_timesteps\n";
    std::exit(EXIT_FAILURE);
  }

  // Tx Ty Tz are the number of tile stripes 
  // (mountain bottom + valley bottom, excluding the 1st mountain and the last valley)
  size_t Tx = std::atoi(argv[1]);
  size_t Ty = std::atoi(argv[2]);
  size_t Tz = std::atoi(argv[3]);
  size_t num_timesteps = std::atoi(argv[4]);

  // BLT = 4, block_size = 4 for cpu simulation
  int block_size = 4;
  int valley_bottom = block_size;
  int mountain_bottom = block_size + 2 * (BLT_UB - 1) + 1;
  size_t Nx = mountain_bottom - (BLT_UB - 1) + Tx * (valley_bottom + mountain_bottom) + valley_bottom;
  size_t Ny = mountain_bottom - (BLT_UB - 1) + Ty * (valley_bottom + mountain_bottom) + valley_bottom;
  size_t Nz = mountain_bottom - (BLT_UB - 1) + Tz * (valley_bottom + mountain_bottom) + valley_bottom;

  std::cout << "simulation space: Nx = " << Nx << ", Ny = " << Ny << ", Nz = " << Nz << "\n";
  gdiamond::gDiamond exp(Nx, Ny, Nz); 

  exp.update_FDTD_cpu_simulation_dt_1_D(num_timesteps, Tx, Ty, Tz);

  return 0;
}
