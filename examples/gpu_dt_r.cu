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

  int valley_bottomX = NTX;
  int mountain_bottomX = NTX + 2 * (BLT_UB - 1) + 1;
  int valley_bottomY = NTY;
  int mountain_bottomY = NTY + 2 * (BLT_UB - 1) + 1;
  size_t Nx = mountain_bottomX - (BLT_UB - 1) + Tx * (valley_bottomX + mountain_bottomX) + valley_bottomX;
  size_t Ny = mountain_bottomY - (BLT_UB - 1) + Ty * (valley_bottomY + mountain_bottomY) + valley_bottomY;
  size_t Nz = Tz * (valley_bottomX + mountain_bottomX);

  std::cout << "simulation space: Nx = " << Nx << ", Ny = " << Ny << ", Nz = " << Nz << "\n";
  gdiamond::gDiamond exp(Nx, Ny, Nz); 

  exp.update_FDTD_seq_check_result(num_timesteps);
  exp.update_FDTD_cpu_simulation_dt_2_D(num_timesteps, Tx, Ty, Tz);

  exp.print_results();

  return 0;
}
