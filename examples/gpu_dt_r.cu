#include "gdiamond_gpu.cuh"

int main(int argc, char* argv[]) {

  if(argc != 5) {
    std::cerr << "usage: ./example/gpu_dt Tx Ny Nz num_timesteps\n";
    std::exit(EXIT_FAILURE);
  }

  // Tx Ty Tz are the number of tile stripes 
  // (mountain bottom + valley bottom, excluding the 1st mountain and the last valley)
  size_t Tx = std::atoi(argv[1]);
  size_t Ny = std::atoi(argv[2]);
  size_t Nz = std::atoi(argv[3]);
  size_t num_timesteps = std::atoi(argv[4]);

  int valley_bottomX = NTX;
  int mountain_bottomX = NTX + 2 * (BLT_UB - 1) + 1;
  size_t Nx = mountain_bottomX - (BLT_UB - 1) + Tx * (valley_bottomX + mountain_bottomX) + valley_bottomX;

  std::cout << "simulation space: Nx = " << Nx << ", Ny = " << Ny << ", Nz = " << Nz << "\n";
  gdiamond::gDiamond exp(Nx, Ny, Nz); 

  exp.update_FDTD_gpu_3D_warp_underutilization_fix(num_timesteps);

  return 0;
}
