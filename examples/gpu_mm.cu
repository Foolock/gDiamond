#include "gdiamond_gpu.cuh"
#include "gdiamond_gpu_mm.cuh"

int main(int argc, char* argv[]) {

  std::cerr << "tiling parameters: BLT_MM = " << BLT_MM 
            << ", NTX_MM = " << NTX_MM << ", NTY_MM = " << NTY_MM << ", NTZ_MM = " << NTZ_MM << "\n";
  std::cerr << "X dimension: MOUNTAIN_X = " << MOUNTAIN_X << ", VALLEY_X = " << VALLEY_X << "\n"; 
  std::cerr << "Y dimension: MOUNTAIN_Y = " << MOUNTAIN_Y << ", VALLEY_Y = " << VALLEY_Y << "\n"; 
  std::cerr << "Z dimension: MOUNTAIN_Z = " << MOUNTAIN_Z << ", VALLEY_Z = " << VALLEY_Z << "\n"; 
    
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

  size_t Nx = Tx * (MOUNTAIN_X + VALLEY_X) + MOUNTAIN_X - (BLT_MM - 1) + VALLEY_X; 
  size_t Ny = Ty * (MOUNTAIN_Y + VALLEY_Y) + MOUNTAIN_Y - (BLT_MM - 1) + VALLEY_Y; 
  size_t Nz = Tz * (MOUNTAIN_Z + VALLEY_Z) + MOUNTAIN_Z - (BLT_MM - 1) + VALLEY_Z; 

  std::cout << "simulation space: Nx = " << Nx << ", Ny = " << Ny << ", Nz = " << Nz << "\n";
  gdiamond::gDiamond exp(Nx, Ny, Nz); 

  exp.update_FDTD_gpu_3D_warp_underutilization_fix(num_timesteps);  
  exp.update_FDTD_mix_mapping_gpu(num_timesteps, Tx, Ty, Tz);  

  if(!exp.check_correctness_gpu()) {
    std::cerr << "results are wrong!\n";
    std::exit(EXIT_FAILURE);
  }

  std::cerr << "results are matched.\n";

  return 0;
}
