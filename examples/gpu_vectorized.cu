#include "gdiamond_gpu_naive_vectorized.cuh"
#include "gdiamond_gpu_mm_ver3.cuh"

int main(int argc, char* argv[]) {

  std::cerr << "mix mapping ver3->consider valley as mountain top.\n";
  std::cerr << "tiling parameters: BLT_MM_V3 = " << BLT_MM_V3 
            << ", NTX_MM_V3 = " << NTX_MM_V3 << ", NTY_MM_V3 = " << NTY_MM_V3 << ", NTZ_MM_V3 = " << NTZ_MM_V3 << "\n";
  std::cerr << "X dimension: MOUNTAIN_X_V3 = " << MOUNTAIN_X_V3 << ", VALLEY_X_V3 = " << VALLEY_X_V3 << "\n"; 
  std::cerr << "Y dimension: MOUNTAIN_Y_V3 = " << MOUNTAIN_Y_V3 << ", VALLEY_Y_V3 = " << VALLEY_Y_V3 << "\n"; 
  std::cerr << "Z dimension: MOUNTAIN_Z_V3 = " << MOUNTAIN_Z_V3 << ", VALLEY_Z_V3 = " << VALLEY_Z_V3 << "\n"; 
    
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

  size_t Nx = Tx * VALLEY_X_V3; 
  size_t Ny = Ty * VALLEY_Y_V3; 
  size_t Nz = Tz * VALLEY_Z_V3; 

  std::cout << "simulation space: Nx = " << Nx << ", Ny = " << Ny << ", Nz = " << Nz << "\n";
  gdiamond::gDiamond exp(Nx, Ny, Nz); 

  exp.update_FDTD_gpu_3D_warp_underutilization_fix_vectorized(num_timesteps); 
  // exp.update_FDTD_mix_mapping_gpu_ver3(num_timesteps, Tx, Ty, Tz);  

  // exp.print_results();

  // if(!exp.check_correctness_gpu()) {
  //   std::cerr << "results are wrong!\n";
  //   std::exit(EXIT_FAILURE);
  // }

  // std::cerr << "results are matched.\n";

  return 0;
}
