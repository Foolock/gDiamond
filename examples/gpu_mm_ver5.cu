#include "gdiamond_gpu.cuh"
#include "gdiamond_gpu_mm_ver5.cuh"

int main(int argc, char* argv[]) {

  std::cerr << "mix mapping ver4->break one diamond into one small diamond and multiple parallelograms.\n";
  std::cerr << "tiling parameters: BLT_MM_V5 = " << BLT_MM_V5
            << ", NTX_MM_V5 = " << NTX_MM_V5 << ", NTY_MM_V5 = " << NTY_MM_V5 << ", NTZ_MM_V5 = " << NTZ_MM_V5 << "\n";
  std::cerr << "X dimension: \n";
  std::cerr << "MOUNTAIN_X_V5 = " << MOUNTAIN_X_V5 << ", VALLEY_X_V5 = " << VALLEY_X_V5 << "\n";
  std::cerr << "BLX_R = " << BLX_R << ", BLX_P = " << BLX_P << ", NUM_P_X = " << NUM_P_X << "\n";
  std::cerr << "Y dimension: \n";
  std::cerr << "MOUNTAIN_Y_V5 = " << MOUNTAIN_Y_V5 << ", VALLEY_Y_V5 = " << VALLEY_Y_V5 << "\n";
  std::cerr << "BLY_R = " << BLY_R << ", BLY_P = " << BLY_P << ", NUM_P_Y = " << NUM_P_Y << "\n";
  std::cerr << "Z dimension: \n";
  std::cerr << "MOUNTAIN_Z_V5 = " << MOUNTAIN_Z_V5 << ", VALLEY_Z_V5 = " << VALLEY_Z_V5 << "\n";
  std::cerr << "BLZ_R = " << BLZ_R << ", BLZ_P = " << BLZ_P << ", NUM_P_Z = " << NUM_P_Z << "\n";
   
  if(argc != 5) {
    std::cerr << "usage: ./example/gpu_dt Tx Ty Tz num_timesteps\n";
    std::exit(EXIT_FAILURE);
  }

  // Tx Ty Tz are the number of tile stripes 
  size_t Tx = std::atoi(argv[1]);
  size_t Ty = std::atoi(argv[2]);
  size_t Tz = std::atoi(argv[3]);
  size_t num_timesteps = std::atoi(argv[4]);

  size_t Nx = Tx * VALLEY_X_V5;
  size_t Ny = Ty * VALLEY_Y_V5;
  size_t Nz = Tz * VALLEY_Z_V5;

  std::cout << "simulation space: Nx = " << Nx << ", Ny = " << Ny << ", Nz = " << Nz << "\n";
  gdiamond::gDiamond exp(Nx, Ny, Nz);

  exp.update_FDTD_mix_mapping_sequential_ver5(num_timesteps, Tx, Ty, Tz);

  return 0;
}











