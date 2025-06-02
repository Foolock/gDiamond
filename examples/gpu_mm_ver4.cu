#include "gdiamond_gpu.cuh"
#include "gdiamond_gpu_mm_ver4.cuh"

int main(int argc, char* argv[]) {

  std::cerr << "mix mapping ver4->break one diamond into one small diamond and multiple parallelograms.\n";
   
  if(argc != 5) {
    std::cerr << "usage: ./example/gpu_dt Tx Ty Tz num_timesteps\n";
    std::exit(EXIT_FAILURE);
  }

  // Tx Ty Tz are the number of tile stripes 
  size_t Tx = std::atoi(argv[1]);
  size_t Ty = std::atoi(argv[2]);
  size_t Tz = std::atoi(argv[3]);
  size_t num_timesteps = std::atoi(argv[4]);

  return 0;
}
