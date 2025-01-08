#include "gdiamond_gpu.cuh"

int main() {

  gdiamond::gDiamond exp(100, 100, 100); 

  exp.update_FDTD_seq(100);
  exp.update_FDTD_gpu(100);
  return 0;
}
