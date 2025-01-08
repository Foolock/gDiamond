#include "gdiamond_omp.hpp"

int main() {

  gdiamond::gDiamond exp(100, 100, 100); 

  exp.update_FDTD_seq(100);
  exp.update_FDTD_omp(100);
  return 0;
}
