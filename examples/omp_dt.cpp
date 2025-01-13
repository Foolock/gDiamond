#include "gdiamond_omp.hpp"

int main() {

  gdiamond::gDiamond exp(100, 100, 100); 

  exp.update_FDTD_omp_dt(100);
  return 0;
}
