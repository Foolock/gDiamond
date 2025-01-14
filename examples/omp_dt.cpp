#include "gdiamond_omp.hpp"

int main() {

  gdiamond::gDiamond exp(100, 100, 100); 

  exp.update_FDTD_seq(100);
  exp.update_FDTD_omp_dt(31, 31, 31, 10, 100);

  if(!exp.check_correctness_omp_dt()) {
    std::cerr << "error: omp_dt result wrong\n";
    std::exit(EXIT_FAILURE);
  }
  return 0;
}
