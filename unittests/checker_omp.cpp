#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include "gdiamond_omp.hpp"

// --------------------------------------------------------
// Testcase: check correctness of openmp compared to single thread implementation 
// --------------------------------------------------------
TEST_CASE("check correctness of openmp" * doctest::timeout(300)) {

  gdiamond::gDiamond exp(100, 100, 100); 

  exp.update_FDTD_seq(100);
  exp.update_FDTD_omp(100);

  REQUIRE(exp.check_correctness_omp() == true);

}







