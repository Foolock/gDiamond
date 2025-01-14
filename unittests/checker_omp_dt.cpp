#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include "gdiamond_omp.hpp"

// --------------------------------------------------------
// Testcase: check correctness of openmp compared to single thread implementation 
// --------------------------------------------------------
TEST_CASE("check correctness of openmp, BLX=BLY=BLZ=16, BLT=5" * doctest::timeout(300)) {

  gdiamond::gDiamond exp(100, 100, 100); 

  exp.update_FDTD_seq(100);
  exp.update_FDTD_omp_dt(16, 16, 16, 5, 100);

  REQUIRE(exp.check_correctness_omp_dt() == true);

}

TEST_CASE("check correctness of openmp, BLX=BLY=BLZ=25, BLT=10" * doctest::timeout(300)) {

  gdiamond::gDiamond exp(100, 100, 100); 

  exp.update_FDTD_seq(100);
  exp.update_FDTD_omp_dt(25, 25, 25, 10, 100);

  REQUIRE(exp.check_correctness_omp_dt() == true);

}

TEST_CASE("check correctness of openmp, BLX=BLY=BLZ=31, BLT=20" * doctest::timeout(300)) {

  gdiamond::gDiamond exp(100, 100, 100); 

  exp.update_FDTD_seq(100);
  exp.update_FDTD_omp_dt(31, 31, 31, 20, 100);

  REQUIRE(exp.check_correctness_omp_dt() == true);

}








