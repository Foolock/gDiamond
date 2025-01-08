#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include "gdiamond_gpu.cuh"

// --------------------------------------------------------
// Testcase: check cycle by manually adding nodes and edges
// --------------------------------------------------------
TEST_CASE("check correctness of gpu" * doctest::timeout(300)) {

  gdiamond::gDiamond exp(100, 100, 100); 

  exp.update_FDTD_seq(100);
  exp.update_FDTD_gpu(100);

  REQUIRE(exp.check_correctness() == true);

}





