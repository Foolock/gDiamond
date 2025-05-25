set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/examples)
add_executable(gpu ${GDIAMOND_EXAMPLE_DIR}/gpu.cu)
add_executable(gpu_figures ${GDIAMOND_EXAMPLE_DIR}/gpu_figures.cu)
add_executable(gpu_dt ${GDIAMOND_EXAMPLE_DIR}/gpu_dt.cu)
add_executable(gpu_pt ${GDIAMOND_EXAMPLE_DIR}/gpu_pt.cu)
add_executable(gpu_fq ${GDIAMOND_EXAMPLE_DIR}/gpu_fq.cu)
add_executable(gpu_upper_bound ${GDIAMOND_EXAMPLE_DIR}/gpu_upper_bound.cu)
add_executable(gpu_dt_r ${GDIAMOND_EXAMPLE_DIR}/gpu_dt_r.cu)
add_executable(omp ${GDIAMOND_EXAMPLE_DIR}/omp.cpp)
add_executable(omp_figures ${GDIAMOND_EXAMPLE_DIR}/omp_figures.cpp)
add_executable(omp_dt ${GDIAMOND_EXAMPLE_DIR}/omp_dt.cpp)
add_executable(seq ${GDIAMOND_EXAMPLE_DIR}/seq.cpp)
add_executable(seq_figures ${GDIAMOND_EXAMPLE_DIR}/seq_figures.cpp)
add_executable(gpu_mm ${GDIAMOND_EXAMPLE_DIR}/gpu_mm.cu)
add_executable(gpu_mm_ver2 ${GDIAMOND_EXAMPLE_DIR}/gpu_mm_ver2.cu)
add_executable(gpu_mm_ver3 ${GDIAMOND_EXAMPLE_DIR}/gpu_mm_ver3.cu)
add_executable(gpu_vectorized ${GDIAMOND_EXAMPLE_DIR}/gpu_vectorized.cu)

list(APPEND GDIAMOND_EXAMPLES
  gpu
  gpu_figures
  gpu_dt
  gpu_pt
  gpu_fq 
  gpu_upper_bound
  gpu_dt_r
  omp
  omp_figures
  omp_dt
  seq
  seq_figures
  gpu_mm
  gpu_mm_ver2
  gpu_mm_ver3
  gpu_vectorized
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(omp PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(omp_dt PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(omp_figures PUBLIC OpenMP::OpenMP_CXX)
endif()

foreach(example IN LISTS GDIAMOND_EXAMPLES) 
  target_include_directories(${example}
    PUBLIC 
    ${PROJECT_SOURCE_DIR}/gDiamond)

endforeach()
