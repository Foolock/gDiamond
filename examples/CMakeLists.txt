set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/examples)
add_executable(gpu ${GDIAMOND_EXAMPLE_DIR}/gpu.cu)
add_executable(gpu_figures ${GDIAMOND_EXAMPLE_DIR}/gpu_figures.cu)
add_executable(gpu_dt ${GDIAMOND_EXAMPLE_DIR}/gpu_dt.cu)
add_executable(omp ${GDIAMOND_EXAMPLE_DIR}/omp.cpp)
add_executable(omp_figures ${GDIAMOND_EXAMPLE_DIR}/omp_figures.cpp)
add_executable(omp_dt ${GDIAMOND_EXAMPLE_DIR}/omp_dt.cpp)
add_executable(seq ${GDIAMOND_EXAMPLE_DIR}/seq.cpp)
add_executable(seq_figures ${GDIAMOND_EXAMPLE_DIR}/seq_figures.cpp)

list(APPEND GDIAMOND_EXAMPLES
  gpu
  gpu_figures
  gpu_dt
  omp
  omp_figures
  omp_dt
  seq
  seq_figures
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
