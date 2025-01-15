set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/examples)
add_executable(gpu ${GDIAMOND_EXAMPLE_DIR}/gpu.cu)
add_executable(omp ${GDIAMOND_EXAMPLE_DIR}/omp.cpp)
add_executable(omp_dt ${GDIAMOND_EXAMPLE_DIR}/omp_dt.cpp)
add_executable(seq ${GDIAMOND_EXAMPLE_DIR}/seq.cpp)

list(APPEND GDIAMOND_EXAMPLES
  gpu
  omp
  omp_dt
  seq
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(omp PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(omp_dt PUBLIC OpenMP::OpenMP_CXX)
endif()

foreach(example IN LISTS GDIAMOND_EXAMPLES) 
  target_include_directories(${example}
    PUBLIC 
    ${PROJECT_SOURCE_DIR}/gDiamond)

endforeach()
