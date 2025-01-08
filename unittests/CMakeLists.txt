enable_testing()

include(${GDIAMOND_3RD_PARTY_DIR}/doctest/doctest.cmake)

add_executable(checker_gpu checker_gpu.cu)
add_executable(checker_omp checker_omp.cpp)
list(APPEND GDIAMOND_UNITTESTS 
  checker_gpu
  checker_omp
)

string(FIND '${CMAKE_CXX_FLAGS}' "-fsanitize" sanitize)
#message("sanitize='${sanitize}'")

# if(${sanitize} EQUAL -1)
#   message(STATUS "Add test_exception to unit tests")
#   list(APPEND ITAP_UNITTESTS test_exceptions)
# endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(checker_omp PUBLIC OpenMP::OpenMP_CXX)
endif()

find_package(Threads REQUIRED)
foreach(unittest IN LISTS GDIAMOND_UNITTESTS)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/unittests)
  target_link_libraries(${unittest} 
    PRIVATE Threads::Threads
    )
  target_include_directories(${unittest} 
    PRIVATE 
    ${GDIAMOND_3RD_PARTY_DIR}/doctest
    PUBLIC
    ${PROJECT_SOURCE_DIR}/gDiamond
  )
  doctest_discover_tests(${unittest})
endforeach()



