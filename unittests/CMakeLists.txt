enable_testing()

include(${GDIAMOND_3RD_PARTY_DIR}/doctest/doctest.cmake)

list(APPEND ITAP_UNITTESTS 
  checker
)

string(FIND '${CMAKE_CXX_FLAGS}' "-fsanitize" sanitize)
#message("sanitize='${sanitize}'")

# if(${sanitize} EQUAL -1)
#   message(STATUS "Add test_exception to unit tests")
#   list(APPEND ITAP_UNITTESTS test_exceptions)
# endif()


find_package(Threads REQUIRED)
foreach(unittest IN LISTS ITAP_UNITTESTS)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/unittests)
  add_executable(${unittest} ${unittest}.cu)
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



