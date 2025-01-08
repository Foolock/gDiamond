set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/examples)
add_executable(simple ${GDIAMOND_EXAMPLE_DIR}/simple.cu)

list(APPEND GDIAMOND_EXAMPLES
  simple
)

foreach(example IN LISTS GDIAMOND_EXAMPLES) 
  target_include_directories(${example}
    PUBLIC 
    ${PROJECT_SOURCE_DIR}/gDiamond)

endforeach()
