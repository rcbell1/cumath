# Set minimum CMake version
cmake_minimum_required(VERSION 3.17)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type specified. Defaulting to Debug.")
  set(CMAKE_BUILD_TYPE
      Debug
      CACHE STRING "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)"
            FORCE)
endif()

# set CUDA toolkit root directory
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.6/bin/nvcc)
set(CUDAToolkit_ROOT /usr/local/cuda-12.6)

# Set the project name
project(CUMSUM LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 70 80 86)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED YES)

# Enable CUDA separable compilation
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Include project directories
include_directories(${PROJECT_SOURCE_DIR}/src)

# Link CUDA runtime library find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Add the test executable
add_executable(test_cumsum tests/TestCumsum.cu)
target_link_libraries(test_cumsum PRIVATE CUDA::cudart)

# Ensure CUDA is enabled for the test executable
set_target_properties(test_cumsum PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Define compiler flags for Debug and Release
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Werror -Wextra")
# set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
# set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} --use_fast_math")

# Add debug and release flags per target
target_compile_options(
  test_cumsum
  PRIVATE $<$<CONFIG:Debug>: -g -G> # Debug flags for CUDA
          $<$<CONFIG:Release>: --use_fast_math> # Release optimization flags
)

# Access the default flags for the current build type
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(SELECTED_CXX_FLAGS "${CMAKE_CXX_FLAGS_DEBUG}")
  set(SELECTED_CUDA_FLAGS "${CMAKE_CUDA_FLAGS_DEBUG}")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -G")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(SELECTED_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE}")
  set(SELECTED_CUDA_FLAGS "${CMAKE_CUDA_FLAGS_RELEASE}")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  set(SELECTED_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
  set(SELECTED_CUDA_FLAGS "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO}")
elseif(CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
  set(SELECTED_CXX_FLAGS "${CMAKE_CXX_FLAGS_MINSIZEREL}")
  set(SELECTED_CUDA_FLAGS "${CMAKE_CUDA_FLAGS_MINSIZEREL}")
else()
  set(SELECTED_CXX_FLAGS "Unknown Build Type")
  set(SELECTED_CUDA_FLAGS "Unknown Build Type")
endif()

# Add a post-build step to display a build summary after `test_cumsum` is built
add_custom_command(
  TARGET test_cumsum
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E echo
          "--------------------------------------------"
  COMMAND ${CMAKE_COMMAND} -E echo "Build Summary:"
  COMMAND ${CMAKE_COMMAND} -E echo "    Build type: ${CMAKE_BUILD_TYPE}"
  COMMAND ${CMAKE_COMMAND} -E echo "    Output directory: ${CMAKE_BINARY_DIR}"
  COMMAND ${CMAKE_COMMAND} -E echo
          "    CXX Compiler flags: ${SELECTED_CXX_FLAGS}"
  COMMAND ${CMAKE_COMMAND} -E echo
          "    CUDA Compiler flags: ${SELECTED_CUDA_FLAGS}"
  COMMAND ${CMAKE_COMMAND} -E echo
          "--------------------------------------------"
  COMMENT "Displaying build summary at the end")

# Define custom targets to simplify use of `make` for debug and release builds
add_custom_target(
  debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --config Debug
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Building Debug version")
add_custom_target(
  release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --config Release
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Building Release version")
