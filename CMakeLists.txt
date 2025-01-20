# Minimum CMake version
cmake_minimum_required(VERSION 3.22)

# Project setup
project(
  cumath
  LANGUAGES CXX CUDA
  VERSION 1.0)

# Options
option(BUILD_TESTING "Enable testing" ON) # Add option to toggle testing

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create the library
add_library(cumath src/cumath.cu)

# Include directories for the library
target_include_directories(
  cumath PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                $<INSTALL_INTERFACE:include>)

# Conditional testing support
if(BUILD_TESTING)
  # Enable testing
  enable_testing()
  add_subdirectory(tests)
endif()

# Examples (optional)
add_subdirectory(examples)

# Installation settings
install(
  TARGETS cumath
  EXPORT cumathTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin)

install(DIRECTORY include/ DESTINATION include)
