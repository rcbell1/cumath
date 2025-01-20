# GoogleTest FetchContent
include(FetchContent)
FetchContent_Declare(
  gtest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.tar.gz
)
# Download and add GTest/GoogleMock
FetchContent_MakeAvailable(gtest)

# Add test executable
add_executable(UnitTests test_cumath.cpp)

# Likely link your library with the tests
target_link_libraries(UnitTests PRIVATE cumath gtest_main)

# Include directories for testing
target_include_directories(UnitTests PRIVATE ${cumath_SOURCE_DIR}/include)

# Register tests
add_test(NAME cumathTests COMMAND UnitTests)
