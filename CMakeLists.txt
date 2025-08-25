# Minimum version of CMake required
cmake_minimum_required(VERSION 3.10)

# Set the name of your project
project(WaveEquationSolver VERSION 1.0)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Include directory for header files
include_directories(${PROJECT_SOURCE_DIR}/include)

# Find deal.II
find_package(deal.II CONFIG REQUIRED)
# Add source files
file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp)

# Solver 
add_executable(wave_solver ${SOURCES})
target_include_directories(wave_solver PRIVATE ${PROJECT_SOURCE_DIR}/include)
deal_ii_setup_target(wave_solver) 

# Tests
enable_testing()
add_executable(run_tests tests/test.cpp)
target_include_directories(run_tests PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_compile_definitions(run_tests PRIVATE BIN=\"${CMAKE_BINARY_DIR}/wave_solver\")
deal_ii_setup_target(run_tests)   
add_test(NAME run_tests COMMAND run_tests)