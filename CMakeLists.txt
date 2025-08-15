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
find_package(deal.II 9.7.0 REQUIRED HINTS ~/Documents/software/deal.II-v9.7.0)

# Add source files
file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp)

# Define the executable
add_executable(wave_solver ${SOURCES})

# Link deal.II 
deal_ii_setup_target(wave_solver)