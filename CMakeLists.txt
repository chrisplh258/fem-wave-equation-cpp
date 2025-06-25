# Minimum version of CMake required
cmake_minimum_required(VERSION 3.10)

# Set the name of your project
project(HeatEquationSolver VERSION 1.0)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


# Include directory for header files
include_directories(${PROJECT_SOURCE_DIR}/include)

# Add source files
file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp)

# Define the executable
add_executable(heat_solver ${SOURCES})