cmake_minimum_required(VERSION 3.10)
project(cmake_example)

# environment variables
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# complier
if (MSVC)
    add_compile_options(/openmp /O2 /W4 /arch:AVX2)
else()
    add_compile_options(-fopenmp -O2 -Wall -march=native)
endif()

# dependencies
include_directories("include")
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11)
find_package(OpenMP)

# sources
set(SOURCES src/utilities.cpp src/List.cpp src/normalize.cpp src/api.cpp)
Python_add_library(pybind_cabess src/pywrap.cpp ${SOURCES})

# links
target_link_libraries(pybind_cabess PRIVATE pybind11::headers)
if(OpenMP_CXX_FOUND)
    target_link_libraries(pybind_cabess PUBLIC OpenMP::OpenMP_CXX)
endif()
