cmake_minimum_required(VERSION 3.10)
project(cmake_example)

# environment variables
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

if(MSVC)
    SET(CMAKE_CXX_FLAGS_RELEASE "/openmp /bigobj /w /O2 /arch:AVX2") 
    SET(CMAKE_CXX_FLAGS_DEBUG "/openmp /bigobj /W4 /arch:AVX2")
    SET(CMAKE_CXX_FLAGS "/EHsc")
else()
    SET(CMAKE_CXX_FLAGS_RELEASE "-fopenmp -w -O2 -march=native") 
    SET(CMAKE_CXX_FLAGS_DEBUG "-fopenmp -Wall -march=native")
endif()

# dependencies
include_directories("include")
find_package(Python COMPONENTS Interpreter Development)
find_package(OpenMP)
find_package(pybind11)

# sources
set(SOURCES src/pywrap.cpp src/utilities.cpp src/List.cpp src/normalize.cpp src/api.cpp)
Python_add_library(pybind_cabess ${SOURCES})

# links
target_link_libraries(pybind_cabess PRIVATE pybind11::headers)
if(OpenMP_CXX_FOUND)
    target_link_libraries(pybind_cabess PUBLIC OpenMP::OpenMP_CXX)
endif()
