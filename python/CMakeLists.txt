cmake_minimum_required(VERSION 3.10)
project(cmake_example)

# environment variables
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

if(MSVC)
    SET(CMAKE_CXX_FLAGS "/EHsc /openmp /arch:AVX2")
    SET(CMAKE_CXX_FLAGS_DEBUG "/W4")
else()
    SET(CMAKE_CXX_FLAGS "-fopenmp -march=native")
    SET(CMAKE_CXX_FLAGS_DEBUG "-Wall")
endif()

# dependencies
include_directories("include")
include_directories("src")
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP)

# sources
set(SOURCES src/pywrap.cpp src/utilities.cpp src/List.cpp src/normalize.cpp src/api.cpp)
Python_add_library(pybind_cabess ${SOURCES})

# links
target_link_libraries(pybind_cabess PRIVATE pybind11::module 
                                            pybind11::opt_size 
                                            pybind11::windows_extras)
if(OpenMP_CXX_FOUND)
    target_link_libraries(pybind_cabess PRIVATE OpenMP::OpenMP_CXX)
endif()
