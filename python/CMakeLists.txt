cmake_minimum_required(VERSION 3.10)
project(pybind_cabess LANGUAGES CXX)

# environment variables
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

if(MSVC)
    SET(CMAKE_CXX_FLAGS "/EHsc /arch:AVX2")
    SET(CMAKE_CXX_FLAGS_DEBUG "/W4")
elseif(DARWIN)
    SET(CMAKE_CXX_FLAGS "-Wno-int-in-bool-context -std=c++11")
    SET(CMAKE_CXX_FLAGS_DEBUG "-Wall")
else()
    SET(CMAKE_CXX_FLAGS "-march=native -Wno-int-in-bool-context -std=c++11 -mavx -mfma")
    SET(CMAKE_CXX_FLAGS_DEBUG "-Wall")
endif()

# dependencies
find_package(pybind11)
find_package(OpenMP)

# sources
SET(SOURCES
    src/pywrap.cpp
    src/utilities.cpp
    src/List.cpp
    src/normalize.cpp
    src/api.cpp
)
add_library(pybind_cabess MODULE ${SOURCES})

# links
target_include_directories(pybind_cabess PRIVATE include
                                                 src)
target_link_libraries(pybind_cabess PRIVATE pybind11::module 
                                            pybind11::opt_size 
                                            pybind11::lto
                                            pybind11::windows_extras)
if(OpenMP_CXX_FOUND)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    target_link_libraries(pybind_cabess PRIVATE OpenMP::OpenMP_CXX)
endif()

# build extension
pybind11_extension(pybind_cabess)
pybind11_strip(pybind_cabess)

set_target_properties(pybind_cabess PROPERTIES CXX_VISIBILITY_PRESET "hidden"
                                               CUDA_VISIBILITY_PRESET "hidden")
