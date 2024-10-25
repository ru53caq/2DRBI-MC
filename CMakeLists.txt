cmake_minimum_required(VERSION 3.1)

option(gcc "on" OFF)
if(gcc)
SET(CMAKE_C_COMPILER /usr/local/bin/gcc-9)
SET(CMAKE_CXX_COMPILER /usr/local/bin/g++-9)
endif()


add_compile_options(-fdiagnostics-color=always)

set(CMAKE_CXX_STANDARD 14)

project(2DRBI CXX)

if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

# find_package Eigen3 is inclued tksvm CMakeLists
find_package(ALPSCore 2.2.0 REQUIRED)
find_package(Threads REQUIRED)

# First executable:
add_executable(2DRBI_MC main_MC.cpp 2DRBI_MC.cpp)
target_link_libraries(2DRBI_MC Threads::Threads ${ALPSCore_LIBRARIES})

# Second executable:
add_executable(2DRBI_PT main_PT.cpp 2DRBI_PT.cpp)
target_link_libraries(2DRBI_PT Threads::Threads ${ALPSCore_LIBRARIES})

