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


add_executable(${PROJECT_NAME} main.cpp 2DRBI.cpp)
#add_executable(${PROJECT_NAME} main_PTfeedback.cpp Tetra_PTfeedback.cpp)
target_link_libraries(2DRBI Threads::Threads)
# Request the ALPSCore package (with all components)
# The ALPSCore package must be installed in some
# standard place (like /usr/local),
# or somewhere in your PATH.
# Otherwise, point ALPSCore_DIR environment variable
# to the ALPScore installation tree.
find_package(ALPSCore 2.0 REQUIRED)

# Use ALPSCore_LIBRARIES variable to link to ALPSCore
target_link_libraries(${PROJECT_NAME} ${ALPSCore_LIBRARIES})
