cmake_minimum_required(VERSION 3.5)
project(testLIPS)
add_executable(testLIPS src/main.cpp)

set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_FLAGS "-O0 -g -Dlinux -std=gnu++11 -lpthread")

include_directories(
    include
    /usr/local/include/OpenNI2
#    /home/inaho-00/tmp/LIPS-Linux-x64-OpenNI2.2/Include
)

target_link_libraries(
    testLIPS 
    -L/home/inaho-00/tmp/LIPS-Linux-x64-OpenNI2.2/Tools
    -lOpenNI2
)

find_package(PkgConfig REQUIRED)
pkg_check_modules(OpenCV opencv REQUIRED)
include_directories(${OpenCV_CFLAGS})
target_link_libraries(testLIPS ${OpenCV_LDFLAGS})
#target_link_libraries(testLIPS -lopencv_ximgproc)

find_package(PkgConfig REQUIRED)
pkg_check_modules(eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(VTK REQUIRED)

include_directories(${eigen3_INCLUDE_DIRS})
target_link_libraries(testLIPS ${eigen3_LIBRARIES})
include_directories(${VTK_INCLUDE_DIRS})
target_link_libraries(testLIPS ${VTK_LIBRARIES})
include_directories(${PCL_INCLUDE_DIRS})
target_link_libraries(testLIPS ${PCL_LIBRARIES})
