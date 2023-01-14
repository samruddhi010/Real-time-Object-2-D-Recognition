cmake_minimum_required(VERSION 2.8)
project(Project_3)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(${OpenCV_INCLUDE_DIRS})
# add_executable(Project_3 training.cpp features.cpp csv_util.h)
add_executable(Project_3 main_p3.cpp features.cpp csv_util.cpp)

target_link_libraries(Project_3 ${OpenCV_LIBS})