cmake_minimum_required(VERSION 3.30)
project(ZebraFlash)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(FetchContent)

FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG master
)
FetchContent_MakeAvailable(yaml-cpp)

find_package(OpenCV REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(ZebraFlash main.cpp)

target_link_libraries(ZebraFlash PRIVATE yaml-cpp ${OpenCV_LIBS})

target_include_directories(ZebraFlash PRIVATE ${OpenCV_INCLUDE_DIRS})