cmake_minimum_required(VERSION 3.10)
project(ZebraFlash)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g")

include(FetchContent)

FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG master
)
FetchContent_MakeAvailable(yaml-cpp)

if (MSVC)
    # Enable debug info in Release
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /DEBUG")
endif()

if(DEFINED ENV{OPENCV_DIR})
    set(OpenCV_DIR $ENV{OPENCV_DIR})
endif()

find_package(OpenCV REQUIRED)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(ZebraFlash main.cpp
        motion-detector/motion_detector.h
        motion-detector/motion_detector.cpp
        benchmark/benchmark.h
        benchmark/benchmark.cpp
        thread-pool/thread_pool.h
        thread-pool/thread_pool.cpp)

target_link_libraries(ZebraFlash PRIVATE yaml-cpp ${OpenCV_LIBS})

target_include_directories(ZebraFlash PRIVATE ${OpenCV_INCLUDE_DIRS})