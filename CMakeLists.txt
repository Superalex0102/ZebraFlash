cmake_minimum_required(VERSION 3.10)
project(ZebraFlash)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g")

if (MSVC)
    # Enable debug info in Release
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /DEBUG")
endif()

# --- OpenCV ---
if(DEFINED ENV{OPENCV_DIR})
    set(OpenCV_DIR $ENV{OPENCV_DIR})
endif()

find_package(OpenCV REQUIRED)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

# --- FetchContent for yaml-cpp and GoogleTest ---
include(FetchContent)

FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG master
)

FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(yaml-cpp googletest)

# --- Main Application ---
add_executable(ZebraFlash
        main.cpp
        motion-detector/motion_detector.cpp
        benchmark/benchmark.cpp
        thread-pool/thread_pool.cpp
        utils/motion_utils.cpp
)

target_include_directories(ZebraFlash PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/motion-detector
        ${CMAKE_SOURCE_DIR}/benchmark
        ${CMAKE_SOURCE_DIR}/thread-pool
        ${CMAKE_SOURCE_DIR}/utils
)

target_link_libraries(ZebraFlash PRIVATE yaml-cpp ${OpenCV_LIBS})

# --- Google Test Setup ---
enable_testing()

add_executable(ZebraFlashTests
        tests/test_application.cpp
        motion-detector/motion_detector.cpp
        utils/motion_utils.cpp
        thread-pool/thread_pool.cpp
        benchmark/benchmark.cpp
)

target_include_directories(ZebraFlashTests PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/motion-detector
        ${CMAKE_SOURCE_DIR}/benchmark
        ${CMAKE_SOURCE_DIR}/thread-pool
        ${CMAKE_SOURCE_DIR}/utils
)

target_link_libraries(ZebraFlashTests
        gtest_main
        yaml-cpp
        ${OpenCV_LIBS}
)

include(GoogleTest)
gtest_discover_tests(ZebraFlashTests)