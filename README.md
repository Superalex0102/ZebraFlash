# Installation

## OpenCV with CUDA Support (Windows + Visual Studio 2022)

This guide explains how to build OpenCV from source with CUDA support using Visual Studio 2022.

## Prerequisites

Make sure you have the following software installed:

- **CUDA Toolkit**: 12.4  
- **Visual Studio**: 2022 (version 17.8 or newer)
- **CMake**: 3.27 or higher

## Clone Repositories

```bash
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
```

> ⚠️ Make sure both repositories are on matching versions (e.g., `4.9.0` or `4.8.0`). You can check out a specific version with:
>
> ```bash
> cd opencv
> git checkout 4.9.0
> cd ../opencv_contrib
> git checkout 4.9.0
> ```

## Configure the Build

Open a command prompt and run:

```bash
cmake -G "Visual Studio 17 2022" -A x64 ^
  -DWITH_CUDA=ON ^
  -DBUILD_opencv_python3=OFF ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DOPENCV_EXTRA_MODULES_PATH=C:\Users\Alex\opencv_cuda\opencv_contrib\modules ^
  -DBUILD_opencv_core=ON ^
  -DBUILD_opencv_imgproc=ON ^
  -DBUILD_opencv_video=ON ^
  -DBUILD_opencv_videoio=ON ^
  -DBUILD_opencv_highgui=ON ^
  -DBUILD_opencv_cudaoptflow=ON ^
  -DBUILD_opencv_cudaarithm=ON ^
  -DWITH_OPENCL=ON ^
  -DBUILD_opencv_ocl=ON ^
  ..\opencv
```

## Build OpenCV

```bash
cmake --build . --config Release -- /m
```
The build should take around 90-120 minutes to complete and it should take up around 20-25GB storage.
After the build completes, the generated binaries will be located in the `bin\Release` directory.

Don't forget to setup OpenCV environments to the `bin\Release` folder.
