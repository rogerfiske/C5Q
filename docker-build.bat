@echo off
REM C5Q Docker Build Script for Windows
REM Builds both CPU and GPU variants of the C5Q container

echo Building C5Q Docker Images...
echo.

REM Build CPU variant (default)
echo [1/3] Building CPU variant...
docker build -t c5q:latest -t c5q:cpu --build-arg BASE_VARIANT=cpu .
if %ERRORLEVEL% neq 0 (
    echo ERROR: CPU build failed
    exit /b 1
)
echo ✓ CPU build complete

REM Build CUDA 11.8 variant for RunPod compatibility
echo [2/3] Building CUDA 11.8 variant...
docker build -t c5q:cuda -t c5q:cuda11.8 --build-arg BASE_VARIANT=cuda .
if %ERRORLEVEL% neq 0 (
    echo ERROR: CUDA 11.8 build failed
    exit /b 1
)
echo ✓ CUDA 11.8 build complete

REM Build CUDA 12.1 variant (latest)
echo [3/3] Building CUDA 12.1 variant...
docker build -t c5q:cuda12 --build-arg BASE_VARIANT=cuda12 .
if %ERRORLEVEL% neq 0 (
    echo ERROR: CUDA 12.1 build failed
    exit /b 1
)
echo ✓ CUDA 12.1 build complete

echo.
echo All builds completed successfully!
echo.
echo Available images:
docker images | findstr c5q
echo.
echo Usage examples:
echo   CPU:        docker run --rm -v "%cd%\data:/data" -v "%cd%\artifacts:/artifacts" c5q:latest
echo   CUDA 11.8:  docker run --gpus all --rm -v "%cd%\data:/data" -v "%cd%\artifacts:/artifacts" c5q:cuda
echo   CUDA 12.1:  docker run --gpus all --rm -v "%cd%\data:/data" -v "%cd%\artifacts:/artifacts" c5q:cuda12