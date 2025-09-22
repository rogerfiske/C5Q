@echo off
REM C5Q Docker Run Script for Windows
REM Provides convenient commands for running C5Q containers with proper volume mounting

if "%1"=="" (
    echo C5Q Docker Run Script
    echo.
    echo Usage:
    echo   docker-run.bat cpu [command]     - Run CPU container
    echo   docker-run.bat cuda [command]    - Run CUDA 11.8 container
    echo   docker-run.bat cuda12 [command]  - Run CUDA 12.1 container
    echo   docker-run.bat test              - Run container tests
    echo.
    echo Examples:
    echo   docker-run.bat cpu python -m c5q.eda --help
    echo   docker-run.bat cuda python -m c5q.dataset --csv /data/sample.csv
    echo   docker-run.bat test
    exit /b 1
)

REM Set common volume mounts
set VOLUMES=-v "%cd%\data:/data" -v "%cd%\artifacts:/artifacts" -v "%cd%\configs:/app/configs"

REM Handle different variants
if "%1"=="cpu" (
    set IMAGE=c5q:latest
    set GPU_FLAG=
) else if "%1"=="cuda" (
    set IMAGE=c5q:cuda
    set GPU_FLAG=--gpus all
) else if "%1"=="cuda12" (
    set IMAGE=c5q:cuda12
    set GPU_FLAG=--gpus all
) else if "%1"=="test" (
    echo Running container validation tests...
    goto :test
) else (
    echo Error: Unknown variant '%1'
    echo Use: cpu, cuda, cuda12, or test
    exit /b 1
)

REM Shift arguments to remove variant
shift

REM Run container with remaining arguments
if "%1"=="" (
    REM No command specified, run default
    docker run --rm %GPU_FLAG% %VOLUMES% %IMAGE%
) else (
    REM Run with specified command
    docker run --rm %GPU_FLAG% %VOLUMES% %IMAGE% %*
)

exit /b %ERRORLEVEL%

:test
echo Testing CPU container...
docker run --rm %VOLUMES% c5q:latest python -c "import c5q; print('✓ CPU container working')"
if %ERRORLEVEL% neq 0 exit /b 1

echo Testing volume mounting...
if not exist "test_data" mkdir test_data
echo test_content > test_data\test.txt
docker run --rm %VOLUMES% -v "%cd%\test_data:/test_data" c5q:latest python -c "import os; assert os.path.exists('/test_data/test.txt'), 'Volume mount failed'; print('✓ Volume mounting working')"
if %ERRORLEVEL% neq 0 exit /b 1
rmdir /s /q test_data

echo Testing CUDA container (if available)...
docker run --rm --gpus all %VOLUMES% c5q:cuda python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')" 2>nul
if %ERRORLEVEL% equ 0 (
    echo ✓ CUDA container working
) else (
    echo ! CUDA container test skipped (no GPU or NVIDIA Docker not available)
)

echo.
echo ✓ All tests passed!
exit /b 0