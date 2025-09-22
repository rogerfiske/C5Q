# C5Q Docker Setup Guide

This guide covers Docker containerization for the C5Q Quantum Logic Matrix project, supporting both local Windows 11 development and RunPod H200 GPU deployment.

## Prerequisites

### Windows 11 Development
- Docker Desktop for Windows
- WSL2 enabled
- Git for Windows

### RunPod Deployment
- RunPod account with H200 GPU access
- Docker runtime (pre-installed on RunPod)

## Quick Start

### 1. Build All Variants
```bash
# Windows (using batch script)
docker-build.bat

# Linux/WSL (using Makefile)
make build-all
```

### 2. Test Containers
```bash
# Windows
docker-run.bat test

# Linux/WSL
make test
```

### 3. Run Containers
```bash
# CPU variant
docker-run.bat cpu python -m c5q.eda --help

# CUDA variant (requires NVIDIA Docker)
docker-run.bat cuda python -m c5q.dataset --csv /data/sample.csv
```

## Available Docker Images

| Image Tag | Base | Use Case | Size Est. |
|-----------|------|----------|-----------|
| `c5q:latest` | PyTorch CPU | Local development | ~1.5GB |
| `c5q:cpu` | PyTorch CPU | Local development | ~1.5GB |
| `c5q:cuda` | PyTorch CUDA 11.8 | RunPod H200 deployment | ~4GB |
| `c5q:cuda12` | PyTorch CUDA 12.1 | Latest GPU environments | ~4GB |

## Build Commands

### Manual Builds
```bash
# CPU variant
docker build -t c5q:cpu --build-arg BASE_VARIANT=cpu .

# CUDA 11.8 variant (RunPod compatible)
docker build -t c5q:cuda --build-arg BASE_VARIANT=cuda .

# CUDA 12.1 variant (latest)
docker build -t c5q:cuda12 --build-arg BASE_VARIANT=cuda12 .
```

### Using Build Scripts
```bash
# Windows
docker-build.bat

# Linux/macOS
make build-all
```

## Volume Mounting Strategy

### Data Directory Structure
```
C5Q/
├── data/           # Input datasets (gitignored)
│   ├── raw/
│   └── processed/
├── artifacts/      # Model outputs (gitignored)
│   ├── models/
│   ├── logs/
│   └── reports/
└── configs/        # Configuration files
    ├── hparams.yaml
    └── buckets.manual.yaml
```

### Volume Mount Examples
```bash
# Windows paths
-v "%cd%\data:/data"
-v "%cd%\artifacts:/artifacts"
-v "%cd%\configs:/app/configs"

# Linux/WSL paths
-v "$(pwd)/data:/data"
-v "$(pwd)/artifacts:/artifacts"
-v "$(pwd)/configs:/app/configs"
```

## Running Containers

### Development Mode (Interactive)
```bash
# CPU development
docker run -it --rm \
  -v "%cd%:/workspace" \
  -v "%cd%\data:/data" \
  -v "%cd%\artifacts:/artifacts" \
  -w /workspace \
  c5q:latest bash

# CUDA development
docker run -it --rm --gpus all \
  -v "%cd%:/workspace" \
  -v "%cd%\data:/data" \
  -v "%cd%\artifacts:/artifacts" \
  -w /workspace \
  c5q:cuda bash
```

### Production Mode (Single Command)
```bash
# EDA analysis
docker run --rm \
  -v "%cd%\data:/data" \
  -v "%cd%\artifacts:/artifacts" \
  c5q:latest python -m c5q.eda --csv /data/c5_Matrix_binary.csv

# Model training (GPU)
docker run --rm --gpus all \
  -v "%cd%\data:/data" \
  -v "%cd%\artifacts:/artifacts" \
  c5q:cuda python -m c5q.train --config /app/configs/hparams.yaml
```

## RunPod H200 Deployment

### Setup on RunPod
```bash
# Clone repository
git clone https://github.com/rogerfiske/C5Q.git
cd C5Q

# Build CUDA image
docker build -t c5q:cuda --build-arg BASE_VARIANT=cuda .

# Upload your data to /workspace/data/
# Run training
docker run --rm --gpus all \
  -v /workspace/data:/data \
  -v /workspace/artifacts:/artifacts \
  c5q:cuda python -m c5q.train --config /app/configs/hparams.yaml
```

### RunPod Environment Variables
```bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## Container Validation Tests

### Health Checks
```bash
# Test basic functionality
docker run --rm c5q:latest python -c "import c5q; print('✓ Package working')"

# Test GPU access (requires NVIDIA Docker)
docker run --rm --gpus all c5q:cuda python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
"

# Test volume mounting
mkdir test_data
echo "test" > test_data/test.txt
docker run --rm -v "%cd%\test_data:/test_data" c5q:latest ls -la /test_data/
```

### Performance Benchmarks
```bash
# CPU benchmark
docker run --rm c5q:latest python -c "
import torch
import time
x = torch.randn(1000, 1000)
start = time.time()
y = torch.mm(x, x)
print(f'CPU MatMul time: {time.time() - start:.3f}s')
"

# GPU benchmark (if available)
docker run --rm --gpus all c5q:cuda python -c "
import torch
import time
if torch.cuda.is_available():
    x = torch.randn(1000, 1000).cuda()
    torch.cuda.synchronize()
    start = time.time()
    y = torch.mm(x, x)
    torch.cuda.synchronize()
    print(f'GPU MatMul time: {time.time() - start:.3f}s')
"
```

## Troubleshooting

### Common Issues

**Docker Desktop not running:**
```bash
# Start Docker Desktop on Windows
# Check status: docker version
```

**Volume mounting issues on Windows:**
```bash
# Ensure drive sharing is enabled in Docker Desktop
# Use absolute paths: "%cd%\data:/data"
```

**CUDA not detected:**
```bash
# Install NVIDIA Docker runtime
# Verify: docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi
```

**Permission denied errors:**
```bash
# Fix with proper user mapping
docker run --rm -u $(id -u):$(id -g) ...
```

### Image Size Optimization
- Multi-stage builds reduce final image size
- .dockerignore excludes unnecessary files
- Shared layer caching improves build times

### Build Cache Management
```bash
# Clear build cache
docker builder prune

# Clear all unused images
docker image prune -a
```

## Development Workflow

### Daily Development
1. Start Docker Desktop
2. Pull latest code: `git pull`
3. Rebuild if needed: `make build-cpu`
4. Run development container: `make dev-cpu`
5. Test changes: `make test`

### Before Deployment
1. Build all variants: `make build-all`
2. Run full test suite: `make test`
3. Commit changes with Docker files
4. Push to repository
5. Deploy on RunPod using CUDA variant

## Integration with C5Q Modules

### Available Python Modules
- `c5q.io` - Data input/output operations
- `c5q.eda` - Exploratory data analysis
- `c5q.dataset` - Dataset processing
- `c5q.utils` - Utility functions

### Example Workflows
```bash
# Data exploration
docker run --rm -v "%cd%\data:/data" c5q:latest \
  python -m c5q.eda --csv /data/input.csv --output /artifacts/eda_report.html

# Dataset preparation
docker run --rm -v "%cd%\data:/data" c5q:latest \
  python -m c5q.dataset --input /data/raw/ --output /data/processed/

# Model training (GPU)
docker run --rm --gpus all -v "%cd%\data:/data" c5q:cuda \
  python -m c5q.train --data /data/processed/ --output /artifacts/model.pkl
```