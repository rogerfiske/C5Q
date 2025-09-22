# Story 1.3: Docker Containerization Setup

## Story Details
**Story ID:** 1.3
**Epic:** Epic 1 - Foundation & Infrastructure
**Story Points:** 8
**Priority:** High
**Dependencies:** Story 1.2 (Development Environment)
**Status:** Ready for Development

## User Story
> As a **developer**, I want **Docker containers for both CPU and GPU environments** so that I can **ensure reproducible execution locally and on RunPod**.

## Business Value
Enables seamless deployment from local Windows 11 development to RunPod H200 GPU cloud environment. Ensures reproducible builds and eliminates environment inconsistencies between development and production.

## Acceptance Criteria
- [ ] CPU Docker image builds successfully and runs basic commands
- [ ] GPU Docker image builds with CUDA support for RunPod deployment
- [ ] Volume mounting works for `data/` and `artifacts/` directories
- [ ] Container can execute Python package commands (`python -m c5q.--help`)
- [ ] Multi-stage builds optimize image size and layer caching

## Technical Implementation Tasks

### Task 1.3.1: Multi-Target Dockerfile Creation
**Estimated Time:** 4 hours
- [ ] Create `Dockerfile` with ARG-based CPU/GPU selection
- [ ] Implement multi-stage builds for optimization
- [ ] Configure proper base images for each target
- [ ] Add efficient layer ordering for Docker caching

**Dockerfile Implementation:**
```dockerfile
# Multi-target Dockerfile for CPU and GPU
ARG BASE=cpu
FROM pytorch/pytorch:2.3.1-${BASE}-py3.11

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install c5q package in editable mode
RUN pip install -e .

# Create directories for volume mounts
RUN mkdir -p /data /artifacts

# Default command
CMD ["python", "-m", "c5q.eda", "--help"]
```

### Task 1.3.2: Volume Mounting Strategy
**Estimated Time:** 2 hours
- [ ] Design volume mounting for persistent data
- [ ] Configure proper permissions for data access
- [ ] Test volume mounting on Windows Docker Desktop
- [ ] Document volume usage patterns

**Volume Mount Examples:**
```bash
# Local development (Windows)
docker run --rm \
  -v %cd%/data:/data \
  -v %cd%/artifacts:/artifacts \
  c5q:latest python -m c5q.eda --csv /data/c5_Matrix_binary.csv

# RunPod deployment (Linux)
docker run --gpus all --rm \
  -v $PWD/data:/data \
  -v $PWD/artifacts:/artifacts \
  c5q:cuda python -m c5q.train_npl --config /app/configs/hparams.yaml
```

### Task 1.3.3: Build and Tag Strategy
**Estimated Time:** 2 hours
- [ ] Implement build scripts for both variants
- [ ] Create proper tagging strategy
- [ ] Add build automation commands
- [ ] Test builds on Windows Docker Desktop

**Build Commands:**
```bash
# CPU build
docker build -t c5q:latest .
docker build -t c5q:cpu .

# GPU build
docker build --build-arg BASE=cuda -t c5q:cuda .

# Test builds
docker run --rm c5q:latest python -c "import c5q; print('CPU build success')"
docker run --rm c5q:cuda python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Task 1.3.4: Container Functionality Testing
**Estimated Time:** 3 hours
- [ ] Test basic Python package functionality in container
- [ ] Validate data access through volume mounts
- [ ] Test artifact generation and persistence
- [ ] Verify environment variable handling

**Container Test Suite:**
```bash
# Test basic functionality
docker run --rm c5q:latest python -c "import c5q; print('Package import successful')"

# Test data access
docker run --rm -v %cd%/data:/data c5q:latest ls -la /data/

# Test artifact generation
docker run --rm -v %cd%/artifacts:/artifacts c5q:latest \
  python -c "import os; open('/artifacts/test.txt', 'w').write('test')"

# Test Python module execution
docker run --rm c5q:latest python -m c5q.utils --version
```

### Task 1.3.5: RunPod GPU Compatibility
**Estimated Time:** 2 hours
- [ ] Verify CUDA base image compatibility with RunPod
- [ ] Test GPU detection and utilization
- [ ] Document RunPod-specific deployment requirements
- [ ] Create RunPod deployment guide

**RunPod Validation:**
```bash
# GPU availability check
docker run --gpus all --rm c5q:cuda python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA devices: {torch.cuda.device_count()}')
print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')
"
```

## Definition of Done
- [ ] Both CPU and GPU Docker images build successfully
- [ ] Volume mounting works correctly for data persistence
- [ ] Containers can execute all planned c5q module commands
- [ ] Images are optimized with proper layer caching
- [ ] GPU image successfully detects CUDA in test environment
- [ ] Documentation includes complete usage examples
- [ ] Build process is automated and repeatable

## Testing Criteria
- [ ] `docker build -t c5q:latest .` completes without errors
- [ ] `docker build --build-arg BASE=cuda -t c5q:cuda .` completes without errors
- [ ] Volume-mounted data is accessible within containers
- [ ] Generated artifacts persist after container shutdown
- [ ] GPU container successfully detects CUDA (when available)
- [ ] Container size is reasonable (<2GB for CPU, <5GB for GPU)

## Container Validation Script
```bash
#!/bin/bash
echo "Validating C5Q Docker Containers..."

# Build both images
echo "Building CPU image..."
docker build -t c5q:latest . || exit 1

echo "Building GPU image..."
docker build --build-arg BASE=cuda -t c5q:cuda . || exit 1

# Test CPU container
echo "Testing CPU container..."
docker run --rm c5q:latest python -c "import c5q; print('✓ CPU container working')" || exit 1

# Test volume mounting
echo "Testing volume mounting..."
mkdir -p test_data test_artifacts
echo "test_file" > test_data/test.txt
docker run --rm -v %cd%/test_data:/data -v %cd%/test_artifacts:/artifacts c5q:latest \
  python -c "
import os
assert os.path.exists('/data/test.txt'), 'Data volume not mounted'
with open('/artifacts/output.txt', 'w') as f: f.write('success')
print('✓ Volume mounting working')
" || exit 1

# Check artifact persistence
test -f test_artifacts/output.txt && echo "✓ Artifact persistence working" || exit 1

# Cleanup
rm -rf test_data test_artifacts

echo "Docker container validation complete!"
```

## Notes
- Use multi-stage builds to minimize final image size
- Consider using .dockerignore to exclude unnecessary files
- Test thoroughly on Windows Docker Desktop before RunPod deployment
- Document common Docker issues and troubleshooting
- Plan for potential Windows-specific path and permission issues