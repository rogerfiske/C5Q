# Multi-target Dockerfile for C5Q Quantum Logic Matrix project
# Supports both CPU and GPU environments for local development and RunPod deployment

# Build argument to select base image variant
ARG BASE_VARIANT=cpu
ARG PYTORCH_VERSION=2.1.0
ARG PYTHON_VERSION=3.11

# Base image selection
FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda11.8-cudnn8-devel AS base-cuda
FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda12.1-cudnn8-devel AS base-cuda12
FROM pytorch/pytorch:${PYTORCH_VERSION}-cpu-py${PYTHON_VERSION} AS base-cpu

# Select final base image based on variant
FROM base-${BASE_VARIANT} AS base

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create directories for volume mounts
RUN mkdir -p /data /artifacts /app/logs

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project configuration files
COPY configs/ ./configs/
COPY setup.py .

# Copy source code
COPY c5q/ ./c5q/

# Install c5q package in editable mode
RUN pip install -e .

# Copy additional project files
COPY README.md .

# Set proper permissions for mounted volumes
RUN chmod 755 /data /artifacts /app/logs

# Expose port for potential web services (future use)
EXPOSE 8000

# Health check to ensure container is working
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import c5q; print('Container healthy')" || exit 1

# Default command - shows help information
CMD ["python", "-c", "import c5q; print('C5Q Quantum Logic Matrix - Container Ready'); print('Available modules:'); import pkgutil; [print(f'  - c5q.{name}') for _, name, _ in pkgutil.iter_modules(c5q.__path__)]"]