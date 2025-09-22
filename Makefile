# C5Q Quantum Logic Matrix - Docker Management Makefile
# Supports both Windows and Linux environments

.PHONY: help build build-cpu build-cuda build-cuda12 build-all test test-cpu test-cuda clean run-cpu run-cuda run-cuda12

# Default target
help:
	@echo "C5Q Docker Management Commands"
	@echo ""
	@echo "Build Commands:"
	@echo "  make build          - Build CPU variant (default)"
	@echo "  make build-cpu      - Build CPU variant"
	@echo "  make build-cuda     - Build CUDA 11.8 variant"
	@echo "  make build-cuda12   - Build CUDA 12.1 variant"
	@echo "  make build-all      - Build all variants"
	@echo ""
	@echo "Test Commands:"
	@echo "  make test           - Run all container tests"
	@echo "  make test-cpu       - Test CPU container"
	@echo "  make test-cuda      - Test CUDA container"
	@echo ""
	@echo "Run Commands:"
	@echo "  make run-cpu        - Run CPU container interactively"
	@echo "  make run-cuda       - Run CUDA container interactively"
	@echo "  make run-cuda12     - Run CUDA 12.1 container interactively"
	@echo ""
	@echo "Utility Commands:"
	@echo "  make clean          - Remove all C5Q Docker images"
	@echo "  make info           - Show container information"

# Build targets
build: build-cpu

build-cpu:
	@echo "Building CPU variant..."
	docker build -t c5q:latest -t c5q:cpu --build-arg BASE_VARIANT=cpu .

build-cuda:
	@echo "Building CUDA 11.8 variant..."
	docker build -t c5q:cuda -t c5q:cuda11.8 --build-arg BASE_VARIANT=cuda .

build-cuda12:
	@echo "Building CUDA 12.1 variant..."
	docker build -t c5q:cuda12 --build-arg BASE_VARIANT=cuda12 .

build-all: build-cpu build-cuda build-cuda12
	@echo "All builds completed!"

# Test targets
test: test-cpu test-cuda

test-cpu:
	@echo "Testing CPU container..."
	docker run --rm -v "$$(pwd)/data:/data" -v "$$(pwd)/artifacts:/artifacts" c5q:latest python -c "import c5q; print('✓ CPU container working')"

test-cuda:
	@echo "Testing CUDA container..."
	-docker run --rm --gpus all -v "$$(pwd)/data:/data" -v "$$(pwd)/artifacts:/artifacts" c5q:cuda python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')"

# Volume mount detection (Windows vs Linux)
ifeq ($(OS),Windows_NT)
    VOLUME_PREFIX := $(shell cd)
    VOLUME_SEPARATOR := ;
else
    VOLUME_PREFIX := $$(pwd)
    VOLUME_SEPARATOR := :
endif

# Run targets with interactive shells
run-cpu:
	docker run -it --rm \
		-v "$(VOLUME_PREFIX)/data:/data" \
		-v "$(VOLUME_PREFIX)/artifacts:/artifacts" \
		-v "$(VOLUME_PREFIX)/configs:/app/configs" \
		c5q:latest bash

run-cuda:
	docker run -it --rm --gpus all \
		-v "$(VOLUME_PREFIX)/data:/data" \
		-v "$(VOLUME_PREFIX)/artifacts:/artifacts" \
		-v "$(VOLUME_PREFIX)/configs:/app/configs" \
		c5q:cuda bash

run-cuda12:
	docker run -it --rm --gpus all \
		-v "$(VOLUME_PREFIX)/data:/data" \
		-v "$(VOLUME_PREFIX)/artifacts:/artifacts" \
		-v "$(VOLUME_PREFIX)/configs:/app/configs" \
		c5q:cuda12 bash

# Utility targets
clean:
	@echo "Removing C5Q Docker images..."
	-docker rmi c5q:latest c5q:cpu c5q:cuda c5q:cuda11.8 c5q:cuda12 2>/dev/null || true
	@echo "Cleanup complete"

info:
	@echo "C5Q Docker Images:"
	@docker images | grep c5q || echo "No C5Q images found"
	@echo ""
	@echo "Container Status:"
	@docker ps | grep c5q || echo "No running C5Q containers"

# Development helpers
dev-cpu: build-cpu
	@echo "Starting CPU development container..."
	docker run -it --rm \
		-v "$(VOLUME_PREFIX):/workspace" \
		-v "$(VOLUME_PREFIX)/data:/data" \
		-v "$(VOLUME_PREFIX)/artifacts:/artifacts" \
		-w /workspace \
		c5q:latest bash

dev-cuda: build-cuda
	@echo "Starting CUDA development container..."
	docker run -it --rm --gpus all \
		-v "$(VOLUME_PREFIX):/workspace" \
		-v "$(VOLUME_PREFIX)/data:/data" \
		-v "$(VOLUME_PREFIX)/artifacts:/artifacts" \
		-w /workspace \
		c5q:cuda bash