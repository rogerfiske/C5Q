#!/usr/bin/env python3
"""
C5Q Docker Container Validation Script

This script validates that Docker containers are built correctly and can execute
the required functionality for the C5Q Quantum Logic Matrix project.
"""

import subprocess
import sys
import os
import tempfile
from pathlib import Path


def run_command(cmd, capture_output=True):
    """Run a shell command and return result."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=capture_output,
            text=True,
            timeout=120
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)


def check_docker_available():
    """Check if Docker is available and running."""
    print("🔍 Checking Docker availability...")
    success, stdout, stderr = run_command("docker --version")
    if not success:
        print(f"❌ Docker not available: {stderr}")
        return False

    print(f"✅ Docker available: {stdout.strip()}")

    # Check if Docker daemon is running
    success, _, stderr = run_command("docker info")
    if not success:
        print(f"❌ Docker daemon not running: {stderr}")
        return False

    print("✅ Docker daemon running")
    return True


def check_image_exists(image_name):
    """Check if a Docker image exists."""
    success, stdout, _ = run_command(f"docker images -q {image_name}")
    return success and stdout.strip() != ""


def validate_cpu_container():
    """Validate CPU container functionality."""
    print("\n🧪 Testing CPU container...")

    if not check_image_exists("c5q:latest"):
        print("❌ CPU image (c5q:latest) not found. Build it first with: docker build -t c5q:latest .")
        return False

    # Test basic import
    print("  Testing Python package import...")
    cmd = 'docker run --rm c5q:latest python -c "import c5q; print(\'Package imported successfully\')"'
    success, stdout, stderr = run_command(cmd)
    if not success:
        print(f"❌ Package import failed: {stderr}")
        return False
    print(f"  ✅ {stdout.strip()}")

    # Test module availability
    print("  Testing module availability...")
    cmd = 'docker run --rm c5q:latest python -c "import c5q.io, c5q.eda, c5q.dataset, c5q.utils; print(\'All modules available\')"'
    success, stdout, stderr = run_command(cmd)
    if not success:
        print(f"❌ Module availability test failed: {stderr}")
        return False
    print(f"  ✅ {stdout.strip()}")

    return True


def validate_volume_mounting():
    """Validate volume mounting functionality."""
    print("\n📁 Testing volume mounting...")

    # Create temporary test data
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir) / "test_input.txt"
        test_file.write_text("test content for volume mounting")

        # Test volume mounting
        print("  Testing read access...")
        if os.name == 'nt':  # Windows
            volume_mount = f'-v "{temp_dir}:/test_data"'
        else:  # Unix-like
            volume_mount = f'-v "{temp_dir}:/test_data"'

        cmd = f'docker run --rm {volume_mount} c5q:latest python -c "import os; assert os.path.exists(\'/test_data/test_input.txt\'), \'File not found\'; print(\'Volume read test passed\')"'
        success, stdout, stderr = run_command(cmd)
        if not success:
            print(f"❌ Volume read test failed: {stderr}")
            return False
        print(f"  ✅ {stdout.strip()}")

        # Test write access
        print("  Testing write access...")
        cmd = f'docker run --rm {volume_mount} c5q:latest python -c "with open(\'/test_data/output.txt\', \'w\') as f: f.write(\'output test\'); print(\'Volume write test passed\')"'
        success, stdout, stderr = run_command(cmd)
        if not success:
            print(f"❌ Volume write test failed: {stderr}")
            return False
        print(f"  ✅ {stdout.strip()}")

        # Verify file was created
        output_file = Path(temp_dir) / "output.txt"
        if not output_file.exists():
            print("❌ Output file not created on host")
            return False
        print("  ✅ File persisted on host filesystem")

    return True


def validate_cuda_container():
    """Validate CUDA container functionality."""
    print("\n🚀 Testing CUDA container...")

    if not check_image_exists("c5q:cuda"):
        print("⚠️  CUDA image (c5q:cuda) not found. Build it first with: docker build -t c5q:cuda --build-arg BASE_VARIANT=cuda .")
        return True  # Not a failure, just unavailable

    # Check if NVIDIA Docker is available
    print("  Checking NVIDIA Docker runtime...")
    success, stdout, stderr = run_command("docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi")
    if not success:
        print("⚠️  NVIDIA Docker runtime not available. CUDA tests skipped.")
        print(f"     Error: {stderr}")
        return True  # Not a failure, just unavailable

    # Test CUDA availability in container
    print("  Testing CUDA availability...")
    cmd = 'docker run --rm --gpus all c5q:cuda python -c "import torch; print(f\'CUDA available: {torch.cuda.is_available()}\'); print(f\'GPU count: {torch.cuda.device_count()}\') if torch.cuda.is_available() else None"'
    success, stdout, stderr = run_command(cmd)
    if not success:
        print(f"❌ CUDA test failed: {stderr}")
        return False
    print(f"  ✅ {stdout.strip()}")

    return True


def validate_container_health():
    """Validate container health checks."""
    print("\n🏥 Testing container health...")

    # Test health check
    print("  Running health check...")
    cmd = 'docker run --rm c5q:latest python -c "import c5q; print(\'Container healthy\')"'
    success, stdout, stderr = run_command(cmd)
    if not success:
        print(f"❌ Health check failed: {stderr}")
        return False
    print(f"  ✅ {stdout.strip()}")

    return True


def main():
    """Main validation function."""
    print("🐳 C5Q Docker Container Validation")
    print("=" * 50)

    # Check Docker availability
    if not check_docker_available():
        print("\n❌ Docker validation failed - Docker not available")
        sys.exit(1)

    all_tests_passed = True

    # Run validation tests
    tests = [
        ("CPU Container", validate_cpu_container),
        ("Volume Mounting", validate_volume_mounting),
        ("CUDA Container", validate_cuda_container),
        ("Container Health", validate_container_health),
    ]

    for test_name, test_func in tests:
        try:
            if not test_func():
                all_tests_passed = False
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            all_tests_passed = False

    # Summary
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 All Docker container validations passed!")
        print("\nNext steps:")
        print("1. Commit Docker files to repository")
        print("2. Push to GitHub for RunPod deployment")
        print("3. Test on RunPod H200 GPU environment")
    else:
        print("❌ Some Docker container validations failed!")
        print("\nPlease fix the issues and run validation again.")
        sys.exit(1)


if __name__ == "__main__":
    main()