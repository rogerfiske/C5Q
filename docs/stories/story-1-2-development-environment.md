# Story 1.2: Development Environment Configuration

## Story Details
**Story ID:** 1.2
**Epic:** Epic 1 - Foundation & Infrastructure
**Story Points:** 5
**Priority:** Highest
**Dependencies:** Story 1.1 (Project Repository Setup)
**Status:** Ready for Development

## User Story
> As a **developer**, I want a **consistent development environment** so that I can **develop and test locally without environment-related issues**.

## Business Value
Ensures all developers can work in identical environments, reducing "works on my machine" issues and enabling efficient collaboration. Critical for Windows 11 + RunPod GPU deployment strategy.

## Acceptance Criteria
- [ ] Python virtual environment setup instructions work on Windows 11
- [ ] All dependencies from `requirements.txt` install successfully
- [ ] Local Python package (`c5q`) imports correctly
- [ ] Development tools (black, isort, flake8) configured and functional
- [ ] Jupyter notebook environment available for exploration

## Technical Implementation Tasks

### Task 1.2.1: Python Virtual Environment Setup
**Estimated Time:** 2 hours
- [ ] Create Windows 11 compatible virtual environment setup script
- [ ] Test with Python 3.9, 3.10, 3.11 versions
- [ ] Document activation/deactivation procedures
- [ ] Validate environment isolation

**Windows Setup Script:**
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

### Task 1.2.2: Dependency Validation and Installation
**Estimated Time:** 3 hours
- [ ] Validate all dependencies in `requirements.txt` install correctly
- [ ] Test version compatibility matrix
- [ ] Resolve any Windows-specific installation issues
- [ ] Add optional GPU dependencies with clear documentation

**Key Dependencies to Validate:**
- `torch>=2.1.0,<2.3.0` (CPU version for local dev)
- `numpy>=1.24.0,<1.27.0`
- `pandas>=2.0.0,<2.2.0`
- `pytest>=7.4.0,<8.1.0`
- Development tools: `black`, `isort`, `flake8`

### Task 1.2.3: Local Package Installation
**Estimated Time:** 2 hours
- [ ] Configure editable installation of `c5q` package
- [ ] Create `setup.py` or use `pyproject.toml`
- [ ] Test import paths and module accessibility
- [ ] Validate package discovery and imports

**Setup Configuration:**
```python
# setup.py or pyproject.toml
name = "c5q"
version = "0.1.0"
packages = find_packages()
python_requires = ">=3.9"
install_requires = [
    # Read from requirements.txt
]
```

### Task 1.2.4: Development Tools Configuration
**Estimated Time:** 3 hours
- [ ] Configure Black for code formatting
- [ ] Configure isort for import sorting
- [ ] Configure flake8 for linting
- [ ] Create configuration files with project-specific settings
- [ ] Test tool integration and consistency

**Configuration Files:**

`.flake8`:
```ini
[flake8]
max-line-length = 88
exclude = .git,__pycache__,.venv,build,dist
ignore = E203,W503
```

`pyproject.toml`:
```toml
[tool.black]
line-length = 88
target-version = ['py39']

[tool.isort]
profile = "black"
multi_line_output = 3
```

### Task 1.2.5: Jupyter Environment Setup
**Estimated Time:** 2 hours
- [ ] Install Jupyter notebook/lab in development environment
- [ ] Configure kernel with project environment
- [ ] Test notebook functionality with c5q imports
- [ ] Create sample exploratory notebook

## Definition of Done
- [ ] Virtual environment creates successfully on clean Windows 11 system
- [ ] All dependencies install without errors
- [ ] `python -c "import c5q; print('Success')"` executes successfully
- [ ] Development tools run without configuration errors
- [ ] Jupyter notebook can import and use c5q package
- [ ] Environment setup documented with troubleshooting section

## Testing Criteria
- [ ] Fresh virtual environment setup completes end-to-end
- [ ] Package imports work correctly: `from c5q import utils`
- [ ] Code formatting works: `black --check c5q/`
- [ ] Import sorting works: `isort --check-only c5q/`
- [ ] Linting passes: `flake8 c5q/`
- [ ] Jupyter kernel detects and imports c5q package

## Environment Validation Script
```bash
# Test script to validate environment
#!/bin/bash
echo "Testing C5Q Development Environment..."

# Test Python import
python -c "import c5q; print('✓ C5Q package imports successfully')"

# Test dependencies
python -c "import torch, pandas, numpy, sklearn; print('✓ Core dependencies available')"

# Test development tools
black --check c5q/ && echo "✓ Black formatting validated"
isort --check-only c5q/ && echo "✓ Import sorting validated"
flake8 c5q/ && echo "✓ Linting passed"

# Test Jupyter
jupyter --version && echo "✓ Jupyter available"

echo "Development environment validation complete!"
```

## Notes
- Focus on Windows 11 compatibility as primary development environment
- Document common troubleshooting issues (PATH problems, permission issues)
- Consider adding development environment health check command
- Ensure all paths use forward slashes or os.path for cross-platform compatibility