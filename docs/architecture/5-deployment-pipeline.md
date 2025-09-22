# 5. Deployment Pipeline

### Local Development Workflow (Windows 11)

```bash
# Environment setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Data validation and EDA
python -m c5q.eda --csv data/c5_Matrix_binary.csv --out artifacts/eda --k 6

# Local training (CPU-bound)
python -m c5q.train_npl --csv data/c5_Matrix_binary.csv --config configs/hparams.yaml --buckets k6

# Evaluation and reporting
python -m c5q.eval --run artifacts/npl/run_*/ --out artifacts/npl/report
```

### Docker Containerization

```bash
# CPU image for local development
docker build -t c5q:latest .

# GPU image for RunPod deployment
docker build --build-arg BASE=cuda -t c5q:cuda .

# Local EDA execution
mkdir -p out
docker run --rm -v %cd%/data:/data -v %cd%/out:/out c5q:latest \
  python -m c5q.eda --csv /data/c5_Matrix_binary.csv --out /out/eda --k 6
```

### RunPod GPU Deployment

```bash
# Heavy training workloads (>1 hour ETA)
docker run --gpus all --rm -v $PWD/data:/data -v $PWD/out:/out c5q:cuda \
  python -m c5q.train_npl --csv /data/c5_Matrix_binary.csv \
  --config configs/hparams.yaml --buckets k6

# Advanced model training
docker run --gpus all --rm -v $PWD/data:/data -v $PWD/out:/out c5q:cuda \
  python -m c5q.train_subsetdiff --csv /data/c5_Matrix_binary.csv \
  --config configs/hparams.yaml --buckets k6
```

### Testing Pipeline

```bash
# Testing framework execution
pytest tests/ -v --cov=c5q --cov-report=html

# Specific test categories
pytest tests/test_io.py -v           # Data validation tests
pytest tests/test_models.py -v       # Model architecture tests
pytest tests/test_integration.py -v  # End-to-end pipeline tests

# Performance and long-running tests
pytest tests/ -m "not slow" -v      # Fast tests only
pytest tests/ -m "slow" -v          # Long-running tests

# Docker testing
docker run --rm -v %cd%:/app c5q:latest pytest tests/ -v
```

### CI/CD Pipeline

```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest tests/ --cov=c5q --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  docker:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - name: Build CPU Docker image
        run: docker build -t c5q:latest .
      - name: Test Docker image
        run: docker run --rm c5q:latest pytest tests/ -v
```

### Automation via Makefile

```bash
# Testing targets
make test         # Run full test suite with coverage
make test-fast    # Run fast tests only
make test-slow    # Run comprehensive tests including integration
make test-docker  # Test within Docker container

# Development targets
make eda          # Run complete EDA pipeline
make train-npl    # Train Neural Plackett-Luce model
make train-diff   # Train Subset Diffusion model
make evaluate     # Generate comprehensive evaluation reports

# Infrastructure targets
make docker-cpu   # Build CPU Docker image
make docker-gpu   # Build GPU Docker image
make lint         # Run code quality checks
make format       # Format code with black/isort
```

---