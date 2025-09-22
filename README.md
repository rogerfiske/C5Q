# C5 Quantum Modeling — Least‑20 Prediction

A comprehensive machine learning framework for quantum state prediction with Neural Plackett-Luce and Discrete Subset Diffusion models.

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install C5Q package in development mode
pip install -e .
```

### Basic Usage

```python
# Load and validate dataset
from c5q.io import load_primary_dataset
df = load_primary_dataset('data/c5_Matrix_binary.csv')

# Run exploratory data analysis
from c5q.eda import run_comprehensive_eda
results = run_comprehensive_eda(df, 'artifacts/eda')

# Create dataset for training
from c5q.dataset import create_train_val_split
train_ds, val_ds = create_train_val_split('data/c5_Matrix_binary.csv')
```

### Command Line Interface

```bash
# Run EDA analysis
python -m c5q.eda --csv data/c5_Matrix_binary.csv --out artifacts/eda

# Train models (coming in Epic 3)
python -m c5q.train_npl --config configs/hparams.yaml

# Evaluate models (coming in Epic 3)
python -m c5q.eval --model artifacts/models/npl_model.pth
```

## Project Structure

```
C5Q/
├─ c5q/                    # Main Python package
├─ configs/                # Configuration files
├─ data/                   # Dataset storage (gitignored)
├─ artifacts/              # Model outputs and reports
├─ tests/                  # Test suite
├─ docs/                   # Documentation and epics
└─ requirements.txt        # Dependencies
```

## Features

- **Data Validation:** Comprehensive dataset integrity checking
- **EDA Framework:** Automated exploratory data analysis with clustering
- **Model Architectures:** Neural Plackett-Luce and Subset Diffusion models
- **Constraint Enforcement:** Feasible range and without-replacement sampling
- **Containerization:** Docker support for local and cloud deployment
- **Testing:** Comprehensive test suite with >90% coverage

## Development Status

**Current:** Epic 1 - Foundation & Infrastructure ✅
**Next:** Epic 2 - Data Pipeline & EDA
**MVP Target:** Precision@20 >80% for least-20 predictions

## Hardware Requirements

### Local Development (Windows 11)
- AMD Ryzen 9 6900HX CPU
- 64GB RAM
- 2TB NVMe storage
- Python 3.9+

### Cloud Training (RunPod)
- NVIDIA H200 GPU
- Docker containerization
- Automated deployment pipeline

## Documentation

- [Epic Structure & Roadmap](docs/epic-summary-roadmap.md)
- [Architecture Documentation](docs/architecture/)
- [Product Requirements](docs/prd/)
- [Story Breakdown](docs/stories/)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `make test`
4. Submit a pull request

## License

MIT License - See LICENSE file for details.

## Contact

C5Q Development Team - dev@c5quantum.com