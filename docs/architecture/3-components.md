# 3. Components

### 3.1 Core Infrastructure

```
C5Q/
├─ configs/
│   ├─ hparams.yaml                  # Model & training configuration
│   └─ buckets.manual.yaml           # Manual bucket overrides
├─ data/                             # Dataset storage (gitignored)
├─ artifacts/                        # EDA & training outputs
└─ c5q/                             # Core package
```

### 3.2 Data Processing & Validation

* **`io.py`:** CSV loading with comprehensive integrity checks
  - QS/QV consistency validation
  - Ascending/unique constraint verification
  - Range bounds checking (1-39)
  - Cylindrical adjacency validation

* **`data_manager.py`:** Data integrity and backup management
  - Dataset checksumming (SHA256) for integrity verification
  - Automated backup creation before any data modification
  - Version tracking for datasets and processed artifacts
  - Corruption detection and recovery procedures

#### Data Directory Structure (Enhanced)
```
data/
├─ raw/                          # Original immutable datasets
│   └─ c5_Matrix_binary.csv      # Primary dataset (read-only)
├─ processed/                    # Preprocessed datasets
│   ├─ eda_artifacts/            # EDA outputs
│   └─ model_inputs/             # Model-ready data
├─ backups/                      # Dataset backups and checksums
│   ├─ checksums.json            # SHA256 hashes
│   └─ snapshots/                # Timestamped backups
artifacts/
├─ models/                       # Model checkpoints and weights
├─ logs/                         # Training and evaluation logs
└─ reports/                      # Generated reports and analysis
```

#### Backup and Recovery Protocol
* **Primary Dataset:** `data/raw/c5_Matrix_binary.csv` (read-only, checksummed)
* **Backup Locations:**
  - Local: `data/backups/` with timestamps
  - Cloud: RunPod persistent storage during training
  - External: User-managed backup to external storage
* **Retention Policy:** Keep last 5 versions of processed data, full backup history for raw data
* **Recovery Procedures:** Automated integrity checks on startup, manual recovery commands

#### Disaster Recovery Strategy
* **Data Loss Mitigation:** Multiple backup copies with checksums
* **Model Recovery:** Checkpoint versioning with metadata
* **Environment Recovery:** Docker images as immutable environments
* **Documentation Recovery:** Version-controlled documentation in git

* **`eda.py`:** Range-aware exploratory data analysis
  - Per-position entropy computation
  - Global and segmented least-20 analysis
  - Jaccard similarity-based clustering (k=4,5,6)
  - Cylindrical adjacency pattern detection
  - Comprehensive JSON/CSV artifact generation

### 3.3 Dataset Management

* **`dataset.py`:** PyTorch sliding-window dataset
  - Configurable context windows (default: 128)
  - Dual representation handling (QS + QV)
  - Batch processing with efficient memory usage
  - Support for stride-based sampling

### 3.4 Constraint & Bucket Systems

* **`masks.py`:** Feasible range enforcement
  - Per-position range masks (QS_1: 1-35, QS_2: 2-36, etc.)
  - Without-replacement sampling logic
  - Dynamic masking during sequential selection

* **`buckets.py`:** State clustering and conditioning
  - Data-driven k=6 default clustering
  - Manual bucket override support
  - Bucket-to-state mapping utilities
  - Support for k=4,5,6 configurations

### 3.5 Encoder Architectures

* **`encoder_itchan.py`:** iTransformer-style channel-as-tokens encoder
  - Multi-head attention over channel dimension
  - Layer normalization and dropout regularization
  - Configurable depth and width parameters

* **`encoder_s4.py`:** Structured State Space encoder
  - State-space equations for long-range dependencies
  - Linear scaling with sequence length
  - Efficient convolution-based implementation

### 3.6 Model Implementations

* **`model_npl.py`:** Neural Plackett–Luce with advanced conditioning
  - Masked Plackett–Luce loss with without-replacement sampling
  - Bucket bias terms and positional embeddings
  - Cost-sensitive weighting for least-20 optimization

* **`model_subsetdiff.py`:** Discrete subset diffusion
  - Fixed-cardinality diffusion (K=5)
  - Cosine noise schedule with 200 steps
  - Bucket-aware token embeddings

* **`model_itransformer.py`:** Inverted Transformer architecture
  - Attention applied on inverted dimensions
  - Enhanced multivariate correlation modeling
  - State-of-the-art time series forecasting performance

* **`model_deepar.py`:** Probabilistic autoregressive model
  - Uncertainty quantification via predictive distributions
  - Multi-variate time series support
  - Likelihood-based training objectives

### 3.7 Training Infrastructure

* **`train_npl.py`:** NPL training pipeline
  - Bucket-aware masked PL loss
  - Gradient clipping and regularization
  - Mixed-precision training support

* **`train_subsetdiff.py`:** Subset diffusion training
  - Denoising score matching objectives
  - Configurable noise schedules
  - GPU-optimized diffusion sampling

* **`train_ensemble.py`:** Advanced ensemble training
  - Dynamic ensemble selection
  - Stacking meta-learner implementation
  - Mixture-of-experts with gating networks

### 3.8 Evaluation & Metrics

* **`eval.py`:** Comprehensive evaluation framework
  - Precision@20/Recall@20 for least-20 predictions
  - Calibration analysis (Brier score, NLL)
  - Per-bucket and per-position breakdowns
  - Uncertainty quantification metrics

* **`metrics.py`:** Specialized metric implementations
  - Bottom-20 precision/recall calculators
  - Calibration plotting utilities
  - Cross-validation and bootstrap helpers

### 3.9 Testing Infrastructure

* **`tests/`:** Comprehensive testing framework
  - **`test_io.py`:** Unit tests for data loading and validation
    - QS/QV consistency validation tests
    - Ascending/unique constraint verification tests
    - Edge case handling (malformed data, missing files)
    - Performance tests for large datasets
  - **`test_eda.py`:** EDA pipeline validation tests
    - Entropy calculation correctness
    - Bucket clustering reproducibility
    - Output artifact validation
  - **`test_models.py`:** Model architecture and training tests
    - Forward pass shape validation
    - Loss function correctness
    - Gradient flow verification
    - Convergence smoke tests
  - **`test_dataset.py`:** Dataset and preprocessing tests
    - Sliding window correctness
    - Batch processing validation
    - Memory usage tests
  - **`test_masks.py`:** Constraint system tests
    - Feasible range mask validation
    - Without-replacement sampling correctness
    - Edge case position constraints
  - **`test_integration.py`:** End-to-end pipeline tests
    - Full EDA pipeline execution
    - Training pipeline integration
    - Docker container functionality
    - Multi-environment compatibility

* **`conftest.py`:** pytest configuration and fixtures
  - Sample dataset fixtures
  - Model initialization fixtures
  - Temporary directory management
  - Seed management for reproducibility

* **`pytest.ini`:** Testing configuration
  - Test discovery patterns
  - Coverage reporting settings
  - Parallel execution configuration
  - Timeout settings for long-running tests

### 3.10 Deployment & Infrastructure

* **`Dockerfile`:** Multi-target containerization
  - CPU base for local Windows 11 development
  - CUDA base for RunPod H200 GPU training
  - Optimized layer caching and dependency management

* **`Makefile`:** Automation and convenience targets
  - EDA pipeline execution
  - Model training workflows
  - Evaluation and reporting automation

* **`utils.py`:** Shared utilities
  - Seed management and reproducibility
  - Logging configuration
  - Checkpoint handling and model persistence

---
