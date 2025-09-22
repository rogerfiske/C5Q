# Architecture Document

## Project: C5 Quantum Modeling — Least‑20 Prediction

---

## 1. System Overview

The system ingests the **C5 Quantum Logic Matrix dataset** and produces per‑state probability distributions for the next event, from which the **20 least‑likely states** are selected. It is designed to run both **locally (CPU)** and on **RunPod GPU (NVIDIA H200)**.

---

## 2. Data Flow

1. **Input Processing:**
   - Primary: `c5_Matrix_binary.csv` (11,541+ events, 45 columns)
   - Supplementary: `QS1_binary_matrix.csv` through `QS5_binary_matrix.csv` (position-specific analysis)
   - Validation: Ascending/unique constraints, QV==QS consistency, range bounds

2. **EDA Pipeline:**
   - Dataset integrity checks with comprehensive error reporting
   - Entropy computation per position with drift analysis
   - Least‑20 baselines (global + temporal segments)
   - Bucket discovery via Jaccard similarity and k-means clustering (k=4,5,6)
   - Cylindrical adjacency pattern analysis
   - Export: JSON summaries, CSV tables, bucket definitions

3. **Dataset Preparation:**
   - Sliding windows of historical events (default: 128 context window)
   - Dual representation handling (QS compact + QV binary)
   - Feasible range masking per position
   - Without-replacement sampling preparation

4. **Model Architecture Stack:**
   - **Primary:** Neural Plackett–Luce (NPL) with bucket conditioning
   - **Primary:** Discrete Subset Diffusion (fixed K=5 cardinality)
   - **Advanced:** Inverted Transformer (iTransformer) for multivariate correlations
   - **Advanced:** Structured State Space (S4) for long-range dependencies
   - **Advanced:** Probabilistic DeepAR for uncertainty quantification
   - **Ensemble:** Dynamic selection with gating networks and stacking meta-learners

5. **Training & Optimization:**
   - Multi-label ranking loss with pairwise objectives
   - Cost-sensitive learning via Focal Loss
   - Monte Carlo Dropout for uncertainty estimation
   - Gradient checkpointing and mixed-precision training

6. **Evaluation Framework:**
   - Precision@20, Recall@20 for least-likely predictions
   - Calibration metrics (Brier score, NLL)
   - Per‑bucket and per‑position performance analysis
   - Cross-validation with bootstrap validation
   - Uncertainty quantification and confidence intervals

7. **Output Generation:**
   - Per‑state probability distributions (39-dimensional)
   - Ranked least‑20 predictions with confidence scores
   - Calibration plots and performance visualizations
   - Comprehensive JSON/CSV reporting with metadata

---

## 3. Components

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

### 3.9 Deployment & Infrastructure

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

## 4. Runtime Environments

### Local Workstation (Detailed Specifications)

* **Operating System:** Windows 11 Home 64‑bit
* **CPU:** AMD Ryzen 9 6900HX (Rembrandt, 6nm technology) @ 56°C
* **RAM:** 64.0GB @ 2393MHz (40-39-39-77 timings)
* **Motherboard:** Shenzhen Meigao Electronic Equipment Co.,Ltd F7BAA (FP7) @ 20°C
* **Graphics:**
  - Primary: AMD Radeon RX 6600M (8176MB VRAM)
  - Integrated: AMD Radeon Graphics (512MB VRAM)
  - CrossFire: Disabled
* **Displays:**
  - VP2780 SERIES (2560x1440@59Hz)
  - ASUS PB278 (2560x1440@60Hz)
* **Storage:**
  - Samsung SSD 990 PRO 1TB (NVMe, Unknown)
  - KINGSTON OM8PGP41024Q-A0 1TB (SATA-2 SSD)
* **Execution Mode:** CPU‑only (EDA, prototyping, light training < 1 hour)
* **Thermal Management:** Adequate cooling with 56°C CPU temps under load
* **Memory Bandwidth:** Sufficient for 128-context window processing

### RunPod GPU

* **GPU:** NVIDIA H200
* **Mode:** CUDA training for NPL and Subset Diffusion.
* **Usage:** Offload heavy jobs via Docker; artifacts synced back.

---

## 5. Deployment Pipeline

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

### Automation via Makefile

```bash
# Convenience targets
make eda          # Run complete EDA pipeline
make train-npl    # Train Neural Plackett-Luce model
make train-diff   # Train Subset Diffusion model
make evaluate     # Generate comprehensive evaluation reports
make docker-cpu   # Build CPU Docker image
make docker-gpu   # Build GPU Docker image
```

---

## 6. Security & Compliance

* Data is local and controlled; no PII.
* Containerized runs ensure reproducibility and portability.

---

## 7. Future Extensions

### Model Architecture Enhancements
* **Advanced Diffusion Kernels:** Implement discrete diffusion with learned transition matrices
* **Regime Change Detection:** Switching Set-Transformer with dynamic regime identification
* **Graph Neural Networks:** Explicit modeling of state co-occurrence relationships
* **Hierarchical Models:** Multi-scale temporal modeling with different time horizons

### Training & Optimization
* **Distributed Training:** Multi-GPU support with data/model parallelism
* **Meta-Learning:** Few-shot adaptation to new temporal regimes
* **Active Learning:** Intelligent selection of training examples for efficiency
* **Federated Learning:** Privacy-preserving training across distributed datasets

### Ensemble & Uncertainty
* **Bayesian Neural Networks:** Full posterior uncertainty quantification
* **Deep Ensembles:** Diverse model initialization and training strategies
* **Conformal Prediction:** Distribution-free uncertainty intervals
* **Adversarial Training:** Robustness to distributional shifts

### Infrastructure & Scalability
* **Real-time Inference:** Low-latency prediction serving with model compression
* **Automated Hyperparameter Tuning:** Bayesian optimization for model configuration
* **MLOps Integration:** Model versioning, experiment tracking, automated deployment
* **Visualization Dashboards:** Interactive exploration of least-20 predictions and model behavior

### Data & Feature Engineering
* **External Data Integration:** Weather, economic indicators, or other contextual features
* **Causal Discovery:** Identification of causal relationships between states
* **Synthetic Data Generation:** Augmentation via learned generative models
* **Feature Selection:** Automated relevance determination for high-dimensional features

---

## 8. Deliverables

### Documentation Suite
* **`prd.md`** — Comprehensive product requirements with advanced modeling considerations
* **`architecture.md`** — Detailed system architecture with component specifications
* **`README.md`** — User-facing documentation with quick-start guides
* **`README_PC_RUNPOD.md`** — Environment-specific deployment instructions
* **`CHANGELOG.md`** — Version history and release notes

### Source Code Repository
* **Core Package (`c5q/`):**
  - Data processing and validation (`io.py`, `eda.py`)
  - Dataset management (`dataset.py`)
  - Constraint systems (`masks.py`, `buckets.py`)
  - Model architectures (NPL, Diffusion, iTransformer, S4, DeepAR)
  - Training pipelines with advanced optimization
  - Comprehensive evaluation framework
  - Utilities and shared components

* **Configuration Management:**
  - `configs/hparams.yaml` — Model and training hyperparameters
  - `configs/buckets.manual.yaml` — Manual bucket override configurations
  - `requirements.txt` — Python dependency specifications

### Containerization & Deployment
* **Multi-target Docker Images:**
  - CPU image for Windows 11 local development
  - CUDA image for RunPod H200 GPU training
  - Optimized builds with dependency caching

* **Automation Infrastructure:**
  - `Makefile` with comprehensive automation targets
  - CI/CD-ready pipeline definitions
  - Environment detection and automatic fallback logic

### Artifacts & Outputs
* **EDA Results:**
  - `artifacts/eda/eda_summary.json` — Comprehensive analysis metadata
  - Global and segmented least-20 CSV tables
  - Bucket clustering results with similarity matrices
  - Visualization plots and statistical summaries

* **Model Artifacts:**
  - Trained model checkpoints with metadata
  - Performance evaluation reports
  - Calibration plots and uncertainty analysis
  - Per-bucket and per-position performance breakdowns

* **Prediction Outputs:**
  - `next_event_marginals.csv` — 39-dimensional probability distributions
  - `least20.csv` — Ranked least-likely state predictions
  - Confidence intervals and uncertainty quantification
  - Model explanation and feature importance analysis
