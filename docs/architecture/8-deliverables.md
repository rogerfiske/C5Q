# 8. Deliverables

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
