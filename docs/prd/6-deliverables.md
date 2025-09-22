# 6. Deliverables

## Core Artifacts
* **EDA Pack:**
  - `eda_summary.json` with entropies, bottom-20 analysis, bucket definitions
  - Global and segmented least-20 CSV tables
  - Bucket clustering results (k=4,5,6) with Jaccard similarity analysis
* **Training Pipelines:**
  - NPL with masked Plackett-Luce loss and bucket conditioning
  - Discrete Subset Diffusion with cosine noise schedule
  - Advanced model implementations (iTransformer, S4, DeepAR)
* **Evaluation Framework:**
  - Per‑bucket, per‑position, and aggregate performance metrics
  - Calibration plots and uncertainty quantification
  - Bottom-20 precision/recall analysis
  - Cross-validation and bootstrap validation strategies

## Technical Infrastructure
* **Repository Structure:** Complete scaffold with configs, artifacts, and modular components
* **Docker Images:** CPU (Windows 11 local) and CUDA (RunPod H200) builds
* **Configuration Management:** YAML-based hyperparameter and bucket configuration
* **Deployment Pipeline:** Makefile with convenience targets for all operations

## Documentation Suite
* **README:** Comprehensive usage instructions for local and cloud execution
* **Environment Specifications:** Detailed hardware requirements and fallback strategies
* **Model Documentation:** Architecture descriptions and implementation guides
* **Performance Benchmarks:** Runtime estimates and resource utilization profiles

---
