# Product Requirements Document (PRD)

## Project: C5 Quantum Modeling — Least‑20 Prediction

---

## 1. Background & Motivation

The **C5 Quantum Logic Matrix Dataset** (`c5_Matrix_binary.csv`) consists of 11,541+ events, each represented as a set of **5 selected states (QS values)** from a universe of 39 possible quantum states. Each event is encoded in two complementary formats:

* **QS columns (QS\_1..QS\_5):** indices of the 5 selected states, strictly ascending and unique (range 1-39).
* **QV columns (QV\_1..QV\_39):** binary vector (exactly 5 ones) indicating the selected states.

### Dataset Characteristics

* **File Size:** ~4.0 MB
* **Format:** CSV with ASCII/UTF-8 encoding
* **Total Columns:** 45 (1 event-ID + 5 QS + 39 QV)
* **Cylindrical Adjacency:** QV positions wrap around (position 39 is adjacent to position 1)
* **Dual Representation:** Enables both compact numerical analysis (QS columns) and sparse binary analysis (QV columns)

### Supplementary QSx Files

The dataset includes five decomposed files (QS1_binary_matrix.csv through QS5_binary_matrix.csv) that isolate each quantum state position for specialized analysis:
* **Purpose:** Position-specific pattern analysis, targeted ML, computational efficiency
* **Structure:** 41 columns each (event-ID + 1 QS position + 39 QV binary matrix)
* **Applications:** Single-position prediction, ensemble modeling, parallel processing

Key property: **QS values are strictly ordered and respect positional feasibility ranges.** Example:

* QS1: 1–35
* QS2: 2–36
* QS3: 3–37
* QS4: 4–38
* QS5: 5–39

Past analyses showed that states cluster into **4–6 distinct buckets (index ranges)** where distributions shift. Predictive models must account for this heterogeneity.

---

## 2. Problem Statement

We need to build a modeling pipeline that predicts **the 20 least likely states** (bottom‑20) for the next event, given historical context. The system must output per‑state probabilities for all 39 states and provide a ranked list of the least‑likely 20.

---

## 3. Success Metrics

* **Precision\@20 (least‑20):** % of predicted least‑likely states that are indeed absent in the next event.
* **Recall\@20:** % of truly least‑likely states captured in predictions.
* **Calibration quality:** Brier score, Negative Log Likelihood (NLL).
* **Per‑bucket performance:** consistency of predictions across 4–6 index ranges.
* **Runtime feasibility:** <1h for local experiments on Windows 11 CPU box; heavy training offloaded to RunPod GPU.

---

## 4. Constraints

* Events must always be represented as **strictly ascending, unique 5‑of‑39 subsets**.
* Predictions must respect positional feasibility constraints (QS\_1..QS\_5 ranges).
* Must support both local workstation (CPU‑only) and RunPod H200 GPU environments.
* Codebase must be **containerized** and reproducible.

---

## 5. Approach

### 5.1 Data Analysis (EDA)

* Validate dataset integrity (ascending, unique, QV==QS consistency).
* Compute per‑position entropy and drift.
* Derive least‑20 states globally and across temporal segments.
* Cluster states into buckets (k=4/5/6) via co‑activation similarity using Jaccard distance and k-means clustering.
* Analyze cylindrical adjacency patterns and positional biases.
* Generate comprehensive EDA artifacts: JSON summaries, CSV exports, bucket definitions.

### 5.2 Advanced Modeling Approaches

#### Primary Models

1. **Neural Plackett–Luce (NPL)**
   * Encoder: channels‑as‑tokens transformer (iTransformer‑style)
   * Loss: masked Plackett–Luce cross‑entropy with bottom‑20 weighting
   * Bucket conditioning: embeddings or per‑bucket bias terms
   * Without-replacement sampling with feasible range constraints

2. **Discrete Subset Diffusion**
   * Fixed-cardinality diffusion process (K=5) over 39‑dimensional space
   * Cosine noise schedule with 200 denoising steps
   * Learns multi‑modal distributions, produces calibrated marginals
   * Bucket‑aware token embeddings

#### Advanced Architecture Considerations

3. **Inverted Transformer (iTransformer)**
   * Applies attention on inverted dimensions (treating each number's time series as token)
   * Better captures multivariate correlations across all 39 QStates
   * State-of-the-art results on multivariate forecasting benchmarks

4. **Structured State Space Models (S4)**
   * Uses state-space equations instead of attention for long sequences
   * Handles very long-range dependencies (16k+ sequence length)
   * Up to 60× faster than Transformers on benchmarks

5. **Probabilistic Sequence Models (DeepAR)**
   * Autoregressive recurrent networks with predictive distributions
   * Multi-label probabilistic modeling for uncertainty quantification

#### Ensemble and Training Enhancements

* **Multi-Label Ranking Loss:** Pairwise ranking objectives (RankNet/LambdaRank style)
* **Cost-Sensitive Learning:** Focal Loss to reduce false positives in least-20 predictions
* **Dynamic Ensemble Selection:** Mixture-of-experts with gating networks
* **Stacking Meta-learners:** Second-level models combining base model predictions
* **Feature Engineering:** Positional patterns, recency trends, graph-based representations

#### Calibration and Validation

* **Monte Carlo Dropout:** Uncertainty estimation during inference
* **Platt Scaling:** Probability calibration on validation sets
* **Two-Stage Prediction:** Conservative filtering + refined selection

### 5.3 Deployment

* Dockerized pipeline supporting CPU and GPU images.
* Local CPU runs: EDA, prototyping, small experiments.
* RunPod GPU: full training and evaluation.
* Outputs: JSON/CSV artifacts, calibration plots, probability tables, least‑20 predictions.

---

## 6. Deliverables

### Core Artifacts
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

### Technical Infrastructure
* **Repository Structure:** Complete scaffold with configs, artifacts, and modular components
* **Docker Images:** CPU (Windows 11 local) and CUDA (RunPod H200) builds
* **Configuration Management:** YAML-based hyperparameter and bucket configuration
* **Deployment Pipeline:** Makefile with convenience targets for all operations

### Documentation Suite
* **README:** Comprehensive usage instructions for local and cloud execution
* **Environment Specifications:** Detailed hardware requirements and fallback strategies
* **Model Documentation:** Architecture descriptions and implementation guides
* **Performance Benchmarks:** Runtime estimates and resource utilization profiles

---

## 7. Risks & Mitigations

### Technical Risks
* **Overfitting to buckets:** Mitigate via cross‑validation, regime splits, and ensemble diversity
* **False positives in least-20 predictions:** Implement cost-sensitive learning and calibrated thresholds
* **Long-range dependency modeling:** Leverage S4 models and structured state spaces for sequence memory
* **Model complexity vs. interpretability:** Balance advanced architectures with explainable bucket-level outputs

### Infrastructure Risks
* **Runtime limits on local machine (AMD Ryzen 9):**
  - Primary mitigation: RunPod NVIDIA H200 GPU fallback
  - Threshold: Tasks exceeding ~1 hour duration
  - Docker containerization ensures seamless environment transfer
* **Memory constraints with large context windows:**
  - Gradient checkpointing and mixed-precision training
  - Sliding window approach with configurable context lengths
* **Storage limitations:** Dual SSD setup (Samsung 990 PRO + Kingston) provides adequate space

### Data Quality Risks
* **Dataset integrity violations:** Comprehensive validation pipeline with strict ascending/unique checks
* **Temporal drift in patterns:** Segmented analysis and regime change detection
* **Bucket instability:** Multiple clustering approaches (k=4,5,6) and manual override capabilities

### Deployment Risks
* **Environment reproducibility:** Docker containerization with locked dependencies
* **Hardware compatibility:** Multi-target builds (CPU/GPU) with automatic device detection
* **Model artifacts management:** Structured output directories with versioning support

---

## 8. Next Steps

* Finalize PRD and hand off to PO.
* Architect team to shard into epics and stories:

  * EDA + validation
  * Model development (NPL, Diffusion)
  * Training + evaluation pipeline
  * Deployment & containerization
  * Documentation & handoff

