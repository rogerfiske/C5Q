# 2. Data Flow

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
