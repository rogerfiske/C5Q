# 5. Approach

## 5.1 Data Analysis (EDA)

* Validate dataset integrity (ascending, unique, QV==QS consistency).
* Compute per‑position entropy and drift.
* Derive least‑20 states globally and across temporal segments.
* Cluster states into buckets (k=4/5/6) via co‑activation similarity using Jaccard distance and k-means clustering.
* Analyze cylindrical adjacency patterns and positional biases.
* Generate comprehensive EDA artifacts: JSON summaries, CSV exports, bucket definitions.

## 5.2 Modeling Approach (MVP Scope)

### MVP Core Models (Epics 1-5)

#### 1. Neural Plackett–Luce (NPL) - Primary MVP Model
   * **Encoder:** Channels‑as‑tokens transformer (simplified architecture)
   * **Loss:** Masked Plackett–Luce cross‑entropy with bottom‑20 weighting
   * **Conditioning:** Basic bucket bias terms and positional embeddings
   * **Constraints:** Without-replacement sampling with feasible range constraints
   * **Priority:** Highest - Core business objective

#### 2. Discrete Subset Diffusion - Secondary MVP Model
   * **Architecture:** Fixed-cardinality diffusion process (K=5) over 39‑dimensional space
   * **Schedule:** Cosine noise schedule with 200 denoising steps
   * **Output:** Calibrated marginal probabilities for least-20 prediction
   * **Conditioning:** Bucket‑aware token embeddings
   * **Priority:** High - Alternative approach for comparison

### MVP Training Framework
* **Loss Functions:** Cross-entropy with cost-sensitive weighting for false positive reduction
* **Optimization:** Standard Adam optimizer with gradient clipping
* **Validation:** Cross-validation with bootstrap confidence intervals
* **Evaluation:** Precision@20, Recall@20, Brier score, NLL
* **Calibration:** Basic Platt scaling on validation sets

## 5.3 Post-MVP Advanced Models (Deferred to Epic 6+)

### Advanced Architecture Considerations (Post-MVP Only)

#### 3. Inverted Transformer (iTransformer) - Post-MVP
   * **Rationale:** Research-grade complexity exceeds MVP scope
   * **Benefits:** Better multivariate correlations, state-of-the-art benchmarks
   * **Defer Reason:** Implementation complexity and uncertain ROI for business objective
   * **Timeline:** Epic 6 (Post-MVP) after core model validation

#### 4. Structured State Space Models (S4) - Post-MVP
   * **Rationale:** Advanced sequence modeling for long-range dependencies
   * **Benefits:** 60× faster than Transformers, handles 16k+ sequences
   * **Defer Reason:** Over-engineering for current dataset size (11,541 events)
   * **Timeline:** Epic 6 (Post-MVP) if performance scaling becomes critical

#### 5. Probabilistic Sequence Models (DeepAR) - Post-MVP
   * **Rationale:** Advanced uncertainty quantification
   * **Benefits:** Full predictive distributions, multi-label probabilistic modeling
   * **Defer Reason:** Basic uncertainty via Monte Carlo Dropout sufficient for MVP
   * **Timeline:** Epic 7 (Production) when advanced uncertainty needed

### Advanced Training Enhancements (Post-MVP Only)
* **Multi-Label Ranking Loss:** Pairwise ranking objectives (RankNet/LambdaRank style)
* **Dynamic Ensemble Selection:** Mixture-of-experts with gating networks
* **Stacking Meta-learners:** Second-level models combining base model predictions
* **Advanced Feature Engineering:** Graph-based representations, complex temporal features

## 5.4 MVP Scope Justification

### Included in MVP (Epics 1-5):
✅ **Core Business Value:** Least-20 prediction with NPL and Subset Diffusion
✅ **Proven Architecture:** Standard transformer encoders with established training methods
✅ **Manageable Complexity:** Implementation feasible within 6-8 week timeline
✅ **Measurable Success:** Clear metrics (Precision@20) with business relevance
✅ **Risk Mitigation:** Two complementary approaches reduce single-point-of-failure

### Deferred to Post-MVP (Epic 6+):
⚪ **Research Exploration:** iTransformer, S4, DeepAR for advanced performance
⚪ **Complex Ensembles:** Mixture-of-experts and stacking meta-learners
⚪ **Production Optimization:** Real-time inference, model compression
⚪ **Advanced Features:** Graph-based modeling, complex temporal engineering

## 5.3 Deployment

* Dockerized pipeline supporting CPU and GPU images.
* Local CPU runs: EDA, prototyping, small experiments.
* RunPod GPU: full training and evaluation.
* Outputs: JSON/CSV artifacts, calibration plots, probability tables, least‑20 predictions.

---
