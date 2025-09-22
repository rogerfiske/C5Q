# Epic 3: Core Modeling (MVP)

## Overview
**Epic ID:** 3
**Priority:** Critical Path - Core Business Value
**Dependencies:** Epic 2 (Data Pipeline & EDA)
**Duration:** 2-3 weeks
**Status:** Blocked by Epic 2

## Epic Goal
Implement the core machine learning models for least-20 quantum state prediction: Neural Plackett-Luce (NPL) and Discrete Subset Diffusion. This epic delivers the primary business value of the MVP with two complementary modeling approaches.

## Business Value
- **Primary Objective:** Deliver least-20 prediction capability with >80% precision
- **Risk Mitigation:** Two independent modeling approaches reduce single-point-of-failure
- **Competitive Advantage:** Advanced diffusion modeling provides unique approach to subset prediction
- **Measurable ROI:** Direct impact on prediction accuracy and business metrics

## Success Criteria
- ✅ NPL model achieves Precision@20 >0.80 on validation set
- ✅ Subset Diffusion model provides calibrated probability distributions
- ✅ Training completes locally (<1 hour) or successfully on RunPod GPU
- ✅ Models respect all feasibility constraints (position ranges, without-replacement)
- ✅ Comprehensive evaluation metrics demonstrate model performance
- ✅ Model checkpoints and artifacts enable reproducible inference

## Epic Stories

### Story 3.1: Neural Plackett-Luce (NPL) Model Implementation
**Story Points:** 13
**Priority:** Highest - Primary MVP Model
**Dependencies:** Epic 2 completion

**User Story:**
> As a **data scientist**, I want a **Neural Plackett-Luce model** so that I can **predict least-20 quantum states with ranking-aware loss functions**.

**Acceptance Criteria:**
- [ ] Channel-as-tokens transformer encoder implemented
- [ ] Masked Plackett-Luce loss with without-replacement sampling
- [ ] Bucket conditioning via bias terms and embeddings
- [ ] Feasible range constraints enforced during training
- [ ] Cost-sensitive weighting for false positive reduction

**Technical Components:**
- `c5q/encoder_itchan.py` - iTransformer-style encoder
- `c5q/model_npl.py` - NPL model with bucket conditioning
- `c5q/masks.py` - Feasible range enforcement
- `c5q/train_npl.py` - Training pipeline

---

### Story 3.2: Discrete Subset Diffusion Model Implementation
**Story Points:** 13
**Priority:** High - Secondary MVP Model
**Dependencies:** Story 3.1 (shared components)

**User Story:**
> As a **data scientist**, I want a **subset diffusion model** so that I can **generate calibrated probability distributions for quantum state subsets**.

**Acceptance Criteria:**
- [ ] Fixed-cardinality diffusion process (K=5) implemented
- [ ] Cosine noise schedule with 200 denoising steps
- [ ] Bucket-aware token embeddings and conditioning
- [ ] Calibrated marginal probability outputs
- [ ] Efficient sampling and inference procedures

**Technical Components:**
- `c5q/model_subsetdiff.py` - Subset diffusion architecture
- `c5q/train_subsetdiff.py` - Diffusion training pipeline
- `c5q/sampling.py` - Diffusion sampling utilities

---

### Story 3.3: Training Pipeline and Optimization
**Story Points:** 8
**Priority:** High
**Dependencies:** Stories 3.1, 3.2

**User Story:**
> As a **ML engineer**, I want **robust training pipelines** so that I can **train models efficiently with proper optimization and monitoring**.

**Acceptance Criteria:**
- [ ] AdamW optimizer with gradient clipping and regularization
- [ ] Cross-validation with bootstrap confidence intervals
- [ ] Mixed-precision training for GPU efficiency
- [ ] Checkpoint management and resumable training
- [ ] Training metrics logging and visualization

**Technical Components:**
- `c5q/training.py` - Shared training utilities
- `c5q/optimization.py` - Optimizer configurations
- `c5q/checkpoints.py` - Model persistence

---

### Story 3.4: Model Evaluation and Metrics Framework
**Story Points:** 8
**Priority:** High
**Dependencies:** Story 3.3

**User Story:**
> As a **product owner**, I want **comprehensive model evaluation** so that I can **measure business impact and model performance**.

**Acceptance Criteria:**
- [ ] Precision@20 and Recall@20 for least-20 predictions
- [ ] Calibration analysis (Brier score, NLL) with plots
- [ ] Per-bucket and per-position performance breakdowns
- [ ] Uncertainty quantification and confidence intervals
- [ ] Model comparison and selection utilities

**Technical Components:**
- `c5q/eval.py` - Evaluation framework
- `c5q/metrics.py` - Specialized metrics
- `c5q/calibration.py` - Calibration utilities
- `c5q/visualization.py` - Performance plots

---

### Story 3.5: Hyperparameter Configuration and Tuning
**Story Points:** 5
**Priority:** Medium
**Dependencies:** Story 3.4

**User Story:**
> As a **data scientist**, I want **flexible hyperparameter management** so that I can **optimize model performance efficiently**.

**Acceptance Criteria:**
- [ ] YAML-based configuration system for all hyperparameters
- [ ] Model architecture parameters (layers, dimensions, dropout)
- [ ] Training parameters (learning rate, batch size, epochs)
- [ ] Bucket configuration options (k=4,5,6, manual overrides)
- [ ] Environment-specific configurations (CPU vs GPU)

**Technical Components:**
- `configs/hparams.yaml` - Default hyperparameters
- `configs/buckets.manual.yaml` - Manual bucket overrides
- `c5q/config.py` - Configuration loading utilities

## Definition of Done
- [ ] Both NPL and Subset Diffusion models implemented and tested
- [ ] Training pipelines execute successfully locally and on RunPod
- [ ] Model evaluation shows Precision@20 >0.80 for at least one model
- [ ] All feasibility constraints properly enforced
- [ ] Comprehensive evaluation reports generated
- [ ] Model artifacts and checkpoints saved for inference
- [ ] Performance benchmarks documented

## Key Performance Targets
- **Precision@20:** >0.80 (primary success metric)
- **Recall@20:** >0.70 (secondary metric)
- **Calibration:** Brier score <0.25
- **Training Time:** <1 hour locally, <30 minutes on GPU
- **Inference Time:** <1 second per prediction

## Risk Mitigation
- **Model Convergence:** Two independent architectures reduce risk
- **Performance Targets:** Conservative baselines with stretch goals
- **Resource Constraints:** Fallback to RunPod GPU for heavy training
- **Complexity Management:** Focus on proven architectures for MVP

## Dependencies for Next Epic
This epic enables **Epic 4: Deployment & Containerization** which requires:
- ✅ Trained model checkpoints
- ✅ Inference pipelines
- ✅ Evaluation frameworks
- ✅ Performance benchmarks