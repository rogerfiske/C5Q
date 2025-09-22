# 4. Constraints

## 4.1 Technical Constraints

* **Data Format:** Events must always be represented as **strictly ascending, unique 5‑of‑39 subsets**
* **Model Constraints:** Predictions must respect positional feasibility constraints (QS\_1..QS\_5 ranges)
* **Environment Support:** Must support both local workstation (CPU‑only) and RunPod H200 GPU environments
* **Containerization:** Codebase must be **containerized** and reproducible with Docker
* **Testing Requirements:** All code must have >90% test coverage with comprehensive unit and integration tests
* **Dependency Management:** All dependencies must be version-locked for reproducibility

## 4.2 MVP Scope Constraints

* **Model Complexity:** MVP limited to NPL and Subset Diffusion models only
* **Advanced Features:** iTransformer, S4, DeepAR, and ensemble methods deferred to post-MVP
* **Timeline:** MVP delivery within 6-8 weeks (Epics 1-5)
* **Resource Allocation:** Single development track focused on critical path

## 4.3 Infrastructure Constraints

* **Local Hardware:** AMD Ryzen 9 6900HX, 64GB RAM, limited to <1 hour training tasks
* **Cloud Fallback:** RunPod H200 GPU for training tasks exceeding local capacity
* **Data Management:** Comprehensive backup and recovery system required
* **CI/CD:** Automated testing and deployment pipeline mandatory

## 4.4 Quality Constraints

* **Code Quality:** Black formatting, isort imports, flake8 linting enforced
* **Documentation:** Comprehensive documentation for all components and workflows
* **Performance:** Precision@20 >0.8 for least-20 predictions (target metric)
* **Reliability:** Zero data loss tolerance with automated backup systems

---
