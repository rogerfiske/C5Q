









# 3. Success Metrics

* **Precision\@20 (least‑20):** % of predicted least‑likely states that are indeed absent in the next event.
* **Recall\@20:** % of truly least‑likely states captured in predictions.
* **Calibration quality:** Brier score, Negative Log Likelihood (NLL).
* **Per‑bucket performance:** consistency of predictions across 4–6 index ranges.
* **Runtime feasibility:** <1h for local experiments on Windows 11 CPU box; heavy training offloaded to RunPod GPU.

---
