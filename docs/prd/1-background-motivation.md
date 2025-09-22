# 1. Background & Motivation

The **C5 Quantum Logic Matrix Dataset** (`c5_Matrix_binary.csv`) consists of 11,541+ events, each represented as a set of **5 selected states (QS values)** from a universe of 39 possible quantum states. Each event is encoded in two complementary formats:

* **QS columns (QS\_1..QS\_5):** indices of the 5 selected states, strictly ascending and unique (range 1-39).
* **QV columns (QV\_1..QV\_39):** binary vector (exactly 5 ones) indicating the selected states.

## Dataset Characteristics

* **File Size:** ~4.0 MB
* **Format:** CSV with ASCII/UTF-8 encoding
* **Total Columns:** 45 (1 event-ID + 5 QS + 39 QV)
* **Cylindrical Adjacency:** QV positions wrap around (position 39 is adjacent to position 1)
* **Dual Representation:** Enables both compact numerical analysis (QS columns) and sparse binary analysis (QV columns)

## Supplementary QSx Files

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
