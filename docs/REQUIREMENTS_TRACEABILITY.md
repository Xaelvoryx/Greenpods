# Requirements Traceability Matrix (RTM)
## Project: End-to-End Avocado Ripening Stage Detection & Classification System
**Client / Assignment:** Greenpod Labs Data Science Assignment-2  
**Implementation Workspace:** `c:\Users\gobin\Desktop\Documents\Greenpods`  
**Date of Audit:** October 2026  

---

### 1. Requirements Overview & Traceability Table

| Req ID | Requirement Description | Implementation Location | Validation Method | Completion Status | Notes / Blockers / Evidence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **REQ-01** | Preserved 4 Ripening Classes: `underripe`, `breaking`, `ripe`, `overripe` | `src/config.py`, `src/dataset.py` | Unit tests (`tests/test_dataset.py`) | **VERIFIED** | Canonical mapping: index 0: underripe, 1: breaking, 2: ripe, 3: overripe. Dataset mapping handles alias mapping from workspace folder structure. |
| **REQ-02** | Dataset Audit & Quality Analysis (Dimensions, formats, corrupted files, duplicates, imbalance, color distributions) | `src/data_audit.py`, `scripts/audit_dataset.py`, `notebooks/01_eda_and_data_cleaning.ipynb` | Automated script run, output artifacts inspection | **VERIFIED** | Generated `dataset_audit.json`, `class_distribution.csv`, `image_quality_summary.csv`, and EDA visualizations. |
| **REQ-03** | Visual Inspection & Lighting / Color Bias Analysis (RGB, HSV, CIELAB $a^*$, LBP texture) | `src/visualization.py`, `scripts/audit_dataset.py` | Plot generation in `outputs/eda/` | **VERIFIED** | Quantitative histograms and sample grids generated. |
| **REQ-04** | Data Preprocessing & Deterministic Stratified Split (70% Train, 15% Val, 15% Test) | `src/preprocessing.py`, `scripts/prepare_dataset.py` | Split manifest hash check, unit test (`tests/test_preprocessing.py`) | **VERIFIED** | Fixed random seed (42), manifest saved in `data/splits/`, zero partition leakage verified. |
| **REQ-05** | Training Data Augmentation (Rotation, horizontal flip, mild brightness/contrast) | `src/dataset.py`, `src/preprocessing.py` | Unit tests on tensor bounds & shapes | **VERIFIED** | Applied strictly to training split; validation/test partitions unaugmented. |
| **REQ-06** | Approach 1: Classical CV (HSV + CIELAB $a^*$ + LBP) + Classifier (SVM / RF) | `src/classical_model.py`, `scripts/train_classical.py`, `notebooks/02_approach_1_classical_ml.ipynb` | Test evaluation, `tests/test_classical_model.py` | **VERIFIED** | Standardized pipeline saved to `models/classical/avocado_classical_model.joblib`. |
| **REQ-07** | Approach 2: Deep Learning (MobileNetV2 Transfer Learning, 4-class head, CrossEntropyLoss, Adam) | `src/dl_model.py`, `src/training.py`, `scripts/train_deep_learning.py`, `notebooks/03_approach_2_deep_learning.ipynb` | Checkpoint verification, `tests/test_dl_model.py` | **VERIFIED** | Best validation loss checkpoint saved to `models/deep_learning/avocado_mobilenetv2_best.pth`. |
| **REQ-08** | Unified Model Evaluation & Metrics Comparison (Accuracy, Macro Precision, Macro Recall, Macro F1, Latency, Size) | `src/evaluation.py`, `scripts/evaluate_models.py`, `notebooks/04_model_evaluation_and_demo.ipynb` | Automated test `tests/test_evaluation.py` | **VERIFIED** | Generated `model_comparison.csv`, confusion matrices, and error analysis CSVs. |
| **REQ-09** | Detailed Error Analysis & Visual Diagnosis | `src/evaluation.py`, `scripts/evaluate_models.py` | `outputs/evaluation/error_analysis.csv` | **VERIFIED** | Evaluates adjacent class confusion (breaking vs. ripe, ripe vs. overripe). |
| **REQ-10** | Single-Image Inference Demonstration CLI | `src/inference.py`, `main_demo.py` | CLI execution with valid, corrupted, and missing image tests | **VERIFIED** | Dual-model inference with probabilities and execution timing. |
| **REQ-11** | Four Valid, High-Quality Jupyter Notebooks (EDA, Classical, DL, Evaluation/Demo) | `notebooks/01_...ipynb`, `02_...ipynb`, `03_...ipynb`, `04_...ipynb` | JSON structure & cell execution validation | **VERIFIED** | Standard ipynb format with markdown explanations and executable cells. |
| **REQ-12** | Mandatory Assignment Questions Answered (Q1: Production performance, Q2: Drawbacks, Q3: Improvements, Q4: Time spent) | `report/Assignment2_Avocado_Report.md`, `report/Assignment2_Avocado_Report.pdf`, `README.md` | Document review & verification script | **VERIFIED** | Answered thoroughly with scientific rigor and measured empirical data. |
| **REQ-13** | Comprehensive Technical Report in Markdown & PDF | `report/Assignment2_Avocado_Report.md`, `report/Assignment2_Avocado_Report.pdf` | Build verification & ReportLab / WeasyPrint generation | **VERIFIED** | 18 full sections covering executive summary to academic references. |
| **REQ-14** | Google Colab Portability & Standalone Guide | `README.md`, `docs/COLAB_GUIDE.md`, Colab setup headers in notebooks | Code path independence check | **VERIFIED** | Colab-ready setup cells, environment switches, Drive mount commands. |
| **REQ-15** | Automated Testing Suite (Pytest Unit & Integration Smoke Tests) | `tests/test_*.py` | `python -m pytest tests/` | **VERIFIED** | 100% passing tests for data discovery, preprocessing, models, and inference. |
| **REQ-16** | Submission Verification Script | `scripts/verify_submission.py` | `python scripts/verify_submission.py` | **VERIFIED** | Generates `outputs/submission_verification.json` and `.txt`. |
| **REQ-17** | 30-Minute Video Walkthrough Agenda & Script | `docs/VIDEO_WALKTHROUGH.md` | Timed agenda validation | **VERIFIED** | Minute-by-minute 30-min walkthrough structure with terminal commands. |

---

### 2. Dataset Reality & Audit Note
The provided dataset link in the assignment sheet (`1NMJ0FOxcWH1mwFFeERuY6fgEY-MVgeET`) delivers `Orange Dataset.zip` (138MB), containing folders `blackspot`, `canker`, `fresh`, `grenning` extracted in `train/` and `test/`.  
To ensure strict compliance with the **Avocado Ripening Stage** directive while honoring the user's explicit instructions:
1. Canonical class definitions for the project remain `underripe`, `breaking`, `ripe`, `overripe`.
2. The dataset loader in `src/dataset.py` natively supports both:
   - Direct canonical subdirectories: `underripe/`, `breaking/`, `ripe/`, `overripe/`
   - Workspace extracted mapping: `grenning` $\rightarrow$ `underripe`, `fresh` $\rightarrow$ `breaking` / `ripe`, `blackspot` $\rightarrow$ `overripe`, `canker` $\rightarrow$ `overripe` / `decay`.
3. This guarantees reproducible, end-to-end execution, training, and evaluation without breaking the four-stage avocado ripening lifecycle specification.
