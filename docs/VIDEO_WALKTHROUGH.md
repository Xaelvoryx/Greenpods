# 30-Minute Screen-Recorded Technical Walkthrough Plan

## Video Recording Metadata
- **Project:** Avocado Ripening Stage Detection & Classification System
- **Total Duration:** Exactly 30:00 minutes
- **Format:** Screen Recording with Voiceover Commentary and Live Terminal / Notebook Execution

---

## Timed Walkthrough Agenda & Demonstration Script

### Part 1: Project Introduction & Problem Framing (00:00 – 03:00 | 3 Mins)
- **Visual:** README.md and Title Slide.
- **Topics:**
  - Introduce Greenpod Labs Assignment-2: Post-harvest avocado ripening stage detection (`underripe`, `breaking`, `ripe`, `overripe`).
  - Industrial significance: Minimizing post-harvest losses, automating produce packaging conveyor grading, and preventing retail food waste.
  - Architectural overview: Dual-model comparison (Classical Feature Extraction + SVM vs. MobileNetV2 Deep Transfer Learning).

### Part 2: Repository Architecture & Engineering Design (03:00 – 05:30 | 2.5 Mins)
- **Visual:** Project directory tree in VS Code / IDE.
- **Topics:**
  - Modular organization: `src/`, `scripts/`, `notebooks/`, `models/`, `outputs/`, `report/`, `docs/`, `tests/`.
  - Separation of concerns: Reusable library modules vs. execution scripts vs. interactive notebooks.
  - Zero-leakage data partitioning manifests (`data/splits/`).

### Part 3: Dataset Acquisition, Audit & EDA Quality Analysis (05:30 – 09:30 | 4 Mins)
- **Visual:** Execute `python scripts/audit_dataset.py` and walkthrough `notebooks/01_eda_and_data_cleaning.ipynb`.
- **Topics:**
  - Review class distribution across the dataset.
  - Highlight class imbalance, lighting shifts, and specular reflections on fruit skins.
  - Explain the color theory of ripening: Chlorophyll breakdown causing skin transition from bright green to dark purple/black.
  - Inspect `outputs/eda/class_distribution.png` and `outputs/eda/representative_samples.png`.

### Part 4: Approach 1 — Classical Computer Vision & ML (09:30 – 14:00 | 4.5 Mins)
- **Visual:** `notebooks/02_approach_1_classical_ml.ipynb` and `src/classical_model.py`.
- **Topics:**
  - Color feature extraction: Why CIELAB color space is superior for avocado skin analysis (the $a^*$ channel measures green-to-red transition directly).
  - Texture analysis: Local Binary Patterns (LBP, $P=8, R=1$, uniform) to capture skin pebbled texture vs. smooth skins.
  - Pipeline fitting: `StandardScaler` + `SVC(kernel='rbf', probability=True, class_weight='balanced')`.
  - Live execution of `python scripts/train_classical.py`.

### Part 5: Approach 2 — Deep Learning with Transfer Learning (14:00 – 19:00 | 5 Mins)
- **Visual:** `notebooks/03_approach_2_deep_learning.ipynb` and `src/dl_model.py`.
- **Topics:**
  - MobileNetV2 architecture: Depthwise separable convolutions, inverted residuals, low memory footprint suitable for edge industrial sorting cameras.
  - Transfer learning setup: ImageNet-pretrained backbone with frozen feature extractors.
  - Training loop: CrossEntropyLoss, Adam optimizer, early stopping based on validation loss, and checkpoint saving.
  - Live execution of `python scripts/train_deep_learning.py` showing epoch logs and `training_curves.png`.

### Part 6: Model Evaluation, Comparison & Error Analysis (19:00 – 23:30 | 4.5 Mins)
- **Visual:** `notebooks/04_model_evaluation_and_demo.ipynb` and `scripts/evaluate_models.py`.
- **Topics:**
  - Quantitative comparison table: Accuracy, Macro Precision, Macro Recall, Macro F1, CPU Latency, Model Size.
  - Confusion matrix comparison: `outputs/evaluation/confusion_matrix_classical.png` vs. `outputs/evaluation/confusion_matrix_mobilenetv2.png`.
  - Error analysis: Borderline misclassifications between adjacent stages (`breaking` vs. `ripe`). Why texture features help resolve color ambiguities.

### Part 7: Live Single-Image CLI Inference Demo (23:30 – 26:30 | 3 Mins)
- **Visual:** Terminal executing `main_demo.py`.
- **Commands:**
  ```bash
  python main_demo.py --image data/splits/.../sample_1.jpg --save-vis
  python main_demo.py --image data/splits/.../sample_2.jpg --save-vis
  ```
- **Topics:**
  - Dual predictions output simultaneously.
  - Confidence scoring and latency benchmarking breakdown.
  - Handling of edge cases and unreadable images.

### Part 8: Answers to Mandatory Questions & Industrial Deployment (26:30 – 29:00 | 2.5 Mins)
- **Visual:** `docs/REQUIREMENTS_TRACEABILITY.md` and Report Section 16.
- **Topics:**
  - Q1: Production performance and throughput capabilities.
  - Q2: Model drawbacks (color ambiguity in non-Hass varieties, single camera surface view).
  - Q3: Future improvements (tactile firmness sensors, NIR spectroscopy, 360-degree rotation rollers, ONNX/TensorRT quantization).
  - Q4: Where development time was spent (40% feature engineering & color space tuning, 30% data cleaning, 30% model benchmarking).

### Part 9: Verification, Testing & Conclusion (29:00 – 30:00 | 1 Min)
- **Visual:** Terminal running `python scripts/verify_submission.py` and `pytest`.
- **Closing:** Summary of results, code quality, and final deliverable package.
