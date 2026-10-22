# Experiment Protocol & Scientific Benchmarking Standard

## 1. Experimental Objectives
Compare the generalization capacity, computational efficiency, and operational trade-offs between:
1. **Approach 1:** Domain-engineered classical CV features (CIELAB color space + LBP micro-texture) paired with an SVM classifier.
2. **Approach 2:** Hierarchical deep feature representation via MobileNetV2 transfer learning.

## 2. Partitioning Strategy
- **Stratified Partitioning:** Exactly 70% Train, 15% Validation, 15% Test.
- **Random Seed:** 42 across Python, NumPy, PyTorch, and scikit-learn.
- **Manifest Tracking:** Every partition is indexed with filename, path, MD5 hash, and ground-truth label. Test partitions are strictly held-out and evaluated once at the conclusion of training.

## 3. Metrics Definition
- **Overall Accuracy:** $\frac{\text{Correct Predictions}}{\text{Total Samples}}$
- **Macro Precision:** Arithmetic mean of per-class precisions, treating all classes with equal priority.
- **Macro Recall:** Arithmetic mean of per-class recalls. Critical to ensure ripe avocados are not missed.
- **Macro F1-Score:** Harmonic mean of macro precision and macro recall.
- **Inference Latency:** Average execution duration over 50 single-image forward passes with warmup runs.
