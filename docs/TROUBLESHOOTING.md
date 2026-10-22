# System Troubleshooting & Diagnostic Guide

## 1. PyTorch / CUDA Compatibility
- **Issue:** `CUDA out of memory` or `CUDA error: no kernel image is available`.
- **Solution:** Pass `--device cpu` or set `DEVICE=cpu` in `.env`. MobileNetV2 is lightweight and runs efficiently on CPU (~30-50ms per sample).

## 2. OpenCV Headless Environment Issues
- **Issue:** `ImportError: libGL.so.1: cannot open shared object file` (on Linux / headless servers).
- **Solution:** Install `opencv-python-headless` instead of `opencv-python`, or run:
  ```bash
  apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0
  ```

## 3. Dataset Discovery & Path Issues
- **Issue:** `ValueError: No records mapped to canonical classes`.
- **Solution:** Ensure your images reside inside either canonical class folders (`underripe/`, `breaking/`, `ripe/`, `overripe/`) or recognized workspace folders (`grenning/`, `fresh/`, `canker/`, `blackspot/`). Run:
  ```bash
  python scripts/audit_dataset.py
  ```

## 4. Missing Checkpoints During Inference
- **Issue:** `ARTIFACT_MISSING: models/classical/avocado_classical_model.joblib`.
- **Solution:** Execute the training script to generate model weights before running `main_demo.py`:
  ```bash
  python scripts/train_classical.py
  python scripts/train_deep_learning.py
  ```
