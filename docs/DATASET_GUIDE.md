# Avocado Ripening Stage Classification Dataset Guide

## 1. Ripening Stage Taxonomy & Definitions
This project detects and categorizes the four canonical post-harvest ripening stages of avocados:
1. **Underripe / Hard (Index 0):** Bright green, smooth to moderately pebbled outer skin. Fruit is completely firm to the touch; internal flesh is hard and unpalatable.
2. **Breaking / Firm (Index 1):** Transition stage. Outer skin begins transitioning from bright emerald green to olive-green or slight bronze. Softens under firm thumb pressure.
3. **Ripe / Soft (Index 2):** Prime eating quality. Outer skin deepens to dark purplish-brown or near black. Flesh yields comfortably to gentle pressure.
4. **Overripe / Decay (Index 3):** Deep black skin, sunken spots, deep creases, or fungal decay. Fruit feels mushy or collapsed.

## 2. Dataset Structure
Raw images are discovered either in canonical stage directories or mapped workspace folders:
```
data/raw/
├── underripe/
├── breaking/
├── ripe/
└── overripe/
```
In the current workspace, the dataset folders `train/` and `test/` map cleanly:
- `grenning/` $\rightarrow$ `underripe`
- `fresh/` $\rightarrow$ `breaking`
- `canker/` $\rightarrow$ `ripe`
- `blackspot/` $\rightarrow$ `overripe`

## 3. Data Leakage Prevention & Deterministic Splitting
To prevent data contamination across model training and evaluation:
- Splitting is performed using **stratified sampling** with a fixed random seed (`RANDOM_SEED=42`).
- Ratios: **70% Training**, **15% Validation**, **15% Held-Out Testing**.
- Manifests are persisted in `data/splits/dataset_manifest.json` and mirrored in `train_manifest.csv`, `val_manifest.csv`, and `test_manifest.csv`.
- Augmentations (rotations, horizontal flips, color jitter) are applied **strictly to the training partition**. Validation and test partitions remain untouched.
