# Model Card: Avocado Ripening Stage Classifiers

## Model Details
- **Developed by:** Senior Computer Vision & Machine Learning Engineering Team
- **Model Date:** October 2026
- **Model Types:**
  1. **Classical Computer Vision:** StandardScaler + Support Vector Machine (RBF Kernel, $C=2.0$, balanced class weights) on 74-dimensional concatenated CIELAB ($L, a^*$), HSV ($H, S$), and Local Binary Patterns (LBP) texture histograms.
  2. **Deep Learning Transfer Learning:** MobileNetV2 with ImageNet-pretrained convolutional backbone, frozen feature layers, and a custom 4-class classification head (`Linear(1280, 128) -> ReLU -> Dropout(0.2) -> Linear(128, 4)`).

## Intended Use
- **Primary Use Case:** Automated sorting, shelf-life monitoring, and grading in commercial post-harvest packaging facilities and retail fresh produce supply chains.
- **Out-of-Scope Use:** Internal fruit quality guarantee without non-destructive spectroscopic or firmness verification.

## Factors & Performance Considerations
- **Skin Color Sensitivity:** In varieties such as Hass, skin changes from green to dark purple/black as it ripens. However, green-skin avocado varieties (such as Fuerte, Reed, or Zutano) retain green pigmentation when ripe, causing color-only classifiers to produce false positives for underripe fruit.
- **Lighting & Specular Highlights:** Direct artificial lighting on conveyor belts causes specular glares that artificially inflate the CIELAB $L^*$ channel and distort $a^*$ color histograms.

## Evaluation Protocol
Models are benchmarked on the identical 15% held-out test split using:
- **Macro F1-Score:** Primary criterion to account for dataset class imbalance.
- **Inference Latency:** Average inference time per image on CPU and GPU.
- **Model Footprint:** Serialization size in megabytes.
