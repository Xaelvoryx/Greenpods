"""
Approach 1: Classical Computer Vision and Machine Learning.
Extracts handcrafted color (HSV, CIELAB) and texture (LBP) features.
Trains and evaluates a Support Vector Machine (SVM) classifier with feature scaling.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

from src.config import CLASSICAL_IMAGE_SIZE, CLASSICAL_MODEL_PATH, CLASSES


def extract_classical_features(image_input) -> np.ndarray:
    """
    Extract color and texture feature vector from an image file path or BGR numpy array.
    
    Features extracted:
    1. CIELAB Color Space: 16-bin L channel histogram + 16-bin a* channel histogram.
       (a* channel explicitly measures green-to-red/purple surface color shifts).
    2. HSV Color Space: 16-bin Hue histogram + 16-bin Saturation histogram.
    3. Texture: Uniform Local Binary Patterns (P=8, R=1) -> 10-bin normalized histogram.
    
    Total feature length: 16 + 16 + 16 + 16 + 10 = 74 dimensions.
    """
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        if img is None:
            raise ValueError(f"Could not read image from {image_input}")
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # Resize to canonical classical resolution
    img = cv2.resize(img, CLASSICAL_IMAGE_SIZE)

    # 1. CIELAB Color Space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hist_l = cv2.calcHist([lab], [0], None, [16], [0, 256]).flatten()
    hist_a = cv2.calcHist([lab], [1], None, [16], [0, 256]).flatten()
    hist_l /= (hist_l.sum() + 1e-7)
    hist_a /= (hist_a.sum() + 1e-7)

    # 2. HSV Color Space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    hist_h /= (hist_h.sum() + 1e-7)
    hist_s /= (hist_s.sum() + 1e-7)

    # 3. Local Binary Pattern (Texture)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist.astype(np.float32)
    lbp_hist /= (lbp_hist.sum() + 1e-7)

    feature_vector = np.concatenate([hist_l, hist_a, hist_h, hist_s, lbp_hist])
    return feature_vector


def build_classical_pipeline(c_param: float = 1.0, gamma: str = "scale") -> Pipeline:
    """Build StandardScaler + SVC pipeline with probability calibration."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            C=c_param,
            kernel="rbf",
            gamma=gamma,
            probability=True,
            class_weight="balanced",
            random_state=42,
        )),
    ])
    return pipeline


def save_classical_model(model_pipeline: Pipeline, save_path: Path = CLASSICAL_MODEL_PATH) -> None:
    """Serialize the trained classical pipeline and metadata."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pipeline": model_pipeline,
        "classes": CLASSES,
        "feature_dim": 74,
        "input_size": CLASSICAL_IMAGE_SIZE,
    }
    joblib.dump(payload, save_path)
    print(f"Classical model successfully saved to {save_path}")


def load_classical_model(model_path: Path = CLASSICAL_MODEL_PATH) -> Dict[str, Any]:
    """Load serialized classical pipeline artifact."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Classical model artifact not found at {model_path}")
    payload = joblib.load(model_path)
    return payload
