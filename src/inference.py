"""
Single-image inference and demonstration pipeline.
Executes both Classical ML (CIELAB+LBP+SVM) and Deep Learning (MobileNetV2)
on a provided avocado image, computing predicted stage, probability distribution,
confidence score, and inference latency.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.config import (
    CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    DEVICE,
    CLASSICAL_MODEL_PATH,
    DL_MODEL_PATH,
)
from src.classical_model import load_classical_model, extract_classical_features
from src.dl_model import load_dl_model
from src.dataset import get_transforms


def predict_single_image(
    image_path: str,
    classical_model_path: Path = CLASSICAL_MODEL_PATH,
    dl_model_path: Path = DL_MODEL_PATH,
    device: str = DEVICE,
) -> Dict[str, Any]:
    """
    Run full dual-model inference on a single image.
    Validates input image readability and artifact availability gracefully.
    """
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Input image not found at: {image_path}")

    # Validate image can be opened
    try:
        pil_img = Image.open(p).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to decode image from {image_path}: {e}")

    results = {
        "image_path": str(p.resolve()),
        "image_name": p.name,
        "classical": {"status": "NOT_RUN", "prediction": None, "confidence": None, "probabilities": {}, "latency_ms": None},
        "deep_learning": {"status": "NOT_RUN", "prediction": None, "confidence": None, "probabilities": {}, "latency_ms": None},
    }

    # 1. Classical Model Inference
    if Path(classical_model_path).exists():
        try:
            t0 = time.perf_counter()
            classical_artifact = load_classical_model(classical_model_path)
            pipeline = classical_artifact["pipeline"]
            feats = extract_classical_features(str(p)).reshape(1, -1)
            pred_idx = pipeline.predict(feats)[0]
            pred_class = IDX_TO_CLASS[pred_idx] if isinstance(pred_idx, (int, np.integer)) else pred_idx
            
            # Predict probabilities if supported
            probs = {}
            confidence = 1.0
            if hasattr(pipeline, "predict_proba"):
                prob_vec = pipeline.predict_proba(feats)[0]
                confidence = float(np.max(prob_vec))
                probs = {CLASSES[i]: round(float(prob_vec[i]), 4) for i in range(len(CLASSES))}

            latency_ms = (time.perf_counter() - t0) * 1000.0
            results["classical"] = {
                "status": "SUCCESS",
                "prediction": pred_class,
                "confidence": round(confidence, 4),
                "probabilities": probs,
                "latency_ms": round(latency_ms, 2),
            }
        except Exception as e:
            results["classical"] = {"status": f"ERROR: {e}", "prediction": None, "confidence": None}
    else:
        results["classical"]["status"] = f"ARTIFACT_MISSING: {classical_model_path}"

    # 2. Deep Learning Inference (MobileNetV2)
    if Path(dl_model_path).exists():
        try:
            t0 = time.perf_counter()
            model = load_dl_model(dl_model_path, device=device)
            val_transform = get_transforms(split="val")
            tensor = val_transform(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(tensor)
                probs_tensor = F.softmax(logits, dim=1).squeeze(0)
                pred_idx = torch.argmax(probs_tensor).item()
                pred_class = IDX_TO_CLASS[pred_idx]
                confidence = probs_tensor[pred_idx].item()
                prob_dict = {
                    CLASSES[i]: round(probs_tensor[i].item(), 4)
                    for i in range(len(CLASSES))
                }

            if device == "cuda":
                torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - t0) * 1000.0

            results["deep_learning"] = {
                "status": "SUCCESS",
                "prediction": pred_class,
                "confidence": round(confidence, 4),
                "probabilities": prob_dict,
                "latency_ms": round(latency_ms, 2),
            }
        except Exception as e:
            results["deep_learning"] = {"status": f"ERROR: {e}", "prediction": None, "confidence": None}
    else:
        results["deep_learning"]["status"] = f"ARTIFACT_MISSING: {dl_model_path}"

    return results
