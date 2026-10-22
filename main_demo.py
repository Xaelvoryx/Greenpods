"""
CLI Demonstration Entry Point: main_demo.py
Accepts an image path, executes both Classical ML and Deep Learning MobileNetV2 models,
displays confidence breakdown, probabilities, latency, and optionally saves visualization.
"""

import argparse
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CLASSES,
    CLASSICAL_MODEL_PATH,
    DL_MODEL_PATH,
    DEVICE,
    PREDICTIONS_OUTPUTS_DIR,
)
from src.inference import predict_single_image


def save_prediction_visualization(image_path: str, results: dict, output_path: Path) -> None:
    """Save an annotated visual showing the image and both model predictions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(image_path).convert("RGB")
    
    # Scale image to comfortable display size
    img.thumbnail((500, 500))
    canvas_w, canvas_h = img.width, img.height + 120
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(245, 245, 245))
    canvas.paste(img, (0, 0))

    draw = ImageDraw.Draw(canvas)
    text_y = img.height + 10

    # Draw titles and predictions
    c_pred = results["classical"].get("prediction", "N/A")
    c_conf = results["classical"].get("confidence", 0.0)
    c_lat = results["classical"].get("latency_ms", 0.0)

    dl_pred = results["deep_learning"].get("prediction", "N/A")
    dl_conf = results["deep_learning"].get("confidence", 0.0)
    dl_lat = results["deep_learning"].get("latency_ms", 0.0)

    line1 = f"Classical (SVM): {c_pred} (Conf: {c_conf*100:.1f}%, {c_lat:.1f}ms)"
    line2 = f"Deep Learning (MobileNetV2): {dl_pred} (Conf: {dl_conf*100:.1f}%, {dl_lat:.1f}ms)"

    draw.text((15, text_y), line1, fill=(20, 60, 120))
    draw.text((15, text_y + 35), line2, fill=(20, 120, 60))

    canvas.save(output_path)
    print(f"Annotated prediction visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Avocado Ripening Stage Detection & Classification - Dual-Model Inference CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", type=str, required=True, help="Path to the avocado image file to classify")
    parser.add_argument("--classical-model", type=str, default=str(CLASSICAL_MODEL_PATH), help="Path to classical model joblib artifact")
    parser.add_argument("--dl-model", type=str, default=str(DL_MODEL_PATH), help="Path to MobileNetV2 checkpoint artifact")
    parser.add_argument("--output-dir", type=str, default=str(PREDICTIONS_OUTPUTS_DIR), help="Output directory to save predictions")
    parser.add_argument("--save-vis", action="store_true", help="Save visual annotation of the prediction")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Provided image file '{args.image}' does not exist.")
        sys.exit(1)

    print("=" * 65)
    print(" AVOCADO RIPENING STAGE INFERENCE DEMO (DUAL-MODEL ENGINE)")
    print("=" * 65)
    print(f"Target Image: {image_path.name}")
    print(f"Device:       {DEVICE}")

    results = predict_single_image(
        str(image_path),
        classical_model_path=Path(args.classical_model),
        dl_model_path=Path(args.dl_model),
        device=DEVICE,
    )

    print("\n--- Model 1: Classical Machine Learning (CIELAB + LBP + SVM) ---")
    c_res = results["classical"]
    print(f"Status:      {c_res['status']}")
    if c_res["status"] == "SUCCESS":
        print(f"Prediction:  {c_res['prediction'].upper()}")
        print(f"Confidence:  {c_res['confidence']*100:.2f}%")
        print(f"Latency:     {c_res['latency_ms']:.2f} ms")
        if c_res["probabilities"]:
            print("Probabilities:")
            for cls, p in c_res["probabilities"].items():
                print(f"  - {cls:<10}: {p*100:5.2f}%")
    else:
        print(f"Details:     {c_res.get('status')}")

    print("\n--- Model 2: Deep Learning (MobileNetV2 Transfer Learning) ---")
    dl_res = results["deep_learning"]
    print(f"Status:      {dl_res['status']}")
    if dl_res["status"] == "SUCCESS":
        print(f"Prediction:  {dl_res['prediction'].upper()}")
        print(f"Confidence:  {dl_res['confidence']*100:.2f}%")
        print(f"Latency:     {dl_res['latency_ms']:.2f} ms")
        if dl_res["probabilities"]:
            print("Probabilities:")
            for cls, p in dl_res["probabilities"].items():
                print(f"  - {cls:<10}: {p*100:5.2f}%")
    else:
        print(f"Details:     {dl_res.get('status')}")

    print("=" * 65)

    if args.save_vis:
        out_vis_path = Path(args.output_dir) / f"demo_pred_{image_path.stem}.png"
        save_prediction_visualization(str(image_path), results, out_vis_path)


if __name__ == "__main__":
    main()
