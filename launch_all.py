"""
One-Click Master Launcher: launch_all.py
Runs:
1. Automated unit test suite (pytest)
2. Submission deliverable verification
3. Model evaluation benchmarks on the held-out test set
4. Live dual-model prediction demo
5. Launches Jupyter Notebook directly in the default web browser opening the notebooks directory.
"""

import sys
import subprocess
import webbrowser
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

def run_step(step_name, command):
    print("\n" + "=" * 65)
    print(f" >>> {step_name}")
    print("=" * 65)
    res = subprocess.run(command, shell=True)
    if res.returncode != 0:
        print(f"Warning: {step_name} exited with code {res.returncode}")
    else:
        print(f"✓ {step_name} completed successfully.")

def main():
    print("*" * 65)
    print(" AVOCADO RIPENING SYSTEM: COMPLETE ALL-IN-ONE RUNNER")
    print("*" * 65)

    # 1. Run Automated Unit & Integration Tests
    run_step("Running Automated Pytest Test Suite", [sys.executable, "-m", "pytest", "-q"])

    # 2. Run Submission Verification Audit
    run_step("Verifying All Deliverables & Reports", [sys.executable, "scripts/verify_submission.py"])

    # 3. Run Benchmark Evaluation
    run_step("Evaluating Models on Held-Out Test Set", [sys.executable, "scripts/evaluate_models.py"])

    # 4. Run Live Single-Image Inference Demo
    sample_img = "test/grenning/g (1).jpg"
    if not (PROJECT_ROOT / sample_img).exists():
        # Fallback to first available test image
        img_files = list(PROJECT_ROOT.glob("test/*/*.*"))
        if img_files:
            sample_img = str(img_files[0].relative_to(PROJECT_ROOT))

    run_step("Running Live Dual-Model Prediction Demo", [sys.executable, "main_demo.py", "--image", sample_img, "--save-vis"])

    # 5. Open Jupyter Notebooks in Browser
    print("\n" + "=" * 65)
    print(" >>> Launching Jupyter Notebook Server & Opening in Browser...")
    print("=" * 65)
    print("Press Ctrl+C in this terminal when you are finished to stop the server.\n")

    # Start Jupyter pointing directly at the notebooks/ folder and open browser
    subprocess.run([sys.executable, "-m", "notebook", "--notebook-dir=notebooks"])

if __name__ == "__main__":
    main()
