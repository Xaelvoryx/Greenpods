"""
Script 7: Verify Submission (scripts/verify_submission.py)
Automated validation of all assignment deliverables:
1. File structure & required directories
2. Four valid Jupyter notebooks
3. Requirements, README, pyproject, and docs
4. Technical reports (Markdown & PDF)
5. Trained models, evaluation metrics, and split manifests
6. CLI inference demonstration entry point
7. Automated tests presence
Outputs structured verification report: outputs/submission_verification.json and .txt
"""

import sys
import json
from pathlib import Path
import nbformat

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CLASSICAL_MODEL_PATH,
    DL_MODEL_PATH,
    SPLIT_MANIFEST_PATH,
    EVALUATION_OUTPUTS_DIR,
    EDA_OUTPUTS_DIR,
    OUTPUTS_DIR,
)


def verify_notebook(nb_path: Path) -> dict:
    """Check if notebook exists and is valid JSON."""
    res = {"path": str(nb_path), "status": "FAIL", "cells": 0, "error": None}
    if not nb_path.exists():
        res["error"] = "File does not exist"
        return res
    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        res["cells"] = len(nb.cells)
        res["status"] = "PASS" if len(nb.cells) > 0 else "FAIL"
    except Exception as e:
        res["error"] = str(e)
    return res


def main():
    print("=" * 65)
    print(" AVOCADO RIPENING PROJECT SUBMISSION AUDITOR")
    print("=" * 65)

    checklist = {}
    all_passed = True

    # 1. Check Notebooks
    notebooks = [
        "notebooks/01_eda_and_data_cleaning.ipynb",
        "notebooks/02_approach_1_classical_ml.ipynb",
        "notebooks/03_approach_2_deep_learning.ipynb",
        "notebooks/04_model_evaluation_and_demo.ipynb",
    ]
    nb_results = {}
    for nb in notebooks:
        p = PROJECT_ROOT / nb
        res = verify_notebook(p)
        nb_results[nb] = res
        if res["status"] != "PASS":
            all_passed = False
    checklist["notebooks"] = nb_results

    # 2. Check Core Python Source Modules
    src_modules = [
        "src/__init__.py",
        "src/config.py",
        "src/reproducibility.py",
        "src/data_audit.py",
        "src/preprocessing.py",
        "src/dataset.py",
        "src/classical_model.py",
        "src/dl_model.py",
        "src/training.py",
        "src/evaluation.py",
        "src/inference.py",
        "src/visualization.py",
    ]
    src_results = {}
    for mod in src_modules:
        exists = (PROJECT_ROOT / mod).exists()
        src_results[mod] = "PASS" if exists else "FAIL"
        if not exists:
            all_passed = False
    checklist["src_modules"] = src_results

    # 3. Check CLI Entry Points & Scripts
    scripts = [
        "main_demo.py",
        "scripts/audit_dataset.py",
        "scripts/prepare_dataset.py",
        "scripts/train_classical.py",
        "scripts/train_deep_learning.py",
        "scripts/evaluate_models.py",
        "scripts/run_all.py",
        "scripts/verify_submission.py",
    ]
    script_results = {}
    for sc in scripts:
        exists = (PROJECT_ROOT / sc).exists()
        script_results[sc] = "PASS" if exists else "FAIL"
        if not exists:
            all_passed = False
    checklist["scripts"] = script_results

    # 4. Check Documentation & Report
    docs = [
        "README.md",
        "requirements.txt",
        "pyproject.toml",
        "LICENSE",
        "docs/REQUIREMENTS_TRACEABILITY.md",
        "docs/VIDEO_WALKTHROUGH.md",
        "docs/MODEL_CARD.md",
        "docs/EXPERIMENT_PROTOCOL.md",
        "docs/TROUBLESHOOTING.md",
        "report/Assignment2_Avocado_Report.md",
        "report/Assignment2_Avocado_Report.pdf",
    ]
    doc_results = {}
    for doc in docs:
        exists = (PROJECT_ROOT / doc).exists()
        doc_results[doc] = "PASS" if exists else "FAIL"
        if not exists:
            all_passed = False
    checklist["documentation"] = doc_results

    # 5. Check Test Suite
    tests = [
        "tests/test_dataset.py",
        "tests/test_preprocessing.py",
        "tests/test_classical_model.py",
        "tests/test_dl_model.py",
        "tests/test_evaluation.py",
        "tests/test_inference.py",
        "tests/test_submission.py",
    ]
    test_results = {}
    for t in tests:
        exists = (PROJECT_ROOT / t).exists()
        test_results[t] = "PASS" if exists else "FAIL"
        if not exists:
            all_passed = False
    checklist["tests"] = test_results

    # 6. Check Artifacts (Models & Metrics)
    artifacts = [
        ("Classical Model (.joblib)", CLASSICAL_MODEL_PATH),
        ("Deep Learning Model (.pth)", DL_MODEL_PATH),
        ("Dataset Split Manifest", SPLIT_MANIFEST_PATH),
        ("Dataset Audit JSON", EDA_OUTPUTS_DIR / "dataset_audit.json"),
        ("Model Comparison CSV", EVALUATION_OUTPUTS_DIR / "model_comparison.csv"),
    ]
    artifact_results = {}
    for name, p in artifacts:
        exists = Path(p).exists()
        artifact_results[name] = "PASS" if exists else "PENDING / NOT RUN"
    checklist["artifacts"] = artifact_results

    # Export report
    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "submission_verification.json", "w", encoding="utf-8") as f:
        json.dump(checklist, f, indent=2)

    txt_lines = [
        "=" * 60,
        " SUBMISSION VERIFICATION AUDIT REPORT",
        "=" * 60,
        f"Overall Core Architecture Status: {'PASS' if all_passed else 'ACTION REQUIRED'}",
        "\n[Notebooks]:",
    ]
    for k, v in nb_results.items():
        txt_lines.append(f"  {k:<45}: {v['status']} ({v['cells']} cells)")
    txt_lines.append("\n[Core Modules]:")
    for k, v in src_results.items():
        txt_lines.append(f"  {k:<45}: {v}")
    txt_lines.append("\n[Documentation & Reports]:")
    for k, v in doc_results.items():
        txt_lines.append(f"  {k:<45}: {v}")
    txt_lines.append("\n[Generated Model & Data Artifacts]:")
    for k, v in artifact_results.items():
        txt_lines.append(f"  {k:<45}: {v}")
    txt_lines.append("=" * 60)

    report_txt = "\n".join(txt_lines)
    with open(out_dir / "submission_verification.txt", "w", encoding="utf-8") as f:
        f.write(report_txt)

    print(report_txt)


if __name__ == "__main__":
    main()
