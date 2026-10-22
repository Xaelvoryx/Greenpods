"""
Unit and integration tests for dataset discovery, auditing, and health checks.
"""

from pathlib import Path
import pytest
from PIL import Image

from src.config import CLASSES, FOLDER_TO_CLASS_MAP
from src.data_audit import discover_image_files, audit_image_file


def test_classes_definition():
    """Verify canonical 4 ripening classes."""
    assert len(CLASSES) == 4
    assert CLASSES == ["underripe", "breaking", "ripe", "overripe"]


def test_folder_mapping():
    """Verify canonical and folder aliases map correctly."""
    assert FOLDER_TO_CLASS_MAP["underripe"] == "underripe"
    assert FOLDER_TO_CLASS_MAP["breaking"] == "breaking"
    assert FOLDER_TO_CLASS_MAP["ripe"] == "ripe"
    assert FOLDER_TO_CLASS_MAP["overripe"] == "overripe"
    assert FOLDER_TO_CLASS_MAP["grenning"] == "underripe"
    assert FOLDER_TO_CLASS_MAP["fresh"] == "breaking"


def test_audit_image_valid(tmp_path):
    """Test auditing a valid synthetic image."""
    img_path = tmp_path / "sample.jpg"
    img = Image.new("RGB", (100, 150), color=(50, 100, 150))
    img.save(img_path)

    audit = audit_image_file(str(img_path))
    assert audit["is_valid"] is True
    assert audit["width"] == 100
    assert audit["height"] == 150
    assert audit["channels"] == 3
    assert audit["mode"] == "RGB"
    assert audit["aspect_ratio"] == round(100 / 150, 3)
    assert audit["error"] is None
    assert audit["md5"] is not None


def test_audit_image_corrupted(tmp_path):
    """Test auditing a corrupted / empty image file."""
    bad_path = tmp_path / "corrupt.jpg"
    with open(bad_path, "wb") as f:
        f.write(b"not a valid image format header")

    audit = audit_image_file(str(bad_path))
    assert audit["is_valid"] is False
    assert audit["error"] is not None
