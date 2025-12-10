from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pytest

from src.utils import BBox, merge_bboxes
from src.pii_detector import detect_pii, PIIItem
from src.ocr_engine import OCRItem
from src.pipeline import process_image


def test_merge_bboxes():
    b1 = BBox(0, 0, 10, 10)
    b2 = BBox(5, 5, 15, 15)
    merged = merge_bboxes([b1, b2], iou_threshold=0.1)
    assert len(merged) == 1
    assert merged[0].to_list() == [0, 0, 15, 15]


def test_regex_pii_detection():
    items = [
        OCRItem([0, 0, 10, 10], "john.doe@example.com", 0.9),
        OCRItem([10, 10, 20, 20], "+1-555-123-4567", 0.8),
        OCRItem([20, 20, 30, 30], "2025-12-10", 0.7),
    ]
    res = detect_pii(items, min_confidence=0.5)
    types = {r.type for r in res}
    assert "EMAIL" in types
    assert "PHONE" in types
    assert "DATE" in types


def test_process_image_with_mocked_ocr(tmp_path: Path, monkeypatch):
    # Create a dummy image
    img = np.full((200, 300, 3), 255, dtype=np.uint8)
    path = tmp_path / "sample.jpg"
    try:
        import cv2

        cv2.putText(img, "+1-555-123-4567", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.imwrite(str(path), img)
    except Exception:
        pytest.skip("OpenCV not available for test")

    # Mock OCR to return predictable items
    def mock_ocr_image(image, lang="en") -> List[OCRItem]:
        return [OCRItem([10, 80, 200, 110], "+1-555-123-4567", 0.9)]

    from src import ocr_engine

    monkeypatch.setattr(ocr_engine, "ocr_image", mock_ocr_image)

    out = tmp_path / "out"
    res = process_image(path, out, lang="en", min_confidence=0.5, redact=True, save_debug=False)
    assert (out / "json" / "sample.json").exists()
    assert (out / "redacted" / "sample_redacted.png").exists()
    data = json.loads((out / "json" / "sample.json").read_text(encoding="utf-8"))
    assert any(p["type"] == "PHONE" for p in data["pii"])  # PII detected
