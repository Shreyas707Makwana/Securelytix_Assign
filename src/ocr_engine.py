from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OCRItem:
    bbox: List[int]  # [x_min, y_min, x_max, y_max]
    text: str
    confidence: float


def _easyocr_reader(lang: str):
    try:
        import easyocr  # type: ignore

        return easyocr.Reader([lang], gpu=False)
    except Exception as e:
        logger.warning("EasyOCR unavailable: %s", e)
        return None


def _pytesseract_extract(image: np.ndarray) -> List[OCRItem]:
    """Fallback OCR using pytesseract.
    Returns coarse line-level boxes.
    """
    try:
        import pytesseract  # type: ignore
        from pytesseract import Output  # type: ignore

        d = pytesseract.image_to_data(image, output_type=Output.DICT)
        items: List[OCRItem] = []
        n = len(d.get("text", []))
        for i in range(n):
            txt = d["text"][i].strip()
            conf = float(d.get("conf", [0])[i])
            if txt:
                x = int(d["left"][i])
                y = int(d["top"][i])
                w = int(d["width"][i])
                h = int(d["height"][i])
                items.append(OCRItem([x, y, x + w, y + h], txt, max(0.0, conf / 100.0)))
        return items
    except Exception as e:
        logger.warning("pytesseract fallback failed: %s", e)
        return []


def ocr_image(image: np.ndarray, lang: str = "en") -> List[OCRItem]:
    """Perform OCR using EasyOCR with pytesseract fallback.

    Returns list of OCRItem with axis-aligned bbox and confidence.
    """
    reader = _easyocr_reader(lang)
    items: List[OCRItem] = []
    if reader is not None:
        try:
            results = reader.readtext(image)
            for (bbox_pts, text, conf) in results:
                xs = [p[0] for p in bbox_pts]
                ys = [p[1] for p in bbox_pts]
                x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)
                items.append(OCRItem([int(x_min), int(y_min), int(x_max), int(y_max)], text, float(conf)))
        except Exception as e:
            logger.warning("EasyOCR read failed: %s", e)
    # fallback if nothing
    if not items:
        items = _pytesseract_extract(image)
    return items
