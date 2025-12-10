from __future__ import annotations

import logging
import re
from typing import List, Tuple

from .ocr_engine import OCRItem

logger = logging.getLogger(__name__)

WHITESPACE_RE = re.compile(r"\s+")
QUOTE_MAP = {"“": '"', "”": '"', "‘": "'", "’": "'"}
DATE_RE = re.compile(r"\b(\d{1,2})[\-/](\d{1,2})[\-/](\d{2,4})\b")


def _normalize_quotes(text: str) -> str:
    for k, v in QUOTE_MAP.items():
        text = text.replace(k, v)
    return text


def _normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def _normalize_dates(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        d, mth, y = m.groups()
        yi = int(y)
        if yi < 100:
            yi += 2000
        try:
            return f"{int(yi):04d}-{int(mth):02d}-{int(d):02d}"
        except Exception:
            return m.group(0)

    return DATE_RE.sub(repl, text)


def _merge_overlaps(items: List[OCRItem]) -> List[OCRItem]:
    """Merge duplicate/overlapping OCR boxes for similar text."""
    merged: List[OCRItem] = []
    for it in items:
        found = False
        for i, m in enumerate(merged):
            # simple text equality and bbox overlap heuristic
            if it.text.strip() == m.text.strip():
                bx1 = it.bbox
                bx2 = m.bbox
                x_min = min(bx1[0], bx2[0])
                y_min = min(bx1[1], bx2[1])
                x_max = max(bx1[2], bx2[2])
                y_max = max(bx1[3], bx2[3])
                merged[i] = OCRItem([x_min, y_min, x_max, y_max], m.text, max(m.confidence, it.confidence))
                found = True
                break
        if not found:
            merged.append(it)
    return merged


def clean_text(ocr_items: List[OCRItem]) -> Tuple[str, List[OCRItem]]:
    """Normalize OCR items and compose full text."""
    cleaned_items: List[OCRItem] = []
    for it in ocr_items:
        t = _normalize_quotes(it.text)
        t = _normalize_whitespace(t)
        t = _normalize_dates(t)
        cleaned_items.append(OCRItem(it.bbox, t, it.confidence))
    cleaned_items = _merge_overlaps(cleaned_items)

    full_text = "\n".join([it.text for it in cleaned_items])
    return full_text, cleaned_items
