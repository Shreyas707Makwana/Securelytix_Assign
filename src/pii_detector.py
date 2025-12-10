from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from rapidfuzz import fuzz
from .ocr_engine import OCRItem
from .utils import BBox, merge_bboxes

logger = logging.getLogger(__name__)

# Robust regex patterns
EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\b(?:\+\d{1,3}[\s-]?)?(?:\(\d{1,4}\)[\s-]?|\d{1,4}[\s-]?)?\d{3,4}[\s-]?\d{3,4}(?:[\s-]?\d{3,4})?\b")
CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
AADHAAR_RE = re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b")
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
PASSPORT_RE = re.compile(r"\b[A-Z]{1}\d{7}\b")
URL_RE = re.compile(r"\bhttps?://[\w.-/]+\b")
DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")

PII_TYPES = {
    "EMAIL": EMAIL_RE,
    "PHONE": PHONE_RE,
    "CARD": CARD_RE,
    "SSN": SSN_RE,
    "AADHAAR": AADHAAR_RE,
    "IBAN": IBAN_RE,
    "PASSPORT": PASSPORT_RE,
    "URL": URL_RE,
    "DATE": DATE_RE,
}

@dataclass
class PIIItem:
    type: str
    text: str
    confidence: float
    bbox: List[int]


def _spacy_ners(text: str) -> List[PIIItem]:
    try:
        import spacy  # type: ignore

        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        items: List[PIIItem] = []
        for ent in doc.ents:
            if ent.label_ in {"PERSON", "ORG", "GPE", "DATE"}:
                items.append(PIIItem(ent.label_, ent.text, 0.7, [0, 0, 0, 0]))
        return items
    except Exception as e:
        logger.warning("spaCy NER unavailable: %s", e)
        return []


def detect_pii(items: List[OCRItem], min_confidence: float = 0.5, use_high_conf_only: bool = False) -> List[PIIItem]:
    """Detect PII from OCR items using regex + NER + fuzzy matching."""
    detections: List[PIIItem] = []

    # Regex pass per item
    for it in items:
        text = it.text
        for typ, pattern in PII_TYPES.items():
            for m in pattern.finditer(text):
                bbox = it.bbox
                conf = max(it.confidence, 0.6)
                if conf >= min_confidence:
                    detections.append(PIIItem(typ, m.group(0), conf, bbox))
        # Fuzzy phone detection: normalize digits
        digits = re.sub(r"\D", "", text)
        if 10 <= len(digits) <= 15:
            score = fuzz.partial_ratio(digits, digits) / 100.0
            conf = max(it.confidence, 0.5 * score)
            if conf >= min_confidence:
                detections.append(PIIItem("PHONE", text, conf, it.bbox))

    # NER fallback on combined text
    combined_text = "\n".join([it.text for it in items])
    detections.extend(_spacy_ners(combined_text))

    # Merge bboxes for same text/type
    merged: List[PIIItem] = []
    grouped: dict[tuple[str, str], List[BBox]] = {}
    texts: dict[tuple[str, str], float] = {}
    for d in detections:
        key = (d.type, d.text)
        if d.bbox != [0, 0, 0, 0]:
            grouped.setdefault(key, []).append(BBox(*d.bbox))
        texts[key] = max(texts.get(key, 0.0), d.confidence)
    for key, bbs in grouped.items():
        mb = merge_bboxes(bbs)
        for bb in mb:
            merged.append(PIIItem(key[0], key[1], texts[key], bb.to_list()))
    # Add NER-only items without bbox
    for d in detections:
        if d.bbox == [0, 0, 0, 0]:
            merged.append(d)

    if use_high_conf_only:
        merged = [m for m in merged if m.confidence >= max(0.75, min_confidence)]

    return merged
