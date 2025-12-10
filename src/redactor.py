from __future__ import annotations

import logging
from typing import Iterable

import cv2
import numpy as np

from .pii_detector import PIIItem

logger = logging.getLogger(__name__)


def redact_image(image: np.ndarray, pii_items: Iterable[PIIItem]) -> np.ndarray:
    """Return image with PII regions blacked out."""
    out = image.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    for item in pii_items:
        bx = item.bbox
        if len(bx) == 4 and any(bx):
            x_min, y_min, x_max, y_max = bx
            cv2.rectangle(out, (x_min, y_min), (x_max, y_max), (0, 0, 0), thickness=-1)
    return out
