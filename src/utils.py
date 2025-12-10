from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


def setup_logging(output_dir: Path) -> None:
    """Configure logging to file and console.

    Avoid logging PII by default; use debug level cautiously.
    """
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "pipeline.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    fh.setFormatter(formatter)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    # Reset handlers to avoid duplicates across multiple runs
    logger.handlers = []
    logger.addHandler(fh)
    logger.addHandler(ch)


@dataclass
class BBox:
    """Axis-aligned bounding box in image coordinates."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def to_list(self) -> List[int]:
        return [self.x_min, self.y_min, self.x_max, self.y_max]

    @staticmethod
    def from_quad(quad: Iterable[Tuple[int, int]]) -> "BBox":
        xs = [p[0] for p in quad]
        ys = [p[1] for p in quad]
        return BBox(min(xs), min(ys), max(xs), max(ys))

    def union(self, other: "BBox") -> "BBox":
        return BBox(
            min(self.x_min, other.x_min),
            min(self.y_min, other.y_min),
            max(self.x_max, other.x_max),
            max(self.y_max, other.y_max),
        )

    def area(self) -> int:
        return max(0, self.x_max - self.x_min) * max(0, self.y_max - self.y_min)

    def iou(self, other: "BBox") -> float:
        ix_min = max(self.x_min, other.x_min)
        iy_min = max(self.y_min, other.y_min)
        ix_max = min(self.x_max, other.x_max)
        iy_max = min(self.y_max, other.y_max)
        inter = max(0, ix_max - ix_min) * max(0, iy_max - iy_min)
        union = self.area() + other.area() - inter
        return inter / union if union > 0 else 0.0


def ensure_dirs(base: Path) -> dict[str, Path]:
    json_dir = base / "json"
    redacted_dir = base / "redacted"
    logs_dir = base / "logs"
    json_dir.mkdir(parents=True, exist_ok=True)
    redacted_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return {"json": json_dir, "redacted": redacted_dir, "logs": logs_dir}


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    img = np.clip(image, 0, 255).astype(np.uint8)
    return img


def quad_to_bbox(quad: List[List[int]]) -> BBox:
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return BBox(min(xs), min(ys), max(xs), max(ys))


def merge_bboxes(bboxes: List[BBox], iou_threshold: float = 0.3) -> List[BBox]:
    """Merge overlapping bboxes using IoU threshold."""
    merged: List[BBox] = []
    for b in bboxes:
        found = False
        for i, m in enumerate(merged):
            if b.iou(m) >= iou_threshold:
                merged[i] = m.union(b)
                found = True
                break
        if not found:
            merged.append(b)
    return merged


def is_jpeg(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg"}


def list_images(input_path: Path) -> List[Path]:
    if input_path.is_dir():
        return [p for p in sorted(input_path.iterdir()) if is_jpeg(p)]
    elif input_path.is_file() and is_jpeg(input_path):
        return [input_path]
    return []
