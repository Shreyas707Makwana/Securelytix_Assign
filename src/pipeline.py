from __future__ import annotations

import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .preprocessing import preprocess_image, preprocess_debug
from .ocr_engine import ocr_image, OCRItem
from .text_cleaning import clean_text
from .pii_detector import detect_pii, PIIItem
from .redactor import redact_image
from .utils import ensure_dirs, iso_utc_now, save_json, setup_logging

logger = logging.getLogger(__name__)


def process_image(
    image_path: Path,
    output_dir: Path,
    lang: str = "en",
    min_confidence: float = 0.5,
    redact: bool = False,
    save_debug: bool = False,
) -> Dict:
    start = time.time()
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    pre_img = preprocess_image(img_bgr)
    ocr_items: List[OCRItem] = ocr_image(pre_img, lang=lang)
    full_text, cleaned_items = clean_text(ocr_items)
    pii_items: List[PIIItem] = detect_pii(cleaned_items, min_confidence=min_confidence)

    json_out = output_dir / "json" / (image_path.stem + ".json")
    data = {
        "image_filename": image_path.name,
        "ocr_text": "\n".join([it.text for it in ocr_items]),
        "cleaned_text": full_text,
        "pii": [asdict(p) for p in pii_items],
        "redacted_image": None,
        "metadata": {
            "processed_at": iso_utc_now(),
            "preprocessing_steps": [
                "deskew",
                "clahe",
                "adaptive-threshold",
                "median-blur",
                "auto-crop",
                "perspective-correction",
            ],
            "ocr_engine": "easyocr",
            "duration_seconds": round(time.time() - start, 3),
        },
    }

    if redact:
        red_img = redact_image(pre_img, pii_items)
        red_path = output_dir / "redacted" / (image_path.stem + "_redacted.png")
        cv2.imwrite(str(red_path), red_img)
        data["redacted_image"] = red_path.name

    save_json(json_out, data)

    if save_debug:
        dbg_dir = output_dir / "debug" / image_path.stem
        dbg_dir.mkdir(parents=True, exist_ok=True)
        dbg_imgs = preprocess_debug(img_bgr)
        for k, v in dbg_imgs.items():
            cv2.imwrite(str(dbg_dir / f"{k}.png"), v)

    return data


def run(
    input_path: Path,
    output_dir: Path,
    lang: str = "en",
    min_confidence: float = 0.5,
    redact: bool = False,
    save_debug: bool = False,
    workers: int = 1,
) -> List[Dict]:
    dirs = ensure_dirs(output_dir)
    setup_logging(output_dir)

    imgs = []
    if input_path.is_dir():
        imgs = [p for p in sorted(input_path.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg"}]
    elif input_path.is_file():
        imgs = [input_path]
    else:
        raise RuntimeError("Invalid input path")

    results: List[Dict] = []
    logger.info("Processing %d images", len(imgs))

    if workers <= 1:
        for p in imgs:
            try:
                results.append(
                    process_image(p, output_dir, lang=lang, min_confidence=min_confidence, redact=redact, save_debug=save_debug)
                )
                logger.info("Processed %s", p.name)
            except Exception as e:
                logger.exception("Failed processing %s: %s", p, e)
    else:
        # Simple parallelism using concurrent.futures
        import concurrent.futures as cf

        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(process_image, p, output_dir, lang, min_confidence, redact, save_debug)
                for p in imgs
            ]
            for f in cf.as_completed(futs):
                try:
                    results.append(f.result())
                except Exception as e:
                    logger.exception("Worker failed: %s", e)

    # Write summary report
    report = output_dir / "report.md"
    total_pii = sum(len(r.get("pii", [])) for r in results)
    report.write_text(
        f"# PII Summary\n\nImages processed: {len(results)}\n\nTotal PII items: {total_pii}\n",
        encoding="utf-8",
    )

    return results
