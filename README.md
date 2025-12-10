# Handwritten OCR + PII Extraction Pipeline

> Quick validation: The `samples/` folder contains 3 handwritten JPEGs that were processed successfully. Their JSON outputs are available under `out/json/`, with optional redacted images in `out/redacted/`. This confirms the end-to-end pipeline (preprocessing → OCR → cleaning → PII detection → JSON + redaction) runs correctly.

Example run that produced these outputs:

```bash
python -m src.cli --input samples/ --output-dir out --redact --lang en --min-confidence 0.6 --save-debug
```

You can open `out/json/` to review per-image results and `out/report.md` for the summary.

This repository provides a production-ready, modular Python pipeline that performs OCR on handwritten JPEG images, cleans text, detects PII (Personally Identifiable Information), and optionally produces redacted images. It includes a CLI, tests, and clear outputs per image.

## Features
- Preprocessing: deskew, denoise, contrast enhancement (CLAHE), adaptive thresholding, resize, optional perspective correction.
- OCR: EasyOCR primary with optional pytesseract fallback.
- Text Cleaning: whitespace normalization, quote unification, duplicate removal, date normalization.
- PII Detection: robust regexes, spaCy NER fallback, fuzzy matching to handle OCR noise.
- Redaction: black boxes over detected PII with PNG output.
- CLI: batch processing of a folder with progress logging and parallel workers.

## Quick Start

### 1) Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Windows users using bash should ensure `tesseract` is installed if you plan to use pytesseract fallback (optional).

### 2) Run the CLI
Place handwritten JPEGs in `samples/`, then run:
```bash
python -m src.cli --input samples/ --output-dir out --redact --lang en --min-confidence 0.6 --save-debug
```

Outputs:
- JSON per image in `out/json/`
- Optional redacted PNG in `out/redacted/`
- Log files in `out/logs/`
- A summary report in `out/report.md`

### JSON Schema (example)
```json
{
  "image_filename": "sample1.jpg",
  "ocr_text": "full raw OCR text ...",
  "cleaned_text": "normalized text ...",
  "pii": [
    {
      "type": "PHONE",
      "text": "+1-555-123-4567",
      "confidence": 0.92,
      "bbox": [100, 50, 240, 80]
    }
  ],
  "redacted_image": "sample1_redacted.png",
  "metadata": {
    "processed_at": "2025-12-10T12:34:56Z",
    "preprocessing_steps": ["deskew","clahe","adaptive-threshold"],
    "ocr_engine": "easyocr",
    "duration_seconds": 1.23
  }
}
```

## Tips for Better Handwriting OCR
- Scan at higher DPI (300+), good lighting, minimal shadows.
- Keep forms flat; avoid heavy perspective distortion.
- Use `--save-debug` to review preprocessing steps.

## Development
- Run tests: `pytest -q`
- Linting/styles can be added; code includes type hints and docstrings.

## Swapping OCR Engine
`src/ocr_engine.py` is designed as a thin wrapper. You can implement a new `ocr_image` that returns the same `OCRItem` structure to integrate cloud OCR services.

## License
This project is provided as-is, without warranty. Intended for educational and practical demonstrations. Review local laws and regulations when handling PII.
