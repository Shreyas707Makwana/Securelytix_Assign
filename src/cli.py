from __future__ import annotations

import logging
from pathlib import Path

import typer

from .pipeline import run

app = typer.Typer(add_completion=False)


@app.command()
def main(
    input: str = typer.Option(..., help="Input JPEG file or folder"),
    output_dir: str = typer.Option("out", help="Output directory"),
    redact: bool = typer.Option(False, help="Produce redacted images"),
    lang: str = typer.Option("en", help="OCR language code"),
    min_confidence: float = typer.Option(0.5, help="Minimum confidence for PII"),
    save_debug: bool = typer.Option(False, help="Save preprocessing debug images"),
    workers: int = typer.Option(1, help="Number of parallel workers"),
):
    """Run the OCR + PII pipeline over input images."""
    input_path = Path(input)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logging.getLogger().setLevel(logging.INFO)

    run(
        input_path=input_path,
        output_dir=output_path,
        lang=lang,
        min_confidence=min_confidence,
        redact=redact,
        save_debug=save_debug,
        workers=workers,
    )


if __name__ == "__main__":
    app()
