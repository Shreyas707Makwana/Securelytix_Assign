# Demo Usage

Process a folder of handwritten JPEGs:

```bash
python -m src.cli --input samples/ --output-dir out --redact --lang en --min-confidence 0.6 --save-debug
```

Outputs:
- JSON files in `out/json/`
- Redacted images in `out/redacted/`
- Logs in `out/logs/`
- Summary report in `out/report.md`

Adjust workers for parallel processing:
```bash
python -m src.cli --input samples/ --output-dir out --workers 4
```
