# Free historical newspaper OCR — prototype v0.1

No paid OCR services, accounts, or API keys. This prototype uses Tesseract and OpenCV locally. It is an **experimental baseline**, not a validated reading-order reconstruction algorithm. It may misidentify columns and article boundaries. Crucially, it does not yet include an LLM verifier; that can be added after evaluating the OCR baseline.

## Install on macOS

```bash
brew install tesseract
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download three authentic 1907 newspaper pages

```bash
python download_samples.py
```

Source: Library of Congress, *The Evening Star* (Washington, DC), September 6, December 7, and December 12, 1907.

- https://www.loc.gov/resource/sn83045462/1907-09-06/ed-1/?sp=1
- https://www.loc.gov/resource/sn83045462/1907-12-07/ed-1/?sp=1
- https://www.loc.gov/resource/sn83045462/1907-12-12/ed-1/?sp=1

If the archive rejects automated downloads, download JPGs manually into `samples/`.

```bash
python pipeline.py samples/*.jpg
```

Or use your own image:

```bash
python pipeline.py /path/to/newspaper.jpg
```

For each image, inspect `outputs/<image_name>/`:

- `baseline.txt`: direct, standard Tesseract OCR
- `column_order.txt`: heuristically segmented transcription
- `regions.json`: region coordinates, words, OCR confidence, source
- `layout_overlay.jpg`: visualize estimated columns

**Quality limitation:** The column detector is geometric. It does not reliably recover articles, notes, or complex reading order. Even high OCR confidence does not prove transcription correctness. Compare with human-corrected references using CER/WER before claiming accuracy.

## Next iteration

1. Manually annotate 10–20 pages (column boundaries, article IDs, reading order).
2. Benchmark Tesseract PSM modes and region-based recognition.
3. Integrate a free pretrained layout model, subject to license and hardware constraints.
4. Add local vision-language verification and keep original OCR separate from corrections.
5. Add a researcher review interface and NVivo export.
