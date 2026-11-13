# Newspaper OCR Lab

A **free, local-first research prototype** for batch transcription of scanned historical newspapers. Upload multiple PDFs/images, detect candidate text regions, OCR each region independently, correct text/order in the browser, and download one ZIP containing DOCX, TXT, and structured JSON for every input.

**Important:** This release fixes the earlier *cross-column mixing* problem on synthetic tests by performing **layout segmentation before OCR**. Layout detection uses an OpenCV **XY-cut whitespace algorithm**, not a pretrained newspaper-article model. On real historical newspapers, headline boundaries, illustrations, marginal notes, and article continuations may still be incorrectly grouped. Do not assume transcriptions are research-ready until reviewed.

## Installation

macOS:

```bash
git clone https://github.com/thinoj01/layout-aware-OCR-pipeline-with-LLM-assisted-verification..git
cd layout-aware-OCR-pipeline-with-LLM-assisted-verification.
brew install tesseract
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Ubuntu: `sudo apt-get install tesseract-ocr`. For French historical material, install the French language pack and set the application language to `fra` or `eng+fra`.

## Using the interface

1. Upload multiple PDFs or JPG/PNG/TIFF/WebP files.
2. Choose language and PDF rendering resolution.
3. Click **Transcribe all**.
4. Inspect the region boxes drawn on the original page preview. Region numbers represent **proposed** reading order.
5. Edit each region's transcription, reading order and optional article ID. These manual changes are included in the ZIP.
6. Download the ZIP: one `transcription.txt`, `transcription.docx`, and `structured.json` per uploaded file, plus `manifest.json`.

The JSON records original source filenames, page numbers, region bounding boxes, OCR confidences, manual article labels, and the proposed or revised region order. Article IDs are labels: they do **not** automatically create separate per-article files yet.

## Optional OpenAI verification

OCR is fully local and does not require payment. OpenAI verification is off by default, separately billed, and sends selected image crops to OpenAI. To enable, enter a key in the interface (password field) or set the `OPENAI_API_KEY` environment variable. **Never commit a key to GitHub.** Obtain permission to send the source material externally. Model suggestions are displayed for inspection; they do not silently replace the OCR text.

## Architecture

- `layout.py`: open-source CPU-only OpenCV XY-cut layout regions and overlay
- `ocr_engine.py`: page rendering, preprocessing, independent region OCR, structured outputs, optional vision verifier
- `app.py`: Streamlit interface, manual correction and reordering, batch export
- `evaluate.py`: character/word error rate compared with human-corrected references
- `tests/test_engine.py`: smoke tests and regression tests
- `pipeline.py`: legacy command-line baseline (not the improved UI engine)

## Evaluation

```bash
pip install pytest
python -m pytest -q
python evaluate.py reference.txt predicted.txt
```

**Current synthetic regression tests:** five local tests passed after implementing the new engine. They cover basic OCR, multi-page PDFs, error handling, column separation and propagation of corrections to exports. The full GitHub UI has not been tested end-to-end on historical scans. CER and WER on the September 6, 1907 *Evening Star* are **not yet measured** because a manually corrected reference is needed.

## Next research engineering steps

- Evaluate 20–30 manually annotated historical newspaper pages, including the professors' worst examples.
- Benchmark a genuinely pretrained newspaper layout detector against this XY-cut baseline.
- Add article-linked continuation tracking and a reviewer interface for merge/split regions.
- Add persistent page-level job queues, resumability and large-batch load testing.
- Produce NVivo-ready documents and source metadata suitable for actual qualitative analysis.

**Do not represent this as a validated article-level OCR system.** It remains a working baseline with human review.
