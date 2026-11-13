# Newspaper OCR Lab

Local-first, batch transcription of newspaper PDFs and images using free Tesseract OCR and Python. The Streamlit interface allows researchers to upload multiple files and download transcriptions as DOCX, TXT and JSON in one ZIP.

**Current status:** functional prototype. It uses heuristic column grouping, *not* a validated article segmentation model. Historic newspaper layout, side notes, continuation links and reading order may be incorrect. OpenAI suggestions are not automatically substituted.

## Install (macOS)

```bash
git clone https://github.com/thinoj01/layout-aware-OCR-pipeline-with-LLM-assisted-verification..git
cd layout-aware-OCR-pipeline-with-LLM-assisted-verification.
brew install tesseract
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

If using French texts, install Tesseract's French language data and set the language code to `fra` or `eng+fra`. Linux: `sudo apt-get install tesseract-ocr`.

## Workflow

1. Open the URL shown by Streamlit (typically http://localhost:8501).
2. Upload multiple PDFs or PNG/JPEG/TIFF/WebP files.
3. Adjust PDF resolution and language; leave optional AI verification turned off for completely free offline processing.
4. Click **Transcribe all**.
5. Review OCR text and low-confidence regions.
6. Download the ZIP: each input becomes its own folder containing `transcription.txt`, `transcription.docx`, `structured.json`, plus a root `manifest.json`.

Source images are processed in memory; the app does not create a cloud service or persistent database. `structured.json` retains word coordinates, page number, OCR confidence and optional unreviewed verification proposals.

## Optional OpenAI verification

Not required. The OpenAI API is **separately billed** from a ChatGPT subscription. To enable it, choose the verification checkbox and enter your key in the password field, or set an environment variable before starting the app:

```bash
export OPENAI_API_KEY='your_key_goes_here'
streamlit run app.py
```

**Never add an API key to GitHub or commit it to source control.** When enabled, image crops are sent to OpenAI; get the professors' consent and check research/copyright requirements. A suggested transcription is **not automatically accepted**. The server may make multiple API calls per page, up to the limit chosen in the sidebar.

## Limitations and next research steps

- Tesseract is free, but handwriting, degraded scans, multiple articles in one column and margin notes are hard. Existing reading order heuristics are not a validated article reconstruction solution.
- There is currently no manual correction editor that persists changes to exports; the preview is read-only.
- No persistent queue, cancel/resume, user accounts, scalability benchmark, or production deployment.
- Large multi-page files consume RAM and can run slowly; the interface executes the batch synchronously.
- Measure character error rate, word error rate, region recall and article reading order against manually corrected research samples before claiming improved accuracy.
- Future improvements: train/test a free document layout model; create manual article grouping and correction UI; per-page job queue and resumable batches; NVivo-compatible metadata import.

## Tests

The source includes `tests/test_engine.py`. Run `python -m pytest -q` after installing `pytest`. Tests cover a synthetic two-column image, a two-page PDF, ZIP/DOCX/JSON output and a damaged PDF.
