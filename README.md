# Historical Newspaper OCR Lab — local research application

An experimental, **local-first** web application for batch transcription of scanned historical newspapers.

**What it supports:** multi-PDF and image upload, durable SQLite jobs, a separate background worker, pretrained DocLayout-YOLO or explicit XY-cut layout detection, region-wise Tesseract OCR, page previews, saved manual corrections and article labels, separate per-article DOCX/TXT output, JSON metadata, and optional image-grounded suggestions via local Ollama or OpenAI.

**Research status:** A tested software prototype, **not a validated high-accuracy historical newspaper transcriber**. On old, busy newspaper pages, a generic pretrained layout model may still miss regions or group articles incorrectly. No measured CER/WER exists yet for the 1907 *Evening Star* example because there is no manually corrected page-level reference.

## Quick start (Mac)

Use Python **3.11 or 3.12**, preferably in a new virtual environment:

```bash
git clone https://github.com/thinoj01/layout-aware-OCR-pipeline-with-LLM-assisted-verification..git
cd layout-aware-OCR-pipeline-with-LLM-assisted-verification.
brew install tesseract
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-ml.txt
python launch.py
```

Open **http://127.0.0.1:8501**. The app runs only on your computer. `launch.py` starts both the Streamlit web UI and the queue worker. Stop with Ctrl+C. For a lightweight test without the model, select **XY-cut** in the sidebar, which uses heuristic segmentation and requires no `requirements-ml.txt`.

**First pretrained use requires internet:** DocLayout-YOLO weights are downloaded from Hugging Face and cached. It does **not** incur an API charge. The model can use substantial RAM and may be slow on CPU.

For French scans, install Tesseract French trained data and select `fra` or `eng+fra`. Ubuntu can install Tesseract via `apt install tesseract-ocr`.

## How professors would use it

1. Upload several PDF or image newspapers.
2. Pick the **pretrained** layout model, choose OCR language, DPI, and review threshold.
3. Add files to queue. The separate worker processes one file at a time; submitted jobs survive restarting the UI.
4. Refresh queue to see completed/failed jobs. Each completed file has page previews and region boxes.
5. Review text and proposed reading order region by region; **Save correction** persists changes to disk.
6. Give fragments from one newspaper story the same **Article ID** to combine them in the article export.
7. Choose **Prepare collection ZIP** to download all completed newspapers in one archive, or download an individual job ZIP.
8. Individual ZIPs contain complete `transcription.txt`, `transcription.docx`, `structured.json` and `articles/<id>.txt/.docx`.

**Security:** Local-only UI has no authentication. Never expose port 8501 directly on a public network. Files live in `~/.newspaper_ocr_lab` and are removed when you delete a completed/failed job. No API keys are saved in SQLite. The local upload limit is 250 MB per file; PDF page cap defaults to 500 pages per file; processing speed depends on available RAM/CPU/GPU.

## Model options

- **Pretrained** (default): DocLayout-YOLO DocStructBench checkpoint. The detector excludes figures and abandoned regions, tries to recognize text and title regions, then each text region is transcribed independently with Tesseract. A sparse-OCR recovery pass flags text outside detected boxes as **unassigned** for researcher review; this can recover missed headlines but can also introduce false positives. Source model: https://github.com/opendatalab/DocLayout-YOLO and https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench.
- **XY-cut**: original geometric segmentation, available without optional model dependencies. Used for smoke tests and as a diagnostic baseline.
- **Vision verification off** (default): fully local, no API.
- **Local Ollama**: requires `ollama serve` and an installed vision model; select Local Ollama in sidebar. Vision suggestions are **unreviewed**, never auto-applied.
- **OpenAI API**: optionally set `export OPENAI_API_KEY=...` **before launching**; separate usage charges and document permissions apply. The key is not saved in job metadata.

**Licensing:** DocLayout-YOLO software is AGPL-3.0. Review that license and the model weights' terms before distributing or hosting the program.

## Tests

```bash
pip install pytest
python -m pytest -q
```

The GitHub Actions **OCR regression tests** suite covers synthetic OCR, multi-page PDFs, invalid uploads, crash isolation, background queue state, review persistence, per-article and collection exports, a Streamlit smoke test, and mocked pretrained detector outputs. A second **Pretrained model integration** workflow downloads the actual weights and performs real inference when triggered.

A passing synthetic test **does not demonstrate accuracy on archival newspapers**. Run the real-scan benchmark:

```bash
# Download up to 30 September 1907 Evening Star pages, where LOC permits API access:
python benchmark_newspapers.py --download --limit 30 --layout pretrained

# Or put manually downloaded JPG/PNG/PDF scans in benchmark/images,
# then run without --download:
python benchmark_newspapers.py --layout pretrained
```

Audit results are written to `benchmark/results/audit.csv`; for measured CER/WER, place **human-verified** reference transcripts under `benchmark/references/<scan_stem>.txt`. If downloads fail, obtain image scans directly from https://www.loc.gov/collections/chronicling-america/ .

## Remaining limitations

- Pretrained model is **general document layout**, not specifically trained to separate articles in 1907 newspapers. Article continuation across columns or pages requires explicit human labeling.
- Large PDF batches are queued and resumable between app restarts but processing is currently **one local worker**; there is no multi-user production deployment or parallel GPU scaling.
- Page preview provides region rectangles, not a fully interactive drag-and-drop segmentation editor; article linking and reading order are controlled by forms.
- A failed page currently fails the job; previously completed page drafts remain on disk for recovery by a restarted worker.
- User must assess completeness, reading order, OCR substitutions and licensing before using results for qualitative research.
