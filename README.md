# Historical Newspaper OCR Lab — local research application

An experimental, **local-first** web application for batch transcription of scanned historical newspapers.

**What it supports:** multi-PDF and image upload, durable SQLite jobs, a separate background worker, pretrained DocLayout-YOLO or explicit XY-cut layout detection, region-wise Tesseract OCR, column-aware provisional reading order, page previews, saved manual corrections and article labels, separate per-article DOCX/TXT output, JSON metadata, and optional image-grounded suggestions via local Ollama or OpenAI.

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

## Research analysis workspace (new)

Choose **Research analysis** from the sidebar to work with previously completed newspaper OCR jobs, including any saved manual corrections. The application also accepts a UTF-8 `.txt` transcript directly. This workspace:

- Retrieves passages from the selected newspapers with locally computed **BM25 lexical search**. Each hit includes its original filename, page, region ID, article label and review status. Search is not yet an embedding-based semantic search.
- Lets you ask natural-language questions, continue with follow-up questions in a **multi-turn chat**, and request a **source-grounded answer** from Ollama (local) or the OpenAI API (optional external cloud); the answer is instructed to cite evidence as `[S1]`, `[S2]`, etc. Expand each evidence passage to inspect the exact saved text. The application explicitly warns when the answer has no valid citations. LLM citations are not proof of correctness: researchers should still inspect the underlying source image.
- Shows repeated terms across OCR passages (passage frequency, **not** article prevalence). An optional theme-analysis button asks the model to describe recurring themes using selected evidence. This is exploratory; it is not a validated qualitative coding system.
- Exports the answer or theme analysis **together with its supporting passages as JSON**, preserving provenance.

**Recommended architecture:** Use retrieval over the corrected corpus with an existing model, called via a local Ollama API (no paid service) or your own application backend that calls a hosted API. Do not train a new language model for the initial version. A later stage can add embedding search, researcher-approved codebooks, categorical annotation, historical date/entity metadata, inter-annotator agreement, and topic trends.

## Ollama verification progress and troubleshooting

When OCR processing uses Ollama, the queue now displays stages including *Checking Ollama model*, *Downloading Ollama model (XX%)*, *OCR region N/M*, and *verifying region N/M*. If the model is already cached it should move directly to the OCR stage. Verifying a large newspaper crop can still be slow; the current local verification request times out after 90 seconds and saves a failed suggestion rather than freezing the entire job. Failed suggestions do not automatically replace OCR text. A model download that stops responding produces a visible error.

After a job completes, use **Refresh completed results** and inspect *Unreviewed vision-model suggestion* inside the region expander. If that section is absent, check that the Ollama checkbox was enabled **before the job was submitted**. Selecting it does not retroactively verify older OCR jobs.

## How professors would use it

1. Upload several PDF or image newspapers.
2. Pick the **pretrained** layout model, choose OCR language, DPI, and review threshold.
3. Add files to queue. The separate worker processes one file at a time; submitted jobs survive restarting the UI.
4. The queue refreshes every five seconds. It shows each waiting file's position, page count, and estimated time before processing begins. Estimates are calculated from actual observed processing speed and show **Calculating...** when there is not enough evidence yet. Use **Refresh completed results** to load finished jobs into the review interface.
   - Select **Remove** beside a waiting newspaper to delete it immediately.
   - Select **Cancel running newspaper** to request a stop after the current page. When cancellation completes, its saved job can be deleted.
   - Queue status refreshes independently of the editing forms so unsaved corrections are not wiped while typing.
5. Check the numbers on the annotated newspaper: wide top headlines and masthead are read first, followed by each inferred newspaper column from top to bottom and left to right. A wide section header spanning multiple columns is marked uncertain. This is not automated article reconstruction. Review text and proposed reading order region by region; **Save correction** persists changes to disk.
6. Give fragments from one newspaper story the same **Article ID** to combine them in the article export.
7. Choose **Prepare collection ZIP** to download all completed newspapers in one **directly extractable** archive with a separate folder for each completed newspaper, or download an individual job ZIP. After clicking a download, the UI asks whether to keep the results or clear the local batch and begin again. **Nothing is erased automatically by a download.** Confirmation is required; the reset button remains disabled while a file is running.
8. To prevent old requests accumulating, open **Clear previous results (N finished requests)**, confirm deletion and select **Delete all previously finished requests**. This removes completed/failed/cancelled records and their local data, but leaves queued and running files untouched. Individual ZIPs contain complete `transcription.txt`, `transcription.docx`, `structured.json` and `articles/<id>.txt/.docx`.

**Security:** Local-only UI has no authentication. Never expose port 8501 directly on a public network. Files live in `~/.newspaper_ocr_lab` and are removed when you delete a completed/failed job. No API keys are saved in SQLite. The local upload limit is 250 MB per file; PDF page cap defaults to 500 pages per file; processing speed depends on available RAM/CPU/GPU.

## Model options

- **Pretrained** (default): DocLayout-YOLO DocStructBench checkpoint. The detector excludes figures and abandoned regions, tries to recognize text and title regions, then each text region is transcribed independently with Tesseract. A sparse-OCR recovery pass flags text outside detected boxes as **unassigned** for researcher review; this can recover missed headlines but can also introduce false positives. Source model: https://github.com/opendatalab/DocLayout-YOLO and https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench.
- **XY-cut**: original geometric segmentation, available without optional model dependencies. Used for smoke tests and as a diagnostic baseline.
- **Vision verification off** (default): fully local, no API.
- **Local Ollama**: install the free Ollama executable once (`brew install ollama` on macOS); then check **Enable local Ollama verification** in the sidebar. The worker automatically starts or reuses the local Ollama process, and downloads the chosen vision model on first use. **You do not have to run `ollama serve` separately.** The inference process still runs internally at localhost, not on a cloud service. First use needs internet, enough disk space and memory; model inference may take additional time. Vision suggestions are **unreviewed**, never auto-applied.
- **OpenAI API**: optionally set `export OPENAI_API_KEY=...` **before launching**; separate usage charges and document permissions apply. The key is not saved in job metadata.

**Licensing:** DocLayout-YOLO software is AGPL-3.0. Review that license and the model weights' terms before distributing or hosting the program.

For a full breakdown of every project file, how Ollama is started automatically, the data flow, and ZIP structure, see **[TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md)**.

## Tests

```bash
pip install pytest
python -m pytest -q
```

The GitHub Actions **OCR regression tests** suite covers synthetic OCR, multi-page PDFs, invalid uploads, durable queue state, page-count metadata, estimated-time arithmetic, queue removal, cancellation, atomic reset, review persistence, per-article and collection exports, a Streamlit smoke test, and mocked pretrained detector outputs. A second **Pretrained model integration** workflow downloads the actual weights and performs real inference when triggered.

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

- The column-aware order is a heuristic, not an article-linking model; complex section headings and article continuations may still require correction. **Already processed jobs retain their old order**; remove and requeue those files to regenerate the preview and exported transcription using this improvement.
- Pretrained model is **general document layout**, not specifically trained to separate articles in 1907 newspapers. Article continuation across columns or pages requires explicit human labeling.
- Large PDF batches are queued and resumable between app restarts but processing is currently **one local worker**; there is no multi-user production deployment or parallel GPU scaling.
- Page preview provides region rectangles, not a fully interactive drag-and-drop segmentation editor; article linking and reading order are controlled by forms.
- A failed page currently fails the job; previously completed page drafts remain on disk for recovery by a restarted worker.
- User must assess completeness, reading order, OCR substitutions and licensing before using results for qualitative research.
