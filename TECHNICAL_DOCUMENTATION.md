# Technical documentation — Historical Newspaper OCR Lab

This guide describes **every tracked source, test, configuration, and workflow file** in the repository, how information flows through the application, the limitations of its OCR, and how to operate it on a local Mac. Read `README.md` first for installation. This file is for contributors and researchers auditing the implementation.

## 1. What the project does

The application converts scanned newspaper images and PDFs into candidate text regions and transcriptions suitable for review. It **does not** fully solve article identification or reading order in historical newspapers; a human must correct region grouping and article continuation before using text as research evidence.

The default pipeline is:

1. A researcher uploads scanned newspaper PDFs or image files in the Streamlit UI.
2. `jobs.py` validates the file, counts pages, saves the source into a unique job directory and writes a `queued` record to SQLite.
3. `worker.py` claims one queued job. Other jobs remain queued. The Streamlit app periodically reads the queue and shows progress and a provisional wait estimate.
4. `ocr_engine.py` renders each PDF page as an image (or reads an uploaded image) and optionally preprocesses it.
5. With `pretrained` mode, `pretrained_layout.py` loads DocLayout-YOLO and predicts region bounding boxes and classes; `text_recovery.py` finds possible text left outside the boxes. With `xycut` mode, `layout.py` uses whitespace separation instead.
6. `ocr_engine.py` passes each crop to the locally installed Tesseract executable, obtaining region text and OCR confidence. `reading_order.py` provides **provisional** column-major numbering rather than trusting the detection model's output order.
7. If requested, `ollama_verify.py` or the OpenAI API compares an image crop with its OCR candidate and offers an **unreviewed** suggested correction. These suggestions never replace the original region text automatically.
8. The worker writes `result.json` and marks the job `completed`. Researchers inspect the annotated image and change text, order, or article ID in the UI.
9. `review.py` saves the corrections and produces a directly extractable collection ZIP or a per-newspaper ZIP with plain text, Word and structured JSON exports.

### OCR versus layout versus article identification

- **Layout detection** finds visible text boxes; it is not the same as identifying an article.
- **OCR** recognizes the characters in a given image crop, but can misread old fonts, smudged text and page decorations.
- **Reading order** guesses the sequence of regions; it is not the same as correctly linking an article that continues elsewhere.
- **Verification** gives a second model a crop and a Tesseract candidate. A vision model can still hallucinate, omit or change historically significant words.
- **Article ID** is a manual label used to collect multiple regions into one exported article. The code does not autonomously infer robust article IDs.

## 2. Local setup and execution

From the repository root on a supported Mac with Homebrew:

```bash
brew install python@3.11 tesseract
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-ml.txt
python launch.py
```

Open `http://127.0.0.1:8501`. The launcher starts the UI and a single background worker on the same computer. Stop with Ctrl+C. A port conflict means another instance is listening on 8501.

To run tests independently:

```bash
python -m pip install pytest
python -m pytest -q
```

To use **local Ollama verification**: install the free executable once with `brew install ollama`. Then simply check **Enable local Ollama verification** in the app before uploading a job. The worker automatically connects to an already running Ollama, or starts `ollama serve` itself if needed, and downloads the selected public vision model on first use. **You do not have to run a second Terminal manually.** Model weights may require several gigabytes of storage and memory, and the first download can be long. After the first download the model remains cached. Ollama is a separate inference *process* internally; that is how its local API works, not a separate internet service or separate application for the professor to operate. If the app started it, it stops it when the worker exits.

The default model name in the UI is `qwen3-vl:4b`. You may choose another vision-capable Ollama model installed on your computer. Merely enabling verification does not change the OCR text automatically; corrections are shown for manual acceptance. Disabling this option avoids running Ollama entirely.

### Typical operation

1. Choose pretrained layout or XY-cut, language and confidence threshold.
2. Check **Enable local Ollama verification** if you want local AI suggestions; otherwise leave it unchecked.
3. Upload one or more PDFs/JPGs and add them to the queue.
4. Monitor queue position and completion. The first pages may display `Calculating...` for ETA until there is a measured processing rate.
5. Review each completed newspaper's original image, reading order and editable region text; save corrections and give related regions the same article ID.
6. Download a per-newspaper ZIP or the collection ZIP. **The collection ZIP has no nested archives:** each newspaper has its own folder.
7. Use **Clear previous results** to delete completed, failed, and cancelled jobs without touching queued/running files. Confirm carefully: source files and saved corrections are removed from the local job directory. Existing ZIP files already saved to your Downloads folder are not affected.
8. Alternatively, use **Start a new batch / clear queue**, which clears all records only when nothing is running.

## 3. Source file inventory

### `app.py` — Streamlit user interface

The UI entry point and user workflows. Builds the sidebar with detector, language, PDF DPI, confidence threshold, and the Ollama-verification checkbox; collects uploads and queues them through `jobs.submit`; displays live queue status and buttons to remove waiting jobs or request cancellation; shows completed scanned-page previews and correction forms; offers ZIP download buttons. The dedicated cleanup control calls `jobs.delete_finished` after a confirmation checkbox. This deletes only finished request records.

Queue status refreshes independently from article-editing forms to avoid erasing unsaved text while typing. Streamlit reruns are not background tasks; actual OCR is run in the separate worker.

### `launch.py` — local launcher

Starts `worker.py` and `streamlit run app.py` as child processes. Binds Streamlit to `127.0.0.1:8501`, not a publicly reachable interface, and stops its children on normal shutdown. Launch one instance at a time.

### `jobs.py` — SQLite queue and local file storage

Stores job metadata in `~/.newspaper_ocr_lab/jobs.sqlite3`, creates per-job folders under `~/.newspaper_ocr_lab/<job_id>/`, and validates input extensions, signatures and page counts before queueing. A UUID prevents file-name collisions. SQLite transactions make job claims and batch deletion atomic with respect to queue updates.

Key operations:

- `submit`: create the job and save source bytes.
- `list_jobs`, `get_job`: inspect job status.
- `claim`: atomically select the oldest queued job for the worker.
- `change`: update progress/status and errors.
- `queue_estimates`: calculate queue position and estimated wait using previous observed seconds/page; these are not guarantees.
- `request_cancel`: cancel a waiting job or flag a running job to stop after its current page.
- `delete`, `delete_many`: remove selected jobs, never a running job.
- `delete_finished`: delete **only completed, failed and cancelled** jobs and their local data; queued/running jobs are preserved.
- `restart_interrupted`: requeue jobs that were marked running when the single worker was interrupted.

SQLite metadata and the raw uploaded scan are local unless an optional external API is explicitly enabled.

### `worker.py` — durable background processing

Claims one job at a time, creates OCR settings, reads PDF/image pages, calls `ocr_engine.transcribe_page`, writes incremental `draft.json`, updates progress, and on success creates `result.json`. It catches exceptions and marks the job failed.

For Ollama-enabled jobs it calls `ollama_runtime.ensure_ollama` automatically. If Ollama installation, startup, or model download fails, the job should display the error rather than remaining forever marked running. Cancelled jobs stop at page boundaries; cancellation is not instantaneous during a long model inference.

### `ocr_engine.py` — OCR orchestration

Contains the `Settings` dataclass and the main page OCR workflow:

- `pages_from_bytes`: render PDFs using PyMuPDF or read image frames with Pillow.
- `prepare`: contrast-enhanced grayscale preprocessing with OpenCV.
- `recognize_lines` / `group_columns`: legacy helpers for OCR line extraction and approximate line grouping.
- `transcribe_page`: detect layout, OCR each region with Tesseract, calculate average token confidence, call `reading_order.order_regions`, optionally request AI proposals, and create an annotated preview.
- `process_files` and `export_bundle`: legacy synchronous batch processing/export utilities used by tests and benchmarks.

Tesseract confidence is a model's own estimate, **not a measured percentage of correct words**.

### `pretrained_layout.py` — DocLayout-YOLO detection

Downloads and caches publicly available DocLayout-YOLO model weights from Hugging Face. Loads the pretrained YOLO model, predicts text-region bounding boxes, normalizes detector class names, and optionally includes figure boxes to avoid transcribing illustrations. It returns region rectangles and class labels.

This detects generic document components, **not historical newspaper articles**. If the model is unavailable, it raises an error rather than silently claiming that a heuristic detector is pretrained. Review DocLayout-YOLO's AGPL-3.0 license and checkpoint terms before public deployment.

### `layout.py` — non-ML geometric fallback and preview

Contains the `Block` data class, a recursive whitespace-based XY-cut detector, and `overlay`, which paints region rectangles and reading-order numbers on the original image. XY-cut does not require the pretrained weights and is useful for comparison, but may fragment closely packed newspapers.

### `text_recovery.py` — detect possible missed text

Runs a sparse Tesseract pass to find substantial text outside predicted regions, excluding already covered areas. The resulting fragments carry the `sparse-ocr-recovery` source label, are marked for human review and should not automatically be attached to unrelated articles. Decorative lines may produce false positives.

### `reading_order.py` — provisional newspaper ordering

Separates wide page headings near the top from the body, estimates vertical column lanes using region coordinates, sorts body blocks top-to-bottom inside each lane and lanes left-to-right, then assigns numbered `reading_order_guess` values. Marks some spanning blocks uncertain.

This improved the previous behavior of numbering regions in arbitrary detector order. Complex section banners, inset images and articles continuing across columns can still require manual corrections.

### `ollama_runtime.py` — manage the free local vision model

New local-process lifecycle helper. Checks whether Ollama's local HTTP API is already available at `127.0.0.1:11434`. If not, and the executable is installed, launches `ollama serve` automatically. Checks installed model names and runs `ollama pull <model>` when the requested model is absent. On worker exit, stops only the Ollama process it started; it does not kill an Ollama service that another application owns. It never sends a request to a remote Ollama host.

### `ollama_verify.py` — generate image-grounded suggestions

Converts the OCR region's image crop to base64, sends it and the existing Tesseract transcription to the Ollama vision model through the loopback API, then returns a `proposed_unreviewed` suggestion. This suggestion is displayed separately from the real OCR text; the user must decide what to copy or correct.

**Ollama is not used for the initial detection and recognition by default.** The intended workflow is Tesseract OCR first, then optional extra checking. It may improve readability or help find errors but cannot guarantee faithful historical transcription.

### `review.py` — saved edits and document exports

`load_result` reads saved data; `save_region` writes researcher-corrected text, reading order and article IDs back to disk. `export_job` creates an individual ZIP with:

```text
transcription.txt
transcription.docx
structured.json
manifest.json
articles/<article_id>.txt
articles/<article_id>.docx
```

Only explicitly labeled articles get separate article files. All other regions remain in the full transcription. The ZIP includes manually corrected text rather than replacing it with raw OCR.

`export_completed_collection` creates **one directly extractable ZIP**, with a `newspaper_<job_id>/` directory for every completed newspaper and a collection manifest. Each folder has the same TXT, DOCX, JSON and per-article exports listed above. Incomplete requests are recorded as skipped when explicitly included in a collection request.

### `pipeline.py` — older command-line baseline

A standalone original prototype, separate from the Streamlit application. It loads a PDF's first page or an image, preprocesses it, uses simple column heuristics, runs Tesseract, and writes `baseline.txt`, `regions.json`, `column_order.txt` and `layout_overlay.jpg`. It is useful as a comparison baseline, not the preferred pipeline for multi-page jobs.

Example:

```bash
python pipeline.py samples/example.jpg --out outputs
```

### `download_samples.py` — optional archival downloader

Attempts to retrieve sample 1907 *Evening Star* front-page scans using Library of Congress metadata and save them under `samples/`. It depends on the external API, whose metadata formats and rate limits can change. If downloads fail, manually obtained scans can be uploaded to the app directly.

### `benchmark_newspapers.py` — archival OCR audit

Accepts a folder of real scans or optionally attempts to retrieve dated issues from Library of Congress. Processes each scan and writes OCR text plus `audit.csv`, with status, pages, region count and mean OCR confidence. If you supply independently verified `benchmark/references/<stem>.txt`, it computes CER/WER with `evaluate.py`.

The download mode stops when the LOC service rate-limits requests. Successfully running OCR on a scan does **not** demonstrate the article grouping is correct.

### `evaluate.py` — quantitative transcription error metrics

Provides memory-efficient edit distance and word/character error rates. Example:

```bash
python evaluate.py reference.txt predicted.txt
```

CER is character substitutions + insertions + deletions divided by reference character count; WER is the same concept over words. This measurement needs an accurate reference transcription. Whitespace is normalized; historic spelling is otherwise preserved.

### `research.py` — source-grounded retrieval and model access

Builds a passage-level corpus from **saved, corrected** completed job regions, with filename, page number, region ID, article ID, OCR confidence and reviewed status retained. Supports supplementary UTF-8 TXT passages. A lightweight BM25 lexical scorer ranks passages for a research question (no embeddings or model training required), and document-frequency counts surface common terms. Constructs strict evidence-only prompts with explicit `[S1]` source references, and can call either the existing local Ollama API or an optional hosted OpenAI API. The code can check that cited source IDs exist, but cannot independently prove the LLM used their text correctly. Results are hypotheses to verify.

### `research_app.py` — second Streamlit workspace

A separate **Research analysis** view accessed using the sidebar switch in `app.py`. The researcher selects completed newspapers or uploads a UTF-8 TXT transcription. The view provides two tools: (1) evidence search and question answering with inspected source excerpts and downloadable evidence JSON, and (2) exploratory repeated-term counts plus a model-assisted theme overview. It provides the original document, page, region and edit-review status for every evidence excerpt.

The prototype does not yet implement vector semantic embeddings, automated article reconstruction, longitudinal topic prevalence, named-entity disambiguation, configurable thematic codebooks or statistical validation. Future work should build these carefully against annotated ground truth.

## 4. Requirements and project configuration

### `requirements.txt`

Core Python dependencies: Streamlit for UI, OpenCV and NumPy for images, Pillow for image loading, PyMuPDF for PDFs, pytesseract (Python wrapper for the **separately installed** Tesseract binary), python-docx for Word export, OpenAI for optional external verification and Requests for Ollama communication and external sample download.

### `requirements-ml.txt`

Optional large machine-learning libraries for the pretrained detector: `doclayout-yolo` and `huggingface-hub`. Install this only if you want the pretrained layout model rather than the lightweight XY-cut baseline.

### `.gitignore`

Prevents common local artifacts and credentials from entering Git: virtual environments, Python caches, pytest cache, `.env`, Streamlit secrets, generated outputs, downloaded samples and ZIPs.

### `README.md`

Shorter user-facing installation, operation, model choices, test commands and limitations. Use this Markdown document for deeper implementation details.

## 5. Automated tests: what each file checks

- `tests/test_engine.py`: OCR on synthetic newspaper columns, multi-page PDF conversion, processing failure isolation, basic export and corrected text propagation.
- `tests/test_end_to_end.py`: job submission, worker processing on a two-page PDF, corrected transcription export, and a Streamlit UI render smoke test.
- `tests/test_local_workflow.py`: SQLite job queue, invalid uploads, persisted corrections, model-detector stubs, region ordering metadata, and batch ZIP contents.
- `tests/test_queue_controls.py`: precomputed page counts, queue positions, observed-time ETA arithmetic, waiting-item removal, cancellation and atomic full reset.
- `tests/test_reading_order.py`: approximate 3- and 4-column layouts, masthead ordering, spanning headings, uncertain blocks and deterministic order.
- `tests/test_research.py`: corpus retrieval from saved manual edits, BM25 relevance, citation marker validation, term frequencies and TXT provenance.
- `tests/test_cleanup_exports_ollama.py`: terminal-job bulk cleanup, preservation of active requests, actual ZIP and Word-file opening, corrected contents, multiple-newspaper folder structure and mocked Ollama startup/availability.

**Test limitations:** Unit tests and synthetic sample scans cannot prove accurate transcription of real nineteenth- or twentieth-century newspapers. Model installation and sample inference are covered by a separate CI workflow; local Ollama availability and downloads depend on the machine running the app.

## 6. GitHub Actions workflows

- `.github/workflows/tests.yml`: Python 3.11, local Tesseract installation, core Python requirements and automatic pytest regression tests on commits/PRs.
- `.github/workflows/pretrained.yml`: downloads actual pretrained DocLayout-YOLO weights and tests model inference plus OCR on a synthetic two-column image. This job requires network and more dependencies.
- `.github/workflows/archival.yml`: **manually triggered** Library of Congress sample-run workflow, not run on every push because the archive can rate-limit or refuse downloads.

## 7. Data storage, privacy and security

Local job database: `~/.newspaper_ocr_lab/jobs.sqlite3`.

Each active/completed job has `~/.newspaper_ocr_lab/<job_id>/`, holding an uploaded `source.pdf` or image, `draft.json` during processing and `result.json` when complete. Deleted records have their folders removed by the cleanup and reset controls.

**Important:** Deleting a job removes its local source scan and saved human corrections. Export a ZIP and verify it exists before deleting. Deletion does not remove downloads stored in another directory. The application does not have authentication or a production access-control system and should not be exposed to the public internet. Ollama runs locally; enabling an optional **OpenAI API** mode sends selected crop images to that external provider. Keep copyright, privacy and research-data policies in mind.

## 8. Known limitations and future work

1. Need human-verified transcription and article segmentation datasets covering many different historical newspaper layouts, languages, scans and dates before any accuracy claim.
2. A layout box is not an article; page-to-page and column-to-column continuation is manually labeled.
3. Queue time estimates may vary widely between small images and long, high-DPI PDFs, especially when first loading a model or enabling Ollama.
4. The worker handles a single job at a time. Public multi-user hosting would require authentication, resource limits, persistent server storage and a production queue.
5. The first Ollama model download requires adequate storage/internet, and supported vision models require enough memory.
6. The automated verification suggestion is not a validated ground-truth correction and should not overwrite original text without researcher review.
7. Distinguish synthetic regression success from measured OCR CER/WER and article-level recall on real newspaper scans.
