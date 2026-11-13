"""Optional pretrained DocLayout-YOLO inference.

Model weights download from Hugging Face on first use; offline use works after caching.
This module intentionally NEVER silently falls back to heuristic detection.
DocLayout-YOLO is AGPL-3.0; review license before public deployment.
"""
from __future__ import annotations

from functools import lru_cache
import numpy as np
from PIL import Image

WEIGHTS_REPO = "juliozhao/DocLayout-YOLO-DocStructBench"
WEIGHTS_FILE = "doclayout_yolo_docstructbench_imgsz1024.pt"
SKIP_CLASSES = {"figure", "abandon"}
TEXT_CLASSES = {"title", "plain_text", "figure_caption", "table_caption", "table_footnote",
                "isolate_formula", "formula_caption", "table"}

@lru_cache(maxsize=1)
def load_model():
    try:
        from huggingface_hub import hf_hub_download
        from doclayout_yolo import YOLOv10
    except ImportError as exc:
        raise RuntimeError(
            "Pretrained detection requires: pip install -r requirements-ml.txt"
        ) from exc
    try:
        weights = hf_hub_download(repo_id=WEIGHTS_REPO, filename=WEIGHTS_FILE)
        return YOLOv10(weights)
    except Exception as exc:
        raise RuntimeError(
            "Could not load pretrained DocLayout-YOLO weights. Connect to the internet "
            "for the first download, or choose the explicit XY-cut fallback."
        ) from exc

def detect_pretrained(image: Image.Image, min_confidence: float = 0.20, include_figures=False):
    model = load_model()
    try:
        results = model.predict(source=np.asarray(image.convert("RGB")),
                                imgsz=1024, conf=min_confidence,
                                device="cpu", verbose=False)
    except Exception as exc:
        raise RuntimeError(f"Pretrained layout inference failed: {exc}") from exc
    detections = []
    width, height = image.size
    for result in results:
        names = result.names
        for row in result.boxes:
            label = str(names[int(row.cls[0])]).lower().replace(" ", "_")
            if label in SKIP_CLASSES:
                continue
            if label not in TEXT_CLASSES and not (include_figures and label=="figure"):
                continue
            x1, y1, x2, y2 = [round(float(x)) for x in row.xyxy[0].tolist()]
            x1, x2 = max(0,x1), min(width,x2)
            y1, y2 = max(0,y1), min(height,y2)
            if x2-x1<15 or y2-y1<12:
                continue
            detections.append({
                "bbox":[x1,y1,x2,y2],
                "type":label,
                "confidence":round(float(row.conf[0]),4),
                "source":"doclayout-yolo"
            })
    # Spatial reading order is tentative; headlines can span multiple columns.
    detections.sort(key=lambda r:(r["bbox"][0],r["bbox"][1]))
    for index,region in enumerate(detections,1):
        region["reading_order_guess"]=index
    return detections
