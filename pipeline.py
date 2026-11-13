#!/usr/bin/env python3
"""Free local prototype: image/PDF -> layout candidates -> OCR -> text/JSON/overlay."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from pytesseract import Output


def load_page(path, max_dim=5200):
    if path.suffix.lower() == ".pdf":
        import fitz
        with fitz.open(str(path)) as doc:
            pix = doc[0].get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False)
            img = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot open image: {path}")
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def prep(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def regions(img):
    """Estimate full-height columns using ink projection; NOT article detection."""
    h, w = img.shape[:2]
    bw = prep(img)
    top, bottom = int(h * .11), int(h * .96)
    ink = (bw[top:bottom, :] < 128).astype(np.uint8)
    density = np.convolve(ink.sum(axis=0).astype(float), np.ones(11) / 11, mode="same")
    active = density > max(2, (bottom - top) * .007)
    close_width = max(9, int(w * .009))
    active = cv2.morphologyEx(
        active.astype("uint8")[None, :],
        cv2.MORPH_CLOSE,
        np.ones((1, close_width), np.uint8)
    )[0] > 0
    spans, start = [], None
    for i, active_here in enumerate(active.tolist() + [False]):
        if active_here and start is None:
            start = i
        elif not active_here and start is not None:
            if i - start > max(45, w * .085):
                spans.append((start, i))
            start = None
    if not spans:
        spans = [(0, w)]
    return [
        {"bbox": [max(0, x1 - 8), top, min(w, x2 + 8), h],
         "type": "column_candidate"}
        for x1, x2 in spans
    ]


def recognize(img, box, lang="eng"):
    x1, y1, x2, y2 = box
    sub = img[y1:y2, x1:x2]
    dat = pytesseract.image_to_data(
        sub, lang=lang, config="--oem 1 --psm 6", output_type=Output.DICT
    )
    lines, conf, words = {}, [], []
    for i, token in enumerate(dat["text"]):
        token = token.strip()
        if not token:
            continue
        score = float(dat["conf"][i])
        if score >= 0:
            conf.append(score)
        key = (dat["block_num"][i], dat["par_num"][i], dat["line_num"][i])
        lines.setdefault(key, []).append(token)
        words.append({
            "text": token, "confidence": round(score, 1),
            "bbox": [
                x1 + dat["left"][i], y1 + dat["top"][i],
                x1 + dat["left"][i] + dat["width"][i],
                y1 + dat["top"][i] + dat["height"][i],
            ],
        })
    return {
        "text": "\n".join(" ".join(tokens) for tokens in lines.values()),
        "mean_ocr_confidence": round(float(np.mean(conf)), 2) if conf else None,
        "words": words,
    }


def process(path, outdir, lang="eng"):
    img = load_page(path)
    h, w = img.shape[:2]
    boxes = regions(img)
    baseline = pytesseract.image_to_string(img, lang=lang, config="--oem 1 --psm 3")
    entries, painted = [], img.copy()
    for k, reg in enumerate(boxes, 1):
        box = reg["bbox"]
        result = recognize(img, box, lang)
        entries.append({
            "region_id": k, "type": reg["type"], "bbox": box,
            "reading_order_guess": k, **result,
        })
        cv2.rectangle(painted, (box[0], box[1]), (box[2]-1, box[3]-1),
                      (0, 0, 255), max(2, w // 800))
        cv2.putText(painted, str(k), (box[0] + 5, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    output = {
        "source": str(path), "width": w, "height": h,
        "warning": "Column candidates are heuristic. Article boundaries, marginal notes and reading order require manual verification. OCR confidence is not calibrated accuracy.",
        "regions": entries,
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "baseline.txt").write_text(baseline, encoding="utf8")
    (outdir / "regions.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf8")
    (outdir / "column_order.txt").write_text(
        "\n\n".join(f"=== REGION {r['region_id']} ===\n{r['text']}" for r in entries),
        encoding="utf8"
    )
    cv2.imwrite(str(outdir / "layout_overlay.jpg"), painted)
    print(f"{path.name}: {w}x{h}, {len(entries)} candidate columns; output: {outdir}")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--lang", default="eng")
    args = parser.parse_args()
    for path in args.paths:
        if not path.exists():
            print(f"NOT FOUND {path}", file=sys.stderr)
            continue
        try:
            process(path, args.out / path.stem, args.lang)
        except Exception as exc:
            print(f"FAILED {path}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
