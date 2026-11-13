"""Local-first multi-page OCR and structured outputs. LLM verification is opt-in."""
from __future__ import annotations
import base64
import io
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import fitz
import numpy as np
import pytesseract
from PIL import Image
from pytesseract import Output
from layout import detect_regions, overlay

SUPPORTED = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}

@dataclass
class Settings:
    language: str = "eng"
    dpi: int = 250
    confidence_threshold: float = 65.0
    verify: bool = False
    model: str = "gpt-4.1-mini"
    max_verify_regions: int = 12
    api_key: str | None = None
    max_pages: int = 500
    use_layout: bool = True
    layout_backend: str = "xycut"

def sanitize(name):
    return (re.sub(r"[^a-zA-Z0-9._-]+", "_", Path(name).stem).strip("._") or "newspaper")[:85]

def pages_from_bytes(filename, data, dpi=250, max_pages=500):
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError("Unsupported file type")
    if not data:
        raise ValueError("Empty file")
    if ext == ".pdf":
        with fitz.open(stream=data, filetype="pdf") as doc:
            if doc.page_count > max_pages:
                raise ValueError("Too many pages in file")
            for i in range(doc.page_count):
                pix = doc[i].get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72), alpha=False)
                yield i+1, Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    else:
        with Image.open(io.BytesIO(data)) as image:
            n = getattr(image, "n_frames", 1)
            if n > max_pages:
                raise ValueError("Too many image frames")
            for i in range(n):
                image.seek(i)
                yield i+1, image.convert("RGB").copy()

def prepare(img):
    gray = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return Image.fromarray(cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8)).apply(gray))

def recognize_lines(img, language):
    # PSM 4 handles simple printed columns more accurately than full-auto PSM 3
    data = pytesseract.image_to_data(img, lang=language, config="--oem 1 --psm 4",
                                    output_type=Output.DICT)
    lines = {}
    for i, raw in enumerate(data["text"]):
        token = raw.strip()
        if not token:
            continue
        box = [int(data["left"][i]), int(data["top"][i]),
               int(data["left"][i]+data["width"][i]),
               int(data["top"][i]+data["height"][i])]
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        line = lines.setdefault(key, {"words": [], "bbox": box[:], "scores": []})
        confidence = float(data["conf"][i])
        line["words"].append({"text": token, "bbox": box, "confidence": round(confidence, 2)})
        line["bbox"] = [min(line["bbox"][0],box[0]),min(line["bbox"][1],box[1]),
                        max(line["bbox"][2],box[2]),max(line["bbox"][3],box[3])]
        if confidence >= 0:
            line["scores"].append(confidence)
    result = []
    for (block, para, index), line in lines.items():
        line["text"] = " ".join(x["text"] for x in line["words"])
        line["confidence"] = round(float(np.mean(line.pop("scores"))),1) if line["scores"] else 0
        line.update(block_id=block, paragraph_id=para, line_id=index)
        result.append(line)
    return result

def group_columns(lines, width):
    """Heuristic grouping based on word-line left edges; NOT article detection."""
    if len(lines) < 3:
        return [lines] if lines else []
    starts = sorted(line["bbox"][0] for line in lines)
    tolerance = max(25, int(width * .035))
    clusters = []
    for x in starts:
        if not clusters or x - clusters[-1][-1] > tolerance:
            clusters.append([x])
        else:
            clusters[-1].append(x)
    robust = [float(np.median(g)) for g in clusters if len(g) >= max(3, len(lines)//15)]
    if not robust:
        robust = [float(np.median(starts))]
    anchors = []
    for x in robust:
        if not anchors or x - anchors[-1] > max(tolerance*2,width*.07):
            anchors.append(x)
    grouped = [[] for x in anchors]
    for line in lines:
        x = line["bbox"][0]
        idx = min(range(len(anchors)), key=lambda i: abs(x - anchors[i]))
        grouped[idx].append(line)
    return [sorted(g, key=lambda x:(x["bbox"][1],x["bbox"][0])) for g in grouped if g]

def verify_region(image, text, settings):
    """Image-grounded, unreviewed suggestions; do not silently replace OCR."""
    from openai import OpenAI
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(image_bytes.getvalue()).decode("ascii")
    client = OpenAI(api_key=settings.api_key)
    response = client.responses.create(
        model=settings.model,
        input=[{"role":"user","content":[
            {"type":"input_text","text":(
                "Compare the historic newspaper image to this OCR candidate. "
                "Return only a literal transcription. Preserve historic spelling, punctuation, names and numbers. "
                "Mark unreadable words [ILLEGIBLE], never guess. OCR:\n"+text)},
            {"type":"input_image","image_url":uri}
        ]}], max_output_tokens=1500)
    return {"suggestion":response.output_text.strip(), "model":settings.model,
            "status":"proposed_unreviewed"}

def transcribe_page(img, page_no, settings):
    """Layout-first independent region OCR; no reading across column gutters."""
    prepared=prepare(img)
    if settings.layout_backend == "pretrained":
        from pretrained_layout import detect_pretrained
        predicted = detect_pretrained(img)
        if not predicted:
            raise RuntimeError("Pretrained model detected no usable text blocks; review source image.")
        from layout import Block
        blocks = [Block(r["bbox"],i+1,kind=r["type"],source=r["source"])
                  for i,r in enumerate(predicted)]
    elif settings.layout_backend == "xycut":
        blocks = detect_regions(prepared) if settings.use_layout else []
    else:
        raise ValueError("Unknown layout backend: "+settings.layout_backend)
    if not blocks:
        from layout import Block
        blocks=[Block([0,0,img.width,img.height],1,source="fallback")]
    regions=[]
    for i,block in enumerate(blocks,1):
        x0,y0,x1,y1=block.bbox
        pad=6
        crop=prepared.crop((max(0,x0-pad),max(0,y0-pad),
                            min(img.width,x1+pad),min(img.height,y1+pad)))
        data=pytesseract.image_to_data(crop,lang=settings.language,
                config="--oem 1 --psm 6",output_type=Output.DICT)
        lines={}
        scores=[]
        for j,raw in enumerate(data["text"]):
            word=raw.strip()
            if not word:
                continue
            conf=float(data["conf"][j])
            if conf>=0:
                scores.append(conf)
            key=(data["block_num"][j],data["par_num"][j],data["line_num"][j])
            lines.setdefault(key,[]).append(word)
        content="\n".join(" ".join(words) for words in lines.values())
        if not content.strip():
            continue
        confidence=round(float(np.mean(scores)),2) if scores else 0.0
        regions.append({"id":f"p{page_no}_r{i}","type":block.kind,
            "reading_order_guess":len(regions)+1,"article_id":None,
            "bbox":block.bbox,"text":content,
            "ocr_confidence":confidence,
            "needs_review":confidence<settings.confidence_threshold,
            "verification":None,"source":block.source})
    if settings.verify:
        if not settings.model.startswith("ollama:") and not settings.api_key:
            raise ValueError("An API key is required for OpenAI verification")
        for region in sorted(regions,key=lambda r:r["ocr_confidence"])[:settings.max_verify_regions]:
            x1,y1,x2,y2=region["bbox"]
            crop=img.crop((max(0,x1-12),max(0,y1-12),
                           min(img.width,x2+12),min(img.height,y2+12)))
            try:
                if settings.model.startswith("ollama:"):
                    from ollama_verify import verify_ollama
                    region["verification"]=verify_ollama(crop,region["text"],
                                                       settings.model.split(":",1)[1])
                else:
                    region["verification"]=verify_region(crop,region["text"],settings)
            except Exception as exc:
                region["verification"]={"status":"failed","error":str(exc)[:250]}
    import base64
    viz=overlay(img,regions)
    if viz.width>1500:
        viz=viz.resize((1500,round(1500*viz.height/viz.width)),Image.Resampling.LANCZOS)
    thumb=io.BytesIO()
    viz.save(thumb,format="JPEG",quality=72)
    return {"page":page_no,"size":list(img.size),"regions":regions,
        "layout_preview":base64.b64encode(thumb.getvalue()).decode("ascii"),
        "text":"\n\n".join(r["text"] for r in regions),
        "warning":"XY-cut order is provisional; article grouping and side notes need human review."}

def process_files(files,settings,progress=None):
    results=[]
    used={}
    for index,(filename,raw) in enumerate(files,1):
        stem=sanitize(filename)
        used[stem]=used.get(stem,0)+1
        if used[stem]>1:
            stem+=f"_{used[stem]}"
        result={"filename":filename,"slug":stem,"pages":[]}
        try:
            for number,img in pages_from_bytes(filename,raw,settings.dpi,settings.max_pages):
                result["pages"].append(transcribe_page(img,number,settings))
        except Exception as exc:
            result["error"]=str(exc)
        results.append(result)
        if progress:
            progress(index,len(files),filename)
    return results

def export_bundle(results):
    """Export corrected region text and manual reading order."""
    from docx import Document
    buffer=io.BytesIO()
    manifest=[]
    with zipfile.ZipFile(buffer,"w",zipfile.ZIP_DEFLATED) as archive:
        for item in results:
            if item.get("error"):
                manifest.append({"source":item["filename"],"error":item["error"]})
                continue
            stem=item["slug"]
            doc=Document()
            doc.add_heading(item["filename"],0)
            parts=[]
            for page in item["pages"]:
                doc.add_heading(f"Page {page['page']}",level=1)
                parts.append(f"=== PAGE {page['page']} ===")
                for region in sorted(page["regions"],key=lambda r:r["reading_order_guess"]):
                    parts.append(region["text"])
                    doc.add_paragraph(region["text"])
            archive.writestr(f"{stem}/transcription.txt","\n\n".join(parts))
            clean_pages=[{k:v for k,v in page.items() if k!="layout_preview"}
                         for page in item["pages"]]
            archive.writestr(f"{stem}/structured.json",json.dumps({
                "source_filename":item["filename"],"pages":clean_pages},
                ensure_ascii=False,indent=2))
            docio=io.BytesIO()
            doc.save(docio)
            archive.writestr(f"{stem}/transcription.docx",docio.getvalue())
            manifest.append({"source":item["filename"],"pages":len(item["pages"])})
        archive.writestr("manifest.json",json.dumps(manifest,indent=2))
    return buffer.getvalue()
