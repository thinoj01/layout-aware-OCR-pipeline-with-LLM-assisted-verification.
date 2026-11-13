"""Functional smoke tests; do not establish accuracy on historical newspapers."""
import io
import json
import zipfile
import fitz
from PIL import Image, ImageDraw, ImageFont
from ocr_engine import Settings, pages_from_bytes, process_files, export_bundle, sanitize

def sample_image():
    image = Image.new("RGB", (1500, 1150), "white")
    draw = ImageDraw.Draw(image)
    # Portable fallback to PIL default font when DejaVu is unavailable
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 27)
    except OSError:
        font = ImageFont.load_default()
    for x,prefix in [(80,"ALPHA"),(830,"BETA")]:
        for i in range(12):
            draw.text((x,130+i*62),f"{prefix} newspaper line number {i+1}",fill="black",font=font)
    buffer=io.BytesIO()
    image.save(buffer,format="PNG")
    return buffer.getvalue()

def test_batch_export_and_error_isolation():
    results=process_files([("news.png",sample_image()),("broken.pdf",b"invalid")],Settings())
    assert len(results)==2 and len(results[0]["pages"])==1
    text=results[0]["pages"][0]["text"]
    assert "newspaper" in text and "BETA" in text
    assert results[1].get("error")
    with zipfile.ZipFile(io.BytesIO(export_bundle(results))) as archive:
        assert "news/transcription.docx" in archive.namelist()
        assert "news/transcription.txt" in archive.namelist()
        assert json.loads(archive.read("news/structured.json"))["pages"][0]["regions"]
        assert json.loads(archive.read("manifest.json"))[1].get("error")

def test_multipage_pdf():
    doc=fitz.open()
    for _ in range(2):
        page=doc.new_page(width=400,height=300)
        page.insert_text((40,80),"Hello historical newspaper",fontsize=20)
    data=doc.tobytes()
    assert len(list(pages_from_bytes("test.pdf",data,dpi=150)))==2
    result=process_files([("test.pdf",data)],Settings(dpi=150))
    assert len(result[0]["pages"])==2
    assert "Hello" in result[0]["pages"][0]["text"]

def test_sanitize():
    assert sanitize("../../bad file.pdf")=="bad_file"


def test_no_cross_column_merging():
    result=process_files([("columns.png",sample_image())],Settings())
    page=result[0]["pages"][0]
    assert len(page["regions"])>=2
    assert page["text"].find("ALPHA")<page["text"].find("BETA")
    assert "BETA" not in page["regions"][0]["text"]

def test_corrections_propagate_to_exports():
    result=process_files([("columns.png",sample_image())],Settings())
    region=result[0]["pages"][0]["regions"][0]
    region["text"]="RESEARCHER CORRECTED TEXT"
    region["reading_order_guess"]=10
    with zipfile.ZipFile(io.BytesIO(export_bundle(result))) as archive:
        transcription=archive.read("columns/transcription.txt").decode("utf8")
        assert "RESEARCHER CORRECTED TEXT" in transcription
        obj=json.loads(archive.read("columns/structured.json"))
        assert "layout_preview" not in obj["pages"][0]
        assert obj["pages"][0]["regions"][0]["reading_order_guess"]==10
