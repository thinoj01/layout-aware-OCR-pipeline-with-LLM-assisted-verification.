"""Review edits and per-article exports for local research workflow."""
import io
import json
import re
import zipfile
from pathlib import Path
from docx import Document
from jobs import get_job, job_dir

def load_result(job_id):
    record=get_job(job_id)
    if not record:
        raise ValueError("Unknown job")
    path=job_dir(job_id)/"result.json"
    if not path.exists():
        raise FileNotFoundError("Job has not finished yet")
    return json.loads(path.read_text(encoding="utf8"))

def save_region(job_id,page_no,region_id,text,order,article_id):
    if get_job(job_id)["status"] != "completed":
        raise ValueError("Can only edit completed jobs")
    if not isinstance(order,int) or order<1:
        raise ValueError("Order must be a positive integer")
    data=load_result(job_id)
    match=[p for p in data["pages"] if p["page"]==page_no]
    if len(match)!=1:
        raise ValueError("Unknown page")
    found=[r for r in match[0]["regions"] if r["id"]==region_id]
    if len(found)!=1:
        raise ValueError("Unknown region")
    if len(text)>200000 or len(article_id)>120:
        raise ValueError("Review text too large")
    found[0]["text"]=text
    found[0]["reading_order_guess"]=order
    found[0]["article_id"]=article_id or None
    found[0]["reviewed"]=True
    path=job_dir(job_id)/"result.json"
    temp=path.with_suffix(".tmp")
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf8")
    temp.replace(path)
    return found[0]

def _slug(name):
    return re.sub(r"[^a-zA-Z0-9_-]+","_",name).strip("_")[:75] or "article"

def export_job(job_id):
    data=load_result(job_id)
    out=io.BytesIO()
    with zipfile.ZipFile(out,"w",zipfile.ZIP_DEFLATED) as archive:
        whole=Document()
        whole.add_heading(data["filename"],0)
        transcript=[]
        article_groups={}
        for page in sorted(data["pages"],key=lambda p:p["page"]):
            whole.add_heading(f"Page {page['page']}",1)
            transcript.append(f"=== PAGE {page['page']} ===")
            for region in sorted(page["regions"],key=lambda r:r["reading_order_guess"]):
                text=region["text"]
                transcript.append(text)
                whole.add_paragraph(text)
                if region.get("article_id"):
                    article_groups.setdefault(region["article_id"],[]).append(
                        (page["page"],region["reading_order_guess"],text))
        archive.writestr("transcription.txt","\n\n".join(transcript))
        buf=io.BytesIO();whole.save(buf)
        archive.writestr("transcription.docx",buf.getvalue())
        safe={**data,"pages":[{k:v for k,v in p.items() if k!="layout_preview"}
                                 for p in data["pages"]]}
        archive.writestr("structured.json",json.dumps(safe,ensure_ascii=False,indent=2))
        used=set()
        for article_id,parts in article_groups.items():
            parts.sort(key=lambda part:(part[0],part[1]))
            basename=_slug(article_id)
            while basename in used:
                basename+="_copy"
            used.add(basename)
            content="\n\n".join(p[2] for p in parts)
            archive.writestr(f"articles/{basename}.txt",content)
            doc=Document();doc.add_heading(article_id,0)
            for page,_,text in parts:
                doc.add_paragraph(text)
            b=io.BytesIO();doc.save(b)
            archive.writestr(f"articles/{basename}.docx",b.getvalue())
        archive.writestr("manifest.json",json.dumps({
            "filename":data["filename"],"pages":len(data["pages"]),
            "article_ids":list(article_groups),
            "notice":"Article grouping comes from researcher-supplied IDs. Unlabeled regions remain only in the complete transcription."
        },indent=2))
    return out.getvalue()

def export_completed_collection(job_ids):
    """Create one archive of completed job archives without cross-job name collisions."""
    buffer=io.BytesIO()
    manifest=[]
    with zipfile.ZipFile(buffer,"w",zipfile.ZIP_DEFLATED) as combined:
        for job_id in job_ids:
            record=get_job(job_id)
            if record is None or record["status"]!="completed":
                continue
            folder=f"newspaper_{job_id}"
            combined.writestr(folder+"/transcriptions.zip",export_job(job_id))
            manifest.append({"job_id":job_id,"filename":record["filename"]})
        combined.writestr("collection_manifest.json",
                          json.dumps(manifest,ensure_ascii=False,indent=2))
    return buffer.getvalue()
