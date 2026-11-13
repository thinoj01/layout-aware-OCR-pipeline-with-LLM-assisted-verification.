"""One-process local worker for persistent newspaper OCR jobs."""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
from jobs import get_job, claim, change, job_dir, restart_interrupted
from ocr_engine import Settings, pages_from_bytes, transcribe_page

def atomic_json(path: Path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf8")
    tmp.replace(path)

def process_job(job_id):
    record=get_job(job_id)
    if not record:
        raise ValueError("Unknown job")
    folder=job_dir(job_id)
    inputs=[f for f in folder.iterdir() if f.name.startswith("source.")]
    if len(inputs)!=1:
        raise ValueError("Exactly one input file is required")
    src=inputs[0]
    opts=json.loads(record["options"])
    settings=Settings(**opts)
    if settings.verify and not settings.model.startswith("ollama:"):
        settings.api_key=os.environ.get("OPENAI_API_KEY")
    pages=[]
    try:
        if get_job(job_id)["cancel_requested"]:
            change(job_id,"cancelled")
            return
        for index,image in pages_from_bytes(record["filename"],src.read_bytes(),
                                             settings.dpi,settings.max_pages):
            if get_job(job_id)["cancel_requested"]:
                change(job_id,"cancelled",done=len(pages))
                return
            page=transcribe_page(image,index,settings)
            # Write incremental progress so a failed page doesn't discard previous pages.
            pages.append(page)
            atomic_json(folder/"draft.json",{"filename":record["filename"],"pages":pages})
            change(job_id,"running",done=index,total=record["total"])
            if get_job(job_id)["cancel_requested"]:
                change(job_id,"cancelled",done=len(pages))
                return
        atomic_json(folder/"result.json",{"filename":record["filename"],"pages":pages})
        (folder/"draft.json").unlink(missing_ok=True)
        change(job_id,"completed",done=len(pages),total=len(pages))
    except Exception as exc:
        change(job_id,"failed",done=len(pages),error=f"{type(exc).__name__}: {exc}")
        raise

def run(once=False,poll_seconds=2):
    restart_interrupted()
    while True:
        job_id=claim()
        if job_id:
            try:
                process_job(job_id)
            except Exception as exc:
                print(f"FAILED {job_id}: {exc}",flush=True)
        elif once:
            return
        else:
            time.sleep(poll_seconds)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--once",action="store_true")
    args=parser.parse_args()
    run(once=args.once)
