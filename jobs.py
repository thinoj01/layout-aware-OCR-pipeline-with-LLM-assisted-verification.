"""Durable SQLite-backed local transcription jobs. One dedicated worker process.

No server authentication: bind Streamlit to localhost only.
Input files are stored on the local disk. Delete jobs to remove source and results.
"""
from __future__ import annotations
import json
import os
import shutil
import sqlite3
import time
import uuid
from datetime import timedelta
from pathlib import Path

ROOT=Path(os.environ.get("NEWSPAPER_OCR_DATA", str(Path.home()/".newspaper_ocr_lab"))).expanduser().resolve()
DB=ROOT/"jobs.sqlite3"

def connect():
    ROOT.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(DB,timeout=30)
    db.row_factory=sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("""CREATE TABLE IF NOT EXISTS jobs(
        id TEXT PRIMARY KEY, filename TEXT NOT NULL, status TEXT NOT NULL,
        done INTEGER NOT NULL DEFAULT 0,total INTEGER NOT NULL DEFAULT 0,
        options TEXT NOT NULL, error TEXT, created REAL NOT NULL, updated REAL NOT NULL
    )""")
    columns={row["name"] for row in db.execute("PRAGMA table_info(jobs)")}
    if "started_at" not in columns:
        db.execute("ALTER TABLE jobs ADD COLUMN started_at REAL")
    if "cancel_requested" not in columns:
        db.execute("ALTER TABLE jobs ADD COLUMN cancel_requested INTEGER NOT NULL DEFAULT 0")
    db.commit()
    return db

def submit(filename: str, content: bytes, options: dict) -> str:
    if not content:
        raise ValueError("Empty upload")
    if len(content)>250*1024*1024:
        raise ValueError("Upload exceeds 250 MB limit")
    ext=Path(filename).suffix.lower()
    if ext not in {".pdf",".png",".jpg",".jpeg",".tif",".tiff",".webp"}:
        raise ValueError("Unsupported extension")
    # Reject obviously invalid uploads before accepting them into the queue.
    if ext==".pdf":
        if not content.lstrip()[:8].startswith(b"%PDF-"):
            raise ValueError("Invalid PDF signature")
    else:
        from io import BytesIO
        from PIL import Image
        try:
            with Image.open(BytesIO(content)) as image:
                image.verify()
        except Exception as exc:
            raise ValueError("Invalid or damaged image upload") from exc
    job_id=uuid.uuid4().hex
    folder=ROOT/job_id
    folder.mkdir(mode=0o700,parents=True)
    (folder/("source"+ext)).write_bytes(content)
    # Determine page count before queueing, so pending files have meaningful progress.
    # Valid PDFs are validated by fitz here, and images by Pillow above.
    if ext==".pdf":
        import fitz
        try:
            with fitz.open(stream=content,filetype="pdf") as pdf:
                page_count=pdf.page_count
        except Exception as exc:
            shutil.rmtree(folder,ignore_errors=True)
            raise ValueError("Invalid or unreadable PDF") from exc
    else:
        from io import BytesIO
        from PIL import Image
        with Image.open(BytesIO(content)) as img:
            page_count=getattr(img,"n_frames",1)
    if page_count<1 or page_count>int(options.get("max_pages",500)):
        shutil.rmtree(folder,ignore_errors=True)
        raise ValueError("Invalid page count or exceeds the configured page limit")
    now=time.time()
    with connect() as db:
        db.execute("INSERT INTO jobs(id,filename,status,done,total,options,error,created,updated) VALUES(?,?,?,?,?,?,?,?,?)",
                   (job_id,Path(filename).name,"queued",0,page_count,json.dumps(options),None,now,now))
    return job_id

def list_jobs():
    with connect() as db:
        return [dict(row) for row in db.execute("SELECT * FROM jobs ORDER BY created DESC,id DESC")]

def get_job(job_id):
    with connect() as db:
        row=db.execute("SELECT * FROM jobs WHERE id=?",(job_id,)).fetchone()
        return dict(row) if row else None

def change(job_id,status,done=None,total=None,error=None):
    with connect() as db:
        db.execute("""UPDATE jobs SET status=?, done=COALESCE(?,done),
            total=COALESCE(?,total), error=?, updated=? WHERE id=?""",
            (status,done,total,error,time.time(),job_id))

def claim():
    """Atomic claim prevents duplicate processing across workers."""
    with connect() as db:
        db.execute("BEGIN IMMEDIATE")
        row=db.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY created,id LIMIT 1").fetchone()
        if row:
            db.execute("UPDATE jobs SET status='running', updated=?, started_at=?, cancel_requested=0 WHERE id=?",
                       (time.time(),time.time(),row["id"]))
            return row["id"]
    return None

def delete(job_id):
    row=get_job(job_id)
    if not row:
        return False
    # Jobs must not be deleted while processing: worker could recreate files.
    if row["status"]=="running":
        raise ValueError("Cannot delete a running job")
    with connect() as db:
        db.execute("DELETE FROM jobs WHERE id=?",(job_id,))
    shutil.rmtree(ROOT/job_id,ignore_errors=True)
    return True

def restart_interrupted():
    """Call only when starting a *single* worker to resume interrupted jobs."""
    with connect() as db:
        db.execute("UPDATE jobs SET status='queued',updated=? WHERE status='running'",
                   (time.time(),))

def job_dir(job_id):
    if not (len(job_id)==32 and all(c in "0123456789abcdef" for c in job_id)):
        raise ValueError("Invalid job ID")
    return ROOT/job_id

def request_cancel(job_id):
    """Cancel a queued job immediately, or a running job after the current page."""
    with connect() as db:
        db.execute("BEGIN IMMEDIATE")
        row=db.execute("SELECT status FROM jobs WHERE id=?",(job_id,)).fetchone()
        if row is None:
            return False
        if row["status"]=="queued":
            db.execute("UPDATE jobs SET status='cancelled',updated=? WHERE id=?",
                       (time.time(),job_id))
        elif row["status"]=="running":
            db.execute("UPDATE jobs SET cancel_requested=1,updated=? WHERE id=?",
                       (time.time(),job_id))
        else:
            return False
        return True

def delete_many(job_ids):
    """Clear a chosen batch. Never delete a running file or its working directory."""
    unique=list(dict.fromkeys(job_ids))
    for job_id in unique:
        job_dir(job_id)
    with connect() as db:
        db.execute("BEGIN IMMEDIATE")
        for job_id in unique:
            row=db.execute("SELECT status FROM jobs WHERE id=?",(job_id,)).fetchone()
            if row and row["status"]=="running":
                raise ValueError("A file is still processing. Cancel it and wait for cancellation before clearing.")
        for job_id in unique:
            db.execute("DELETE FROM jobs WHERE id=?",(job_id,))
    for job_id in unique:
        shutil.rmtree(job_dir(job_id),ignore_errors=True)
    return len(unique)

def _duration(seconds):
    if seconds is None:
        return "Calculating..."
    sec=max(0,round(seconds))
    if sec<60:
        return f"{sec} sec"
    if sec<3600:
        return f"{sec//60} min {sec%60:02d} sec"
    return f"{sec//3600} hr {(sec%3600)//60} min"

def queue_estimates(jobs, now=None):
    """Per-job ETA, queue position, and actual page count; no invented ETA without evidence.

    Prefer the current worker's seconds/page; use average from previously completed
    jobs if available. Estimates are approximate and may vary by document size/model.
    """
    now=time.time() if now is None else now
    rates=[]
    for job in jobs:
        if job["status"]=="completed" and job.get("started_at") is not None and job["done"]>0:
            elapsed=job["updated"]-job["started_at"]
            if elapsed>0:
                rates.append((elapsed,job["done"]))
    history=(sum(x[0] for x in rates)/sum(x[1] for x in rates)) if rates else None
    running=next((j for j in jobs if j["status"]=="running"),None)
    active_rate=None
    if running is not None and running["done"]>0 and running.get("started_at"):
        elapsed=now-running["started_at"]
        if elapsed>0:
            active_rate=elapsed/running["done"]
    rate=active_rate or history
    remaining_running=(max(0,running["total"]-running["done"])*rate
                       if running is not None and rate is not None else 0)
    estimate={}
    if running:
        estimate[running["id"]]={
            "position":0, "seconds_to_start":0,
            "seconds_to_finish":remaining_running if rate is not None else None,
            "rate_seconds_per_page":rate,
        }
    pending=sorted((j for j in jobs if j["status"]=="queued"),
                   key=lambda j:(j["created"],j["id"]))
    ahead=remaining_running
    for i,job in enumerate(pending,1):
        wait=ahead if rate is not None else None
        ahead += job["total"]*rate if rate is not None else 0
        estimate[job["id"]]={
            "position":i,"seconds_to_start":wait,
            "seconds_to_finish":ahead if rate is not None else None,
            "rate_seconds_per_page":rate,
        }
    return estimate
