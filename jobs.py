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
    job_id=uuid.uuid4().hex
    folder=ROOT/job_id
    folder.mkdir(mode=0o700,parents=True)
    (folder/("source"+ext)).write_bytes(content)
    now=time.time()
    with connect() as db:
        db.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?)",
                   (job_id,Path(filename).name,"queued",0,0,json.dumps(options),None,now,now))
    return job_id

def list_jobs():
    with connect() as db:
        return [dict(row) for row in db.execute("SELECT * FROM jobs ORDER BY created DESC")]

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
        row=db.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
        if row:
            db.execute("UPDATE jobs SET status='running', updated=? WHERE id=?",
                       (time.time(),row["id"]))
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
