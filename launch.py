"""Launch local OCR worker and Streamlit UI together.

Use python launch.py; stop with Ctrl+C. Bind only to localhost.
"""
import os
import signal
import subprocess
import sys
from pathlib import Path

def main():
    here=Path(__file__).resolve().parent
    worker=subprocess.Popen([sys.executable,str(here/"worker.py")],cwd=here)
    try:
        ui=subprocess.Popen([
            sys.executable,"-m","streamlit","run",str(here/"app.py"),
            "--server.address","127.0.0.1","--server.port","8501"
        ],cwd=here)
        ui.wait()
    finally:
        worker.terminate()
        try:
            worker.wait(timeout=5)
        except subprocess.TimeoutExpired:
            worker.kill()
        if "ui" in locals() and ui.poll() is None:
            ui.terminate()
            ui.wait(timeout=5)

if __name__=="__main__":
    main()
