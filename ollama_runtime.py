"""Local Ollama lifecycle managed by the OCR worker.

Ollama always needs its own local inference process, but users do not need
to start it manually. The worker connects to an existing local instance
or starts it on demand and stops only a process it created.

The configured vision model is fetched on first use only when not present.
All requests remain at 127.0.0.1; no paid API key is needed.
"""
from __future__ import annotations
import os
import shutil
import subprocess
import time
from urllib.parse import urlsplit
import requests

BASE_URL="http://127.0.0.1:11434"
_spawned=None

def _check_local_url(base_url):
    parsed=urlsplit(base_url)
    if parsed.scheme!="http" or parsed.hostname not in {"127.0.0.1","localhost"}:
        raise ValueError("Ollama URL must use the local computer, not a remote server")
    return base_url.rstrip("/")

def is_running(base_url=BASE_URL):
    base_url=_check_local_url(base_url)
    try:
        response=requests.get(base_url+"/api/tags",timeout=2)
        response.raise_for_status()
        return isinstance(response.json().get("models"),list)
    except (requests.RequestException,ValueError):
        return False

def _available_models(base_url):
    response=requests.get(base_url+"/api/tags",timeout=10)
    response.raise_for_status()
    return {str(x.get("name","")) for x in response.json().get("models",[])}

def ensure_ollama(model="qwen3-vl:4b",base_url=BASE_URL):
    """Ensure the local server and requested model are ready.

    On first use, model download may be large and need internet. This runs
    inside the worker, not inside Streamlit's UI event loop.
    """
    global _spawned
    base_url=_check_local_url(base_url)
    if not model or any(c in model for c in "\n\r\0"):
        raise ValueError("Invalid Ollama model name")
    if not is_running(base_url):
        executable=shutil.which("ollama")
        if not executable:
            raise RuntimeError("Ollama is not installed. Run 'brew install ollama' once, then select Local Ollama again.")
        if _spawned is None or _spawned.poll() is not None:
            _spawned=subprocess.Popen(
                [executable,"serve"],
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                env={**os.environ,"OLLAMA_HOST":"127.0.0.1:11434"})
        for _ in range(30):
            if is_running(base_url):
                break
            if _spawned.poll() is not None:
                raise RuntimeError("Ollama exited while starting. Check Ollama installation.")
            time.sleep(0.5)
        else:
            raise RuntimeError("Ollama did not become ready on localhost:11434.")
    installed=_available_models(base_url)
    if model not in installed and not any(name.split(":")[0]==model.split(":")[0]
                                          and (":" not in model) for name in installed):
        # Let Ollama itself download the specified public model.
        executable=shutil.which("ollama")
        if not executable:
            raise RuntimeError("Ollama command not found for model installation.")
        try:
            subprocess.run([executable,"pull",model],check=True,timeout=1200,
                           stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,
                           text=True)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Ollama model download timed out; check internet and disk space.") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError("Unable to download Ollama model: "+(exc.stderr or "")[-400:]) from exc
    return base_url

def stop_managed_ollama():
    global _spawned
    if _spawned is not None and _spawned.poll() is None:
        _spawned.terminate()
        try:
            _spawned.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _spawned.kill()
    _spawned=None
