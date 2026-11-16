"""Optional local Ollama vision verification. Never auto-accept model proposals."""
import base64
import io
import requests

def verify_ollama(image, ocr_text, model="qwen3-vl:4b", base_url="http://127.0.0.1:11434"):
    buffer=io.BytesIO()
    image.save(buffer,format="PNG")
    prompt=(
        "Compare the attached historical newspaper crop with OCR text. "
        "Write ONLY a literal transcription of what is visible. Preserve punctuation, "
        "historic spelling and line order. Mark unreadable parts [ILLEGIBLE], do not invent. "
        "This is an unreviewed suggestion.\nOCR candidate:\n"+ocr_text
    )
    try:
        response=requests.post(base_url.rstrip("/")+"/api/chat",
        json={"model":model,"stream":False,
              "messages":[{"role":"user","content":prompt,
                           "images":[base64.b64encode(buffer.getvalue()).decode("ascii")]}]},
        timeout=(10,90))
    except requests.Timeout as exc:
        raise RuntimeError("Ollama did not return a verification within 90 seconds. The model may be too slow for this Mac; try fewer regions or a smaller vision model.") from exc
    try:
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc
    proposal=response.json().get("message",{}).get("content","").strip()
    if not proposal:
        raise RuntimeError("Ollama returned an empty verification response")
    return {"status":"proposed_unreviewed","model":model,"suggestion":proposal}
