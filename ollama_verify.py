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
    response=requests.post(base_url.rstrip("/")+"/api/chat",
        json={"model":model,"stream":False,
              "messages":[{"role":"user","content":prompt,
                           "images":[base64.b64encode(buffer.getvalue()).decode("ascii")]}]},
        timeout=180)
    response.raise_for_status()
    proposal=response.json().get("message",{}).get("content","").strip()
    return {"status":"proposed_unreviewed","model":model,"suggestion":proposal}
