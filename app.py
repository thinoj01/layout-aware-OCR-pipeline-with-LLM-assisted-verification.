"""Local research GUI. Start with python launch.py for background processing."""
import base64
import json
import os
from pathlib import Path
import streamlit as st
from jobs import submit, list_jobs, get_job, delete
from review import load_result,save_region,export_job,export_completed_collection

st.set_page_config(page_title="Historical Newspaper OCR Lab",layout="wide")
st.title("Historical Newspaper OCR Lab")
st.caption("Local research prototype: uploads, persistent queue, human review, article-level export.")
st.warning("Model detections and reading order must be checked against the original scans. Do not expose this unauthenticated UI to the public Internet.")

with st.sidebar:
    st.header("New transcription job")
    backend=st.selectbox("Layout detector",[
        "pretrained","xycut"],index=0,
        help="Pretrained requires optional ML dependencies and model weights. XY-cut is a geometric fallback.")
    language=st.text_input("Tesseract languages","eng")
    dpi=st.slider("PDF rendering DPI",150,400,250,25)
    threshold=st.slider("Review below OCR confidence",0,100,80)
    verification=st.selectbox("Vision verification",["Off","Local Ollama","OpenAI API"])
    model="gpt-4.1-mini"
    key=None
    if verification=="Local Ollama":
        model="ollama:"+st.text_input("Local Ollama model","qwen3-vl:4b")
        st.caption("Ollama must be running at 127.0.0.1:11434. Suggestions are never auto-applied.")
    elif verification=="OpenAI API":
        key=os.getenv("OPENAI_API_KEY","")
        st.caption("Set OPENAI_API_KEY in your terminal before starting the app. It is not saved to the job database.")
        model=st.text_input("OpenAI model",model)
    limit=st.number_input("Maximum vision-verified regions/page",1,40,5)
    st.caption("All source files and output remain on this computer unless verification is explicitly enabled.")
uploads=st.file_uploader("Upload multiple newspaper PDFs and scanned images",accept_multiple_files=True,
    type=["pdf","jpg","jpeg","png","tif","tiff","webp"])
if st.button("Add files to processing queue",disabled=not uploads,type="primary"):
    if verification=="OpenAI API" and not key:
        st.error("Enter an OpenAI API key or disable verification.")
    else:
        opts={"layout_backend":backend,"language":language,"dpi":dpi,
              "confidence_threshold":threshold,
              "verify":verification!="Off","model":model,
              "max_verify_regions":int(limit)}
        successes=0
        for f in uploads:
            try:
                submit(f.name,f.getvalue(),opts)
                successes+=1
            except Exception as exc:
                st.error(f"{f.name}: {exc}")
        st.success(f"Queued {successes} file(s). Start the background worker using python launch.py.")
st.divider()
st.subheader("Processing queue")
if st.button("Refresh jobs"):
    st.rerun()
jobs=list_jobs()
for job in jobs:
    status=f"{job['status']} — {job['done']} pages"
    with st.expander(f"{job['filename']} | {status}",expanded=job["status"]=="completed"):
        st.caption(f"Job {job['id']}")
        if job.get("error"):
            st.error(job["error"])
        if job["status"]=="completed":
            try:
                data=load_result(job["id"])
            except Exception as exc:
                st.error(str(exc));continue
            for page in data["pages"]:
                st.markdown(f"### Page {page['page']}")
                left,right=st.columns([2,3])
                with left:
                    if page.get("layout_preview"):
                        st.image(base64.b64decode(page["layout_preview"]),
                                 caption="Detected blocks and proposed reading order")
                with right:
                    for region in page["regions"]:
                        with st.expander(f"{region['id']} | {region['type']} | confidence {region['ocr_confidence']}"):
                            with st.form(key=f"edit_{job['id']}_{region['id']}"):
                                text=st.text_area("Transcription",region["text"],height=150)
                                order=st.number_input("Reading order",min_value=1,
                                    value=int(region["reading_order_guess"]))
                                article=st.text_input("Article ID (same ID to merge fragments)",
                                                       region.get("article_id") or "")
                                if st.form_submit_button("Save correction"):
                                    try:
                                        save_region(job["id"],page["page"],region["id"],
                                                    text,int(order),article)
                                        st.success("Saved to disk")
                                    except Exception as exc:
                                        st.error(str(exc))
                            verification_result=region.get("verification")
                            if verification_result:
                                st.caption("Unreviewed vision-model suggestion")
                                st.json(verification_result)
            st.download_button("Download corrected documents and articles (ZIP)",
                data=export_job(job["id"]),file_name=f"{job['id']}_transcriptions.zip",
                mime="application/zip",key=f"download_{job['id']}")
        if job["status"]!="running" and st.button("Delete job and stored files",key="delete_"+job["id"]):
            delete(job["id"]);st.rerun()
if not jobs:
    st.info("No jobs. Upload newspaper scans using the file uploader above.")

completed=[job["id"] for job in jobs if job["status"]=="completed"]
if completed:
    st.divider()
    if st.button(f"Prepare collection ZIP ({len(completed)} completed newspapers)"):
        with st.spinner("Bundling all completed newspaper exports..."):
            st.session_state["collection_archive"]=export_completed_collection(completed)
    if "collection_archive" in st.session_state:
        st.download_button("Download all completed newspapers (ZIP)",
            data=st.session_state["collection_archive"],
            file_name="newspaper_collection.zip",
            mime="application/zip")
