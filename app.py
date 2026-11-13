"""Local OCR research interface with live queue, safe removal and explicit batch reset."""
import base64
import os
import streamlit as st
from jobs import (submit,list_jobs,delete,delete_many,request_cancel,
                  queue_estimates,_duration)
from review import (load_result,save_region,export_job,export_completed_collection)

st.set_page_config(page_title="Historical Newspaper OCR Lab",layout="wide")
st.title("Historical Newspaper OCR Lab")
st.caption("Local research prototype: batch OCR, live queue, saved corrections and exports.")
st.warning("OCR regions and reading order still require human verification. Run locally only.")

def mark_download():
    # Download events are not proof that the user saved/opened the ZIP.
    st.session_state["reset_prompt"]=True

def invalidate_archive():
    st.session_state.pop("collection_archive",None)
    st.session_state.pop("collection_ids",None)

with st.sidebar:
    st.header("New transcription job")
    backend=st.selectbox("Layout detector",["pretrained","xycut"],index=0)
    language=st.text_input("Tesseract languages","eng")
    dpi=st.slider("PDF rendering DPI",150,400,250,25)
    threshold=st.slider("Review below OCR confidence",0,100,80)
    verification=st.selectbox("Vision verification",["Off","Local Ollama","OpenAI API"])
    model="gpt-4.1-mini"
    key=None
    if verification=="Local Ollama":
        model="ollama:"+st.text_input("Local Ollama model","qwen3-vl:4b")
    elif verification=="OpenAI API":
        key=os.getenv("OPENAI_API_KEY","")
        st.caption("Set OPENAI_API_KEY in your shell. The key is not stored in SQLite.")
        model=st.text_input("OpenAI model",model)
    limit=st.number_input("Maximum verified regions per page",1,40,5)
    st.caption("The UI and files are local. Ollama requires a separate local server.")

if "upload_nonce" not in st.session_state:
    st.session_state["upload_nonce"]=0
uploads=st.file_uploader("Upload multiple newspaper PDFs and scanned images",
    accept_multiple_files=True,
    type=["pdf","jpg","jpeg","png","tif","tiff","webp"],
    key=f"newspaper_upload_{st.session_state['upload_nonce']}")
if st.button("Add files to processing queue",disabled=not uploads,type="primary"):
    if verification=="OpenAI API" and not key:
        st.error("Set OPENAI_API_KEY or disable OpenAI verification.")
    else:
        opts={"layout_backend":backend,"language":language,"dpi":dpi,
              "confidence_threshold":threshold,"verify":verification!="Off",
              "model":model,"max_verify_regions":int(limit)}
        successes=0
        for f in uploads:
            try:
                submit(f.name,f.getvalue(),opts)
                successes+=1
            except Exception as exc:
                st.error(f"{f.name}: {exc}")
        invalidate_archive()
        st.success(f"Added {successes} file(s) to the persistent queue.")
        st.session_state["upload_nonce"]+=1
        st.rerun()

st.divider()
st.subheader("Processing queue")
st.caption("Automatically refreshed every 5 seconds. Estimates are approximate and use measured "
           "seconds per page; the first file may show 'Calculating...' until a page finishes.")

@st.fragment(run_every="5s")
def live_queue():
    """Refresh only the status pane; never reset unsaved article edits."""
    records=list_jobs()
    estimates=queue_estimates(records)
    counts={status:sum(j["status"]==status for j in records)
            for status in ("running","queued","completed","failed","cancelled")}
    cols=st.columns(4)
    for col,title,value in zip(cols,["Running","Waiting","Completed","Failed/cancelled"],
                               [counts["running"],counts["queued"],counts["completed"],
                                counts["failed"]+counts["cancelled"]]):
        col.metric(title,value)
    st.caption("Queue status refreshes every 5 seconds. Select Refresh completed results "
               "below when a job finishes.")
    active=sorted((j for j in records if j["status"] in ("running","queued")),
                  key=lambda j:(j["status"]!="running",j["created"],j["id"]))
    if not active:
        st.info("No files currently waiting or processing.")
    for job in active:
        estimate=estimates.get(job["id"])
        if job["status"]=="running":
            remaining=_duration(estimate["seconds_to_finish"]) if estimate else "Calculating..."
            st.markdown(f"**Processing: {job['filename']}**")
            st.progress(min(1.0,job["done"]/max(1,job["total"])),
                        text=f"{job['done']}/{job['total']} pages — estimated remaining {remaining}")
            if job.get("cancel_requested"):
                st.caption("Cancellation requested; finishes current page before stopping.")
            elif st.button("Cancel running newspaper",key="cancel_"+job["id"]):
                request_cancel(job["id"])
                st.rerun(scope="fragment")
        else:
            position=estimate["position"] if estimate else "?"
            wait=_duration(estimate["seconds_to_start"]) if estimate else "Calculating..."
            x,y=st.columns([4,1])
            x.write(f"**Waiting #{position}: {job['filename']}** — {job['total']} pages — "
                    f"estimated start in {wait}")
            if y.button("Remove",key="remove_"+job["id"]):
                try:
                    delete(job["id"])
                    invalidate_archive()
                    st.rerun(scope="fragment")
                except ValueError:
                    st.warning("The worker already started this job; use Cancel instead.")

live_queue()
st.divider()
st.subheader("Completed newspapers and review")
if st.button("Refresh completed results",key="refresh_results"):
    st.rerun()
jobs=list_jobs()
completed=[job for job in jobs if job["status"]=="completed"]
finished=[job for job in jobs if job["status"] in ("completed","failed","cancelled")]
for job in sorted(finished,key=lambda x:(-x["created"],x["id"])):
    with st.expander(f"{job['status'].title()}: {job['filename']}",
                     expanded=False):
        if job.get("error"):
            st.error(job["error"])
        if job["status"]=="completed":
            try:
                data=load_result(job["id"])
            except Exception as exc:
                st.error(f"Cannot read saved transcription: {exc}")
                data=None
            if data:
                for page in data["pages"]:
                    st.markdown(f"**Page {page['page']}**")
                    left,right=st.columns([2,3])
                    with left:
                        if page.get("layout_preview"):
                            st.image(base64.b64decode(page["layout_preview"]),
                                caption="Detected regions and proposed order")
                    with right:
                        for region in page["regions"]:
                            with st.expander(
                                f"{region['id']} | {region['type']} | confidence {region['ocr_confidence']}"
                            ):
                                with st.form(key=f"edit_{job['id']}_{region['id']}"):
                                    candidate=st.text_area("Transcription",
                                        value=region["text"],height=140)
                                    order=st.number_input("Reading order",min_value=1,
                                        value=int(region["reading_order_guess"]))
                                    article=st.text_input("Article ID",value=region.get("article_id") or "")
                                    if st.form_submit_button("Save correction"):
                                        try:
                                            save_region(job["id"],page["page"],
                                                region["id"],candidate,int(order),article)
                                            invalidate_archive()
                                            st.success("Saved to disk")
                                        except Exception as exc:
                                            st.error(str(exc))
                                if region.get("verification"):
                                    st.caption("Unreviewed vision-model suggestion")
                                    st.json(region["verification"])
                st.download_button("Download newspaper ZIP",
                    data=export_job(job["id"]),
                    file_name=f"{job['id']}_transcriptions.zip",
                    mime="application/zip",key="download_"+job["id"],
                    on_click=mark_download)
        if st.button("Delete this newspaper and stored files",key="delete_"+job["id"]):
            try:
                delete(job["id"])
                invalidate_archive()
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

if completed:
    st.divider()
    ids=tuple(j["id"] for j in completed)
    if st.session_state.get("collection_ids")!=ids:
        invalidate_archive()
    if st.button(f"Prepare collection ZIP ({len(completed)} completed newspapers)"):
        with st.spinner("Building archive..."):
            st.session_state["collection_archive"]=export_completed_collection(ids)
            st.session_state["collection_ids"]=ids
    if "collection_archive" in st.session_state:
        st.download_button("Download all completed newspapers (ZIP)",
            data=st.session_state["collection_archive"],
            file_name="newspaper_collection.zip",
            mime="application/zip",on_click=mark_download,key="download_collection")

st.divider()
if not st.session_state.get("reset_prompt"):
    if jobs and st.button("Start a new batch / clear queue"):
        st.session_state["reset_prompt"]=True
        st.rerun()
else:
    st.info("ZIP download started or reset requested. Did you save your ZIP? "
            "To start over, confirm below. Results are never deleted just by downloading.")
    if any(j["status"]=="running" for j in list_jobs()):
        st.warning("Cancel the running newspaper first and wait until it stops.")
    a,b=st.columns(2)
    if a.button("Keep results",key="keep_results"):
        st.session_state["reset_prompt"]=False
        st.rerun()
    if b.button("Clear all jobs and start a new batch",key="confirm_reset",
                disabled=any(j["status"]=="running" for j in list_jobs())):
        try:
            delete_many([job["id"] for job in list_jobs()])
            invalidate_archive()
            st.session_state["reset_prompt"]=False
            st.session_state["upload_nonce"]+=1
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
