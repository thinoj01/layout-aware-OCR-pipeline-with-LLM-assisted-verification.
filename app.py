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
def queue_panel():
    jobs=list_jobs()
    estimates=queue_estimates(jobs)
    running=[j for j in jobs if j["status"]=="running"]
    pending=[j for j in jobs if j["status"]=="queued"]
    completed=[j for j in jobs if j["status"]=="completed"]
    cols=st.columns(4)
    for col,label,count in zip(cols,["Running","Waiting","Completed","Failed/cancelled"],
                               [len(running),len(pending),len(completed),
                                sum(j["status"] in ("failed","cancelled") for j in jobs)]):
        col.metric(label,count)

    if st.button("Refresh now",key="refresh_queue"):
        st.rerun(scope="fragment")
    for job in sorted(jobs,key=lambda j:(j["created"],j["id"])):
        status=job["status"]
        estimate=estimates.get(job["id"])
        if status=="running":
            remaining=_duration(estimate["seconds_to_finish"]) if estimate else "Calculating..."
            label=f"Processing: {job['filename']} — {job['done']}/{job['total']} pages — ETA {remaining}"
        elif status=="queued":
            position=estimate["position"] if estimate else "?"
            waiting=_duration(estimate["seconds_to_start"]) if estimate else "Calculating..."
            label=f"Waiting #{position}: {job['filename']} — {job['total']} pages — Starts in {waiting}"
        else:
            label=f"{status.title()}: {job['filename']} — {job['done']}/{job['total']} pages"

        with st.expander(label,expanded=status=="running"):
            st.caption(f"Job ID: {job['id']}")
            if status=="running":
                st.progress(min(1.0,job["done"]/max(1,job["total"])),
                            text=f"{job['done']} of {job['total']} pages completed")
                if job.get("cancel_requested"):
                    st.info("Cancellation requested. This file stops after the current page.")
                elif st.button("Cancel this running newspaper",key="cancel_"+job["id"]):
                    request_cancel(job["id"])
                    st.rerun(scope="fragment")
            elif status=="queued":
                st.progress(0,text=f"Queue position {estimate['position']} of {len(pending)}")
                st.caption("Estimated waiting time: "+
                           (_duration(estimate["seconds_to_start"]) if estimate else "Calculating..."))
                if st.button("Remove from queue",key="remove_"+job["id"]):
                    delete(job["id"])
                    invalidate_archive()
                    st.rerun(scope="fragment")
            else:
                if job.get("error"):
                    st.error(job["error"])
                if status=="completed":
                    try:
                        data=load_result(job["id"])
                    except Exception as exc:
                        st.error(str(exc))
                        data=None
                    if data:
                        for page in data["pages"]:
                            st.markdown(f"**Page {page['page']}**")
                            left,right=st.columns([2,3])
                            with left:
                                if page.get("layout_preview"):
                                    st.image(base64.b64decode(page["layout_preview"]),
                                             caption="Detected regions and provisional order")
                            with right:
                                for region in page["regions"]:
                                    with st.expander(
                                        f"{region['id']} | {region['type']} | confidence {region['ocr_confidence']}"
                                    ):
                                        with st.form(key=f"edit_{job['id']}_{region['id']}"):
                                            candidate=st.text_area("Transcription",
                                                region["text"],height=140)
                                            order=st.number_input("Reading order",min_value=1,
                                                value=int(region["reading_order_guess"]))
                                            article=st.text_input("Article ID",
                                                region.get("article_id") or "")
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
                if st.button("Delete newspaper and its saved files",key="delete_"+job["id"]):
                    delete(job["id"])
                    invalidate_archive()
                    st.rerun(scope="fragment")

    if completed:
        st.divider()
        current_ids=tuple(j["id"] for j in completed)
        if st.session_state.get("collection_ids")!=current_ids:
            invalidate_archive()
        if st.button(f"Prepare collection ZIP ({len(completed)} completed newspapers)",
                     key="prepare_collection"):
            with st.spinner("Building ZIP..."):
                st.session_state["collection_archive"]=export_completed_collection(current_ids)
                st.session_state["collection_ids"]=current_ids
        if "collection_archive" in st.session_state:
            st.download_button("Download all completed newspapers (ZIP)",
                data=st.session_state["collection_archive"],
                file_name="newspaper_collection.zip",mime="application/zip",
                on_click=mark_download,key="download_collection")

    if st.session_state.get("reset_prompt"):
        st.divider()
        st.info("ZIP download initiated. Do you want to restart with an empty queue? "
                "Confirm that the ZIP is saved before clearing. No files are deleted automatically.")
        if running:
            st.warning("A newspaper is processing. Cancel it and wait for cancellation "
                       "before clearing this batch.")
        first,second=st.columns(2)
        if first.button("Keep results and continue",key="keep_all"):
            st.session_state["reset_prompt"]=False
            st.rerun(scope="fragment")
        if second.button("Clear all jobs and start a new batch",key="reset_all",
                         disabled=bool(running)):
            try:
                delete_many([j["id"] for j in jobs])
                invalidate_archive()
                st.session_state["reset_prompt"]=False
                st.session_state["upload_nonce"]+=1
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    elif jobs:
        if st.button("Start a new batch / clear queue",key="prompt_manual_reset"):
            st.session_state["reset_prompt"]=True
            st.rerun(scope="fragment")
    else:
        st.info("No newspapers in the queue. Add PDFs or image scans above.")

queue_panel()
