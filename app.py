"""Human-in-the-loop newspaper OCR; Streamlit local interface."""
import base64
import os
import streamlit as st
from ocr_engine import Settings, process_files, export_bundle

st.set_page_config(page_title="Newspaper OCR Lab",layout="wide")
st.title("Newspaper OCR Lab")
st.caption("Free, local, multi-PDF transcription. Optional OpenAI image verification is billed separately.")
st.warning("Region detection and reading order are experimental. Always review historical sources.")

with st.sidebar:
    st.header("Settings")
    language=st.text_input("Tesseract language","eng")
    dpi=st.slider("PDF rendering DPI",150,400,250,25)
    threshold=st.slider("Review below OCR confidence",0,100,65)
    verify=st.checkbox("Optional OpenAI image verification",False)
    key=""
    model="gpt-4.1-mini"
    limit=12
    if verify:
        st.warning("Image crops will leave your machine. Check rights and institutional permissions.")
        key=st.text_input("OpenAI API key",os.getenv("OPENAI_API_KEY",""),type="password")
        model=st.text_input("Vision model",model)
        limit=st.number_input("Maximum regions verified per page",1,50,12)
    st.caption("Never add API keys to GitHub. Local OCR requires none.")

files=st.file_uploader("Upload multiple newspaper PDFs or images",
    type=["pdf","png","jpg","jpeg","tif","tiff","webp"],
    accept_multiple_files=True)
if st.button("Transcribe all",disabled=not bool(files),type="primary"):
    if verify and not key:
        st.error("Enter an API key or disable optional verification.")
    else:
        bar=st.progress(0)
        settings=Settings(language=language,dpi=dpi,confidence_threshold=threshold,
            verify=verify,model=model,api_key=key or None,max_verify_regions=int(limit))
        with st.spinner("Running layout-aware OCR..."):
            st.session_state["results"]=process_files(
                [(f.name,f.getvalue()) for f in files],settings,
                lambda done,total,name:bar.progress(done/total,text=f"{done}/{total}: {name}"))
        st.success("OCR completed. Review the text and reading order before exporting.")

if "results" in st.session_state:
    results=st.session_state["results"]
    st.subheader("Review and correct")
    st.caption("Edits to transcription and reading order are included in exports. "
        "Article labels are stored in JSON for later analysis; automatic article linking is not yet reliable.")
    for doc in results:
        with st.expander(f"{doc['filename']} — {len(doc['pages'])} pages",
                         expanded=len(results)==1):
            if doc.get("error"):
                st.error(doc["error"])
                continue
            for page in doc["pages"]:
                st.markdown(f"**Page {page['page']}**")
                left,right=st.columns([2,3])
                with left:
                    if page.get("layout_preview"):
                        st.image(base64.b64decode(page["layout_preview"]),
                            caption="Layout regions and proposed sequence")
                with right:
                    for region in page["regions"]:
                        suffix=" — REVIEW" if region["needs_review"] else ""
                        with st.expander(f"Region {region['id']} — OCR confidence "
                                         f"{region['ocr_confidence']:.1f}{suffix}"):
                            region["text"]=st.text_area("Correct transcription",
                                value=region["text"],height=150,
                                key=f"text_{doc['slug']}_{region['id']}")
                            region["reading_order_guess"]=st.number_input(
                                "Reading order",min_value=1,max_value=9999,
                                value=int(region["reading_order_guess"]),
                                key=f"order_{doc['slug']}_{region['id']}")
                            region["article_id"]=st.text_input("Article identifier",
                                value=region.get("article_id") or "",
                                key=f"article_{doc['slug']}_{region['id']}") or None
                            verification=region.get("verification") or {}
                            if verification.get("suggestion"):
                                st.caption("Unreviewed AI suggestion — compare to the scan:")
                                st.code(verification["suggestion"])
    st.download_button("Download corrected DOCX, TXT and JSON (ZIP)",
        export_bundle(results),file_name="newspaper_transcriptions.zip",
        mime="application/zip",type="primary")
