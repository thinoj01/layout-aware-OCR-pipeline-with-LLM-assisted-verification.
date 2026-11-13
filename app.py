"""Streamlit UI for local batch newspaper transcription."""
import os
import streamlit as st
from ocr_engine import Settings, process_files, export_bundle

st.set_page_config(page_title="Newspaper OCR Lab", layout="wide")
st.title("Newspaper OCR Lab")
st.caption("Free local newspaper OCR. Optional paid OpenAI verification is opt-in.")
st.info("Historical article boundaries and reading order require human review. AI corrections are suggestions only.")

with st.sidebar:
    st.header("Processing settings")
    language = st.text_input("Tesseract language", "eng")
    dpi = st.slider("PDF rendering DPI", 150, 400, 250, 25)
    threshold = st.slider("Review confidence threshold", 0, 100, 65)
    verify = st.checkbox("Use optional OpenAI image verification", False)
    key = ""
    model = "gpt-4.1-mini"
    limit = 12
    if verify:
        st.warning("This sends image crops to OpenAI, can incur charges, and requires permission for your source material.")
        key = st.text_input("OpenAI API key", os.environ.get("OPENAI_API_KEY", ""), type="password")
        model = st.text_input("Vision model", model)
        limit = st.number_input("Maximum verified regions per page", 1, 50, 12)
    st.caption("API keys must never be committed to GitHub.")

uploaded = st.file_uploader("Upload multiple newspaper PDFs or images",
    type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "webp"], accept_multiple_files=True)

if st.button("Transcribe all", disabled=not uploaded, type="primary"):
    if verify and not key:
        st.error("Provide an API key or disable verification.")
    else:
        bar = st.progress(0, text="Starting")
        def progress(done, total, name):
            bar.progress(done / total, text=f"{done}/{total}: {name}")
        options = Settings(language=language, dpi=dpi, confidence_threshold=threshold,
            verify=verify, model=model, api_key=key or None, max_verify_regions=int(limit))
        with st.spinner("Processing images locally..."):
            results = process_files([(f.name, f.getvalue()) for f in uploaded], options, progress)
            st.session_state["results"] = results
            st.session_state["archive"] = export_bundle(results)
        st.success("Batch finished. Inspect transcriptions before research use.")

if "results" in st.session_state:
    results = st.session_state["results"]
    st.write(f"{sum(not x.get('error') for x in results)} successful / {len(results)} files")
    st.download_button("Download ZIP with all TXT, DOCX and JSON documents",
        st.session_state["archive"], file_name="newspaper_transcriptions.zip",
        mime="application/zip", type="primary")
    for result in results:
        with st.expander(result["filename"]):
            if result.get("error"):
                st.error(result["error"])
            else:
                for page in result["pages"]:
                    st.subheader(f"Page {page['page']}")
                    st.text_area("Transcription", page["text"], height=230,
                                 key=f"{result['slug']}_{page['page']}")
                    for region in page["regions"]:
                        if region["needs_review"] or region.get("verification"):
                            st.caption(f"{region['id']}: confidence={region['ocr_confidence']}, review={region['needs_review']}")
                            if region["verification"]:
                                st.json(region["verification"])
