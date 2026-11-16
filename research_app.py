"""Research workspace for searching OCR passages and exploring themes with citations.

Accessed from the main Streamlit app's Research analysis view. This keeps
transcription and analysis separate and preserves source provenance.
"""
from __future__ import annotations
import json
import streamlit as st
from research import (
    corpus_from_jobs,uploaded_passages,retrieve,top_terms,
    build_prompt,local_answer,openai_answer,mentioned_citations
)
from jobs import list_jobs

def show_sources(hits):
    if not hits:
        st.info("No matching saved passages.")
        return
    for i,hit in enumerate(hits,1):
        description=(f"Page {hit['page']}, region {hit['region_id']}"
                     if hit["page"] is not None else hit["region_id"])
        source=f"[S{i}] {hit['filename']} — {description}"
        with st.expander(source,expanded=i<=2):
            st.caption(
                f"Source: {hit['origin']} | article ID: {hit.get('article_id') or 'unassigned'}"
                f" | reviewed: {'yes' if hit['reviewed'] else 'no'}"
                f" | OCR confidence: {hit['ocr_confidence'] if hit['ocr_confidence'] is not None else 'N/A'}"
            )
            st.text(hit["text"])
            if hit["job_id"]:
                st.caption("Open this completed newspaper in the OCR processing view "
                           "to compare the passage against the scanned page.")

def show():
    st.title("Newspaper research and thematic analysis")
    st.caption("Search saved newspaper passages, explore recurring terms, and ask "
               "questions with traceable source excerpts. This is a research prototype, "
               "not a validated topic model or autonomous qualitative coder.")
    jobs=[j for j in list_jobs() if j["status"]=="completed"]
    ids=[j["id"] for j in jobs]
    choices={f"{j['filename']} [{j['id'][:8]}]":j["id"] for j in jobs}
    labels=st.multiselect("Completed newspapers to analyze",
        list(choices),default=list(choices),
        help="Only completed documents and their saved manual corrections are included.")
    selected=[choices[x] for x in labels]
    passages=corpus_from_jobs(selected) if selected else []
    uploaded=st.file_uploader("Optionally add a transcribed TXT file",
                              type=["txt"],key="research_txt")
    if uploaded is not None:
        try:
            passages+=uploaded_passages(uploaded.name,uploaded.getvalue())
        except (UnicodeDecodeError,ValueError) as exc:
            st.error(f"Could not read TXT: {exc}")
    if not passages:
        st.info("Complete an OCR job in the Processing view or upload a UTF-8 TXT transcript.")
        return
    doc_count=len({p["source_key"].split(":")[0] if p["job_id"] is None
                   else p["job_id"] for p in passages})
    a,b=st.columns(2)
    a.metric("Passages available",len(passages))
    b.metric("Selected documents",len(selected)+(1 if uploaded else 0))
    mode=st.radio("Analysis tool",["Evidence search and chatbot","Theme exploration"],
                  horizontal=True,key="research_mode")
    with st.expander("Model configuration",expanded=False):
        provider=st.radio("Answer provider",
            ["Local Ollama (private)","OpenAI API (optional cloud)"],
            help="Retrieval is local. Only retrieved passages are sent to a cloud API if selected.")
        model=st.text_input("Model",value="qwen3-vl:4b" if provider.startswith("Local") else "gpt-4.1-mini",
            key=f"research_model_{'local' if provider.startswith('Local') else 'openai'}")
        st.caption("Ollama is started automatically when used, provided the executable is installed. "
                   "A model must fit the Mac's available memory. An external API sends retrieved "
                   "newspaper excerpts outside this computer.")

    if mode=="Theme exploration":
        st.subheader("Repeated terms in selected passages")
        st.caption("Counts are numbers of OCR passages containing each word, not numbers "
                   "of articles or validated semantic topic categories. Historical spelling "
                   "and OCR errors affect these counts.")
        frequencies=top_terms(passages,top_k=30)
        st.dataframe([{"term":term,"passages_containing_term":count}
                     for term,count in frequencies],use_container_width=True,hide_index=True)
        if st.button("Generate evidence-backed theme analysis",type="primary"):
            # Use a reproducible lexical query to retrieve representative evidence.
            tokens=" ".join(term for term,_ in frequencies[:12])
            hits=retrieve(passages,tokens,limit=8)
            if not hits:
                st.warning("No evidence passages could be selected.")
                return
            prompt=build_prompt("What recurring topics and themes appear?",hits,theme_mode=True)
            try:
                with st.status("Analyzing passages...",expanded=True) as status:
                    if provider.startswith("Local"):
                        result=local_answer(prompt,model,progress=lambda message:st.write(message))
                    else:
                        result=openai_answer(prompt,model)
                    status.update(label="Theme analysis complete",state="complete")
                st.markdown(result)
                st.subheader("Source passages used")
                show_sources(hits)
                st.caption("Themes are AI-generated interpretations from a limited sample, not "
                           "an exhaustive thematic analysis. Verify them against the full corpus.")
                st.download_button("Download theme analysis and evidence (JSON)",
                    data=json.dumps({"analysis":result,"sources":hits},ensure_ascii=False,indent=2),
                    file_name="newspaper_themes_evidence.json",mime="application/json")
            except Exception as exc:
                st.error(f"Theme analysis failed: {exc}")
        return

    st.subheader("Research chatbot")
    st.caption("Follow-up questions use your recent question history to retrieve related "
               "passages. Each answer has its own evidence, which remains visible in this session.")
    if "research_conversation" not in st.session_state:
        st.session_state["research_conversation"]=[]
    history=st.session_state["research_conversation"]
    control,other=st.columns([1,3])
    if control.button("Clear chat",key="clear_research_chat"):
        st.session_state["research_conversation"]=[]
        history=[]
        st.rerun()
    limit=st.slider("Maximum source passages per answer",2,10,6)
    for turn in history:
        with st.chat_message("user"):
            st.write(turn["query"])
        with st.chat_message("assistant"):
            if turn.get("answer"):
                st.markdown(turn["answer"])
            elif turn.get("error"):
                st.error(turn["error"])
            with st.expander(f"Inspect {len(turn['hits'])} source passages"):
                show_sources(turn["hits"])
            if turn["hits"]:
                st.download_button("Export this answer and its evidence (JSON)",
                    data=json.dumps(turn,ensure_ascii=False,indent=2),
                    file_name=f"research_answer_{turn['index']}.json",
                    key=f"export_research_{turn['index']}",mime="application/json")
    query=st.chat_input("Ask about the newspapers, or follow up on a previous theme")
    if query:
        previous=[item["query"] for item in history[-2:]]
        retrieval_query=" ".join(previous+[query]) if previous else query
        hits=retrieve(passages,retrieval_query,limit=limit)
        record={"index":len(history)+1,"query":query,"hits":hits,
                "provider":provider,"model":model}
        if not hits:
            record["error"]="No matching text found. Try another keyword or spelling."
        else:
            try:
                with st.status("Retrieving and interpreting evidence...",expanded=True) as status:
                    prompt=build_prompt(query,hits)
                    if history:
                        context="Previous dialogue (background only; do not use as evidence):\\n"
                        for turn in history[-3:]:
                            context+=("Researcher: "+turn["query"]+"\\n"
                                      "Assistant: "+(turn.get("answer") or "")[:1500]+"\\n")
                        prompt=context+"\\n"+prompt
                    if provider.startswith("Local"):
                        response=local_answer(prompt,model,
                            progress=lambda message:st.write(message))
                    else:
                        response=openai_answer(prompt,model)
                    status.update(label="Answer generated",state="complete")
                record["answer"]=response
                if not mentioned_citations(response,len(hits)):
                    record["citation_warning"]="The model did not provide valid source citation markers."
            except Exception as exc:
                record["error"]=str(exc)
        st.session_state["research_conversation"].append(record)
        st.rerun()
