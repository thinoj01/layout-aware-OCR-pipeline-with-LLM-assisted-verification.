"""One source-grounded research chatbot; OpenAI used only for selected passages."""
from __future__ import annotations

import json
import streamlit as st
from jobs import list_jobs
from research import (
    corpus_from_jobs, uploaded_passages, retrieve, top_terms, openai_answer,
    get_openai_api_key, build_research_request, verified_citations,
)

MODEL = "gpt-4.1-mini"
NO_EVIDENCE = "I could not find supporting evidence in the selected newspaper passages."


def show_sources(hits):
    """Expose the exact text, filename, page, region and researcher review status."""
    for n, hit in enumerate(hits, 1):
        page = f"page {hit['page']}" if hit.get("page") is not None else "uploaded TXT"
        with st.expander(f"[S{n}] {hit['filename']} — {page}, {hit['region_id']}"):
            if hit.get("article_id"):
                st.caption(f"Article: {hit['article_id']}")
            st.caption("Manually reviewed" if hit.get("reviewed") else
                       "OCR text — not manually verified")
            st.text(hit["text"])


def overview_question(text):
    """Recognize broad theme-summary questions; use a representative sample."""
    q=" ".join(text.lower().split())
    return any(phrase in q for phrase in (
        "main themes","major themes","recurring themes","common themes",
        "main topics","major topics","recurring topics",
        "summarize the collection","summarise the collection",
        "overview of the newspapers","what are these newspapers about",
        "what do these articles discuss",
    ))


def choose_evidence(passages, question, history, limit=12):
    """Favor direct matches; for broad theme queries sample across documents.

    Selection is bounded to keep API costs predictable. It is not exhaustive
    qualitative coding or a semantic vector search.
    """
    if overview_question(question):
        terms=" ".join(term for term, _ in top_terms(passages,top_k=16))
        ranked=retrieve(passages,terms,limit=max(limit*3,20))
        # Ensure diversity across source documents when possible.
        chosen=[]
        seen=set()
        for hit in ranked:
            doc=hit["job_id"] or hit["filename"]
            if doc not in seen:
                chosen.append(hit)
                seen.add(doc)
            if len(chosen)>=limit:
                return chosen
        for hit in ranked:
            if hit["source_key"] not in {p["source_key"] for p in chosen}:
                chosen.append(hit)
            if len(chosen)>=limit:
                break
        return chosen
    # Use current research question first. Add prior question only for elliptical
    # follow-ups, never as an independent factual source.
    hits=retrieve(passages,question,limit=limit)
    if not hits and history:
        previous=str(history[-1].get("query",""))
        hits=retrieve(passages,previous+" "+question,limit=limit)
    return hits


def show():
    st.title("Newspaper research chatbot")
    st.caption("Ask about people, events, topics, themes, framing, and differences "
               "across the selected transcripts. Answers must be grounded in the passages.")

    completed=[j for j in list_jobs() if j["status"]=="completed"]
    choices={f"{j['filename']} [{j['id'][:8]}]":j["id"] for j in completed}
    chosen=st.multiselect("Newspapers to analyze",list(choices),
                          default=list(choices),key="research_docs")
    selected=[choices[label] for label in chosen]
    passages=corpus_from_jobs(selected) if selected else []

    uploaded=st.file_uploader("Or add an existing transcription (.txt)",type=["txt"],
                              key="research_txt")
    if uploaded is not None:
        try:
            passages.extend(uploaded_passages(uploaded.name,uploaded.getvalue()))
        except (UnicodeDecodeError,ValueError) as exc:
            st.error(f"Could not read transcript: {exc}")

    total_docs=len(selected)+(1 if uploaded is not None else 0)
    st.caption(f"{total_docs} selected document(s) | {len(passages)} searchable text passages")

    with st.expander("OpenAI API setup",expanded=not bool(get_openai_api_key())):
        if get_openai_api_key():
            st.success("OpenAI API key configured. Research model: "+MODEL)
        else:
            st.warning("OpenAI API key not configured.")
        st.code('OPENAI_API_KEY = "paste-your-own-key-here"',language="toml")
        st.caption("Put this line in .streamlit/secrets.toml on your Mac "
                   "(inside this project folder). Never commit your key to GitHub. "
                   "Restart the application after changing the file.")

    if not passages:
        st.info("No transcripts selected. Complete a newspaper OCR job or upload "
                "a TXT transcript to begin source-grounded research.")

    if "research_conversation" not in st.session_state:
        st.session_state["research_conversation"]=[]
    history=st.session_state["research_conversation"]

    if st.button("Clear conversation",key="clear_research_chat"):
        st.session_state["research_conversation"]=[]
        st.rerun()

    for turn in history:
        with st.chat_message("user"):
            st.write(turn["query"])
        with st.chat_message("assistant"):
            st.markdown(turn["answer"])
            if turn.get("note"):
                st.caption(turn["note"])
            if turn["hits"]:
                with st.expander(f"View supporting newspaper passages ({len(turn['hits'])})"):
                    show_sources(turn["hits"])
                st.download_button("Export answer with source passages",
                    data=json.dumps(turn,ensure_ascii=False,indent=2),
                    file_name=f"research_answer_{turn['index']}.json",
                    mime="application/json",key=f"download_research_{turn['index']}")

    question=st.chat_input("Ask a question about the selected newspapers",
                           disabled=not bool(passages))
    if not question:
        return

    record={"index":len(history)+1,"query":question,"hits":[],"answer":NO_EVIDENCE}
    matches=choose_evidence(passages,question,history)
    record["hits"]=matches

    if matches:
        if not get_openai_api_key():
            record["answer"]=(
                "OpenAI is not configured. Add OPENAI_API_KEY to "
                ".streamlit/secrets.toml to analyze the selected passages.")
        else:
            try:
                with st.status("Reading newspaper evidence...",expanded=False):
                    request=build_research_request(question,matches,history)
                    result=openai_answer(request,model=MODEL,strict=True)
                if verified_citations(result,len(matches)):
                    record["answer"]=result
                elif result.strip()==NO_EVIDENCE:
                    record["answer"]=NO_EVIDENCE
                elif result.strip()=="I can only discuss the selected newspaper passages.":
                    record["answer"]=result.strip()
                else:
                    # Reject uncited or fabricated [S...] numbers; fail closed.
                    record["answer"]=(
                        "I couldn't produce a sufficiently cited answer from the "
                        "retrieved passages. Please rephrase the question.")
                    record["note"]="The model response lacked valid source citations."
            except Exception as exc:
                record["answer"]="OpenAI could not complete this request: "+str(exc)
    elif not passages:
        record["answer"]="Select newspaper transcripts before asking a question."
    else:
        record["note"]=(
            "No passage matched the question using the current text search. "
            "This is not proof that the topic is absent from the full collection.")

    if overview_question(question) and matches:
        record["note"]=(
            "This overview uses representative retrieved passages, not the entire "
            "corpus. Verify broader claims before coding themes or estimating prevalence.")
    st.session_state["research_conversation"].append(record)
    st.rerun()
