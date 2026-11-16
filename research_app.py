"""Research workspace for searching OCR passages and exploring themes with citations.

Accessed from the main Streamlit app's Research analysis view. This keeps
transcription and analysis separate and preserves source provenance.
"""
from __future__ import annotations
import json
import streamlit as st
from research import (
    corpus_from_jobs,uploaded_passages,retrieve,top_terms,
    build_prompt,local_answer,openai_answer,mentioned_citations,
    classify_chat_message,interface_reply
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
        st.warning("No newspaper passages are loaded. Complete an OCR job, select a completed newspaper above, or upload a TXT transcript. You can still check the chat interface without documents.")
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
        if st.button("Test model connection",key="research_model_check"):
            try:
                with st.spinner("Contacting the selected model..."):
                    if provider.startswith("Local"):
                        answer=local_answer("Reply with the single word READY.",model)
                    else:
                        answer=openai_answer("Reply with the single word READY.",model)
                st.success(f"Model responded: {answer[:160]}")
            except Exception as exc:
                st.error(f"Model connection failed: {exc}")

    if mode=="Theme exploration" and not passages:
        st.info("Load at least one transcript to explore themes.")
        return
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
    limit=6  # Internal retrieval budget; not a researcher-facing decision.
    for turn in history:
        with st.chat_message("user"):
            st.write(turn["query"])
        with st.chat_message("assistant"):
            if turn.get("answer"):
                st.markdown(turn["answer"])
            elif turn.get("error"):
                st.error(turn["error"])
            if turn["hits"]:
                with st.expander(f"Inspect {len(turn['hits'])} source passages"):
                    show_sources(turn["hits"])
            if turn.get("evidence_note"):
                st.info(turn["evidence_note"])
            if turn.get("citation_warning"):
                st.warning(turn["citation_warning"])
            if turn["hits"]:
                st.download_button("Export this answer and its evidence (JSON)",
                    data=json.dumps(turn,ensure_ascii=False,indent=2),
                    file_name=f"research_answer_{turn['index']}.json",
                    key=f"export_research_{turn['index']}",mime="application/json")
    query=st.chat_input("Ask about the newspapers, or follow up on a previous theme")
    if query:
        kind=classify_chat_message(query)
        record={"index":len(history)+1,"query":query,"hits":[],
                "provider":provider,"model":model}
        if kind!="research":
            record["answer"]=interface_reply(
                kind,len(passages),len(selected)+(1 if uploaded is not None else 0))
        else:
            previous=[item["query"] for item in history[-2:]
                      if classify_chat_message(item["query"])=="research"]
            retrieval_query=" ".join(previous+[query]) if previous else query
            hits=retrieve(passages,retrieval_query,limit=limit) if passages else []
            record["hits"]=hits
            if hits:
                prompt=build_prompt(query,hits)
                if history:
                    context="Previous dialogue (background only; not evidence):\\n"
                    for turn in history[-3:]:
                        context+=("Researcher: "+turn["query"]+"\\n"
                                  "Assistant: "+(turn.get("answer") or "")[:1500]+"\\n")
                    prompt=context+"\\n"+prompt
            else:
                # No passages or no lexical match must not disable the chatbot.
                # Explicitly forbid claiming to have documentary evidence.
                prompt=(
                    "You are an assistant in a historical newspaper research workspace. "
                    "Respond helpfully to the user's message. "
                    "There are no retrieved newspaper passages relevant to this question. "
                    "Do not pretend you consulted a newspaper or invent document citations. "
                    "If the question asks for claims about this collection, explain that "
                    "the available text does not establish an answer, and suggest search "
                    "terms or importing transcriptions. For general questions and "
                    "conversation, answer normally.\\n\\n"
                    "User: "+query
                )
                if history:
                    for turn in history[-3:]:
                        prompt+=("\\nPrevious user: "+turn["query"]
                                 +"\\nPrevious assistant: "
                                 +(turn.get("answer") or "")[:1000])
            try:
                with st.status("Generating answer...",expanded=True) as status:
                    if provider.startswith("Local"):
                        response=local_answer(prompt,model,
                            progress=lambda message:st.write(message))
                    else:
                        response=openai_answer(prompt,model)
                    status.update(label="Answer generated",state="complete")
                record["answer"]=response
                if hits and not mentioned_citations(response,len(hits)):
                    record["citation_warning"]=(
                        "The model did not provide valid source citation markers.")
                if not hits and passages:
                    record["evidence_note"]=(
                        "No matching source passages were retrieved. "
                        "This response is not evidence about the newspaper collection.")
            except Exception as exc:
                record["error"]=(
                    "The selected AI model could not respond: "+str(exc)
                    +". Check Model configuration and Test model connection.")
        st.session_state["research_conversation"].append(record)
        st.rerun()
