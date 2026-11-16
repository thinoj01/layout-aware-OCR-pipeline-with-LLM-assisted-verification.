"""Offline tests for source-grounded research, saved corrections and citation prompts."""
import json
import pytest
from research import (
    words,retrieve,top_terms,uploaded_passages,evidence_context,build_prompt,
    mentioned_citations,corpus_from_jobs,
)

def passage(doc,region,text,reviewed=False):
    return {"source_key":f"{doc}:{region}","job_id":doc,
        "filename":doc+".pdf","page":1,"region_id":region,
        "article_id":None,"text":text,"reviewed":reviewed,
        "ocr_confidence":80,"origin":"completed_newspaper"}

def test_retrieval_ranks_relevant_regions_and_avoids_unrelated():
    docs=[
        passage("a","r1","The government adopted new immigration restrictions in Parliament."),
        passage("a","r2","Sports and weather reports from yesterday."),
        passage("b","r1","Immigration policies faced criticism from provincial leaders."),
    ]
    hits=retrieve(docs,"immigration restrictions",limit=5)
    assert len(hits)==2
    assert hits[0]["region_id"]=="r1"
    assert hits[0]["filename"]=="a.pdf"
    assert all("immigration" in h["text"].lower() for h in hits)
    assert retrieve(docs,"astrobiology",limit=3)==[]

def test_evidence_prompt_exposes_exact_source_ids():
    docs=[passage("a","r1","A newspaper mentioned railroads.")]
    prompt=build_prompt("What about railroads?",docs)
    assert "[S1]" in prompt
    assert "a.pdf" in prompt
    assert "A newspaper mentioned railroads." in prompt
    assert "only" in prompt.lower()
    assert mentioned_citations("Supported [S1], invented [S8]",1)==[1]

def test_term_counts_are_passage_frequency_not_raw_word_frequency():
    passages=[
        passage("a","r1","Factories factories factories unemployment"),
        passage("b","r1","Factories jobs wages"),
        passage("c","r1","Voters and elections"),
    ]
    assert dict(top_terms(passages))["factories"]==2

def test_uploaded_txt_is_segmented_with_filename_provenance():
    data=b"First paragraph about trade.\n\nSecond paragraph about employment."
    segments=uploaded_passages("transcribed.txt",data)
    assert len(segments)==2
    assert segments[0]["origin"]=="uploaded_text"
    assert segments[0]["filename"]=="transcribed.txt"
    assert segments[0]["page"] is None
    with pytest.raises(ValueError,match="2 MB"):
        uploaded_passages("large.txt",b"X"*2_000_001)

def test_research_uses_saved_manual_correction(tmp_path,monkeypatch):
    import jobs
    from PIL import Image
    import io
    monkeypatch.setattr(jobs,"ROOT",tmp_path)
    monkeypatch.setattr(jobs,"DB",tmp_path/"jobs.sqlite3")
    img=io.BytesIO()
    Image.new("RGB",(25,25),"white").save(img,"PNG")
    jid=jobs.submit("verified_newspaper.png",img.getvalue(),{})
    record={"filename":"verified_newspaper.png","pages":[{"page":2,"regions":[{
        "id":"p2_r1","text":"Original OCR error","reading_order_guess":1,
        "article_id":None,"type":"plain_text","bbox":[1,1,10,10],
        "ocr_confidence":43}]}]}
    (jobs.job_dir(jid)/"result.json").write_text(json.dumps(record))
    jobs.change(jid,"completed",done=1,total=1)
    from review import save_region
    save_region(jid,2,"p2_r1","Corrected railway workers",1,"railway_story")
    passages=corpus_from_jobs([jid])
    assert len(passages)==1
    assert passages[0]["text"]=="Corrected railway workers"
    assert passages[0]["reviewed"] is True
    assert passages[0]["article_id"]=="railway_story"
    assert passages[0]["page"]==2
    assert retrieve(passages,"railway workers")[0]["job_id"]==jid

def test_ollama_progress_when_server_already_running(monkeypatch):
    import ollama_runtime as rt
    monkeypatch.setattr(rt,"is_running",lambda base_url=True:True)
    monkeypatch.setattr(rt,"_available_models",lambda url:{"qwen3-vl:4b"})
    messages=[]
    assert rt.ensure_ollama("qwen3-vl:4b",progress=messages.append)
    assert "ready" in messages[-1]


def test_research_workspace_render_without_completed_jobs(tmp_path,monkeypatch):
    import jobs
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(jobs,"ROOT",tmp_path)
    monkeypatch.setattr(jobs,"DB",tmp_path/"jobs.sqlite3")
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/"app.py"),
                          default_timeout=30).run()
    assert not app.exception
    app.radio[0].set_value("Research analysis").run()
    assert not app.exception


def test_stage_updates_are_persisted_when_worker_running(tmp_path,monkeypatch):
    import jobs
    from PIL import Image
    import io
    monkeypatch.setattr(jobs,"ROOT",tmp_path)
    monkeypatch.setattr(jobs,"DB",tmp_path/"jobs.sqlite3")
    b=io.BytesIO()
    Image.new("RGB",(30,30),"white").save(b,"PNG")
    jid=jobs.submit("testing.png",b.getvalue(),{})
    assert jobs.claim()==jid
    jobs.set_stage(jid,"Ollama downloading model (50%)")
    assert jobs.get_job(jid)["stage"]=="Ollama downloading model (50%)"
    jobs.change(jid,"completed",done=1,total=1)
    jobs.set_stage(jid,"incorrect change")
    assert jobs.get_job(jid)["stage"]=="Ollama downloading model (50%)"
