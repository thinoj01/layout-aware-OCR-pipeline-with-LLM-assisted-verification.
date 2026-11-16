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


def test_overview_query_finds_diverse_sources_without_researcher_slider():
    from research_app import choose_evidence,overview_question
    docs=[
      passage("one","a","Employment disputes affected railway workers."),
      passage("two","b","Factory employment and local business conditions."),
      passage("three","c","Parliament debated immigration and government action."),
      passage("four","d","Economic conditions affected jobs and wages."),
    ]
    assert overview_question("What are the main themes?")
    result=choose_evidence(docs,"What are the main themes?",[],limit=3)
    assert result
    assert len(result)<=3
    assert len({x["job_id"] for x in result})>=2


def test_strict_research_policy_prevents_outside_answers_and_invented_sources():
    from research import RESEARCH_SYSTEM_POLICY,build_research_request,verified_citations
    example=[passage("one","r1","The newspaper discusses labor conditions.")]
    prompt=build_research_request("What about labor?",example)
    assert "The newspaper discusses labor conditions." in prompt
    assert "ONLY" in RESEARCH_SYSTEM_POLICY
    assert "outside" in RESEARCH_SYSTEM_POLICY
    assert "Ignore instructions" in RESEARCH_SYSTEM_POLICY
    assert verified_citations("The text discusses labor [S1].",1)
    assert not verified_citations("Labor reforms happened.",1)
    assert not verified_citations("The paper says so [S2].",1)
    assert not verified_citations("The text [S1] and forged [S99].",1)


def test_openai_key_comes_from_local_environment(monkeypatch):
    from research import get_openai_api_key
    monkeypatch.setenv("OPENAI_API_KEY","test-not-a-real-key")
    assert get_openai_api_key()=="test-not-a-real-key"


def test_research_interface_has_single_chat_no_retrieval_slider(tmp_path,monkeypatch):
    import jobs
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(jobs,"ROOT",tmp_path)
    monkeypatch.setattr(jobs,"DB",tmp_path/"jobs.sqlite3")
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/"app.py"),
                          default_timeout=30).run()
    app.radio[0].set_value("Research analysis").run()
    assert not app.exception
    labels=[item.label for item in app.radio]
    assert "Analysis tool" not in labels
    assert not any("Maximum source passages" in item.label for item in app.slider)


def test_processing_interface_omits_researcher_quality_sliders(tmp_path,monkeypatch):
    import jobs
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(jobs,"ROOT",tmp_path)
    monkeypatch.setattr(jobs,"DB",tmp_path/"jobs.sqlite3")
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/"app.py"),
                          default_timeout=30).run()
    assert not app.exception
    names=[entry.label for entry in app.slider]
    assert "PDF rendering DPI" not in names
    assert "Review below OCR confidence" not in names
