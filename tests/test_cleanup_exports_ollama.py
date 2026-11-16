"""Tests for cleanup, export integrity, and Ollama lifecycle (no model download needed)."""
import io
import json
import zipfile
from pathlib import Path
from unittest.mock import Mock
import pytest
from PIL import Image
from docx import Document


def _image():
    stream=io.BytesIO()
    Image.new("RGB",(80,70),"white").save(stream,"PNG")
    return stream.getvalue()


@pytest.fixture
def jobs_in_tmp(tmp_path,monkeypatch):
    import jobs
    monkeypatch.setattr(jobs,"ROOT",tmp_path)
    monkeypatch.setattr(jobs,"DB",tmp_path/"jobs.sqlite3")
    return jobs


def _completed(jobs,name,text,article):
    jid=jobs.submit(name,_image(),{"layout_backend":"xycut"})
    data={"filename":name,"pages":[{"page":1,"layout_preview":"FAKE_B64",
        "regions":[{"id":"p1_r1","type":"plain_text","bbox":[10,20,60,50],
                    "reading_order_guess":1,"article_id":article,
                    "text":text,"ocr_confidence":88}]}]}
    (jobs.job_dir(jid)/"result.json").write_text(json.dumps(data))
    jobs.change(jid,"completed",done=1,total=1)
    return jid


def test_delete_finished_clears_only_terminal_jobs(jobs_in_tmp):
    jobs=jobs_in_tmp
    done=_completed(jobs,"done.png","Story A","A")
    failed=jobs.submit("failed.png",_image(),{})
    cancelled=jobs.submit("cancel.png",_image(),{})
    queued=jobs.submit("queued.png",_image(),{})
    running=jobs.submit("running.png",_image(),{})
    jobs.change(failed,"failed",error="test error")
    jobs.request_cancel(cancelled)
    assert jobs.claim()==queued
    assert jobs.claim()==running  # both running; none should be removed
    count=jobs.delete_finished()
    assert count==3
    for jid in [done,failed,cancelled]:
        assert jobs.get_job(jid) is None
        assert not jobs.job_dir(jid).exists()
    for jid in [queued,running]:
        assert jobs.get_job(jid)["status"]=="running"
        assert jobs.job_dir(jid).exists()
    assert jobs.delete_finished()==0


def test_single_zip_openable_docx_and_corrected_text(jobs_in_tmp):
    jobs=jobs_in_tmp
    jid=_completed(jobs,"a.png","Original","StoryA")
    from review import save_region,export_job
    save_region(jid,1,"p1_r1","Researcher corrected text",1,"StoryA")
    with zipfile.ZipFile(io.BytesIO(export_job(jid))) as z:
        assert z.testzip() is None
        assert "Researcher corrected text" in z.read("transcription.txt").decode()
        structured=json.loads(z.read("structured.json"))
        assert structured["pages"][0]["regions"][0]["text"]=="Researcher corrected text"
        doc=Document(io.BytesIO(z.read("transcription.docx")))
        assert any("Researcher corrected text" in p.text for p in doc.paragraphs)
        article=Document(io.BytesIO(z.read("articles/StoryA.docx")))
        assert any("Researcher corrected text" in p.text for p in article.paragraphs)


def test_collection_zip_flat_openable_and_preserves_all_files(jobs_in_tmp):
    jobs=jobs_in_tmp
    a=_completed(jobs,"first.png","First","same_article")
    b=_completed(jobs,"second.png","Second","same_article")
    from review import export_completed_collection
    with zipfile.ZipFile(io.BytesIO(export_completed_collection([a,b]))) as z:
        assert z.testzip() is None
        manifest=json.loads(z.read("collection_manifest.json"))
        assert [x["job_id"] for x in manifest["newspapers"]]==[a,b]
        assert manifest["skipped_job_ids"]==[]
        for jid,expected in [(a,"First"),(b,"Second")]:
            prefix=f"newspaper_{jid}/"
            assert expected in z.read(prefix+"transcription.txt").decode()
            assert expected in z.read(prefix+"articles/same_article.txt").decode()
            assert prefix+"structured.json" in z.namelist()
            assert prefix+"manifest.json" in z.namelist()
            document=Document(io.BytesIO(z.read(prefix+"transcription.docx")))
            assert any(expected in p.text for p in document.paragraphs)


def test_collection_skips_unfinished_jobs(jobs_in_tmp):
    jobs=jobs_in_tmp
    done=_completed(jobs,"done.png","Done","article")
    pending=jobs.submit("pending.png",_image(),{})
    from review import export_completed_collection
    with zipfile.ZipFile(io.BytesIO(export_completed_collection([done,pending]))) as z:
        manifest=json.loads(z.read("collection_manifest.json"))
        assert len(manifest["newspapers"])==1
        assert manifest["skipped_job_ids"]==[pending]


def test_ollama_uses_existing_local_service_without_launch(monkeypatch):
    import ollama_runtime as runtime
    monkeypatch.setattr(runtime,"is_running",lambda base_url=True: True)
    monkeypatch.setattr(runtime,"_available_models",lambda base_url: {"qwen3-vl:4b"})
    called=[]
    monkeypatch.setattr(runtime.subprocess,"Popen",lambda *args,**kwargs: called.append(args))
    assert runtime.ensure_ollama("qwen3-vl:4b")=="http://127.0.0.1:11434"
    assert called==[]


def test_ollama_starts_on_demand(monkeypatch):
    import ollama_runtime as runtime
    class FakeProcess:
        def poll(self): return None
        def terminate(self): pass
        def wait(self,timeout=None): return 0
    running=[False]
    monkeypatch.setattr(runtime,"_spawned",None)
    monkeypatch.setattr(runtime,"is_running",lambda base_url: running[0])
    monkeypatch.setattr(runtime,"_available_models",lambda base_url: {"qwen3-vl:4b"})
    monkeypatch.setattr(runtime.shutil,"which",lambda name:"/usr/local/bin/ollama")
    monkeypatch.setattr(runtime.subprocess,"Popen",
                        lambda *args,**kwargs:(running.__setitem__(0,True) or FakeProcess()))
    assert runtime.ensure_ollama("qwen3-vl:4b")
    runtime.stop_managed_ollama()


def test_ollama_missing_install_error(monkeypatch):
    import ollama_runtime as runtime
    monkeypatch.setattr(runtime,"is_running",lambda base_url:False)
    monkeypatch.setattr(runtime.shutil,"which",lambda name:None)
    with pytest.raises(RuntimeError,match="brew install ollama"):
        runtime.ensure_ollama("qwen3-vl:4b")


@pytest.mark.parametrize("bad_url",["https://example.com:443",
                                    "http://192.168.1.1:11434",
                                    "file:///tmp/model"])
def test_ollama_rejects_nonlocal_urls(bad_url):
    from ollama_runtime import _check_local_url
    with pytest.raises(ValueError,match="local"):
        _check_local_url(bad_url)
