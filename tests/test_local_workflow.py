"""Offline regression tests for queue, review, layout selection and exports."""
import io
import json
import os
import zipfile
from pathlib import Path
from unittest.mock import patch
import pytest
from PIL import Image,ImageDraw

def test_queue_claim_and_delete(tmp_path,monkeypatch):
    import jobs
    monkeypatch.setattr(jobs,"ROOT",tmp_path)
    monkeypatch.setattr(jobs,"DB",tmp_path/"jobs.sqlite3")
    ids=[jobs.submit(f"paper_{i}.png",b"placeholder",{"layout_backend":"xycut"}) for i in range(12)]
    assert len(set(ids))==12
    assert [j["status"] for j in jobs.list_jobs()]==["queued"]*12
    claimed=[jobs.claim() for _ in range(12)]
    assert set(claimed)==set(ids)
    assert jobs.claim() is None
    with pytest.raises(ValueError):
        jobs.delete(claimed[0])
    jobs.change(claimed[0],"completed",done=3,total=3)
    assert jobs.get_job(claimed[0])["done"]==3
    assert jobs.delete(claimed[0])
    assert not jobs.job_dir(claimed[0]).exists()

@pytest.mark.parametrize("filename",[
    "../bad.sh","empty.pdf","big.exe","not-an-image.csv",
    "bad.xyz","script.py","archive.zip","file.svg"
])
def test_reject_unsupported_files(tmp_path,monkeypatch,filename):
    import jobs
    monkeypatch.setattr(jobs,"ROOT",tmp_path)
    monkeypatch.setattr(jobs,"DB",tmp_path/"jobs.sqlite3")
    with pytest.raises(ValueError):
        jobs.submit(filename,b"123",{})

def test_review_preserves_changes_and_exports(tmp_path,monkeypatch):
    import jobs
    monkeypatch.setattr(jobs,"ROOT",tmp_path)
    monkeypatch.setattr(jobs,"DB",tmp_path/"jobs.sqlite3")
    jid=jobs.submit("sample.png",b"some bytes",{})
    data={"filename":"sample.png","pages":[
        {"page":1,"layout_preview":"THIS_IS_NOT_A_REAL_IMAGE","regions":[
            {"id":"p1_r1","reading_order_guess":1,"article_id":None,"text":"First passage","bbox":[1,1,10,10],"ocr_confidence":81,"type":"plain_text"},
            {"id":"p1_r2","reading_order_guess":2,"article_id":None,"text":"Second passage","bbox":[11,1,20,10],"ocr_confidence":62,"type":"plain_text"},
        ]},
        {"page":2,"layout_preview":"THUMB","regions":[
            {"id":"p2_r1","reading_order_guess":1,"article_id":None,"text":"Continuing","bbox":[1,1,20,20],"ocr_confidence":70,"type":"plain_text"}
        ]}
    ]}
    (jobs.job_dir(jid)/"result.json").write_text(json.dumps(data))
    jobs.change(jid,"completed")
    from review import save_region,load_result,export_job
    save_region(jid,1,"p1_r1","Corrected start",2,"story_1")
    save_region(jid,1,"p1_r2","Corrected second",1,"story_1")
    save_region(jid,2,"p2_r1","Corrected end",1,"story_1")
    updated=load_result(jid)
    assert updated["pages"][0]["regions"][0]["reviewed"]
    with zipfile.ZipFile(io.BytesIO(export_job(jid))) as archive:
        text=archive.read("transcription.txt").decode("utf8")
        assert text.index("Corrected second")<text.index("Corrected start")
        assert archive.read("articles/story_1.txt").decode("utf8").count("Corrected")==3
        assert "articles/story_1.docx" in archive.namelist()
        structured=json.loads(archive.read("structured.json"))
        assert "layout_preview" not in structured["pages"][0]
    with pytest.raises(ValueError):
        save_region(jid,1,"fake","fake",1,"story_1")
    with pytest.raises(ValueError):
        save_region(jid,1,"p1_r1","text",0,"story_1")

def test_pretrained_detection_mock(monkeypatch):
    import pretrained_layout as model
    class DummyArray:
        def __init__(self,values):self.values=values
        def __getitem__(self,index):return self.values[index]
        def tolist(self):return self.values
    class DummyBox:
        def __init__(self,xy,klass,conf):
            self.xyxy=[DummyArray(xy)]
            self.cls=[klass]
            self.conf=[conf]
    class DummyResult:
        names={0:"plain_text",1:"figure",2:"title"}
        boxes=[DummyBox([5,5,80,95],0,.95),DummyBox([90,5,160,120],1,.85),
               DummyBox([6,1,140,30],2,.96)]
    class Fake:
        def predict(self,**kwargs):return [DummyResult()]
    monkeypatch.setattr(model,"load_model",lambda:Fake())
    detections=model.detect_pretrained(Image.new("RGB",(200,150),"white"))
    assert len(detections)==2
    assert all(r["source"]=="doclayout-yolo" for r in detections)
    assert all(r["type"]!="figure" for r in detections)
    assert sorted(r["reading_order_guess"] for r in detections)==[1,2]

def test_pretrained_failure_is_explicit(monkeypatch):
    import pretrained_layout as m
    def fail():
        raise RuntimeError("weights unavailable")
    monkeypatch.setattr(m,"load_model",fail)
    with pytest.raises(RuntimeError,match="weights unavailable"):
        m.detect_pretrained(Image.new("RGB",(200,150)))

@pytest.mark.parametrize("missing",range(12))
def test_empty_image_no_spurious_regions(missing):
    from layout import detect_regions
    image=Image.new("RGB",(230+missing,180),"white")
    assert detect_regions(image)==[]

@pytest.mark.parametrize("value",["", "  ","../../", "wrong-id","a"*100])
def test_job_id_path_rejected(value):
    from jobs import job_dir
    with pytest.raises(ValueError):
        job_dir(value)
