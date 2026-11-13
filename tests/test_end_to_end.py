"""Full local workflow integration, using an actual two-page PDF and SQLite queue."""
import io
import json
import fitz
from PIL import Image

def test_queued_pdf_processed_into_reviewable_result(tmp_path,monkeypatch):
    import jobs
    monkeypatch.setattr(jobs,"ROOT",tmp_path)
    monkeypatch.setattr(jobs,"DB",tmp_path/"jobs.sqlite3")
    document=fitz.open()
    for page_index in range(2):
        page=document.new_page(width=600,height=500)
        page.insert_text((70,110),f"Newspaper page {page_index+1}",fontsize=24)
        page.insert_text((70,160),"Historic newspaper archive",fontsize=22)
    content=document.tobytes()
    jid=jobs.submit("historical.pdf",content,{
        "layout_backend":"xycut","dpi":200,"language":"eng",
        "verify":False,"max_pages":5
    })
    assert jobs.get_job(jid)["status"]=="queued"
    from worker import process_job
    process_job(jid)
    assert jobs.get_job(jid)["status"]=="completed"
    assert jobs.get_job(jid)["done"]==2
    from review import load_result,save_region,export_job
    result=load_result(jid)
    assert len(result["pages"])==2
    assert any("Newspaper" in region["text"]
               for page in result["pages"] for region in page["regions"])
    region=result["pages"][0]["regions"][0]
    save_region(jid,1,region["id"],"Verified by researcher",1,"newspaper_article")
    assert load_result(jid)["pages"][0]["regions"][0]["text"]=="Verified by researcher"
    assert b"Verified by researcher" in export_job(jid)

def test_streamlit_import_and_render(tmp_path,monkeypatch):
    import jobs
    monkeypatch.setattr(jobs,"ROOT",tmp_path)
    monkeypatch.setattr(jobs,"DB",tmp_path/"jobs.sqlite3")
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file("app.py",default_timeout=30).run()
    assert not app.exception
