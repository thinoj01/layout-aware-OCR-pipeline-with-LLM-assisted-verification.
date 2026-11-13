"""Regression tests for queue ordering, ETA estimates, cancellation and reset."""
import io
import time
import pytest
import fitz
from PIL import Image

@pytest.fixture
def isolated(tmp_path,monkeypatch):
    import jobs
    monkeypatch.setattr(jobs,"ROOT",tmp_path)
    monkeypatch.setattr(jobs,"DB",tmp_path/"jobs.sqlite3")
    return jobs

def _png():
    b=io.BytesIO()
    Image.new("RGB",(90,80),"white").save(b,format="PNG")
    return b.getvalue()

def _pdf(pages=3):
    doc=fitz.open()
    for _ in range(pages):
        doc.new_page()
    return doc.tobytes()

def test_upload_records_page_count_before_processing(isolated):
    jobs=isolated
    a=jobs.submit("first.pdf",_pdf(3),{})
    b=jobs.submit("second.png",_png(),{})
    assert jobs.get_job(a)["total"]==3
    assert jobs.get_job(b)["total"]==1

def test_queue_order_and_unknown_eta_without_observations(isolated):
    jobs=isolated
    identifiers=[jobs.submit(f"{k}.png",_png(),{}) for k in range(4)]
    estimates=jobs.queue_estimates(jobs.list_jobs(),now=1000)
    assert [estimates[k]["position"] for k in identifiers]==[1,2,3,4]
    assert all(estimates[k]["seconds_to_start"] is None for k in identifiers)

def test_queued_wait_estimates_follow_measured_rate(isolated):
    jobs=isolated
    first=jobs.submit("one.pdf",_pdf(4),{})
    second=jobs.submit("two.pdf",_pdf(2),{})
    third=jobs.submit("three.pdf",_pdf(3),{})
    with jobs.connect() as db:
        db.execute("UPDATE jobs SET status='running',done=2,total=4,started_at=1000,updated=1020 WHERE id=?",(first,))
    estimates=jobs.queue_estimates(jobs.list_jobs(),now=1020)
    assert estimates[first]["seconds_to_finish"]==20
    assert estimates[second]["seconds_to_start"]==20
    assert estimates[second]["seconds_to_finish"]==40
    assert estimates[third]["seconds_to_start"]==40
    assert estimates[third]["seconds_to_finish"]==70

def test_history_supports_eta_when_first_page_in_progress(isolated):
    jobs=isolated
    previous=jobs.submit("past.png",_png(),{})
    with jobs.connect() as db:
        db.execute("UPDATE jobs SET status='completed',done=1,started_at=10,updated=22 WHERE id=?",(previous,))
    current=jobs.submit("new.pdf",_pdf(3),{})
    queued=jobs.submit("queued.pdf",_pdf(2),{})
    assert jobs.claim()==current
    estimate=jobs.queue_estimates(jobs.list_jobs(),now=101)
    assert estimate[current]["seconds_to_finish"]==36
    assert estimate[queued]["seconds_to_start"]==36

def test_remove_queued_item_changes_positions(isolated):
    jobs=isolated
    a=jobs.submit("one.png",_png(),{})
    b=jobs.submit("two.png",_png(),{})
    c=jobs.submit("three.png",_png(),{})
    assert jobs.delete(b)
    assert not jobs.get_job(b)
    assert not jobs.job_dir(b).exists()
    estimate=jobs.queue_estimates(jobs.list_jobs())
    assert estimate[a]["position"]==1
    assert estimate[c]["position"]==2
    assert jobs.claim()==a

def test_cancel_queued_and_running(isolated):
    jobs=isolated
    a=jobs.submit("a.png",_png(),{})
    b=jobs.submit("b.png",_png(),{})
    assert jobs.request_cancel(a)
    assert jobs.get_job(a)["status"]=="cancelled"
    assert jobs.claim()==b
    assert jobs.request_cancel(b)
    assert jobs.get_job(b)["cancel_requested"]==1
    assert jobs.get_job(b)["status"]=="running"
    assert jobs.delete(a)
    with pytest.raises(ValueError,match="processing"):
        jobs.delete(b)
    jobs.change(b,"cancelled")
    assert jobs.delete(b)

def test_atomic_reset_refuses_active_worker_without_partial_deletion(isolated):
    jobs=isolated
    a=jobs.submit("a.png",_png(),{})
    b=jobs.submit("b.png",_png(),{})
    assert jobs.claim()==a
    with pytest.raises(ValueError,match="still processing"):
        jobs.delete_many([a,b])
    assert jobs.get_job(a) and jobs.get_job(b)
    jobs.change(a,"cancelled")
    assert jobs.delete_many([a,b])==2
    assert jobs.list_jobs()==[]
    assert not jobs.job_dir(a).exists()
    assert not jobs.job_dir(b).exists()

@pytest.mark.parametrize("pages",[0,1,2,5,11])
def test_pdf_page_count_limits(isolated,pages):
    jobs=isolated
    if pages==0:
        with pytest.raises(ValueError):
            jobs.submit("bad.pdf",b"%PDF- not real",{})
    else:
        if pages>5:
            with pytest.raises(ValueError,match="page count"):
                jobs.submit("large.pdf",_pdf(pages),{"max_pages":5})
        else:
            jid=jobs.submit(f"sample_{pages}.pdf",_pdf(pages),{"max_pages":5})
            assert jobs.get_job(jid)["total"]==pages

def test_worker_honors_cancellation_before_first_page(isolated):
    jobs=isolated
    jid=jobs.submit("one.png",_png(),{"layout_backend":"xycut"})
    assert jobs.claim()==jid
    jobs.request_cancel(jid)
    from worker import process_job
    process_job(jid)
    assert jobs.get_job(jid)["status"]=="cancelled"

def test_duration_formatting(isolated):
    assert isolated._duration(None)=="Calculating..."
    assert isolated._duration(35)=="35 sec"
    assert isolated._duration(125)=="2 min 05 sec"
    assert isolated._duration(3660)=="1 hr 1 min"

def test_no_partial_reset_of_running(isolated):
    jobs=isolated
    running=jobs.submit("running.png",_png(),{})
    queued=jobs.submit("queued.png",_png(),{})
    assert jobs.claim()==running
    with pytest.raises(ValueError):
        jobs.delete_many([queued,running])
    assert jobs.get_job(queued)["status"]=="queued"
    assert jobs.get_job(running)["status"]=="running"
