"""Collect public historical newspaper scans for reproducible OCR audits.

Network-dependent; archive pages can be unavailable, so do not treat missing downloads as success.
Generates OCR outputs and audit.csv. Accuracy requires independently reviewed reference text.
"""
import argparse
import csv
import json
from pathlib import Path
import requests
from ocr_engine import Settings,process_files
from evaluate import metrics

COLLECTION="https://www.loc.gov/resource/sn83045462/{date}/ed-1/?sp=1"

def download_scan(session,date):
    # The LOC page's JSON metadata includes image variants. The endpoint can change.
    page=COLLECTION.format(date=date)
    r=session.get(page+"&fo=json",timeout=35)
    r.raise_for_status()
    data=r.json()
    candidates=[]
    def walk(value):
        if isinstance(value,dict):
            for key,sub in value.items():
                if key=="image" and isinstance(sub,list):
                    for v in sub:
                        if isinstance(v,dict):
                            for u in v.get("url",[]):
                                if isinstance(u,str):candidates.append(u)
                        elif isinstance(v,str):candidates.append(v)
                elif isinstance(sub,(dict,list)):
                    walk(sub)
        elif isinstance(value,list):
            for sub in value:walk(sub)
    walk(data.get("resources",[]))
    # Prefer highest resolution on available LOC metadata; avoid OCR texts.
    images=[u for u in candidates if u.startswith("https://") and
            (".jpg" in u.lower() or ".jpeg" in u.lower() or "image" in u.lower())]
    if not images:
        raise RuntimeError("No JPEG image URL found; download manually from "+page)
    result=session.get(images[-1],timeout=120)
    result.raise_for_status()
    return result.content,page

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--images",type=Path,default=Path("benchmark/images"))
    p.add_argument("--output",type=Path,default=Path("benchmark/results"))
    p.add_argument("--download",action="store_true")
    p.add_argument("--limit",type=int,default=30)
    p.add_argument("--layout",choices=["xycut","pretrained"],default="xycut")
    p.add_argument("--dpi",type=int,default=250)
    args=p.parse_args()
    args.images.mkdir(parents=True,exist_ok=True)
    args.output.mkdir(parents=True,exist_ok=True)
    if args.download:
        session=requests.Session()
        session.headers["User-Agent"]="HistoricalNewspaperOCRResearch/0.1"
        # Thirty dated issues in September 1907 (Sunday editions may vary).
        for day in range(1,31):
            if len(list(args.images.glob("*.jpg")))>=args.limit:break
            date=f"1907-09-{day:02d}"
            target=args.images/(date+".jpg")
            if target.exists():continue
            try:
                data,_=download_scan(session,date)
                target.write_bytes(data)
                print("Downloaded",date)
            except Exception as exc:
                print("Unavailable",date,str(exc)[:150])
    rows=[]
    for filename in sorted(args.images.iterdir()):
        if filename.suffix.lower() not in {".jpg",".jpeg",".png",".tif",".tiff",".webp",".pdf"}:
            continue
        try:
            result=process_files([(filename.name,filename.read_bytes())],
                Settings(layout_backend=args.layout,dpi=args.dpi))[0]
            if result.get("error"):
                raise RuntimeError(result["error"])
            text="\n\n".join(page["text"] for page in result["pages"])
            (args.output/(filename.stem+".txt")).write_text(text,encoding="utf8")
            conf=[r["ocr_confidence"] for page in result["pages"] for r in page["regions"]]
            row={"file":filename.name,"status":"processed","pages":len(result["pages"]),
                 "regions":len(conf),"mean_ocr_confidence":
                 round(sum(conf)/len(conf),2) if conf else None,
                 "cer":"","wer":"","review_notes":""}
            reference=Path("benchmark/references")/(filename.stem+".txt")
            if reference.exists():
                scores=metrics(reference.read_text(encoding="utf8"),text)
                row["cer"]=scores["character_error_rate"]
                row["wer"]=scores["word_error_rate"]
            rows.append(row)
        except Exception as exc:
            rows.append({"file":filename.name,"status":"failed: "+str(exc)[:150],
                         "pages":"","regions":"","mean_ocr_confidence":"",
                         "cer":"","wer":"","review_notes":""})
    with (args.output/"audit.csv").open("w",newline="",encoding="utf8") as output:
        cols=["file","status","pages","regions","mean_ocr_confidence","cer","wer","review_notes"]
        writer=csv.DictWriter(output,fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    print("Audit:",args.output/"audit.csv", "Pages with gold references:",
          sum(bool(r.get("cer") or r.get("cer")==0) for r in rows))
    print("Manual inspection is required for article completeness and reading order.")

if __name__=="__main__":
    main()
