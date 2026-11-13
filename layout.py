"""CPU-only XY-cut segmentation baseline for historical newspaper pages.

This uses whitespace separators to propose text regions and a reading order.
It is not a pretrained article-classification model; human review is essential.
"""
from dataclasses import dataclass
import cv2
import numpy as np
from PIL import Image

@dataclass
class Block:
    bbox: list[int]
    reading_order: int = 0
    kind: str = "text_region"
    source: str = "xycut"

def _runs(mask):
    padded=np.r_[False,mask,False].astype(np.int8)
    return list(zip(np.where(np.diff(padded)==1)[0].tolist(),
                    np.where(np.diff(padded)==-1)[0].tolist()))

def detect_regions(image: Image.Image, max_depth: int = 9) -> list[Block]:
    gray=np.asarray(image.convert("L"))
    gray=cv2.GaussianBlur(gray,(3,3),0)
    binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    ink=(binary<140).astype(np.uint8)
    ink=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
    h,w=ink.shape
    ys,xs=np.where(ink>0)
    if len(xs)==0:
        return []
    root=(max(0,int(xs.min())-4),max(0,int(ys.min())-4),
          min(w,int(xs.max())+5),min(h,int(ys.max())+5))
    regions=[]
    def rec(box,depth):
        x0,y0,x1,y1=box
        ww,hh=x1-x0,y1-y0
        patch=ink[y0:y1,x0:x1]
        if depth>=max_depth or ww<75 or hh<45 or np.count_nonzero(patch)<35:
            regions.append(box);return
        v=np.mean(patch,axis=0)<0.008
        horizontal=np.mean(patch,axis=1)<0.008
        vs=[(a,b) for a,b in _runs(v)
            if b-a>=max(12,int(ww*.018))
            and a>=int(ww*.12) and ww-b>=int(ww*.12)]
        hs=[(a,b) for a,b in _runs(horizontal)
            if b-a>=max(24,int(hh*.045))
            and a>=int(hh*.10) and hh-b>=int(hh*.10)]
        best_v=max(vs,key=lambda p:p[1]-p[0]) if vs else None
        best_h=max(hs,key=lambda p:p[1]-p[0]) if hs else None
        use_h=best_h and (not best_v or
              (best_h[1]-best_h[0])/hh>(best_v[1]-best_v[0])/ww*1.65)
        if use_h:
            a,b=best_h
            rec((x0,y0,x1,y0+a),depth+1)
            rec((x0,y0+b,x1,y1),depth+1)
        elif best_v:
            a,b=best_v
            rec((x0,y0,x0+a,y1),depth+1)
            rec((x0+b,y0,x1,y1),depth+1)
        else:
            regions.append(box)
    rec(root,0)
    output=[]
    for box in regions:
        x0,y0,x1,y1=box
        yy,xx=np.where(ink[y0:y1,x0:x1]>0)
        if len(xx)<20:
            continue
        output.append([max(0,x0+int(xx.min())-3),
                       max(0,y0+int(yy.min())-3),
                       min(w,x0+int(xx.max())+4),
                       min(h,y0+int(yy.max())+4)])
    return [Block(box,i+1) for i,box in enumerate(output)]

def overlay(image,regions):
    view=np.asarray(image.convert("RGB")).copy()
    for region in regions:
        x0,y0,x1,y1=(region.bbox if hasattr(region,"bbox") else region["bbox"])
        index=(region.reading_order if hasattr(region,"reading_order")
               else region.get("reading_order_guess",0))
        cv2.rectangle(view,(x0,y0),(x1,y1),(230,45,35),3)
        cv2.putText(view,str(index),(x0+4,max(24,y0+24)),
                    cv2.FONT_HERSHEY_SIMPLEX,.8,(0,50,230),2)
    return Image.fromarray(view)
