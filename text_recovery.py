"""Recover high-confidence text outside pretrained layout boxes.

Unassigned text is *not* automatically attached to articles. Always flag it for review.
"""
from collections import defaultdict
import pytesseract
from pytesseract import Output

def recover_unassigned(image, covered, min_confidence=55):
    data=pytesseract.image_to_data(image,config="--oem 1 --psm 11",
                                  output_type=Output.DICT)
    groups=defaultdict(list)
    def inside(x,y):
        return any(a<=x<=c and b<=y<=d for a,b,c,d in covered)
    for i,word in enumerate(data["text"]):
        word=word.strip()
        if len(word)<2:
            continue
        try: confidence=float(data["conf"][i])
        except (ValueError,TypeError): continue
        if confidence<min_confidence:continue
        x=int(data["left"][i]);y=int(data["top"][i])
        w=int(data["width"][i]);h=int(data["height"][i])
        if inside(x+w/2,y+h/2):continue
        key=(data["block_num"][i],data["par_num"][i],data["line_num"][i])
        groups[key].append((x,y,x+w,y+h,word))
    regions=[]
    for words in groups.values():
        if not words:continue
        bbox=[min(w[0] for w in words),min(w[1] for w in words),
              max(w[2] for w in words),max(w[3] for w in words)]
        # Only recover sufficiently substantive lines; ignore isolated noise.
        if sum(len(w[4]) for w in words)<8:
            continue
        regions.append({"bbox":bbox,"type":"unassigned_text",
                        "source":"sparse-ocr-recovery"})
    return sorted(regions,key=lambda r:(r["bbox"][1],r["bbox"][0]))
