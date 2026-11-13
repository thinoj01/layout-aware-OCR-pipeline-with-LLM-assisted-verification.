"""Evaluate OCR against a manually corrected ground-truth transcription.

Usage: python evaluate.py reference.txt predicted.txt
Preserves historical spelling; normalizes only repeated whitespace.
"""
import argparse
import json
import re
from pathlib import Path

def distance(a,b):
    # Memory-efficient Levenshtein edit distance.
    if len(a)<len(b): a,b=b,a
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        curr=[i]
        for j,y in enumerate(b,1):
            curr.append(min(curr[j-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=curr
    return prev[-1]

def metrics(reference,predicted):
    ref=re.sub(r"\s+"," ",reference).strip()
    pred=re.sub(r"\s+"," ",predicted).strip()
    ref_words=ref.split()
    pred_words=pred.split()
    return {
        "reference_characters":len(ref),
        "reference_words":len(ref_words),
        "character_error_rate":distance(ref,pred)/max(1,len(ref)),
        "word_error_rate":distance(ref_words,pred_words)/max(1,len(ref_words))
    }

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Compare OCR to human-corrected text")
    parser.add_argument("reference",type=Path)
    parser.add_argument("predicted",type=Path)
    args=parser.parse_args()
    print(json.dumps(metrics(args.reference.read_text(encoding="utf8"),
                             args.predicted.read_text(encoding="utf8")),indent=2))
