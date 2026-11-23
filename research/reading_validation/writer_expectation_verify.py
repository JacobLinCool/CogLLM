"""Independently check visible Gradio values against the frozen native GPU response."""
import argparse
import json
from pathlib import Path

from research.backbone_transfer.extract import digest

OUT=Path("research/results/reading-validation-writer")


def main(log):
    body=log.read_text().split("### Result\n",1)[1]
    screens,_=json.JSONDecoder().raw_decode(body)
    source=OUT/"expectation-source/responses.json"
    sample=json.loads(source.read_text())["1"]
    words=sample["english"]["expectations"]["words"]
    if [(s["theme"],s["width"]) for s in screens]!=[("light",1280),("dark",390)]:
        raise ValueError("Both display sizes are required")
    verified=[]
    for s in screens:
        if len(s["rows"])!=len(words) or len(s["points"])!=len(words):
            raise ValueError("Words are missing from the curve or the table")
        def label(a,b,signed=False):
            fmt="+.1f" if signed else ".0f";ends=[format(100*t,fmt) for t in (a,b)]
            return (ends[0] if ends[0]==ends[1] else " to ".join(ends))+("" if signed else "%")
        for j,(row,point,upper,w) in enumerate(zip(s["rows"],s["points"],s["upper"],words,strict=True)):
            before=w["before"];delta=w["change"]
            hi=min(1,w["after"]+w["other_after"]**2)
            oldhi=min(1,before+w["other_before"]**2) if before is not None else None
            expected=[str(j+1),w["text"],"—" if before is None else label(before,oldhi),label(w["after"],hi),
                      "—" if delta is None else label(w["after"]-oldhi,hi-before,True)]
            if row!=expected or abs(point[2]-(99-98*w["after"]))>.00501 or abs(upper[2]-(99-98*hi))>.00501:
                raise ValueError("A visual value disagrees with its native probability")
        flagged={}
        for direction,sign in (("increase",1),("decrease",-1)):
            want=[w for w in words if w["change"] is not None and (
                w["after"]-min(1,w["before"]+w["other_before"]**2)>=.2 if sign>0 else
                w["before"]-min(1,w["after"]+w["other_after"]**2)>=.2)]
            flagged[direction]=len(want)
            exact={i for w in want for i in range(w["start"],w["end"])}
            actual=set()
            for mark in s["marks"][direction]:
                a,b=mark["start"],mark["end"]
                if sample["text"][a:b]!=mark["text"]:
                    raise ValueError("Visible word highlight moved")
                actual.update(range(a,b))
            if actual!=exact:
                raise ValueError("Source word coordinates disagree with displayed changes")
        verified.append(dict(theme=s["theme"],viewport_width=s["width"],curve_points=len(words),
            verified_probability_bounds=2*len(words),exact_highlighted_words=flagged,additional_model_requests=s["requestsWhileInspecting"]))
    result=dict(source_sha256=digest(source),browser_log_sha256=digest(log),
                localhost_verified_replay=True,outcomes=verified)
    result_path=OUT/"expectation-browser-verification.json"
    temporary=result_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    temporary.replace(result_path)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("browser_log",type=Path);main(p.parse_args().browser_log)
