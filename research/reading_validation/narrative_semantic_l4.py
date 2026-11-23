"""Add native semantic line summaries to the original single-pass full-story readout."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from research.backbone_transfer.extract import digest,stream_rows,write_json
from research.reading_validation.identity_run_l4 import run_phases
from research.reading_validation.narrative_course_l4 import prefix
from research.reading_validation.narrative_long import native


@torch.inference_mode()
def run(runtime,reader,packet,source,output):
    rows=list(stream_rows(source))
    if len(rows)!=4:
        raise ValueError("Expected four source-held stories")
    output.mkdir()
    counter=0
    def count(module,args):
        nonlocal counter
        counter+=1
    elapsed=perf_counter()
    totals=[]
    arrays={}
    with runtime.encoder.register_forward_pre_hook(count):
        for r in rows:
            v=native(runtime,reader,packet,r,semantic_lines=True)
            k=r["id"]
            for name in ("line_semantic","trace","pairs","offsets","lines","ends","state"):
                arrays[k+"."+name]=v[name]
            totals.append(dict(id=k,tokens=len(v["trace"]),lines=len(v["ends"])))
    if counter!=4:
        raise RuntimeError("A single story was encoded more than once")
    short=prefix(rows[0],6)
    small=native(runtime,reader,packet,short,semantic_lines=True)
    k=rows[0]["id"]
    error=float(np.abs(small["line_semantic"]-arrays[k+".line_semantic"][:len(small["ends"])]).max())
    if error>0.002:
        raise ArithmeticError(f"Prefix evidence for earlier lines changed: {error}")
    np.savez_compressed(output/"forecasts.npz",**arrays)
    write_json(output/"receipt.json",dict(articles=totals,content_forwards=counter,technical_forwards=1,
            native_revision=runtime.revision,source=digest(source),file=digest(output/"forecasts.npz"),
            prefix_max_abs=error,seconds=perf_counter()-elapsed))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("model","stimuli","output"):
        p.add_argument("--"+key,type=Path,required=True)
    for key in ("candidate","protocol-sha256"):
        p.add_argument("--"+key,required=True)
    a=p.parse_args()
    run_phases(a.model,a.candidate,a.output,a.protocol_sha256,[("narrative-semantic",run,a.stimuli)])
