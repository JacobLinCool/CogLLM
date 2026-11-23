"""Capture causal source and token effects across four complete German narratives."""

from argparse import ArgumentParser
import gc
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from research.backbone_transfer.extract import digest,stream_rows,write_json
from research.reading_validation.identity_run_l4 import run_phases
from research.reading_validation.narrative_long import CHANNELS,geometry,native
from research.reading_validation.narrative_long_verify import dense


def prefix(row,n):
    bounds=row["spans"][:n]
    return dict(text=row["text"][:bounds[-1][1]],spans=bounds)


@torch.inference_mode()
def run(runtime,reader,packet,source,output):
    output.mkdir()
    rows=list(stream_rows(source))
    if len(rows)!=4:
        raise ValueError("Four complete original narratives are required")
    sample=prefix(rows[0],5)
    offset=runtime.tokenizer.encode(sample["text"],add_special_tokens=False).offsets
    if len(offset)>512:
        raise ValueError("Unexpected length of direct native comparison")
    units,ends=geometry(sample["text"],sample["spans"],offset)
    forwards=0
    def count(*args):
        nonlocal forwards
        forwards+=1
    clock=perf_counter()
    with runtime.encoder.register_forward_pre_hook(count):
        normal=runtime.encode(sample["text"])
        reference,links=dense(reader,packet,normal,sample["text"],units,ends)
        test=native(runtime,reader,packet,sample)
        verification=dict(old_trace=float(np.max(abs(reference.cpu().numpy()-test["trace"][:,:4]))),
                          candidate_lines=float(np.max(abs(links.cpu().numpy()-test["pairs"]))))
        if max(verification.values())>1e-3:
            raise ArithmeticError(f"Short passage disagrees with original deployed trace: {verification}")
        del normal,reference,links
        receipt={}
        for row in sorted(rows,key=lambda r:len(r["text"])):
            start=perf_counter()
            torch.cuda.reset_peak_memory_stats()
            result=native(runtime,reader,packet,row)
            trace=result["trace"]
            conservation=float(abs(np.diff(trace[:,0],prepend=0)-trace[:,1]+trace[:,2]).max())
            if conservation>2e-4 or np.any(np.triu(result["pairs"][:,:,0],1)>1e-6):
                raise ArithmeticError("Unresolved mass is not conserved or attends to a later source")
            if row["id"]==rows[0]["id"]:
                length=len(test["trace"])
                if not np.array_equal(test["offsets"],result["offsets"][:length]):
                    raise ValueError("Diagnostic source is not a certified native token prefix")
                verification["suffix_trace"]=float(np.max(abs(test["trace"]-trace[:length])))
                verification["suffix_state"]=float(np.max(abs(test["state"].astype(float)-result["state"][:length])))
                span=len(ends)
                # The final visible line of the short text has the same extent in the full text.
                verification["suffix_sources"]=float(np.max(abs(test["pairs"]-result["pairs"][:span,:span])))
                if max(verification["suffix_trace"],verification["suffix_sources"])>1e-3 or verification["suffix_state"]>.007:
                    raise ArithmeticError(f"Native prefix certification failed: {verification}")
            np.savez_compressed(output/(row["id"]+".npz"),**result)
            receipt[row["id"]]=dict(tokens=len(trace),lines=len(result["ends"]),seconds=perf_counter()-start,
                                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),conservation=conservation)
            print("Completed whole story",row["id"][:8],len(trace),"tokens",flush=True)
            del result,trace
            gc.collect()
            torch.cuda.empty_cache()
    if forwards!=6:
        raise ValueError(f"Expected four content and two diagnostic forwards, got {forwards}")
    packed={p.name:digest(p) for p in output.glob("*.npz")}
    write_json(output/"receipt.json",dict(packet=digest(source),channels=list(CHANNELS),articles=receipt,
               short_dense_check=verification,content_forwards=4,technical_forwards=2,seconds=perf_counter()-clock,
               revision=runtime.revision,torch=str(torch.__version__),files=packed))


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    for n in ("model","stimuli","output"):
        parser.add_argument("--"+n,type=Path,required=True)
    for n in ("candidate","protocol-sha256"):
        parser.add_argument("--"+n,required=True)
    a=parser.parse_args()
    run_phases(a.model,a.candidate,a.output,a.protocol_sha256,[("narrative",run,a.stimuli)])
