"""Verify the writer-facing physical observer against precomputed article-only test forecasts."""

from pathlib import Path
from time import perf_counter
import json

import numpy as np
import torch

from research.backbone_transfer.extract import digest, stream_rows, write_json
from research.reading_validation.state_inference import BUNDLE
from research.token_readout.physical_view import EXAMPLE, render
from research.token_readout.runner import TokenRunner


def max_discrepancy(a,b):
    if [list(x) for x in a["sentences"]]!=[list(x) for x in b["sentences"]]:
        raise ValueError("Original sentence coordinates changed in physical serving.")
    differences=[abs(a["probability_any"]-b["probability_any"])]
    for u,v in zip(a["observations"],b["observations"],strict=True):
        differences.append(float(np.max(abs(np.asarray(u["source_probabilities"])-v["source_probabilities"]))))
    return max(differences)


@torch.inference_mode()
def run(runtime, reader, packet, source, output_dir):
    out=Path(output_dir);out.mkdir()
    original=list(stream_rows(source));expected=json.loads((BUNDLE/"expected.json").read_bytes())
    if len(original)!=10 or set(expected)!={r["id"] for r in original}:
        raise ValueError("Missing preregistered physical verification sample.")
    start=perf_counter()
    runner=TokenRunner(runtime,reader,None,packet["priors"],"physical-verified",scale=packet["scale"],anchor_width=reader.anchor_tokens)
    records={};drifts=[];checks=[];calls=0
    def count(*args):
        nonlocal calls
        calls+=1
    with runtime.encoder.register_forward_pre_hook(count):
        for i, row in enumerate(original+[{"id":"writer-example","text":EXAMPLE}]):
            text=row["text"];key=row["id"]
            result=runner.physical_story(text)
            if runner.forward_count()!=i+1:
                raise ValueError("The normal writer entry point repeated its native forward.")
            if key in expected:
                err=max_discrepancy(result,expected[key]);drifts.append(err)
                if err>.003 or result["review"]!=expected[key]["review"]:
                    raise ArithmeticError(f"Physical review failed offline/native parity: {err}")
            records[key]=dict(text=text, reading=result, html=render(text,result))
        for row in original[:2]:
            exposed=records[row["id"]]["reading"]
            prefix=row["text"][:exposed["sentences"][2][1]]
            early=runner._physical_head.from_features(prefix,runtime,runtime.encode(prefix))
            full=exposed["observations"][:2]
            drift=max(np.max(abs(np.array(a["source_probabilities"])-b["source_probabilities"])) for a,b in zip(early["observations"],full,strict=True))
            if drift>.003:raise ArithmeticError(f"Future words changed an earlier physical forecast: {drift}")
            checks.append(float(drift))
    if calls!=13:raise RuntimeError("Unexpected pass count in physical replay.")
    write_json(out/"responses.json",records)
    write_json(out/"verification.json",dict(articles=len(records),article_forwards=runner.forward_count(),native_calls=calls,
               independent_prefix_forwards=2,cutoff_decisions_unchanged=True,maximum_probability_error=max(drifts),
               max_early_prefix_error=max(checks),source=digest(source),bundle=digest(BUNDLE/"spec.json"),
               reference=digest(BUNDLE/"expected.json"),seconds=perf_counter()-start,files={"responses.json":digest(out/"responses.json")}))
