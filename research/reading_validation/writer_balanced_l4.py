"""Capture the deployed balanced English writer controls on their eight original example inputs."""
import json
from pathlib import Path
from research.backbone_transfer.extract import digest,write_json
from research.reading_validation.local_forecast import CHECKPOINT
from research.reading_validation.local_inference import LocalInference
from research.reading_validation.narrative_inference import DECISIONS_BUNDLE,MarginalNarrativeInference
from research.reading_validation.transition_inference import TransitionInference
from research.reading_validation.writer_verify import DraftText
from research.token_readout.app import parse_analysis
from research.token_readout.runner import TokenRunner
from research.token_readout.writing_review import render,STYLE

def run(runtime,reader,packet,source,out):
    rows=json.loads(Path(source).read_bytes())
    if len(rows)!=8 or any(set(r)!={"view","id","text","narrative"} for r in rows):
        raise ValueError("Need exactly eight text-only writer input/mode selections")
    out=Path(out);out.mkdir()
    gate=MarginalNarrativeInference.for_writers("balanced")
    runner=TokenRunner(runtime,reader,None,packet["priors"],CHECKPOINT,scale=packet["scale"],anchor_width=reader.anchor_tokens,
        english_head=LocalInference.load(),transition_head=TransitionInference.load(),narrative_head=gate)
    calls=0
    def observe(*args):
        nonlocal calls
        calls+=1
    captures={"main":{},"references":{}};flags=0
    with runtime.encoder.register_forward_pre_hook(observe):
        for j,r in enumerate(rows):
            group=r["view"];text=r["text"]
            entry=runner.review_english(text,narrative=r["narrative"],references=group=="references")
            parse_analysis(text,entry);html=render(text,entry["english"]);parsed=DraftText();parsed.feed(html);parsed.close()
            if "".join(parsed.parts)!=text or parsed.depth or calls!=j+1 or runner.forward_count()!=j+1:
                raise ValueError("Source text alignment or single passage exposure failed")
            if r["narrative"]:
                flags+=sum(x["flagged"] for x in entry["english"]["narrative_words"])
            captures[group][r["id"]]=entry
    for group,items in captures.items():
        target=out/group;target.mkdir();write_json(target/"responses.json",items)
        write_json(target/"verification.json",{"encoder_forwards":len(items),"candidate":runtime.candidate,"revision":runtime.revision,
            "threshold_selection":"balanced","decision_bundle":digest(DECISIONS_BUNDLE),"input":digest(source),
            "files":{"responses.json":digest(target/"responses.json")}})
    files={f"{view}/{f}":digest(out/view/f) for view in captures for f in ("responses.json","verification.json")}
    write_json(out/"verification.json",{"forwards":calls,"narrative_flags_across_inputs":flags,"threshold_selection":"balanced","files":files})
