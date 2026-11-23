"""Test normal English callbacks, optional character markers, and an exact text roundtrip."""

import hashlib
import json
from pathlib import Path

from research.reading_validation.local_forecast import CHECKPOINT
from research.reading_validation.local_inference import LocalInference
from research.reading_validation.narrative_inference import MarginalNarrativeInference
from research.reading_validation.transition_inference import TransitionInference
from research.reading_validation.writer_verify import DraftText, run as verify_writing
from research.token_readout.app import parse_analysis
from research.token_readout.runner import TokenRunner
from research.token_readout.writing_review import STYLE, render


def run(runtime,reader,packet,source,output):
    original=verify_writing(runtime,reader,packet,source,output)
    output=Path(output)
    readings=json.loads((output/"responses.json").read_text())
    runner=TokenRunner(runtime,reader,None,packet["priors"],CHECKPOINT,scale=packet["scale"],
                       anchor_width=reader.anchor_tokens,english_head=LocalInference.load(),
                       transition_head=TransitionInference.load(),narrative_head=MarginalNarrativeInference.load())
    example=("Mara and Leo spent the evening repairing their boat. Mara checked the engine while Leo worked on the hull. "
             "By morning they were ready to leave.\n\nNatascha met them at the dock. She handed Mara a faded photograph.")
    live=runner.review_english(example,narrative=True)
    parse_analysis(example,live)
    html=render(example,live["english"])
    spans=DraftText()
    spans.feed(html)
    spans.close()
    if "".join(spans.parts)!=example or spans.depth or runner.forward_count()!=1:
        raise ValueError("The narrative display changed the author's text or repeated inference")
    named=live["english"]["narrative_words"]
    if len(named)<12 or ("class=\"character\"" not in html and any(r["flagged"] for r in named)
                        and 'phrase character' not in html):
        raise ValueError("Character choices were dropped during rendering")
    readings["4"]=live
    (output/"responses.json").write_text(json.dumps(readings,indent=1,allow_nan=False)+"\n")
    (output/"narrative-preview.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>CogLLM narrative review</title>'
        '<style>body{max-width:54rem;margin:6vh auto;padding:0 1rem;font:17px/1.5 system-ui}'+STYLE+
        '</style><main><h1>CogLLM Narrative Review</h1>'+html+'</main></html>')
    result=dict(original,forwards=original["forwards"]+1,passages=original["passages"]+1,
                tokens=original["tokens"]+len(live["tokens"]),narrative_word_estimates=len(named),
                narrative_flagged_words=sum(r["flagged"] for r in named))
    result["files"]={k:hashlib.sha256((output/k).read_bytes()).hexdigest()
                     for k in ("responses.json","preview.html","narrative-preview.html")}
    (output/"verification.json").write_text(json.dumps(result,indent=2)+"\n")
    return result
