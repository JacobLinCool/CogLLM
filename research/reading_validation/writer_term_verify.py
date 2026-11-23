"""Apply the terminology lens to fresh examples through the actual public writer entry point."""

import hashlib
import json
from pathlib import Path

from research.backbone_transfer.extract import digest,write_json
from research.reading_validation.local_inference import LocalInference
from research.reading_validation.terminology_inference import TerminologyInference
from research.reading_validation.transition_inference import CHECKPOINT,TransitionInference
from research.reading_validation.writer_verify import DraftText
from research.token_readout.app import parse_analysis
from research.token_readout.runner import TokenRunner
from research.token_readout.writing_review import TERMINOLOGY_EXAMPLE,render,STYLE


def run(runtime,reader,packet,source,output):
    output.mkdir()
    loaded=TerminologyInference.load()
    app=TokenRunner(runtime,reader,None,packet["priors"],CHECKPOINT,scale=packet["scale"],anchor_width=reader.anchor_tokens,
                    english_head=LocalInference.load(),transition_head=TransitionInference.load(),terminology_head=loaded)
    examples=[TERMINOLOGY_EXAMPLE,
              "A substance dissolved in a liquid is called a solute. Diffusion is the net movement of particles from a region of high concentration to a region of low concentration.",
              "A metal bar rested on the table. They brought the table into the kitchen."]
    all_rows={}
    checks=[]
    for i,text in enumerate(examples):
        count=app.forward_count()
        result=app.review_english(text,terminology=True)
        parsed=parse_analysis(text,result)
        html=render(text,result["english"])
        original=DraftText();original.feed(html);original.close()
        panel=result["english"]["terminology"]
        if (app.forward_count()-count!=1 or original.depth or "".join(original.parts)!=text or
            len(parsed.tokens)!=len(panel["expected"]) or "Terms awaiting an explanation" not in html):
            raise ValueError("Live terminology lens lost content or made a repeated model call")
        checks.append(dict(id=hashlib.sha256(text.encode()).hexdigest(),tokens=len(parsed.tokens),
                           emitted_links=len(panel["links"]),single_forward=True,source_text_roundtrips=True))
        all_rows[str(i+1)]=result
        if i==0:
            (output/"terminology.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1"><title>CogLLM scientific writing review</title>'
                '<style>body{max-width:54rem;margin:6vh auto;padding:0 1rem;font:17px/1.5 system-ui;color:#23352f}'+STYLE+
                '</style><main><h1>Scientific writing review</h1>'+html+'</main></html>')
    write_json(output/"responses.json",all_rows)
    write_json(output/"verification.json",dict(model_revision=runtime.revision,checkpoint=CHECKPOINT,
       encoder_forwards=app.forward_count(),checks=checks,files={f:digest(output/f) for f in ("terminology.html","responses.json")}))
