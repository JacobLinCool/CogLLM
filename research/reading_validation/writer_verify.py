"""Verify the shipped writing readout on real live features and save a replayable response."""

from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path

from research.reading_validation.local_inference import LocalInference, REVISION
from research.token_readout.app import parse_analysis
from research.token_readout.writing_review import EXAMPLE, STYLE, render


class DraftText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth=0
        self.parts=[]

    def handle_starttag(self,tag,attrs):
        if self.depth:
            self.depth+=1
        elif tag=="div" and dict(attrs).get("class")=="draft":
            self.depth=1

    def handle_endtag(self,tag):
        if self.depth:
            self.depth-=1

    def handle_data(self,data):
        if self.depth:
            self.parts.append(data)


def run(runtime,reader,packet,input_path,output):
    from research.reading_validation.local_forecast import CHECKPOINT
    from research.reading_validation.transition_inference import TransitionInference
    from research.token_readout.runner import TokenRunner
    if runtime.revision!=REVISION or runtime.candidate!="qwen":
        raise ValueError("Validation requires the exact supported backbone")
    head=LocalInference.load()
    runner=TokenRunner(runtime,reader,None,packet["priors"],CHECKPOINT,scale=packet["scale"],
                       anchor_width=reader.anchor_tokens,english_head=head,transition_head=TransitionInference.load())
    count=0
    def observe(module,args):
        nonlocal count
        count+=1
    output=Path(output)
    output.mkdir(parents=True)
    captures={}
    prefix=EXAMPLE.split(". ",1)[0]+"."
    examples=(EXAMPLE,prefix+" Nobody mentioned the experiment again.","The result was < 5% & the signal stayed low.")
    with runtime.encoder.register_forward_pre_hook(observe):
        for i,text in enumerate(examples,1):
            result=runner.review_english(text)
            parse_analysis(text,result)
            marked=render(text,result["english"])
            parsed=DraftText()
            parsed.feed(marked)
            parsed.close()
            if "".join(parsed.parts)!=text or parsed.depth!=0 or count!=i or runner.forward_count()!=i:
                raise AssertionError("Writing review changed the draft or used extra backbone calls")
            if any(s["rating"] is None for s in result["english"]["sentences"]):
                raise AssertionError("Expected certified original endpoints for this integration fixture")
            captures[str(i)]=result
            if i==1:
                page='<!doctype html><html lang="en"><meta charset="utf-8"><title>CogLLM writing review</title>'
                page+='<meta name="viewport" content="width=device-width,initial-scale=1">'
                page+='<style>body{max-width:54rem;margin:6vh auto;padding:0 1rem;font:17px/1.5 system-ui}'+STYLE+'</style>'
                page+='<main><h1>CogLLM Writing Review</h1><p>English wording and continuity estimates from a single model pass.</p>'
                (output/"preview.html").write_text(page+marked+'</main></html>')
    a,b=(captures[k]["english"]["sentences"][0] for k in ("1","2"))
    if ([(p["start"],p["end"]) for p in a["phrases"]]!=[(p["start"],p["end"]) for p in b["phrases"]]
            or max(abs(a["rating"][k]-b["rating"][k]) for k in a["rating"])>.002):
        raise AssertionError("Changing the continuation changed the shared first sentence")
    if max(abs(captures["1"]["english"]["transitions"][i]["risk"]) for i in (0,1))>1:
        raise AssertionError("Uncertified or invalid coherence probability in the example")
    (output/"responses.json").write_text(json.dumps(captures,indent=1,allow_nan=False)+"\n")
    verification=dict(model_revision=runtime.revision,curve_checkpoint=CHECKPOINT,bundle_sha256=head.bundle_sha256,
                      forwards=count,passages=len(examples),tokens=sum(len(r["tokens"]) for r in captures.values()),
                      verified_full_text_roundtrip=True,unchanged_prefix_sentence_ranking=True,
                      prefix_rating_max_abs_difference=max(abs(a["rating"][k]-b["rating"][k]) for k in a["rating"]),
                      files={k:sha256((output/k).read_bytes()).hexdigest() for k in ("responses.json","preview.html")})
    (output/"verification.json").write_text(json.dumps(verification,indent=2)+"\n")
    return verification
