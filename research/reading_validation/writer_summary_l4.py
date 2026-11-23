"""Verify the actual one-pass English summary interface and replay sealed human-source input cases."""
import json
from pathlib import Path

from research.backbone_transfer.extract import digest,write_json
from research.reading_validation.local_forecast import CHECKPOINT
from research.reading_validation.local_inference import LocalInference
from research.reading_validation.memory_data import h
from research.reading_validation.summary_understanding import SummaryUnderstanding
from research.reading_validation.transition_inference import TransitionInference
from research.reading_validation.writer_verify import DraftText
from research.token_readout import summary_view
from research.token_readout.runner import TokenRunner
from research.token_readout.writing_review import render

POSITIVE_EXAMPLE="fc9d7d0fc534b7a03e40f18d758cd105058976e7fdafcc9608c71da3a7d6d9bb"

def run(runtime,reader,packet,source,out):
    source=Path(source);out=Path(out);out.mkdir()
    rows=list(map(json.loads,source.read_text().splitlines()))
    if rows[0]["text"]!=summary_view.EXAMPLE or len(rows)<65:
        raise ValueError("Incorrect original demo or held-source subset")
    runner=TokenRunner(runtime,reader,None,packet["priors"],CHECKPOINT,scale=packet["scale"],anchor_width=reader.anchor_tokens,
                    english_head=LocalInference.load(),transition_head=TransitionInference.load())
    observer=SummaryUnderstanding.load();records={};pred={};calls=[0];explicit=0;empty=0
    def count(*_):calls[0]+=1
    with runtime.encoder.register_forward_pre_hook(count):
        for i,r in enumerate(rows):
            text=r["text"];uid=r["id"]
            if set(r)!={"id","text"} or uid!=h(text) or uid in pred:raise ValueError("Unexpected external replay input")
            if i<2 or uid==POSITIVE_EXAMPLE:
                rec=runner.review_english(text,summary=True)
                html=render(text,rec["english"]);doc=DraftText();doc.feed(html);doc.close()
                if doc.depth or "".join(doc.parts)!=text or "Can a reader understand this summary?" not in html:
                    raise ValueError("Original displayed passage or scope was lost")
                records[str(i+1) if i<2 else "3"]=rec
                pred[uid]=rec["english"]["summary"];explicit+=1
            elif not text.strip():
                pred[uid]=observer.score(text,None,runtime)
                empty+=1
            else:
                features=runtime.encode(text)
                pred[uid]=observer.score(text,features,runtime)
            if calls[0]!=i+1-empty:raise ValueError("Each supported summary needs exactly one encoder pass")
    if set(records)!={"1","2","3"}:raise ValueError("All three original writer decisions must be captured")
    write_json(out/"responses.json",records)
    write_json(out/"native-forecasts.json",pred)
    write_json(out/"verification.json",{"source":__name__,"input_sha256":digest(source),"candidate":runtime.candidate,
          "revision":runtime.revision,"encoder_forwards":calls[0],"full_app_forwards":explicit,
          "direct_serving_forwards":len(pred)-explicit-empty,"empty_abstentions_without_forward":empty,
          "files":{k:digest(out/k) for k in ("responses.json","native-forecasts.json")}})
