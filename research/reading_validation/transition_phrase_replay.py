"""Check original Unicode coordinates and every measured English antecedent extension at deployment."""

from collections import defaultdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from research.reading_validation.antecedent_inference import AntecedentInference
from research.reading_validation.antecedent_phrase_inference import PHRASE_MODEL,PHRASE_MODEL_SHA,PhraseInference,expand_links
from research.reading_validation.transition_score import dump,sha
from research.reading_validation.transition_snac_prepare import ROOT


SRC=PHRASE_MODEL.parent.parent
GAP=SRC.parent/"reading-validation-transition-pointer-gap"


def run():
    head=AntecedentInference.load();phrases=PhraseInference.load(PHRASE_MODEL,PHRASE_MODEL_SHA)
    results=json.loads((SRC/"phrases/heldout-phrases.json").read_text())
    coverage={}
    for name,packet,forecasts in (("game","pointer-stimuli.jsonl",SRC/"candidate-word-logits.npz"),
                                  ("gap","gap-pointer-stimuli.jsonl",GAP/"candidate-word-logits.npz")):
        expected=results[name]; keys=defaultdict(list)
        for k in expected:
            d,q=k.rsplit("/",1);keys[d].append((k,int(q)))
        n=0
        with np.load(forecasts) as native:
            for r in map(json.loads,(ROOT/packet).read_text().splitlines()):
                if r["id"] not in keys:continue
                bounds=r["words"];links=[]
                proxy=SimpleNamespace(words=bounds,model=head,queries=r["queries"])
                for k,q in keys[r["id"]]:
                    seed=int(native[k][-1].argmax());a,b=bounds[seed];x,y=bounds[q]
                    links.append({"pronoun_start":x,"pronoun_end":y,"antecedent_start":a,"antecedent_end":b})
                observed=expand_links(r["text"],proxy,links,phrases)
                for qs,l in zip(keys[r["id"]],observed,strict=True):
                    k,q=qs; a,b=expected[k]
                    if (l["phrase_start"],l["phrase_end"])!=(bounds[a][0],bounds[b-1][1]):raise ValueError("Production phrase coordinates differ from source study")
                    n+=1
        if n!=len(expected):raise ValueError("A measured English phrase was omitted")
        coverage[name]=n
    text="Zoë and René met. She waved."
    bounds=[(0,3),(4,7),(8,12),(13,16),(18,21),(22,27)]
    function=lambda t:expand_links(t,SimpleNamespace(words=bounds,model=head,queries=[4]),[{"pronoun_start":18,"pronoun_end":21,
             "antecedent_start":8,"antecedent_end":12}],phrases)
    a=function(text)[0];b=function(text+" A new plot line.")[0]
    assert a==b and a["phrase_start"]<=8 and 12<=a["phrase_end"]<=18
    dump(SRC/"phrases/serving-replay.json",{"model_sha256":PHRASE_MODEL_SHA,"source_specification_sha256":sha(Path("research/reading_validation/antecedent_phrase_inference.py")),
        "original_word_span_exact_reconstruction":coverage,"unicode_and_suffix_invariance":True,"additional_encoder_calls":0})


if __name__=="__main__":run()
