"""One-pass whole-summary forecasts of a trained English rater's comprehensibility judgment."""
from hashlib import sha256
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.special import expit

from research.reading_validation.summary_measurement import summarize

BUNDLE=Path(__file__).resolve().parents[1]/"results/reading-validation-transition-external/seahorse/serving"
SCOPE="english_generated_summary_comprehensibility"


class SummaryUnderstanding:
    def __init__(self,spec,weights):
        self.spec=spec
        self.weights=weights
        self.threshold=float(spec["threshold"])

    @classmethod
    def load(cls,directory=BUNDLE):
        directory=Path(directory);spec=json.loads((directory/"spec.json").read_text())
        weights=directory/"weights.npz"
        if (spec["schema"]!="english-summary-q1-v1" or spec["scope"]!=SCOPE or
                sha256(weights.read_bytes()).hexdigest()!=spec["weights_sha256"]):
            raise ValueError("Unrecognized English summary readout")
        with np.load(weights,allow_pickle=False) as data:
            return cls(spec,dict(data))

    def unavailable(self,reason):
        descriptions = {
            "over_1024_model_tokens":"This summary exceeds the 1,024-token range evaluated in the reading study.",
            "subword_crosses_sentences":"The model could not align the printed sentence boundaries to its reading positions.",
            "no_printed_word":"The summary contains no recognizable printed word.",
            "no_observable_word_surprisal":"This summary has no complete word after the initial model token that can be evaluated.",
            "no_certified_boundary":"This summary has no sentence boundary with a usable whole-word reading measurement.",
        }
        if reason not in descriptions:
            raise ValueError("Unknown reason for a summary without a measurement")
        return {"scope":SCOPE,"available":False,"probability_difficult":None,"threshold":self.threshold,
                "review":None,"reason":descriptions[reason],"reason_code":reason,"eligible_regions":0}

    def from_arrays(self,text,native):
        w=self.weights
        x,regions,reason=summarize(text,native,w["region_center"],w["region_scale"])
        if reason:return self.unavailable(reason)
        if x is None:raise ArithmeticError("Missing document measurement")
        source=w["theta"][0]+((x-w["center"])/w["scale"])@w["theta"][1:]
        p=float(expit(w["calibration"][0]+w["calibration"][1]*source))
        return {"scope":SCOPE,"available":True,"probability_difficult":p,
                "threshold":self.threshold,"review":p>=self.threshold,"reason":None,"reason_code":None,"eligible_regions":regions}

    def score(self,text,features,runtime,*,token_surprisal=None):
        import torch
        from torch.nn import functional as F
        from research.reading_validation.argument_l4 import locations
        from research.reading_validation.expectation_native import coordinates
        from research.reading_validation.identity_causal_features import project_basis
        from research.reading_validation.transition_external_forecast import token_surprisals

        if runtime.candidate!="qwen" or runtime.revision!=self.spec["backbone_revision"]:
            raise ValueError("English summary scoring requires the registered Qwen3.5-2B native states")
        if not text.strip():
            return self.unavailable("no_printed_word")
        if features is None or features.relations is None:
            raise ValueError("A nonempty summary needs the native features of its original text")
        tokenized=runtime.tokenizer.encode(text,add_special_tokens=False)
        if tuple(tokenized.ids)!=tuple(features.token_ids) or tuple(map(tuple,tokenized.offsets))!=tuple(features.offsets):
            raise ValueError("Summary text differs from the encoded passage")
        n=len(features.token_ids)
        if n>1024:return self.unavailable("over_1024_model_tokens")
        if not n:return self.unavailable("no_printed_word")
        parts=locations(text,features.offsets)
        if parts is None:return self.unavailable("subword_crosses_sentences")
        _,spans=parts
        chars=coordinates(text,SimpleNamespace(offsets=features.offsets))
        if not len(chars):return self.unavailable("no_printed_word")
        if not chars[:,4].any():return self.unavailable("no_observable_word_surprisal")
        with torch.inference_mode():
            basis=project_basis(runtime.width,runtime.device)
            state=torch.cat([F.layer_norm(v,v.shape[-1:])@w for v,w in zip((features.mid,features.hidden),basis,strict=True)],dim=1)
            dims=torch.cat((state[:,:32],state[:,96:128]),dim=1)
            running=F.pad(dims.cumsum(0),(0,0,1,0))
            lo,hi=torch.as_tensor(chars[:,2:4].astype(np.int64),device=state.device).T
            word=(running[hi+1]-running[lo])/(hi-lo+1)[:,None]
            width=torch.arange(1,n+1,device=state.device)
            attention=torch.expm1(features.relations+math.log(2)).clamp_min(0)/width[:,None,None]
            rel=torch.stack([torch.stack((attention[hi,a:b,0].sum(1),attention[hi,a:b,1].amax(1)),dim=1) for a,b in spans],dim=1)
            ends=torch.tensor([b for _,b in spans],device=state.device)
            rel=rel.masked_fill((ends[None,:]>lo[:,None])[:,:,None],0.)
            losses=token_surprisals(runtime,features) if token_surprisal is None else token_surprisal
            if len(losses)!=n:raise ValueError("Word log losses and summary token positions differ")
            total=np.r_[0.,np.asarray(losses,dtype=float).cumsum()]
            native=dict(char=chars,words=word.cpu().numpy().astype("float16"),rel=rel.cpu().numpy().astype("float16"),
                        surprisal=(total[chars[:,3]+1]-total[chars[:,2]]).astype("float32"))
        return self.from_arrays(text,native)
