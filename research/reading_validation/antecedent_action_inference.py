"""Pair a source-trained human action decision with an all-prior-word inference."""

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from research.reading_validation.antecedent_inference import AntecedentInference,BUNDLE,BUNDLE_SHA,REVISION
from research.reading_validation.transition_gate_features import NAMES,features as surface_features
from research.reading_validation.transition_pointer_features import CONTROL,PARTS,sentence_ids


@dataclass(frozen=True)
class ApplicableAntecedentInference:
    pointer: AntecedentInference
    method: str
    threshold: float
    mean: np.ndarray
    scale: np.ndarray
    beta: np.ndarray
    intercept: float

    @classmethod
    def load(cls,signature,directory=BUNDLE):
        """Load only a pinned action gate that passed both its source and external release criteria."""
        directory=Path(directory)
        path=directory/"applicable-pronoun.json";raw=path.read_bytes()
        if sha256(raw).hexdigest()!=signature:raise ValueError("Unexpected broad-applicability word pointer")
        schema=json.loads(raw);method=schema["method"]
        if (schema["schema"]!="cogllm-english-applicable-pronoun-v1" or schema["revision"]!=REVISION
                or schema["pointer_bundle_sha256"]!=BUNDLE_SHA or not schema["external_release_criterion_passed"]
                or method not in ("native","surface") or schema["surface_columns"]!=list(NAMES)):
            raise ValueError("Unknown broad-applicability source contract")
        means,scales,beta=[np.asarray(schema["weights"][k],dtype=float) for k in ("mean","sd","beta")]
        expected=len(NAMES)+(2*schema["hidden_width"] if method=="native" else 0)
        if any(v.shape!=(expected,) or not np.isfinite(v).all() for v in (means,scales,beta)) or (scales<=0).any():
            raise ValueError("Invalid source action readout")
        tau=float(schema["threshold"]);bias=float(schema["weights"]["bias"])
        if not np.isfinite(bias) or not 0<tau<1:raise ValueError("Invalid source action operating point")
        return cls(AntecedentInference.load(directory),method,tau,means,scales,beta,bias)

    @contextmanager
    def capture(self,text,runtime):
        with self.pointer.capture(text,runtime) as past:
            row={"text":text,"words":[w[:2] for w in past.words]}
            row["sentences"]=sentence_ids(text,row["words"])
            pos=CONTROL.index("NNP")
            cues=[]
            for q,c in zip(past.queries,past.controls,strict=True):
                onehot=c[:,pos:pos+len(PARTS)]
                if not np.all(onehot.sum(1)==1):raise ValueError("Invalid prior-word part-of-speech codes")
                tags=np.array(PARTS)[onehot.argmax(1)]
                cues.append(surface_features(row,q,tags))
            yield ActionPass(self,past,cues)


@dataclass
class ActionPass:
    model: ApplicableAntecedentInference
    pointer: object
    cues: list

    def forecasts(self,encoded):
        """Report all earlier-word and discourse-action scores from the already computed article states."""
        prior=self.pointer.logits(encoded)
        if not prior:return {}
        x=np.stack(self.cues)
        if self.model.method=="native":
            import torch
            from torch.nn import functional as F
            with torch.inference_mode():
                idx=self.pointer.ends[torch.tensor(self.pointer.queries,device=self.pointer.ends.device)]
                native=torch.cat([F.layer_norm(v[idx],(v.shape[1],)) for v in (encoded.mid,encoded.hidden)],dim=-1).cpu().numpy()
            x=np.column_stack((x,native))
        z=((x.astype(float)-self.model.mean)/self.model.scale)@self.model.beta+self.model.intercept
        p=np.exp(-np.logaddexp(0,-z));values={}
        for i,q in enumerate(self.pointer.queries):
            logits=prior[q];top=int(logits.argmax())
            conditional=float(1/np.exp(logits-logits[top]).sum())
            values[q]={"word":top,"prior_action_score":float(p[i]),"conditional_word_score":conditional,
                       "selection_score":float(p[i]*conditional)}
        return values

    def links(self,encoded):
        result=[]
        for q,row in self.forecasts(encoded).items():
            if row["selection_score"]<self.model.threshold:continue
            a,b=self.pointer.words[q][:2];x,y=self.pointer.words[row["word"]][:2]
            if not 0<=x<y<=a<b:raise ValueError("Invalid original-text antecedent geometry")
            result.append({"pronoun_start":a,"pronoun_end":b,"antecedent_start":x,"antecedent_end":y,
                           "selection_score":row["selection_score"]})
        return result
