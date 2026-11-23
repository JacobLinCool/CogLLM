"""Locate a likely incompatible sentence and its earlier cause in a short physical narrative."""

from pathlib import Path
from types import SimpleNamespace
import json
import math

import numpy as np
import torch
from torch.nn import functional as F

from research.reading_validation.argument_l4 import locations, summarize
from research.reading_validation.expectation_native import coordinates
from research.reading_validation.identity_causal_features import project_basis
from research.reading_validation.memory_data import sha
from research.reading_validation.state_events import event_probabilities, pair_rows
from research.reading_validation.state_words import build_document, predict_attributes
from research.reading_validation.transition_external_forecast import token_surprisals

BUNDLE = Path(__file__).resolve().parents[1] / "results/reading-validation-state-transport/serving"


class PhysicalContinuity:
    def __init__(self, spec, weights):
        self.spec = spec
        self.weights = weights

    @classmethod
    def load(cls, directory=BUNDLE):
        directory = Path(directory)
        spec = json.loads((directory/"spec.json").read_bytes())
        if spec["schema"] != "physical-first-conflict-v1" or sha(directory/"weights.npz") != spec["weights"]:
            raise ValueError("The physical-continuity readout does not match its fitted weights.")
        with np.load(directory/"weights.npz", allow_pickle=False) as f:
            weights = {k: f[k] for k in f.files}
        return cls(spec, weights)

    def sentence_probabilities(self, doc):
        params = self.weights
        states = predict_attributes(doc["x"], params["attribute.mu"], params["attribute.sd"], params["attribute.v"])
        pairs = pair_rows(doc, states)
        arms = self.spec["arms"]
        forecasts = [event_probabilities(pairs, arm, params[arm+".mu"], params[arm+".sd"], params[arm+".v"]) for arm in arms]
        weights = self.spec["mixture"]
        return [np.average(np.stack(measures), axis=0, weights=weights) for measures in zip(*forecasts,strict=True)]

    def from_arrays(self, text, arrays):
        doc = build_document(text, arrays)
        predictions = self.sentence_probabilities(doc)
        before = 1.
        events = []
        for j, prob in enumerate(predictions, 1):
            if not 0 < prob[0] <= 1 or not np.isclose(prob.sum(),1.):
                raise ArithmeticError("Origin probabilities are inconsistent at a reading position.")
            events.append(dict(sentence=j, start=doc["regions"][j][0], end=doc["regions"][j][1],
                               first_conflict_probability=float(before * (1-prob[0])),
                               source_probabilities=(before*prob[1:]).tolist()))
            before *= prob[0]
        if not np.isclose(before+sum(v["first_conflict_probability"] for v in events),1.):
            raise ArithmeticError("First-conflict probabilities lose mass along the reading path.")
        return dict(scope="authored_physical_conflict", sentences=doc["regions"], observations=events,
                    probability_any=float(1-before), review=bool(1-before>=self.spec["threshold"]))

    @torch.inference_mode()
    def from_features(self, text, runtime, features, token_surprisal=None):
        if runtime.revision!=self.spec["revision"] or runtime.candidate!="qwen":
            raise ValueError("Physical continuity requires the verified Qwen3.5-2B model.")
        n=len(features.token_ids)
        if n>1024 or features.relations is None:
            raise ValueError("Physical continuity requires native causal attention and up to 1,024 tokens.")
        loc=locations(text,features.offsets)
        if not loc or not 2 <= len(loc[0]) <= 7:
            raise ValueError("Use two to seven complete sentences about a sequence of physical events.")
        _, spans=loc
        words=coordinates(text,SimpleNamespace(offsets=features.offsets))
        if not len(words):
            raise ValueError("The passage needs visible words at the measured reading positions.")
        basis=project_basis(runtime.width,runtime.device)
        st=torch.cat([F.layer_norm(v,v.shape[-1:])@w for v,w in zip((features.mid,features.hidden),basis,strict=True)],dim=1)
        widths=torch.arange(1,n+1,device=st.device)
        attention=torch.expm1(features.relations+math.log(2)).clamp_min(0)/widths[:,None,None]
        attention=attention.masked_fill(torch.triu(torch.ones(n,n,dtype=torch.bool,device=st.device),diagonal=1).unsqueeze(2),0.)
        embedding=F.layer_norm(runtime.encoder.embed_tokens(torch.tensor(features.token_ids,device=st.device)),(runtime.width,))@basis[0]
        rep,links=summarize(st,embedding,attention[:,:,0],attention[:,:,1],spans)
        dims=torch.cat((st[:,:32],st[:,96:128]),dim=1)
        running=F.pad(dims.cumsum(0),(0,0,1,0))
        lo,hi=torch.as_tensor(words[:,2:4].astype(np.int64),device=st.device).T
        lexical=(running[hi+1]-running[lo])/(hi-lo+1)[:,None]
        surprisal=token_surprisals(runtime,features) if token_surprisal is None else token_surprisal
        cumulative=np.r_[0.,np.asarray(surprisal,dtype="float64").cumsum()]
        native=dict(char=words,rep=rep.cpu().numpy().astype("float16"),links=links.cpu().numpy().astype("float16"),
                    words=lexical.cpu().numpy().astype("float16"),
                    surprisal=(cumulative[words[:,3]+1]-cumulative[words[:,2]]).astype("float32"))
        return self.from_arrays(text,native)
