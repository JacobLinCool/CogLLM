"""Read population next-word disagreement from an already-encoded English passage."""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np
from scipy.special import logsumexp

DEFAULT = Path(__file__).resolve().parents[1] / "results/reading-validation-cloze-population/serving"
SCOPE = (
    "Estimated chance that two English-speaking readers will give different next-word "
    "responses among 8,192 named spellings and one other category. Training used genuine "
    "English sentence completions and evaluation used English paragraphs."
)


@dataclass
class PopulationWordInference:
    words: list
    weights: dict
    revision: str

    @classmethod
    def load(cls, root=DEFAULT):
        root = Path(root)
        spec = json.loads((root/"spec.json").read_text())
        for name, expected in spec["files"].items():
            if sha256((root/name).read_bytes()).hexdigest()!=expected:
                raise ValueError("Population readout file differs: "+name)
        words = json.loads((root/"dictionary.json").read_text())
        with np.load(root/"weights.npz",allow_pickle=False) as w:
            weights = {key:w[key] for key in w.files}
        if len(words)!=8192 or weights["co"].shape!=(275,) or weights["U"].shape!=(8193,18):
            raise ValueError("Incompatible next-word probability observer")
        return cls(words,weights,spec["revision"])

    @staticmethod
    def positions(text, features, tokenizer):
        offsets = features.offsets
        by_end = {end:i for i,(start,end) in enumerate(offsets) if end>start}
        matches=list(re.finditer(r"\S+",text))
        rows=[]
        for j,m in enumerate(matches):
            lo,hi=m.span()
            i=by_end.get(hi)
            if i is None:
                continue
            prefix=tokenizer.encode(text[:hi],add_special_tokens=False).ids
            if len(prefix)!=i+1 or prefix!=list(features.token_ids[:i+1]):
                continue
            rows.append((j,lo,hi,i,m.group()))
        return matches,rows

    def probabilities(self, runtime, features, tokens):
        import torch
        with torch.inference_mode():
            yield from self._probabilities(runtime, features, tokens)

    def _probabilities(self, runtime, features, tokens):
        import torch
        from torch.nn import functional as F
        if runtime.candidate!="qwen" or runtime.revision!=self.revision:
            raise ValueError("Population readout requires its registered Qwen backbone")
        z=self.weights
        unique=torch.as_tensor(z['unique'],device=runtime.device)
        emb=runtime.encoder.embed_tokens.weight
        other=torch.ones(len(emb),dtype=torch.bool,device=runtime.device)
        other[unique]=False
        matrices=[torch.as_tensor(z[k],device=runtime.device) for k in ('mid_proj','last_proj')]
        wordbias=z['U']@z['co'][1:19]
        for lo in range(0,len(tokens),16):
            chunk=tokens[lo:lo+16]
            ii=torch.as_tensor(chunk,device=runtime.device)
            ctx=torch.cat([F.layer_norm(h[ii],(runtime.width,))@p for h,p in
                           zip((features.mid,features.hidden),matrices,strict=True)],dim=1)
            # The source head was trained with the following recorded precision.
            ctx=ctx.half().cpu().numpy().astype(float)
            ctx=((ctx-z['mean'])/z['scale'])@z['basis']
            original=F.linear(features.hidden[ii],emb).float()
            logit=torch.cat([original[:,unique],torch.logsumexp(original[:,other],dim=1)[:,None]],dim=1)
            lg=(logit-torch.logsumexp(original,dim=1)[:,None]).half().cpu().numpy().astype(np.float32)
            lp=np.empty((len(chunk),len(self.words)+1),dtype=np.float32)
            lp[:,:-1]=lg[:,z['route']]+z['share']
            lp[:,-1]=lg[:,-1]
            lp-=logsumexp(lp,axis=1)[:,None]
            calibrated=z['co'][0]*lp+wordbias+(ctx@z['co'][19:].reshape(16,16))@z['V'].T
            p=np.exp(calibrated-logsumexp(calibrated,axis=1)[:,None])
            if np.max(abs(p.sum(1)-1))>1e-8:
                raise ValueError("Word observer lost probability mass")
            yield p

    def analyze(self, text, features, runtime):
        matches,valid=self.positions(text,features,runtime.tokenizer)
        out=[];previous=None
        tokens=[r[3] for r in valid]
        offset=0
        for prob in self.probabilities(runtime,features,tokens):
            for p in prob:
                ordinal,start,end,token,form=valid[offset]
                q=float(1.-p@p);other=float(p[-1])
                before=remaining=None
                if previous is not None and previous[0]+1==ordinal and text[previous[1]:start]==" ":
                    before=previous[2];remaining=previous[3]
                largest=np.argsort(-p,kind="stable")[:3]
                top=[{"word":self.words[k] if k<len(self.words) else None,"probability":float(p[k])} for k in largest]
                out.append({"start":start,"end":end,"text":form,"token":token,"word_index":ordinal,
                            "before":before,"after":q,"change":None if before is None else q-before,
                            "other_before":remaining,"other_after":other,"top_next":top,
                            "document_end":bool(ordinal+1==len(matches))})
                previous=(ordinal,end,q,other)
                offset+=1
        return {"scope":"population_next_word_disagreement","interpretation":SCOPE,"words":out,"coverage":{"surface_words":len(matches),
                "verified_word_ends":len(valid),"adjacent_single_space_updates":sum(r['change'] is not None for r in out)},
                "units":"Probability, from 0 to 1. The exact-spelling upper bound adds other_after squared.",
                "terminal_scope":"At document end, after refers to a hypothetical next word. The source readers predicted actual continuations."}
