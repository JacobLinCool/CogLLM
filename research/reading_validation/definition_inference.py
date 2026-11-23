"""Single-pass, original-offset concept ledger on Qwen3.5-2B frozen backbone features."""

from math import log
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import torch
from torch.nn import functional as F

from research.backbone_transfer.extract import digest,read_json
from research.reading_validation.definition_data import OUT,word_geometry
from research.reading_validation.definition_features import Page
from research.reading_validation.definition_forecast import projection
from research.reading_validation.definition_stock import evidence,render


@torch.inference_mode()
def causal_page(text, features, runtime, revision):
    """Project already-observed native states to the exact coordinates used during training."""
    if runtime.candidate!='qwen' or runtime.revision!=revision or features.relations is None:
        raise ValueError('This ledger requires the certified native Qwen backbone and its attention outputs')
    words,n=word_geometry(text,runtime.tokenizer);enc=runtime.tokenizer.encode(text,add_special_tokens=False)
    if tuple(enc.ids)!=tuple(features.token_ids) or tuple(enc.offsets)!=tuple(features.offsets) or n>1024 or not words:
        raise ValueError('Unsupported document or inconsistent text indices')
    basis=projection(runtime);states=torch.cat([F.layer_norm(x,x.shape[-1:])@weight for x,weight in zip((features.mid,features.hidden),basis,strict=True)],-1)
    last=torch.as_tensor([r[3] for r in words],device=states.device);native=features.relations[last]+log(2.)
    valid=torch.arange(n,device=states.device)[None,:]<=last[:,None];native=native.masked_fill(~valid[:,:,None],0.)
    quant=(32*native).round();minimum,maximum=float(quant.min()),float(quant.max())
    if minimum<0 or maximum>255 or not torch.isfinite(quant).all():raise ValueError('Unexpected native attention range')
    page=Page(text,words,states.half().cpu().numpy(),quant.to(torch.uint8).cpu().numpy())
    return page


@dataclass
class ConceptLedger:
    model_revision: str
    term: dict
    link: dict
    threshold: float
    arm: str

    @classmethod
    def load(cls,root=OUT,arm='semantic'):
        if arm not in ('joint','semantic'):raise ValueError('A serving ledger requires a contextual model')
        root=Path(root);lock=read_json(root/'stock-thresholds.json');name='joint-weights.json' if arm=='joint' else 'weights.json'
        if digest(root/name)!=lock['weights_sha256'][arm]:raise ValueError('Fitted stock and calibration weights differ')
        weights=read_json(root/name)
        term=weights['term'] if arm=='joint' else weights['term'][arm];link=weights['link'] if arm=='joint' else weights['link'][arm]
        return cls(lock['revision'],term,link,lock['thresholds'][arm]['threshold'],arm)

    @torch.inference_mode()
    def from_features(self,text,features,runtime):
        page=causal_page(text,features,runtime,self.model_revision)
        ledger=evidence(page,self.term,self.link,self.arm)
        return render(page,ledger,self.threshold,features.offsets)

    def analyze(self,text,runtime):
        count=0
        def seen(_m,_a):
            nonlocal count
            count+=1
        with runtime.encoder.register_forward_pre_hook(seen):
            out=self.from_features(text,runtime.encode(text),runtime)
        if count!=1:raise RuntimeError('A fresh article must require exactly one encoder call')
        out['encoder_forwards']=count
        return out
