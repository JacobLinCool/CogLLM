"""Read out explanatory links between earlier sentences and arriving tokens in a single document."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from research.backbone_transfer.extract import read_json
from research.reading_validation.identity_causal_features import encode_batch, project_basis, sentence_tokens
from research.reading_validation.identity_causal_fit import trace


@torch.inference_mode()
def analyze(text, runtime, weights, supplied_tokens=None):
    """Return the current evidential gain, each antecedent's retained update, and its token offsets."""
    if runtime.candidate!='qwen' or not isinstance(text,str) or not text.strip():
        raise ValueError('The causal review requires original nonempty text and the registered Qwen runtime')
    model=read_json(Path(weights)) if isinstance(weights,(str,Path)) else weights
    if set(model)!={'semantic','lexical','prior','alert_threshold','calibration_null_views'}:
        raise ValueError('Expected a calibrated causal attribution checkpoint')
    encoded=runtime.tokenizer.encode(text,add_special_tokens=False)
    if not 1<=len(encoded.ids)<=2048:raise ValueError('Document exceeds the validated 2,048-token range')
    spans=sentence_tokens(text,encoded.offsets);seq=encoded.ids
    if supplied_tokens is not None:
        for start,end in supplied_tokens.values():
            if not 0<=start<end<=len(seq) or not any(a<=start and end<=b for a,b in spans):
                raise ValueError('Supplied antecedents must lie within one visible sentence')
    count=0
    def counted(_module,_args):
        nonlocal count
        count+=1
    with runtime.encoder.register_forward_pre_hook(counted):
        states,_,_,mean,peak=encode_batch(runtime,[seq],project_basis(runtime.width,runtime.device))
    if count!=1:raise RuntimeError('Expected exactly one encoder pass')
    ident='article';data=SimpleNamespace(spans={ident:spans},text={ident:text},offsets={ident:encoded.offsets})
    attention=torch.stack((mean[0],peak[0]),dim=-1).half().cpu().numpy()
    levels,raw=trace(data,model,ident,states[0].half().cpu().numpy(),attention,supplied_tokens)
    anchors={str(i):s for i,s in enumerate(spans)} if supplied_tokens is None else supplied_tokens
    def region(span):
        start,end=span
        return [encoded.offsets[start][0],encoded.offsets[end-1][1]]
    return {'model':runtime.revision,'forward_count':count,'tokens':seq,'token_offsets':encoded.offsets,
            'automatic':supplied_tokens is None,'sentence_offsets':[region(s) for s in spans],
            'antecedents':{k:{'tokens':list(s),'chars':region(s),'relative_evidence':raw[k].tolist(),
                              'retained_update':levels[k].tolist()} for k,s in anchors.items()},
            'alert_threshold':model['alert_threshold']}
