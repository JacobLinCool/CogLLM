"""Lock one article-level forecast from previously fitted English reader-response estimators."""
from dataclasses import replace
from pathlib import Path
import re
from time import perf_counter
import numpy as np
import torch

from research.backbone_transfer.extract import digest,stream_rows,write_json
from research.reading_validation.local_inference import LocalInference
from research.reading_validation.transition_inference import TransitionInference
from research.reading_validation.transition_external_forecast import token_surprisals
from research.token_readout.runner import trace_rows


@torch.inference_mode()
def run(runtime,reader,checkpoint,input_path,output_dir):
    source=Path(input_path);output=Path(output_dir);output.mkdir(parents=True)
    local=LocalInference.load();transition=TransitionInference.load();hash_curve=digest(Path('curve.pt'));scale=checkpoint['scale']
    results={};count=0;start=perf_counter()
    def hook(_m,_a):
        nonlocal count
        count+=1
    with runtime.encoder.register_forward_pre_hook(hook):
        for row in stream_rows(source):
            text=row['text'];f=runtime.encode(text)
            s=reader(f);s=replace(s,opening=s.opening*scale['opening_gain'],resolution=s.resolution*scale['resolution_gain']+scale['resolution_bias'])
            trace=trace_rows(text,f.offsets,s,checkpoint['priors'],reader.anchor_tokens)
            y=local.score(text,f,runtime.tokenizer,reader)['sentences'];spans=[(r['start'],r['end']) for r in y]
            surprise=token_surprisals(runtime,f)
            ts=transition.score(text,f,runtime.tokenizer,reader,runtime,spans,token_trace=trace,checkpoint_sha256=hash_curve,token_surprisal=surprise)
            valid=[r['rating'] for r in y if r['rating'] is not None];tips=[r['risk'] for r in ts if r['risk'] is not None]
            if len(valid)<2 or not tips:raise ValueError('Insufficient original sentence or boundary coverage')
            words=re.findall(r"\b[^\W_]+(?:['’][^\W_]+)*",text);W=len(words);S=len(spans);C=sum(c.isalnum() for c in text)
            stock=np.array([r['unresolved'] for r in trace]);results[row['id']]={'chars':len(text),'tokens':len(trace),'words':W,'sentences':S,'rated':len(valid),
                'boundaries':len(ts),'rated_boundaries':len(tips),
                'difficulty':float(np.mean([5-r['understanding'] for r in valid])),'ease_difficulty':float(np.mean([5-r['ease'] for r in valid])),
                'coherence':float(np.mean(tips)),'peak_transition':float(max(tips)),'mean_unresolved':float(stock.mean()),
                'peak_unresolved':float(np.quantile(stock,.9)),'surprisal':float(surprise[1:].mean()),
                'ARI':4.71*C/W+.5*W/S-21.43,'coleman_liau':.0588*(100*C/W)-.296*(100*S/W)-15.8}
            if count!=len(results):raise ValueError('A news presentation requires exactly one backbone pass')
    write_json(output/'predictions.json',results);write_json(output/'receipt.json',{'input':digest(source),'protocol':digest(Path('guardian-protocol.md')),
        'forecasts':digest(output/'predictions.json'),'readout':hash_curve,'domain':'untrained English news articles','queries':len(results),
        'forwards':count,'seconds':perf_counter()-start,'reader_outcomes_loaded':False})
