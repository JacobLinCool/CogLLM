"""Replay source-frozen English next-word predictions without acquiring another encoder state."""
from pathlib import Path
from time import perf_counter
import json

import numpy as np
from research.backbone_transfer.extract import digest, stream_rows, write_json
from research.reading_validation.population_inference import PopulationWordInference
from research.reading_validation.transition_architecture_l4 import native_losses


def run(runtime, reader, checkpoint, source, out):
    out=Path(out);out.mkdir();start=perf_counter()
    observer=PopulationWordInference.load()
    rows=list(stream_rows(source))
    predictions=[]; parity=[]; count=0
    def touch(*_):
        nonlocal count
        count+=1
    with runtime.encoder.register_forward_pre_hook(touch):
        for j,r in enumerate(rows):
            if set(r)!={'id','text','ends'}:raise ValueError("Unregistered fields in replay")
            f=runtime.encode(r['text'])
            mode=observer.analyze(r['text'],f,runtime)
            if mode['scope']!="population_next_word_disagreement":raise ValueError("Unknown word observer")
            endmap={row['word_index']:row for row in mode['words']}
            losses=native_losses(runtime,f)
            wordloss=[]
            for i,(a,b,u,v) in enumerate(r['ends']):
                if not (f.offsets[v][1]==b and r['text'][a:b]):
                    raise ValueError("Measured geometry differs from native application")
                wordloss.append(None if u==0 else float(np.sum(losses[u:v+1])))
                if i in endmap and (a,b,v)!=(endmap[i]['start'],endmap[i]['end'],endmap[i]['token']):
                    raise ValueError("Serving word intervals differ from original materials")
            if j in (0,17,40):
                n=len(r['ends'])//2; l=r['ends'][n][1];x=r['text'][:l]
                prefix=runtime.encode(x); pre=observer.analyze(x,prefix,runtime)
                if list(prefix.token_ids)!=list(f.token_ids[:len(prefix.token_ids)]):
                    raise ValueError("Independent prefix tokenization changed")
                e=[];eother=[]
                for line in pre['words']:
                    full=endmap[line['word_index']]
                    e.append(abs(line['after']-full['after']))
                    eother.append(abs(line['other_after']-full['other_after']))
                error=max(e+eother)
                if error>.001:raise ValueError("Future context changed probabilities")
                parity.append(dict(words=len(e),max_error=error))
            predictions.append(dict(id=r['id'],words=mode['words'],surprisal=wordloss))
    if count!=len(rows)+len(parity):raise ValueError("Unexpected encoder forward count")
    file=out/'article-forecasts.jsonl'
    with file.open('w') as stream:
        for r in predictions:stream.write(json.dumps(r,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
    write_json(out/'receipt.json',dict(input=digest(source),documents=len(rows),production_forwards=len(rows),
               checks=parity,all_forwards=count,seconds=perf_counter()-start,output=digest(file)))
