"""Reproduce all 26,417 movie word forecasts through the shipped narrative reader."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from research.reading_validation.narrative_inference import CHECKPOINT, NarrativeInference


def prepare():
    from research.reading_validation.transition_snac_fit import inputs
    from research.reading_validation.transition_snac_prepare import ROOT, RESULTS
    docs,meta=inputs();sealed=np.load(RESULTS/'heldout-probabilities.npz');packet=[]
    for m in meta['documents']:
        if m['split']!='test':continue
        doc=docs[m['doc']];p=sealed['CharE__'+m['doc']][:,3]
        packet.append({'text':doc['text'],'segments':[[s['start'],s['end']] for s in m['segments']],
                       'expected':[[w[0],w[1],float(prob)] for w,prob in zip(doc['words'],p,strict=True)]})
    out=ROOT/'narrative-serving-stimuli.jsonl';out.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in packet));print(sha256(out.read_bytes()).hexdigest())


def run(runtime,reader,checkpoint,input_path,output_dir,*,model=None):
    import torch
    from research.reading_validation.transition_external_forecast import token_surprisals
    source=Path(input_path).read_bytes();docs=[json.loads(s) for s in source.splitlines()];model=NarrativeInference.load() if model is None else model;calls=[0];checks=0;error=0.;flags=0
    def observe(_encoder,_args):calls[0]+=1
    def forecast(text,segments,study):
        count=calls[0];features=runtime.encode(text);cached=token_surprisals(runtime,features)
        head=model.score_segments if study else model.score
        score=head(text,features,runtime.tokenizer,reader,runtime,segments,checkpoint_sha256=CHECKPOINT,token_surprisal=cached)
        if calls[0]!=count+1:raise RuntimeError('One raw passage must have exactly one backbone call')
        return score
    with torch.inference_mode(), runtime.encoder.register_forward_pre_hook(observe):
        for doc in docs:
            out=forecast(doc['text'],doc['segments'],True)
            for r,(a,b,p) in zip(out,doc['expected'],strict=True):
                if r['start']!=a or r['end']!=b or r['flagged']!=(p>model.threshold):raise ValueError('A released location/threshold decision changed')
                error=max(error,abs(r['risk']-p));checks+=1;flags+=r['flagged']
        if forecast('…',[(0,1)],False)!=[]:raise ValueError('No words must produce no word alerts')
        shorter='Café guests met at the Cafe\u0301.'
        text=shorter+' After lunch, Rosa arrived. The door closed.'
        stops=[(0,len(shorter)),(len(shorter)+1,text.index(' The door')),(text.index('The door'),len(text))]
        direct=forecast(text,stops,False);prefix=forecast(shorter,[(0,len(shorter))],False)
        if [text[r['start']:r['end']] for r in prefix]!=['Café','guests','met','at','the','Cafe\u0301']:
            raise ValueError('Original Unicode character ranges changed')
        same=[r for r in direct if r['end']<=len(shorter)]
        prefix_error=max(abs(a['risk']-b['risk']) for a,b in zip(same,prefix,strict=True))
    if len(docs)!=39 or checks!=26417 or calls[0]!=42 or max(error,prefix_error)>1e-4:
        raise ValueError(f'Narrative serving verification failed: {checks}, {calls}, {error}, {prefix_error}')
    dst=Path(output_dir);dst.mkdir(exist_ok=True,parents=True);(dst/'receipt.json').write_text(json.dumps({'source_sha256':sha256(source).hexdigest(),
        'movies':len(docs),'all_original_words_checked':checks,'threshold_flags':flags,'matched_all_locations_and_flags':True,
        'inputs_and_encoder_forwards':calls[0],'max_abs_score_difference':error,'max_abs_changed_suffix_difference':prefix_error,
        'unicode_words_checked':['Café','Cafe\u0301'],'source_with_no_words_yields_zero_flags':True},indent=2))


if __name__=='__main__':prepare()
