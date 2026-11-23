"""Replay locked English transition forecasts through the public inference API on actual hardware."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

from research.reading_validation.transition_inference import TransitionInference, CHECKPOINT


def prepare():
    import csv
    from research.reading_validation.transition_external_prepare import ROOT, OUT
    packet={r['id']:r for r in [json.loads(s) for s in (ROOT/'cohesion-stimuli.jsonl').read_text().splitlines()]}
    predicted={r['id']:r for r in csv.DictReader((OUT/'heldout-predictions.csv').open())}
    material=json.loads((OUT/'stimuli.json').read_text())['material']
    examples=[]
    for source in ('cohesentia','decor'):
        members={r['doc'] for r in material if r['source']==source and r['split'] in ('holdout','external')}
        selected=0
        for h in sorted(members):
            row=packet[h];stops=row['stops'];text=row['text']
            if [r['index'] for r in stops]!=list(range(2,len(stops)+2)) or len(stops)<3:
                continue
            spans=[(0,len(text[:stops[0]['start']].rstrip()))]+[(r['start'],r['end']) for r in stops]
            examples.append({'id':h,'text':text,'spans':spans,'expected':[
                {k:float(predicted[r['id']][arm]) for k,arm in [('risk','integration'),('distilled_risk','student'),('surprisal_control_risk','lexical_surprisal')]}
                for r in stops]})
            selected+=1
            if selected==3:
                break
        if selected!=3:
            raise ValueError('Six complete public forecast fixtures are required')
    path=ROOT/'transition-serving-stimuli.jsonl'
    path.write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in examples))
    print(path,sha256(path.read_bytes()).hexdigest())


def run(runtime,reader,packet,input_path,output_dir):
    import torch
    from research.token_readout.runner import trace_rows
    model=TransitionInference.load()
    raw=Path(input_path).read_bytes();examples=[json.loads(s) for s in raw.splitlines()]
    counter=[0]
    def hook(_encoder,_args):
        counter[0]+=1
    def forecast(text,spans):
        before=counter[0]
        features=runtime.encode(text);scores=reader(features);scale=packet['scale']
        scores=replace(scores,opening=scores.opening*scale['opening_gain'],resolution=scores.resolution*scale['resolution_gain']+scale['resolution_bias'])
        trace=trace_rows(text,features.offsets,scores,packet['priors'],reader.anchor_tokens)
        output=model.score(text,features,runtime.tokenizer,reader,runtime,spans,token_trace=trace,checkpoint_sha256=CHECKPOINT)
        if counter[0]!=before+1:
            raise RuntimeError('Transition analysis must retain one backbone call for all sentence boundaries')
        return output
    arms=['risk','distilled_risk','surprisal_control_risk']
    checked=0;uncertified=0;max_replay=0.;max_prefix=0.;prefix_checks=0
    with torch.inference_mode(), runtime.encoder.register_forward_pre_hook(hook):
        for row in examples:
            original=forecast(row['text'],row['spans'])
            for prediction,target in zip(original,row['expected'],strict=True):
                if prediction['risk'] is None:
                    uncertified+=1
                    continue
                checked+=1;max_replay=max(max_replay,*[abs(prediction[k]-target[k]) for k in arms])
            # Keep the first originally scored complete sentence, replace every subsequent sentence.
            end=row['spans'][1][1]; prefix=row['text'][:end]
            changed=prefix+' A clock showed the time.'
            variants=[(prefix,row['spans'][:2]),(changed,row['spans'][:2]+[[end+1,len(changed)]])]
            if original[0]['risk'] is None:
                raise ValueError('The first check requires a certified earlier boundary')
            for altered,spans in variants:
                replay=forecast(altered,spans)[0]
                if replay['risk'] is None:
                    raise ValueError('A changed suffix broke certification of the earlier boundary')
                max_prefix=max(max_prefix,*[abs(original[0][k]-replay[k]) for k in arms]);prefix_checks+=1
    if len(examples)!=6 or counter[0]!=18 or checked<18 or max_replay>1e-4 or max_prefix>1e-4:
        raise ValueError(f'Serving validation failed: {counter}, {checked}, {max_replay}, {max_prefix}')
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    (out/'receipt.json').write_text(json.dumps({'fixture_sha256':sha256(raw).hexdigest(),'source_articles':len(examples),
        'original_and_perturbed_forecasts':18,'observed_backbone_forwards':counter[0],'source_forecasts_checked':checked,
        'uncertified_original_endpoints':uncertified,'suffix_or_truncation_comparisons':prefix_checks,
        'max_abs_released_forecast_difference':max_replay,'max_abs_changed_suffix_difference':max_prefix},indent=2))


if __name__=='__main__':
    prepare()
