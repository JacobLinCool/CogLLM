"""Score frozen hazards at human-selected evidence paragraphs and historical cues."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from research.backbone_transfer.extract import digest, read_json, stream_rows, write_json
from research.reading_validation.identity_fit import ARMS, DATA, RESULT
from research.reading_validation.identity_prepare import PREVIOUS, TOKENIZER
from research.token_readout.score import evidence_summary
from research.token_readout.transfer_data import EncodedText
from research.token_readout.transfer_score import ab_rows, ab_summary, control_summary, controls_rows, evidence_rows, ranking_by_display_order


def intervals(values):
    x=np.asarray(values,dtype=float)
    if not len(x):
        return None
    sample=np.random.default_rng(203912).integers(0,len(x),size=(20000,len(x)))
    estimate=x[sample].mean(1)
    return {'mean':float(x.mean()),'paired_or_group_bootstrap_95':np.quantile(estimate,[.025,.975]).tolist()}


def qasper_metrics(rows,features,forecasts):
    papers=[]
    for row in rows:
        key=row['id']; names=features[key]; windows=row['window_tokens']
        if len(names)!=2 or not set(windows)==set(names):
            raise ValueError('The same two supplied questions must be scored on both passages')
        raw=forecasts[key]
        if not np.isfinite(raw).all() or raw.min()<0 or raw.max()>1:
            raise ValueError('Uncalibrated or missing hazard')
        details={}
        for model, probs in zip(ARMS,raw,strict=True):
            softplus=-np.log1p(-np.minimum(probs,1-1e-15))
            ratings=[]; losses=[]; gains=[]
            for t,who in enumerate(names):
                a,b=windows[who]
                accumulated=softplus[:,a:b].sum(1)
                # Conditional first-hit probability increases strictly with integrated hazard.
                difference=accumulated[t]-accumulated[1-t]
                ratings.append(0.5 if abs(difference)<1e-10 else float(difference>0))
                gains.append(float((-np.expm1(-accumulated[t]))-(-np.expm1(-accumulated[1-t]))))
                losses.append(float(softplus[t,row['risk_start']:a].sum()-np.log(-np.expm1(-accumulated[t]))))
            details[model]={'both':ratings[0]*ratings[1],'average':float(np.mean(ratings)),
                            'nll':float(np.mean(losses)),'question_correct':ratings,'conditional_advantage':gains}
        first_by_position=min(names,key=lambda name:windows[name][0]); first_in_display=min(names,key=lambda name:features.display[key][name][0])
        papers.append({'paper':key,'first_supported_displayed_first':first_by_position==first_in_display,
                       'all_multiple':min(row['annotators'].values())>=2,'arms':details})
    summary={'papers':len(papers),'both_questions_multiple_annotators':sum(p['all_multiple'] for p in papers),'arms':{}}
    for arm in ARMS:
        summary['arms'][arm]={field:intervals([p['arms'][arm][field] for p in papers]) for field in ('both','average','nll')}
        summary['arms'][arm]['order']={str(order):{'papers':len(v:=[p for p in papers if p['first_supported_displayed_first']==order]),
                                                   'both':intervals([p['arms'][arm]['both'] for p in v])} for order in (True,False)}
    summary['paired_candidate_minus_baseline']={arm:{field:intervals([p['arms']['new_readout'][field]-p['arms'][arm][field] for p in papers])
                                                for field in ('both','average','nll')} for arm in ARMS if arm!='new_readout'}
    summary['primary_gate']=all(summary['paired_candidate_minus_baseline'][b]['both']['paired_or_group_bootstrap_95'][0]>0 for b in ('joint','lexical'))
    return summary,papers


class Keys(dict):
    def __init__(self, entries):
        self.display={r['id']:r['anchor_tokens'] for r in entries}
        super().__init__({k:sorted(v) for k,v in self.display.items()})


def historical(keys, forecasts):
    layout={k:v for k,v in read_json(PREVIOUS/'qwen.json')['rows'].items() if v['split']=='test' and v['kind']!='frick'}
    texts={r['id']:r['text'] for r in stream_rows(PREVIOUS/'stimuli.jsonl')}
    receipt=read_json(PREVIOUS/'receipt.json'); design=read_json(PREVIOUS/'design.json'); stops=receipt['structural_alignment']['control_shared_stops']
    tokenizer=Tokenizer.from_file(str(TOKENIZER)); enc={k:EncodedText.encode(texts[k],tokenizer) for k in layout}
    records={}; summaries={}
    for m,arm in enumerate(ARMS):
        predictions={k:{'hazard':{name:v.tolist() for name,v in zip(keys[k],forecasts[k][m],strict=True)}} for k in layout}
        support=evidence_rows([r for r in design['evidence'] if r['split']=='test'],enc,layout,predictions)
        single=controls_rows(design['controls'],enc,layout,predictions,stops)
        both=ab_rows(design['ab'],enc,layout,predictions,stops)
        records[arm]={'evidence':support,'controls':single,'ab':both}
        summaries[arm]={'controls':control_summary(single),'ab':ab_summary(both)}
    for arm in ARMS:
        support=records[arm]['evidence']
        summaries[arm]['evidence']={**evidence_summary(support,records['position']['evidence']),'by_display_order':ranking_by_display_order(support)}
    return summaries,records


def score(cache,output):
    seal=read_json(output/'forecast-seal.json'); lock=read_json(output/'input-seal.json')
    if seal['forecasts_sha256']!=digest(cache/'forecasts.npz') or seal['weights_sha256']!=digest(output/'readout.json'):
        raise ValueError('Forecasts or model weights differ from their pre-evaluation lock')
    if any(lock['files'][split]!=digest(DATA/f'identity-{split}.jsonl') for split in ('train','dev','test')):
        raise ValueError('Changed local human assignment maps')
    keys=Keys(list(stream_rows(DATA/'identity-stimuli.jsonl')))
    # Load once. Repeated decompression of per-paper arrays would alter only runtime.
    with np.load(cache/'forecasts.npz',allow_pickle=False) as archived:
        forecasts={k:archived[k] for k in archived.files}
    scores={}; records={}
    for split in ('test','dev','train'):
        rows=[r for r in stream_rows(DATA/f'identity-{split}.jsonl') if r['source']=='qasper']
        scores[split],records[split]=qasper_metrics(rows,keys,forecasts)
    scores['historical'],records['historical']=historical(keys,forecasts)
    write_json(output/'scores.json',{'forecast_sha256':seal['forecasts_sha256'],'scope':'Human evidence paragraphs for supplied questions; authored controls lack human response timing.','scores':scores})
    write_json(output/'paper-scores.json',records)
    print({s:(v['papers'],v['arms']['new_readout']['both']['mean'],v['primary_gate']) for s,v in scores.items() if s!='historical'})


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,default=DATA/'identity-fit');parser.add_argument('--output',type=Path,default=RESULT)
    score(**vars(parser.parse_args()))
