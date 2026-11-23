"""Score locked causal attribution rankings on independent human choice items."""

import argparse
from bisect import bisect_right
from pathlib import Path

import numpy as np
from scipy.special import expit
from tokenizers import Tokenizer

from research.backbone_transfer.extract import digest, read_json, stream_rows, write_json
from research.reading_validation.identity_causal_fit import IGNORE, OUT, WORD
from research.reading_validation.identity_prepare import DATA, PREVIOUS, TOKENIZER
from research.token_readout.transfer_data import EncodedText
from research.token_readout.transfer_score import ab_rows, ab_summary, control_summary, controls_rows


def confidence(values):
    values=np.asarray(values,dtype=float)
    if len(values)==0:
        return None
    rng=np.random.default_rng(91471)
    means=np.concatenate([values[rng.integers(len(values),size=(100,len(values)))].mean(1) for _ in range(100)])
    return {'mean':float(values.mean()),'paired_item_bootstrap_95':np.quantile(means,[.025,.975]).tolist()}


def item(row,predictions,texts):
    views=row['views']
    if len(views)!=2 or {v['kind'] for v in views}!={'choice-0','choice-1'}:
        raise ValueError('Held-out items require both orders')
    view=next(v for v in views if v['kind']=='choice-0')
    statements=texts[view['id']].splitlines()
    if len(statements)!=3:
        raise ValueError('Unmatched rendered statements')
    words=[set(WORD.findall(s.casefold()))-IGNORE for s in statements]
    arms={}
    for arm in predictions[view['id']]:
        draws=[];scores=[];ll=[];correct_position=[];ties=0
        for v in views:
            values=predictions[v['id']][arm]
            i=v['targets'].index(1)
            diff=values[i]-values[1-i]
            ties+=int(abs(diff)<=1e-9)
            draws.append(.5 if abs(diff)<=1e-9 else float(diff>0))
            scores.append(float(expit(diff)));ll.append(float(np.logaddexp(0.,-diff)));correct_position.append(i)
        arms[arm]={'both':float(np.prod(draws)),'accuracy':float(np.mean(draws)),'ties':ties,
                   'target_p':float(np.mean(scores)),'nll':float(np.mean(ll)),
                   'target_first':draws[correct_position.index(0)],'target_second':draws[correct_position.index(1)]}
    return {'id':row['id'],'ask_for':row['ask_for'],'no_overlap':not (words[2]&(words[0]|words[1])),
            'arms':arms,'probability_gain_after_premise':arms['semantic']['target_p']-arms['prior']['target_p']}


def summary(rows):
    arms=list(rows[0]['arms'])
    by_arm={m:{s:confidence([r['arms'][m][s] for r in rows]) for s in ('both','accuracy','target_first','target_second','nll')} for m in arms}
    delta={m:{s:confidence([r['arms']['semantic'][s]-r['arms'][m][s] for r in rows]) for s in ('both','accuracy','nll')} for m in arms if m!='semantic'}
    subsets={}
    for k,f in [('no_content_word_overlap',lambda r:r['no_overlap']),('cause',lambda r:r['ask_for']=='cause'),('effect',lambda r:r['ask_for']=='effect')]:
        subset=[r for r in rows if f(r)]
        subsets[k]={'n':len(subset),'both':{m:confidence([r['arms'][m]['both'] for r in subset]) for m in arms}}
    return {'n':len(rows),'arms':by_arm,'paired_semantic_differences':delta,'subsets':subsets,
            'superiority_vs_lexical':delta['lexical']['both']['paired_item_bootstrap_95'][0]>0,
            'conditional_target_probability_increase_after_premise':confidence([r['probability_gain_after_premise'] for r in rows]),
            'ties_by_arm':{m:sum(r['arms'][m]['ties'] for r in rows) for m in arms}}


def free_pointer(authored,encoded,layout,design):
    rows=[]
    for trial in design['ab']:
        for target in ('A','B'):
            v=trial['variants'][target];key=v['id'];s=authored[key];data=encoded[key]
            a,b=v['cue_span'];lo=bisect_right(data.ends,a);hi=bisect_right(data.ends,b)
            anchor_start=layout[key]['anchor_tokens'][target][0];predecessors={}
            for j,(start,end) in enumerate(s['candidate_sentences']):
                if end>lo:continue
                hazard=np.asarray(s['all_candidates']['hazard'][str(j)])
                # Change in bounded cumulative state between start and end of the decisive cue.
                before=float(np.prod(1-hazard[:lo]));after=float(np.prod(1-hazard[:hi]))
                predecessors[j]=before-after
            expected=next(j for j,(a,b) in enumerate(s['candidate_sentences']) if a<=anchor_start<b)
            if expected not in predecessors:raise ValueError('The true source was not an earlier completed sentence')
            top=max(predecessors.values());tied=[j for j,delta in predecessors.items() if abs(delta-top)<1e-9]
            # Abstain at empty signal even if every candidate is equal.
            selected=float(expected in tied)/len(tied) if top>1e-9 else 0.
            rows.append({'group':trial['group_id'],'branch':target,'n_candidates':len(predecessors),'expected':expected,
                         'tied_max':tied if top>1e-9 else [],'change':predecessors,'credit':selected})
    groups={k:[r for r in rows if r['group']==k] for k in sorted({r['group'] for r in rows})}
    return {'branches':len(rows),'abstained':sum(not r['tied_max'] for r in rows),
            'branch_accuracy':confidence([np.mean([x['credit'] for x in v]) for v in groups.values()]),
            'both':confidence([np.prod([x['credit'] for x in v]) for v in groups.values()]),'rows':rows}


def historical(authored):
    layout={k:v for k,v in read_json(PREVIOUS/'qwen.json')['rows'].items() if v['kind'] in ('ab','controls')}
    texts={r['id']:r['text'] for r in stream_rows(PREVIOUS/'stimuli.jsonl') if r['id'] in layout}
    tokenizer=Tokenizer.from_file(str(TOKENIZER));encoded={k:EncodedText.encode(t,tokenizer) for k,t in texts.items()}
    design=read_json(PREVIOUS/'design.json')
    stops=read_json(PREVIOUS/'receipt.json')['structural_alignment']['control_shared_stops']
    if set(authored)!=set(layout):raise ValueError('Historical cue cases incomplete')
    result={}
    for mode in ('automatic','supplied'):
        predictions={k:v[mode] for k,v in authored.items()}
        single=controls_rows(design['controls'],encoded,layout,predictions,stops);double=ab_rows(design['ab'],encoded,layout,predictions,stops)
        branches=[r[c] for r in double for c in ('A','B')]
        result[mode]={'early_late':control_summary(single),'two_issues':ab_summary(double),
                      'two_issue_criteria':{'correct_first_divergence':sum(r['timing']=='inside' for r in branches),
                            'positive_selectivity':sum(r['selectivity']>0 for r in branches),
                            'min_sham_adjusted_increase':sum(r['target_gain']-r['sham_target_gain']>=.05 for r in branches),
                            'min_target_minus_rival':sum(r['target_gain']-r['other_gain']>=.05 for r in branches),
                            'no_rival_false_resolution':sum(r['other_gain']-r['sham_other_gain']<.05 for r in branches)},
                      'controls_details':single,'ab_details':double}
    result['full_automatic_antecedent_inventory']=free_pointer(authored,encoded,layout,design)
    return result


def score():
    source=read_json(OUT/'input-seal.json');forecasts=read_json(OUT/'forecast-seal.json')
    for name,sha in [('choice-forecasts','semantic_forecast_sha256'),('authored-forecasts','authored_forecast_sha256'),('weights','weights_sha256')]:
        if digest(OUT/(name+'.json'))!=forecasts[sha]:raise ValueError('Forecast or fitted coefficients changed')
    predictions=read_json(OUT/'choice-forecasts.json');texts={r['id']:r['text'] for r in stream_rows(DATA/'causal-stimuli.jsonl')}
    results={};details={}
    for split in ('copa-test','copa-dev','ecare-dev'):
        path=DATA/f'causal-{split}.jsonl'
        if digest(path)!=source['source_grades_sha256'][split]:raise ValueError('Outcome source no longer matches registration')
        details[split]=[item(row,predictions,texts) for row in stream_rows(path)]
        results[split]=summary(details[split]);print(split,len(details[split]),results[split]['arms']['semantic']['both'],flush=True)
    hist=historical(read_json(OUT/'authored-forecasts.json'))
    write_json(OUT/'scores.json',results);write_json(OUT/'item-scores.json',details);write_json(OUT/'historical-scores.json',hist)


if __name__=='__main__':score()
