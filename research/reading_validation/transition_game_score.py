"""Evaluate frozen, source-calibrated forecasts for an unobserved pair of human readers."""
from argparse import ArgumentParser
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from research.reading_validation.transition_direct_data import OUT as DIRECT,ROOT
from research.reading_validation.transition_direct_score import concordance
from research.reading_validation.transition_external_score import grouped_distribution
from research.reading_validation.transition_game_prepare import OUT,PROTOCOL
from research.reading_validation.transition_score import dump,sha,with_interval


def pooled_values(y,p,idx,draws):
    err=y*(1-p)**2+(1-y)*p**2
    total=np.bincount(idx);denom=draws@total
    def boot(v):return (draws@np.bincount(idx,weights=v))/denom
    return ({'expected_pair_brier':with_interval(err.mean(),boot(err)),
             'mean_risk':with_interval(p.mean(),boot(p)),
             'observed_pair_disagreement':with_interval(y.mean(),boot(y)),
             'calibration_bias':with_interval((p-y).mean(),boot(p-y))},boot(err))


def ranks(rows,methods,match_form):
    out={};bootstrap={};support=None;draws=None;ids=None
    for arm in methods:
        points,counts=concordance(rows,arm,'d',match_form)
        if ids is None:
            ids=sorted(points);support={'documents':len(ids),'unequal_disagreement_pairs':sum(counts.values())}
            if len(ids)<2:return {**support,'models':{}}
            _,_,draws=grouped_distribution(ids)
        if sorted(points)!=ids:raise ValueError('A model uses a different set of pronouns')
        x=np.array([points[k] for k in ids]);boot=(draws@x)/len(ids)
        out[arm]=with_interval(x.mean(),boot);bootstrap[arm]=boot
    pairs=(('direct','surface'),('shared','surface'),('shared','direct'),('direct','constant'),('shared','constant'))
    if 'book_only' in methods:pairs+= (('direct','book_only'),('shared','book_only'))
    diffs={a+'_minus_'+b:with_interval(out[a]['estimate']-out[b]['estimate'],bootstrap[a]-bootstrap[b]) for a,b in pairs}
    return {**support,'models':out,'paired_differences':diffs}


def evaluate(rows,registry,fits):
    if not rows:return {'mentions':0,'documents':0}
    y=np.array([r['d'] for r in rows]);groups=[r['doc'] for r in rows]
    ids,idx,draws=grouped_distribution(groups);perf={};brier={};bins={}
    for arm in registry:
        p=np.array([r[arm] for r in rows])
        if not np.isfinite(p).all() or (p<0).any() or (p>1).any():raise ValueError('Invalid external probability')
        if arm=='book_only':continue
        perf[arm],brier[arm]=pooled_values(y,p,idx,draws)
        edges=fits[arm]['quintile_edges'] if arm!='constant' else []
        which=np.searchsorted(edges,p,side='right')
        bins[arm]=[]
        for cell in range(len(edges)+1):
            m=which==cell
            bins[arm].append({'n':int(m.sum()),'predicted':float(p[m].mean()) if m.any() else None,
                              'observed':float(y[m].mean()) if m.any() else None})
        perf[arm]['weighted_absolute_calibration_error']=float(sum(r['n']*abs(r['predicted']-r['observed']) for r in bins[arm] if r['n'])/len(y))
    diff={}
    for a,b in [('direct','surface'),('shared','surface'),('shared','direct'),('direct','constant'),('shared','constant')]:
        diff[a+'_minus_'+b]=with_interval(perf[a]['expected_pair_brier']['estimate']-perf[b]['expected_pair_brier']['estimate'],
                                          brier[a]-brier[b])
    return {'mentions':len(rows),'documents':len(ids),'observed_disputes':int((y>0).sum()),'estimates':perf,
            'paired_brier_differences':diff,'source_quintile_reliability':bins,
            'within_document':ranks(rows,registry,False),'within_document_and_form':ranks(rows,registry,True)}


def score(directory,zero_shot):
    seal=json.loads((OUT/'input-seal.json').read_text())
    cfg=json.loads((directory/'receipt.json').read_text())
    packet=ROOT/'game-native-stimuli.jsonl';predict=directory/'game-forecasts.npz'
    if (sha(packet)!=seal['input_sha256'] or cfg['input_sha256']!=seal['input_sha256'] or sha(PROTOCOL)!=cfg['protocol_sha256']
            or sha(predict)!=cfg['forecasts_sha256'] or sha(directory/'readouts.npz')!=cfg['weights_sha256']):
        raise ValueError('Gold forecast or training source changed')
    old=json.loads((zero_shot/'receipt.json').read_text());old_predictions=zero_shot/'word-probabilities.npz'
    if old['candidate']!='qwen' or sha(old_predictions)!=old['predictions_sha256']:
        raise ValueError('The book-only transfer forecast is not the registered independent run')
    bound=DIRECT/'external-reader-outcomes.json';exclusive=OUT/'new-reader-outcomes.json'
    if sha(bound)!=seal['registered_gold_sha256'] or sha(exclusive)!=seal['supplement_gold_sha256']:
        raise ValueError('External votes changed')
    game_material=json.loads(bound.read_text())['rows']
    gold={(r['doc'],r['word']):r for r in game_material if r['antecedent_disagreement'] is not None}
    unseen={(r['doc'],r['word']):r for r in json.loads(exclusive.read_text())['rows']}
    data=list(map(json.loads,packet.read_text().splitlines()));probs=np.load(predict);prev=np.load(old_predictions)
    arms=cfg['arms'];available=arms+['constant','book_only'];observations=defaultdict(list)
    for doc in data:
        d=doc['id']
        if doc['fold'] not in ('test','dev'):continue
        table=probs[d]
        if table.shape!=(len(doc['save']),len(arms)):raise ValueError('Single-word forecast geometry changed')
        for k,i in enumerate(doc['save']):
            row={a:float(table[k,j]) for j,a in enumerate(arms)}
            a,b=doc['words'][i][:2];form=doc['text'][a:b].casefold()
            row.update(doc=d,form=form,constant=cfg['fits']['direct']['observed_development_rate'])
            if doc['fold']=='dev':
                target=doc['labels'][k][-1]
                if target>=0:observations['development'].append({**row,'d':target})
                continue
            if (d,i) not in gold:continue
            target=gold[d,i]
            if form!=target['form']:raise ValueError('Original tagged pronoun form changed')
            row['book_only']=float(prev[d][i,old['arms'].index('shared'),old['tasks'].index('CorefE')])
            item={**row,'d':target['antecedent_disagreement']};observations['gold'].append(item)
            observations['gold_'+target['genre']].append(item)
            if (d,i) in unseen:observations['new_reader_gold'].append({**row,'d':unseen[d,i]['d']})
    if len(observations['gold'])!=len(gold):raise ValueError('Some gold pronoun was not scored')
    out={}
    for name,items in observations.items():
        methods=available if name!='development' else arms+['constant']
        if len({q['doc'] for q in items})<2:
            out[name]={'mentions':len(items),'documents':len({q['doc'] for q in items})};continue
        out[name]=evaluate(items,methods,cfg['fits'])
    if 'new_reader_gold' not in out:out['new_reader_gold']={'mentions':0,'documents':0}
    for name in ('receipt.json','readouts.npz','game-forecasts.npz'):
        des=OUT/name;des.write_bytes((directory/name).read_bytes())
        if sha(des)!=sha(directory/name):raise ValueError('Artifact copy changed')
    dump(OUT/'scores.json',{'external_human_sha256':sha(bound),'primary_protocol_sha256':sha(PROTOCOL),'evaluations':out})
    print('held-out gold',out['gold']['mentions'],out['gold']['within_document'],out['gold']['paired_brier_differences'],flush=True)


if __name__=='__main__':
    cli=ArgumentParser(description=__doc__);cli.add_argument('--features',required=True,type=Path)
    cli.add_argument('--book-only',required=True,type=Path);args=cli.parse_args();score(args.features,args.book_only)
