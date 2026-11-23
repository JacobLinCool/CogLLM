"""Transport the news ballot observer to exact completed prefixes before associating with human essay marks."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from tokenizers import Tokenizer

from research.reading_validation.coherence_order_data import OUT as SOURCE, read, texts
from research.reading_validation.coherence_order_fit import source_targets
from research.reading_validation.local_forecast import digest
from research.reading_validation.summeval_ballots import forecast
from research.reading_validation.summeval_controls import lexical
from research.reading_validation.summeval_data import OUT
from research.reading_validation.summeval_forecast import open_old
from research.reading_validation.summeval_score import evaluate,paired
from research.reading_validation.transport_word_archive import article_boundaries

LEGACY=Path('research/results/reading-validation-transition-external')


def source_controls():
    old=SOURCE/'forecast/predictions.npz';historical=read(SOURCE/'forecast/seal.json')
    if digest(old)!=historical['predictions']:raise ValueError('Historical native baseline changed')
    with np.load(old,allow_pickle=False) as w:
        controls={k:dict(zip(w['english.keys'],w['english.'+col],strict=True)) for k,col in [('local_form','form'),('local_word_probability','surprisal'),('local_bilingual','adaptive')]}
    original=LEGACY/'essay-cross-topic/source-anchored-hazards.npz'
    if digest(original)!=read(LEGACY/'essay-cross-topic/runtime-replay.json')['source_result_sha256']:
        # The runtime replay pins the coordinating JSON report rather than the underlying array.
        meta=read(LEGACY/'essay-cross-topic/source-anchored-hazard-scores.json')
        if digest(original) not in json.dumps(meta):raise ValueError('Unknown prior human essay observer')
    ids=[r['id'] for r in read(LEGACY/'stimuli.json')['material'] if r['source']=='decor']
    with np.load(original,allow_pickle=False) as w:
        controls['local_essay_head']=dict(zip(ids,w['source_reliability'],strict=True));controls['local_essay_lexical']=dict(zip(ids,w['lexical_surprisal'],strict=True))
    return controls


def prefix_design(archive):
    rec=read(archive/'receipt.json')
    if rec['file']!=read(SOURCE/'native-features.json')['archive'] or digest(archive/'features.npz')!=rec['file']:raise ValueError('Unverified essay word-level feature archive')
    p,_=open_old();frozen=read(OUT/'ballot-forecast-seal.json');paramfile=OUT/'ballot-weights.npz'
    if digest(paramfile)!=frozen['weights_sha256']:raise ValueError('Changed news cohort weights')
    with np.load(paramfile,allow_pickle=False) as w:
        alpha=w['alpha'].item();method={k:{typ:w[k+'.'+typ] for typ in ('center','scale','slope')} for k in ('form','native')}
    path=(Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots'/rec['revision']/'tokenizer.json');tk=Tokenizer.from_file(str(path));docs=texts();geo=read(SOURCE/'geometry.json');counts=Counter();out={};coverage={}
    with np.load(archive/'features.npz',allow_pickle=False) as w:
        for doc,r in sorted(geo.items()):
            if r['source']!='decor':continue
            counts['all_human_target_rows']+=len(r['boundaries'])
            if doc not in rec['articles']:counts['excluded_essay']+=1;continue
            t=docs[doc];detail=rec['articles'][doc];raw,_=article_boundaries(w,doc,detail,t,include_initial=True);ss=[tuple(v) for v in detail['sentences']];feats={};enc=tk.encode(t,add_special_tokens=False);char=w[doc+'.char'];sur=w[doc+'.surprisal']
            if len(enc.ids)!=detail['tokens']:raise ValueError('Tokenizer length changed')
            for j,(s,e) in enumerate(ss):
                pref=t[:e];tok=tk.encode(pref,add_special_tokens=False);counts['prefix_checked']+=1
                if tok.ids!=enc.ids[:len(tok.ids)] or list(tok.offsets)!=list(enc.offsets[:len(tok.ids)]):counts['ineligible_bpe_prefix']+=1;continue
                if (s,e) not in raw:counts['prefix_missing_native_boundary']+=1;continue
                units=[r for r in (ss[:1] if j==0 else ss[1:j+1]) if r in raw];v=np.stack([raw[u] for u in units]);v=np.clip((v-p['english.center'])/p['english.scale'],-8,8)
                m=(char[:,1]<=e)&(char[:,4]!=0)
                if not m.any():continue
                if np.any(char[m,3]>=len(tok.ids)):raise ArithmeticError('Feature depends on a token outside the visible prefix')
                # Self reference is discarded by construction. The form channel uses original wording only.
                form=np.r_[lexical(pref,[pref])[0],float(np.mean(sur[m]))];native=np.r_[form,v.mean(0),logsumexp(v,axis=0)-np.log(len(v))];c=float(forecast(method['form'],form));z=float(forecast(method['native'],native));feats[j]={'form':c,'native':z,'mixture':(1-alpha)*c+alpha*z,'start':s,'end':e,'token_end':len(tok.ids)}
            mapped={}
            for q in r['boundaries']:
                key=q['id'];s,e=q['start'],q['end']
                if (s,e) not in ss:counts['multi_segment_or_geometry_mismatch']+=1;continue
                j=ss.index((s,e))
                if j not in feats or j-1 not in feats:counts['ineligible_preceding_or_current_prefix']+=1;continue
                current,prev=feats[j],feats[j-1];stub={'doc':doc,'essay':r['group'],'start_char':s,'end_char':e,'token_end_exclusive':current['token_end'],'token_prior_prefix_end':prev['token_end'],'sentence_index_one_based':j+1,'relative_position':j/max(1,len(ss)-1)}
                for n in ('form','native','mixture'):stub[n]=prev[n]-current[n]
                mapped[key]=stub
            coverage[doc]=len(mapped);out.update(mapped)
    lock=OUT/'prefix-forecasts.json';lock.write_text(json.dumps(out,separators=(',',':'),sort_keys=True)+'\n');receipt={'forecast_sha256':digest(lock),'code':digest(Path(__file__)),'protocol':digest(Path(__file__).with_name('news-local-boundary.md')),'tokenizer':digest(path),'underlying_l4':rec['file'],'weights':digest(paramfile),'native_weight':alpha,'counts':dict(counts),'eligible_stops':len(out),'eligible_essays':sum(n>0 for n in coverage.values()),'scoring_coordinates':'first token through the end of the completed sentence; human mark is a whole-sentence decision','new_information_from_decor_used_to_estimate_parameters':False}
    (OUT/'prefix-forecast-seal.json').write_text(json.dumps(receipt,indent=2)+'\n');print('prefix coverage',receipt['counts'],len(out))


def score():
    seal=read(OUT/'prefix-forecast-seal.json')
    if digest(OUT/'prefix-forecasts.json')!=seal['forecast_sha256'] or digest(Path(__file__))!=seal['code']:raise ValueError('The retrospective observer changed')
    q=read(OUT/'prefix-forecasts.json');controls=source_controls();actual=source_targets(True)['english']
    keys=sorted(set(q)&set(actual)&set.intersection(*(set(v) for v in controls.values())))
    y=np.array([actual[k][0]/actual[k][1] for k in keys]);n=np.array([actual[k][1] for k in keys])
    if not np.all(n==1) or set(y)!={0,1}:raise ValueError('Different DECOR rating unit')
    g=np.array([q[k]['essay'] for k in keys]);groups=np.unique(g);methods={name:np.array([q[k][name] for k in keys]) for name in ('mixture','native','form','relative_position')};methods.update({name:np.array([d[k] for k in keys]) for name,d in controls.items()});methods['chance']=np.zeros_like(y)
    out={'evaluated_rows':len(keys),'essays':len(groups),'human_marked':int(y.sum()),'removed_without_coordinate_or_comparator':len(q)-len(keys),'scores':{k:evaluate(y,v,groups,g)[0] for k,v in methods.items()},'primary':{c:paired(y,methods['mixture'],methods[c],groups,g,2) for c in ('chance','form')},'secondary':{c:paired(y,methods['mixture'],methods[c],groups,g,1) for c in ('relative_position','local_form','local_word_probability','local_bilingual','local_essay_head','local_essay_lexical')},'receipt':digest(OUT/'prefix-forecast-seal.json')}
    (OUT/'prefix-scores.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print('prefix primary',out['primary'])

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--archive',type=Path);parser.add_argument('--score',action='store_true');args=parser.parse_args()
    if bool(args.archive)==args.score:parser.error('Select archive or scoring')
    if args.archive:prefix_design(args.archive)
    else:score()
