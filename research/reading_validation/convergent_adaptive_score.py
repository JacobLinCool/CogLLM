"""Compare proper held-out reading likelihoods and transport the unmodified English fits to Spanish."""
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
import csv
import numpy as np
from scipy.stats import spearmanr
from research.backbone_transfer.extract import digest,read_json,stream_rows,write_json
from research.reading_validation.convergent_adaptive import OUT,ROOT,ARMS,TARGETS,PLAN,assemble
from research.reading_validation.spanish_clue_data import ROOT as SPANISH


def interval(x,lo=.025,hi=.975):return np.nanquantile(x,[lo,hi],axis=0).T.tolist()


def run(folder):
    lock=read_json(OUT/'fit-seal.json');files=sorted(OUT.glob('*-fold-*.json'));hashes={digest(f) for f in files}
    if hashes!=set(lock['file_sha']) or lock['plan']!=digest(PLAN):raise ValueError('Incomplete or changed fold checkpoints')
    items={int(r['item_id']):r['sent_id'] for r in csv.DictReader((ROOT/'outcomes.csv').open())};groups=[];metrics={};diff={};P={};L={};raw=None;ns=None;order0=None
    for ctx in ('context','surface'):
        fs=[read_json(OUT/(ctx+f'-fold-{i}.json')) for i in range(5)];order=sum([f['held_items'] for f in fs],[]);y=np.vstack([f['raw_y'] for f in fs]);n=np.concatenate([f['trial_n'] for f in fs]);ids=np.array([items[i] for i in order]);unit,inv=np.unique(ids,return_inverse=True)
        if len(set(order))!=len(order) or (order0 is not None and order0!=order):raise ValueError('Inconsistent held-sentence forecasts')
        order0=order;groups=ids;raw=y;ns=n;yfrac=y.copy();yfrac[:,1]/=n
        for a in ARMS:
            name=ctx+'_'+a;p=np.vstack([f['predictions'][a] for f in fs]);l=np.vstack([f['held_likelihood'][a] for f in fs]);P[name]=p;L[name]=l;metrics[name]=dict(mean_nll=l.mean(0).tolist(),rho=[float(spearmanr(p[:,j],yfrac[:,j]).statistic) for j in range(3)],rmse=np.sqrt(((p-yfrac)**2).mean(0)).tolist())
    unit,inv=np.unique(groups,return_inverse=True);counts=np.bincount(inv);rng=np.random.default_rng(6941);boot=rng.multinomial(len(unit),np.ones(len(unit))/len(unit),size=10000);denom=boot@counts
    pairs=[('context_shared','context_diagonal'),('context_shared','context_fixed'),('context_diagonal','context_fixed'),('context_shared','surface_shared'),('surface_shared','surface_diagonal')]
    for a,b in pairs:
        gains=L[b]-L[a];gains=np.column_stack([gains,gains.mean(1)]);grouped=np.vstack([gains[inv==j].sum(0) for j in range(len(unit))]);rep=boot@grouped/denom[:,None];d=gains.mean(0);sd=rep.std(0,ddof=1);sim=np.quantile(np.max(abs((rep[:,:3]-d[:3])/np.maximum(sd[:3],1e-12)),axis=1),.95)
        diff[a+'__over__'+b]=dict(gain=d.tolist(),sentence_ci=interval(rep),simultaneous_3_endpoint_ci=np.c_[d[:3]-sim*sd[:3],d[:3]+sim*sd[:3]].tolist(),positive_sentence_fraction=(grouped>0).mean(0).tolist())
    (S,Y,weights,_,_,_,_,_),_=assemble(folder);mapping=[r for r in stream_rows(ROOT/'map.jsonl') if r['language']=='Spanish'];es={r['id']:r for r in stream_rows(SPANISH/'mapping.jsonl')};families=np.array([es[r['id']]['target_word'] for r in mapping for _ in range(20)]);source={};native=P;cost={}
    if len(families)!=len(S):raise ValueError('The source protocol included a different set of Spanish noninitial words')
    for ctx in ('context','surface'):
        doc=read_json(OUT/(ctx+'-fold-5.json'));prep=doc['projection'];parts=[np.ones((len(S),1)),(S[:,:6]-prep['mean'])/prep['sd']]
        if ctx=='context':parts.append(((S[:,6:]-prep['context_mean'])/prep['context_scale'])@np.array(prep['basis']))
        z=np.column_stack(parts)
        for a in ARMS:
            m=doc['models'][a];source[ctx+'_'+a]=(z@np.array(m['B']))[:,0]*m['scale'][0]+m['mu'][0];cost[ctx+'_'+a]={'covariance':m['prior'],'iterations':m['iterations'],'gradient':m['max_gradient'],'dispersion':np.exp(m['h'][1])}
    source['raw_Qwen_surprisal']=S[:,5];un,gi=np.unique(families,return_inverse=True);npz=OUT/'transfer-forecasts.npz';np.savez_compressed(npz,**source);from research.reading_validation.convergent_word_score import correlations,table_comparisons
    R=correlations(source,Y[:,:1],families,draws=1200);xfer=dict(rho={k:v[0][0] for k,v in R.items()},paired=table_comparisons(R,[('context_shared','context_diagonal'),('context_shared','surface_shared'),('context_shared','raw_Qwen_surprisal')]),source_word_contexts=len(S),distinct_target_families=len(un),external_forecast_checkpoint=digest(npz),human_word_presentations=int(weights.sum()))
    out=dict(fit_seal=digest(OUT/'fit-seal.json'),primary_units=len(raw),sentences=len(unit),cloze_decisions=int(ns.sum()),target_names=TARGETS,policy_names=list(ARMS),single_endpoint=metrics,paired=diff,full_fit_diagnostics=cost,english_to_spanish=xfer);write_json(OUT/'scores.json',out);print('Sibling-outcome coupling',diff['context_shared__over__context_diagonal']);print('English to Spanish EEG',xfer['rho']['context_shared'])


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();run(a.folder)
