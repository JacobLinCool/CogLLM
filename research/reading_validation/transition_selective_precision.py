"""Calibrate a separate selective editorial head on 30 source-only titles."""
from argparse import ArgumentParser
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from research.reading_validation.transition_external_score import grouped_distribution
from research.reading_validation.transition_full_movie_bounds import threshold_bounds
from research.reading_validation.transition_partial_loss import OUT,design,material,solve
from research.reading_validation.transition_score import dump,sha
from research.reading_validation.transition_snac_fit import inputs


PROTOCOL=Path('research/reading_validation/transition-selective-precision-protocol.md')
GRID=np.arange(10,100,5)/100


def estimates(spans,score,allowed,cutoff):
    counts=defaultdict(lambda:np.zeros(4,dtype=int));flags=(score>cutoff)&allowed
    for s in spans:
        f=flags[s.ids];overlap=(s.masks&f).sum(axis=1)
        counts[s.group]+=[overlap.min(),overlap.max(),f.sum(),len(f)]
    groups=sorted(counts);m=np.array([counts[g] for g in groups]);return m,flags


def precision(total,which):
    arr=np.asarray(total,dtype=float)
    return np.divide(arr[...,which],arr[...,2],out=np.zeros_like(arr[...,which]),where=arr[...,2]>0)


def run(directory):
    X,segments,old_mu,old_sd,receipt,excluded=material(directory)
    if excluded:raise ValueError("Unexpected unmapped character annotations")
    old_train=sorted({s.group for s in segments if s.split=='train'},
                     key=lambda g:sha256(('editor-selective-v1/'+g).encode()).hexdigest())
    titles=set(old_train[:25]);fit=[s for s in segments if s.group in titles]
    calibration=[s for s in segments if s.split in ('train','dev') and s.group not in titles]
    test=[s for s in segments if s.split=='test'];dest=OUT/'selective';dest.mkdir(parents=True,exist_ok=True)
    if len({s.group for s in calibration})!=30 or len(fit)!=len([s for s in fit if s.split=='train']):raise ValueError('Wrong source folds')
    raw=X[:,1:]*old_sd+old_mu
    source=np.concatenate([s.ids for s in fit]);mu=raw[source].mean(0);sd=np.maximum(raw[source].std(0),1e-10)
    X[:,1:]=(raw-mu)/sd;del raw
    model=design(X,fit,'set_marginal');w,record=solve(model,np.zeros(X.shape[1]));p=np.exp(-np.logaddexp(0,-(X@w)))
    docs,meta=inputs();allowed=[]
    for m in meta['documents']:
        text=docs[m['doc']]['text'];seen=set()
        for a,b,*_ in docs[m['doc']]['words']:
            word=text[a:b];k=word.casefold();allowed.append(word[0].isupper() and k not in seen);seen.add(k)
    allowed=np.array(allowed)
    if len(allowed)!=len(p):raise ValueError('First mention gate and frozen word scores do not align')
    eligible=[];details=[];_,_,draw=grouped_distribution(sorted({s.group for s in calibration}))
    for cut in GRID:
        mat,_=estimates(calibration,p,allowed,cut);obs=mat.sum(0)
        valid=obs[2]>=20 and (mat[:,2]>0).sum()>=5
        item={'threshold':float(cut),'minimum_true_positives':int(obs[0]),'maximum_true_positives':int(obs[1]),'alerts':int(obs[2]),
              'titles_with_alerts':int((mat[:,2]>0).sum()),'identified_minimum_precision':float(precision(obs,0)),'eligible':bool(valid)}
        details.append(item)
        if valid:eligible.append((item,precision(draw@mat,0)))
    adjustment=float(np.quantile(np.stack([m['identified_minimum_precision']-b for m,b in eligible]).max(0),.95)) if eligible else None
    for m,b in eligible:m['simultaneous_lower_bound']=max(0.,m['identified_minimum_precision']-adjustment)
    accepted=[m for m,_ in eligible if m['simultaneous_lower_bound']>=.70]
    choice=max(accepted,key=lambda m:(m['minimum_true_positives'],m['simultaneous_lower_bound'],m['threshold'])) if accepted else None
    pack=dest/'source-fit.npz';np.savez_compressed(pack,mean=mu,sd=sd,weights=w)
    lock={'protocol_sha256':sha(PROTOCOL),'source_fit_sha256':sha(pack),'features_sha256':receipt['feature_sha256'],
          'source_train_titles':sorted(titles),'source_calibration_titles':sorted({s.group for s in calibration}),
          'source_fit':record,'candidate_thresholds':details,'simultaneous_adjustment':adjustment,'choice':choice}
    dump(dest/'source-lock.json',lock)
    if choice:
        cut=choice['threshold'];mat,mask=estimates(test,p,allowed,cut);tot=mat.sum(0);_,_,draw=grouped_distribution(sorted({s.group for s in test}))
        gs=np.concatenate([s.ids for s in test]);local={j:i for i,j in enumerate(gs)}
        materialset=[(np.asarray([local[j] for j in s.ids]),s.masks) for s in test]
        extremal=threshold_bounds(materialset,mask[gs],False)
        evaluation={'source_lock_sha256':sha(dest/'source-lock.json'),'word_positions':int(tot[3]),'alerts':int(tot[2]),
                    'movies_with_alerts':int((mat[:,2]>0).sum()),'flags_per_1000_words':float(1000*tot[2]/tot[3]),
                    'true_alerts_range':[int(tot[0]),int(tot[1])],'sharp_operating_ranges':extremal,
                    'movie_bootstrap_precision_envelope':[float(np.quantile(precision(draw@mat,0),.025)),float(np.quantile(precision(draw@mat,1),.975))]}
    else:evaluation={'source_lock_sha256':sha(dest/'source-lock.json'),'selected':False}
    dump(dest/'heldout.json',evaluation)
    print(choice,'max source lower bound',max((m['simultaneous_lower_bound'] for m,_ in eligible),default=None),evaluation)


if __name__=='__main__':
    cli=ArgumentParser(description=__doc__);cli.add_argument('--features',type=Path,required=True)
    run(cli.parse_args().features)
