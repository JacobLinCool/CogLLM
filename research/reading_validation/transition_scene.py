"""Forecast human-marked abrupt scene transitions from five visible new words or a complete segment."""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.reading_validation.transition_external_fit import binomial,predict
from research.reading_validation.transition_external_score import evaluate_one,grouped_distribution,within_document_order
from research.reading_validation.transition_calibration import direct_rank
from research.reading_validation.transition_snac_fit import inputs
from research.reading_validation.transition_snac_prepare import RESULTS,source
from research.reading_validation.transition_score import dump,sha,with_interval

PROTOCOL=Path('research/reading_validation/transition-scene-protocol.md')
MODELS=('surprisal','local','relations')


def labels(raw,doc,seg):
    anns=raw[doc][str(seg)]['errors']
    return max([e['votes'] for e in anns if e['error_type']=='SceneE'], default=0)


def design(matrix,segments,i,visible):
    prev=matrix[segments==i-1];new=matrix[segments==i][:visible]
    if not len(prev) or not len(new):raise ValueError('Every retained reading segment needs a word')
    observed=np.concatenate([new[:,:17].mean(axis=0),new[:,:17].max(axis=0),[prev[:,16].mean(),np.log1p(len(new))]])
    old=prev[:,17:113].mean(axis=0);cur=new[:,17:113].mean(axis=0);den=np.linalg.norm(old)*np.linalg.norm(cur)
    return np.concatenate([observed,cur,old,old*cur,[old@cur/den if den>0 else 0.],new[:,113:].mean(axis=0)])


def fit(directory):
    docs,meta=inputs();raw=source();features=directory/'word-features.npz';rec=json.loads((directory/'receipt.json').read_text())
    if sha(features)!=rec['feature_sha256'] or rec['forwards']!=len(docs) or rec['input_sha256']!=meta['packet_sha256']:
        raise ValueError('Expected the released one-pass SNaC word features')
    matrices=np.load(features,allow_pickle=False);weights={};forecasts=[]
    for exposure,limit in [('early',5),('complete',None)]:
        rows=[];x=[]
        for m in meta['documents']:
            doc=m['doc'];ws=np.asarray(docs[doc]['words']);matrix=matrices[doc].astype(np.float64)
            if matrix.shape!=(len(ws),128):raise ValueError('Wrong SNaC feature width')
            for seg in range(1,len(m['segments'])):
                rows.append({'doc':doc,'group':m['group'],'split':m['split'],'segment':seg,'exposure':exposure})
                x.append(design(matrix,ws[:,4],seg,limit))
        f=pd.DataFrame(rows);x=np.array(x);train=f.split.eq('train').to_numpy();dev=f.split.eq('dev').to_numpy()
        y=np.array([int(labels(raw,z['doc'],z['segment'])>0) if z['split'] in ('train','dev') else -1 for z in rows]);weights[exposure]={}
        for arm,n in zip(MODELS,[36,132,x.shape[1]],strict=True):
            model=binomial(x[train,:n],y[train],np.ones(train.sum(),dtype=int));pred=predict(x[:,:n],model)
            cutoff=float(np.quantile(pred[dev&(y==0)],.9,method='higher'));weights[exposure][arm]={'model':model,'threshold':cutoff}
            f[arm]=pred
        forecasts.append(f[f.split.eq('test')])
    dump(RESULTS/'scene-weights.json',weights);pred=RESULTS/'scene-predictions.csv';pd.concat(forecasts).to_csv(pred,index=False,float_format='%.15g')
    dump(RESULTS/'scene-lock.json',{'forecast_sha256':sha(pred),'weights_sha256':sha(RESULTS/'scene-weights.json'),
        'protocol_sha256':sha(PROTOCOL),'features_receipt_sha256':sha(directory/'receipt.json')})


def score():
    lock=json.loads((RESULTS/'scene-lock.json').read_text());pred=RESULTS/'scene-predictions.csv';model=RESULTS/'scene-weights.json'
    if sha(pred)!=lock['forecast_sha256'] or sha(model)!=lock['weights_sha256'] or sha(PROTOCOL)!=lock['protocol_sha256']:
        raise ValueError('Scene forecasts or models no longer sealed')
    f=pd.read_csv(pred);weights=json.loads(model.read_text());raw=source();results={}
    for exposure in ('early','complete'):
        obs=f[f.exposure.eq(exposure)];strength=np.array([labels(raw,row.doc,row.segment) for row in obs.itertuples()]);y=(strength>0).astype(int)
        n=np.ones(len(y),dtype=int);groups,ids,draws=grouped_distribution(obs.group.to_numpy());stats={};boot={}
        for arm in MODELS:
            p=obs[arm].to_numpy();m,b=evaluate_one(y,n,p,weights[exposure][arm]['threshold'],ids,draws)
            m['within_movie_concordance']=within_document_order(y,n,p,ids,draws)
            ap,auc=direct_rank((strength>=2).astype(int),n,p);m['two_vote_ap']=ap;m['two_vote_auroc']=auc
            stats[arm]=m;boot[arm]=b
        differences={}
        for baseline in ('surprisal','local'):
            differences[baseline]=with_interval(stats['relations']['ap']['estimate']-stats[baseline]['ap']['estimate'],boot['relations']['ap']-boot[baseline]['ap'])
        results[exposure]={'movies':len(groups),'regions':len(y),'regions_marked':int(y.sum()),'regions_with_two_votes':int((strength>=2).sum()),
            'models':stats,'relations_minus_control_ap':differences,'gate':all(r['family_bootstrap_95_interval'][0]>0 for r in differences.values())}
    dump(RESULTS/'scene-scores.json',results);return results


if __name__=='__main__':
    parser=ArgumentParser();parser.add_argument('phase',choices=['fit','score']);parser.add_argument('--features',type=Path);args=parser.parse_args()
    if args.phase=='fit':fit(args.features)
    else:
        result=score()
        for e,v in result.items():print(e,v['regions'],v['regions_marked'],{a:k['ap'] for a,k in v['models'].items()})
