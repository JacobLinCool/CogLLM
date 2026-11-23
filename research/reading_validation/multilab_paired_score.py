"""Pair the predeclared frozen word observers under the same experimental resamples."""
from pathlib import Path
import numpy as np
from research.backbone_transfer.extract import read_json, write_json, digest, stream_rows
from research.reading_validation.convergent_word_score import corr
from research.reading_validation.multilab_materials import OUT, ROOT
from research.reading_validation.multilab_score import human, unpack
from research.reading_validation.physiology_portability_data import OUT as ARCH, CANDIDATES


def run():
    native,cloze,Y,_,sl,items,coverage,_=unpack()
    records=list(stream_rows(ROOT/'conditions.jsonl'))
    table={k:v[:,1]-v[:,0] for k,v in native.items()}
    table["held_human_cloze"]=cloze[:,0]-cloze[:,1]
    for c in CANDIDATES:
        key=read_json(ARCH/c/'forecast-lock.json')
        if key['nine_lab_order']!=[r['id'] for r in records]:raise ValueError("Mismatched external order")
        p=ARCH/c/'forecasts.npz'
        if digest(p)!=key['forecast']:raise ValueError("Unsealed architecture predictions")
        with np.load(p) as a:
            for model in ('raw','surface','context'):
                m=np.full_like(cloze,np.nan)
                pred=a['multi_'+model]
                if pred.ndim==3:pred=pred[:,:,0]
                for z,row in zip(pred,records,strict=True):m[items.index(row['item']),int(row['condition']=='unexpected')]=z
                if not np.isfinite(m).all():raise ValueError("Missing material")
                table[c+'_'+model]=m[:,1]-m[:,0]
    N=len(items);rng=np.random.default_rng(91402);K=int(sl.max()+1)
    I=rng.multinomial(N,np.ones(N)/N,size=3000);sites=rng.multinomial(K,np.ones(K)/K,size=len(I))
    W=np.zeros((len(I),len(sl)))
    for k in range(K):
        g=np.flatnonzero(sl==k);W[:,g]=rng.multinomial(len(g),np.ones(len(g))/len(g),size=len(I))*sites[:,k,None]
    full=human(Y,np.ones((1,len(sl))))[0];d=full[:,1]-full[:,0]
    b=human(Y,W);B=b[:,:,1]-b[:,:,0];keep=np.isfinite(B).all(axis=(1,2));B=B[keep][::2];I=I[keep][::2]
    metrics={}
    pairs=[('english_context_EEG','english_surface_EEG'),('english_context_EEG','raw_surprisal'),('english_context_EEG','held_human_cloze')]
    for c in CANDIDATES:pairs.extend([(c+'_context',c+'_surface'),(c+'_context',c+'_raw')])
    pairs.extend([('qwen_context','lfm3_context'),('qwen_context','gemma4_context'),('lfm3_context','gemma4_context')])
    for n in set(v for pair in pairs for v in pair):
        x=table[n]
        base=[corr(x[:,j],d[:,j]) for j in range(2)]
        alternatives=[[corr(x[:,j],y[:,j],w) for j in range(2)] for y,w in zip(B,I,strict=True)]
        metrics[n]=np.vstack((base,alternatives))
    labels={}; simultaneous=[]
    for a,c in pairs:
        z=metrics[a]-metrics[c]
        key=a+'__over__'+c
        labels[key]={"article_difference":float(z[0,0]),"noun_difference":float(z[0,1]),
                      "paired_crossed_interval":np.quantile(z[1:],[.025,.975],axis=0).T.tolist()}
        if a.split('_')[0] in CANDIDATES and c==a.split('_')[0]+'_surface':
            simultaneous.append((key,z))
    stacked=np.stack([v[:,1] for _,v in simultaneous],axis=1)
    sd=stacked[1:].std(0);critical=np.quantile(np.max(abs((stacked[1:]-stacked[0])/sd),axis=1),.95)
    for i,(name,_) in enumerate(simultaneous):
        labels[name]['three_architecture_noun_interval']=[float(stacked[0,i]-critical*sd[i]),float(stacked[0,i]+critical*sd[i])]
    write_json(OUT/'paired-readout-differences.json',dict(
        interpretation="Secondary uncertainty estimate for pre-existing comparisons; computed after viewing the vector of marginal correlations. No model was fitted or selected.",
        genuine_EEG_trials=coverage,materials=N,independent_resamples=len(B),results=labels))
    print(labels['english_context_EEG__over__english_surface_EEG'],labels['qwen_context__over__qwen_surface'])


if __name__=='__main__':run()
