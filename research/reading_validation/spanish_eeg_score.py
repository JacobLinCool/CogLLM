"""Read matched Spanish word epochs and compare target-specific semantic expectations with N400."""
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from research.backbone_transfer.extract import digest,read_json,stream_rows,write_json
from research.reading_validation.spanish_clue_data import ROOT,OUT
from research.reading_validation.spanish_clue_score import interval,sign_p,SEED


def rho(x,y):
    if len(x)<4 or len(set(x))<2 or len(set(y))<2:return np.nan
    x=rankdata(x);y=rankdata(y);x-=x.mean();y-=y.mean()
    return x@y/(np.linalg.norm(x)*np.linalg.norm(y))


def ci(x):
    v=np.asarray(x,dtype=float);v=v[np.isfinite(v)]
    return np.quantile(v,[.025,.975]).tolist() if len(v) else None


def buffer(families,field):
    folder=ROOT/'wave';people=list(stream_rows(folder/'geometry.jsonl'));seal=read_json(OUT/'eeg-extraction.json');source={r['output'] for r in seal['sources']}
    locator={q[c]:(i,j) for i,q in enumerate(families) for j,c in enumerate(('high','low'))}
    Y=np.full((len(people),len(families),2,21,2),np.nan)
    for k,p in enumerate(people):
        fp=folder/(p['id']+'.npz')
        if digest(fp) not in source:raise ValueError('Unverified EEG sequence')
        with np.load(fp) as f:
            ids=list(f['ids']);v=f['values'];m=f[field]
        if len(ids)!=144 or ids!=p['ids'] or m.shape!=(144,21):raise ValueError('EEG trial order or epoch inventory changed')
        for a,key in enumerate(ids):
            if key in locator:
                i,j=locator[key];Y[k,i,j]=np.where(m[a,:,None],v[a],np.nan)
    return Y


def measure(fam,Y):
    n=len(Y);rng=np.random.default_rng(SEED);count=np.isfinite(Y).sum(0);summed=np.nansum(Y,0);means=np.divide(summed,count,out=np.full_like(summed,np.nan),where=count>=3)
    D=means[:,1]-means[:,0];p=17;good=np.isfinite(D[:,p,0]);x=np.array([r['category_nll'] for r in fam])[good]
    model={'whole_category_surprisal':x,'category_onset':np.array([r['category_onset'] for r in fam])[good],'fixed_embedding_specificity':np.array([r['fixed'][21] for r in fam])[good]}
    y=D[good,p,0];indices=rng.integers(len(y),size=(4000,len(y)));cor={k:dict(rho=float(rho(v,y)),item_ci=ci([rho(v[j],y[j]) for j in indices]),reader_boot=[]) for k,v in model.items()}
    # Reader resampling respects the counterbalanced design: high and low were assigned to different people per target.
    flags=np.isfinite(Y);raw=np.nan_to_num(Y);counts=np.asarray(rng.multinomial(n,np.full(n,1/n),size=4000),float)
    eps={}
    for q,w,name in [(17,0,'category_N400'),(6,0,'before_clue_N400'),(17,1,'early_category'),(6,1,'early_before_clue')]:
        num=counts@raw[:,:,:,q,w].reshape(n,-1);den=counts@flags[:,:,:,q,w].reshape(n,-1);cell=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0).reshape(4000,len(fam),2)
        d=cell[:,:,1]-cell[:,:,0];ok=np.isfinite(D[:,q,w]);obs=D[ok,q,w]
        eps[name]=interval(obs)|{'reader_ci':ci(np.nanmean(d[:,ok],axis=1)),'item_sign_p_positive':sign_p(obs),'word':q+1,'window':['300–500 ms','100–200 ms'][w]}
        if name=='category_N400':
            for k,v in model.items():
                rr=[]
                for sample in d[:,good]:
                    valid=np.isfinite(sample);rr.append(rho(v[valid],sample[valid]))
                cor[k]['reader_ci']=ci(rr);del cor[k]['reader_boot']
    # Common complete family set avoids an unrelated denominator at different boundaries.
    complete=np.isfinite(D[:,:,0]).all(1);d=D[complete,:,0];mu=d.mean(0);se=d.std(0,ddof=1)/np.sqrt(len(d))
    b=[];rng=np.random.default_rng(SEED)
    for _ in range(40):
        draw=rng.integers(len(d),size=(100,len(d)));b.extend(np.max(np.abs((d[draw].mean(1)-mu)/np.maximum(se,1e-10)),axis=1))
    q=float(np.quantile(b,.95));band=np.c_[mu,mu-q*se,mu+q*se]
    return dict(people=n,words=int(good.sum()),retained_paired_word_epochs=int(flags[:,:,:,:,0].sum()),retained_category_by_condition=count[good,:,p,0].sum(0).tolist(),endpoints=eps,model_relationship=cor,complete_timecourse_families=int(complete.sum()),word_effect_simultaneous_band=band.tolist())


def main():
    cue=read_json(OUT/'cue-scores.json');families=cue['families'];result={}
    for m in ('valid','strict'):
        Y=buffer(families,m);result[m]=measure(families,Y)
    write_json(OUT/'eeg-scores.json',{'registered_protocol':digest(Path(__file__).with_name('spanish-eeg-protocol.md')),'unique_original_extraction':digest(OUT/'eeg-extraction.json'),'released_word_scores':digest(OUT/'cue-scores.json'),'results':result})

if __name__=='__main__':main()
