"""Learn independent links at typed evidence; distinct simultaneously explained symbols may co-occur."""
from argparse import ArgumentParser
from bisect import bisect_left
from collections import Counter,defaultdict
from pathlib import Path
import zlib
import joblib
import numpy as np
from scipy.special import expit
from research.backbone_transfer.extract import digest,read_json,write_json
from research.reading_validation.symbol_span_fitting import DIR,CACHE,GRID,PLAN,create_head,file_model
from research.reading_validation.symbol_span_geometry import gold,observation,unit_features
from research.reading_validation.symbol_features import canonical
from research.reading_validation.symbol_model import Bank,case_geometry,anchor
from research.reading_validation.symbol_syntax import Mention
from research.reading_validation.symbol_evaluate import score_events,metric
from research.reading_validation.symbol_factorized import quality

CUT=np.unique(np.r_[np.geomspace(.00001,.4,26),.5,.65,.8,.9,.95,.99])


def roster(page,z,teach=False,doc=None):
    k=page.row['id'];spec=z[k+'_spans'];source=z[k+'_probs'];e=z[k+'_evidence']
    ms=[Mention(int(a),int(b),int(end),'typed') for a,b,end in spec]
    # Preserve the finite calibration-selected model roster also when fitting the edge model.
    original={(m.start,m.end):m for m in page.mentions};ms=[original[(m.start,m.end)] if (m.start,m.end) in original and original[(m.start,m.end)].committed==m.committed else m for m in ms]
    page.mentions=ms;page.by_name=defaultdict(list)
    for m in ms:page.by_name[canonical(page.text[m.start:m.end])].append(m)
    return source,e


def order(page):
    ms=page.mentions;available=np.array([bisect_left(page.end,m.committed) for m in ms],int)
    return available,[canonical(page.text[m.start:m.end]) for m in ms]


def features(page,source,e,k,ts,unit,org,names):
    m=page.mentions[k];s,h=page.features(m,ts);a,b=page.token_span(m.start,m.end);p=source[k];v=unit[k];event=e[ts]
    # First mention and same-name span preference are incremental; names committed in the future do not participate.
    same=np.array([j for j,n in enumerate(names) if n==names[k]],int);age=ts-org[k];other=same[same!=k]
    if len(other):
        available=org[other,None]<=ts[None,:];best=np.max(np.where(available,source[other,None],0),axis=0);seen_count=available.sum(0)
    else:best=np.zeros(len(ts));seen_count=np.zeros(len(ts))
    prefix=p>=best;oldest=org[same].min()==org[k];shown=(org[:,None]<=ts[None,:]).sum(0)
    name_features=np.column_stack((np.full(len(ts),p),p-best,prefix,np.full(len(ts),oldest),np.log1p(seen_count),np.log1p(shown),np.log1p(age)))
    keep=h[:,[0,1,2,3,4,5,6,7]]
    # Qwen state interactions with each live source and earliest source-token keys remain recoverable.
    dense=np.column_stack((s,keep,h[:,8:32],h[:,88:112],h[:,168:200],h[:,248:280],name_features,np.broadcast_to(v[:40],(len(ts),40)),event,event[:,3]+event[:,4]))
    return dense.astype('float32')


def sample_terms(page,doc,source,e):
    rng=np.random.default_rng(int(doc['id'][:8],16)+117);ms=page.mentions;org,names=order(page);n=len(ms)
    if not n:return None
    unit=unit_features(page,ms);by={(m.start,m.end):i for i,m in enumerate(ms)};true=case_geometry(page,doc);loc=defaultdict(set);allzones=set();pairs=defaultdict(dict)
    for target,_,ts in true:
        k=by.get((target.start,target.end));allzones.update(ts.tolist())
        if k is not None:
            for t in ts:
                if t>=org[k]:loc[int(t)].add(k)
    def add(t,validneg):
        yes=loc[t];valid=np.flatnonzero(org<=t)
        if not len(valid):return
        neg=[j for j in valid if j not in yes]
        aliases=[j for j in neg if any(names[j]==names[q] for q in yes)]
        if len(neg)>validneg:neg=rng.choice(neg,validneg,replace=False).tolist()
        for j in set(neg)|set(aliases)|yes:pairs[j][t]=int(j in yes)
    for _,_,ts in true:
        for t in np.unique(ts[np.linspace(0,len(ts)-1,min(5,len(ts)),dtype=int)]):add(int(t),32)
    labels=observation(page,doc);nonfeedback=np.flatnonzero((labels==1)|(labels==2))
    for t in rng.choice(nonfeedback,min(20,len(nonfeedback)),replace=False):add(int(t),12)
    # Such points receive no relation label even if a spurious textual score makes them look like a definition.
    i=np.array([q for q in range(page.n) if q not in allzones]);high=i[np.argsort(e[i,3]+e[i,4])[-min(22,len(i)):]] if len(i) else np.array([],int)
    random=rng.choice(i,min(15,len(i)),replace=False) if len(i) else np.array([],int)
    for t in np.unique(np.r_[high,random]):add(int(t),8)
    X=[];Y=[]
    for j,records in pairs.items():
        t=np.array(sorted(records));X.append(features(page,source,e,j,t,unit,org,names));Y.extend([records[a] for a in t])
    return (np.vstack(X),np.array(Y,'int8')) if X else None


def learn(bank):
    allx=[];ally=[];docs=0;src=read_json(DIR/'input-readouts.json');f=CACHE/'fit.npz'
    if digest(f)!=src['files']['fit']:raise ValueError('Training emissions changed')
    with np.load(f) as z:
        for page,doc in bank.pages('fit'):
            source,e=roster(page,z);train=sample_terms(page,doc,source,e)
            if train is not None:allx.append(train[0]);ally.append(train[1]);docs+=1
    X=np.vstack(allx);y=np.concatenate(ally);del allx,ally
    # Calibration uses full candidate sets; the training sample includes every competing alias of a positive.
    model=create_head(X,y,np.ones(len(y)),2);path=file_model('relation',model)
    write_json(DIR/'relation-fit.json',dict(plan=digest(PLAN),upstream=digest(DIR/'input-readouts.json'),classifier=path,features=X.shape[1],paragraphs=docs,rows=len(y),positive_rows=int(y.sum()),iterations=model.n_iter_))
    print('Learned source-to-current-definition likelihood',len(y),int(y.sum()),flush=True)


def events(page,s,e,model):
    org,names=order(page)
    if not len(org):return [],[]
    unit=unit_features(page,page.mentions);allx=[];co=[]
    for k,m in enumerate(page.mentions):
        ts=np.arange(org[k],page.n)
        if len(ts):allx.append(features(page,s,e,k,ts,unit,org,names));co.append((k,ts))
    X=np.vstack(allx);q=model.predict_proba(X)[:,1];cursor=0;F=np.zeros((len(org),page.n),dtype='float32')
    for k,t in co:
        pred=q[cursor:cursor+len(t)];cursor+=len(t);F[k,t]=pred*(e[t,3]+e[t,4])*s[k]
    return F,names


def decode(F,names,cut):
    if not len(names):return []
    idx=defaultdict(list)
    for i,n in enumerate(names):idx[n].append(i)
    out=[]
    for k,I in idx.items():
        y=F[I];high=y.max(0);ts=np.where(high>=cut)[0]
        if len(ts):
            t=int(ts[0]);i=I[int(y[:,t].argmax())];out.append((i,t,float(high[t])))
    return sorted(out,key=lambda x:x[1])


def utility(rows):
    x=np.sum(rows,axis=0);eligible=[i for i,c in enumerate(x) if c[1]>0 and c[0]/c[1]>=.50]
    return float(CUT[max(eligible,key=lambda i:(x[i,0],x[i,0]/x[i,1]))]) if eligible else None


def summary(rows):
    paper=sorted(rows);A=np.stack([rows[k] for k in paper]);s=A.sum(0);rng=np.random.default_rng(6613);b=A[rng.integers(len(paper),size=(5000,len(paper)))].sum(1)
    out={}
    for name,v in metric(s).items():
        val=metric(b)[name];ok=val[np.isfinite(val)]
        out[name]={'value':float(v) if np.isfinite(v) else None,'paper_ci':np.quantile(ok,[.025,.975]).tolist() if len(ok) else None}
    return dict(papers=len(paper),counts=s.tolist(),metrics=out)


def score(bank,split,model,cal=None):
    src=read_json(DIR/'input-readouts.json');f=CACHE/(split+'.npz')
    if digest(f)!=src['files'][split]:raise ValueError('Unverified source roster or emitted roles')
    sums=defaultdict(lambda:np.zeros((len(CUT) if cal is None else 1,8),int));nulls=Counter();P=[[],[]];baseline=Counter();at=Counter();typed_prediction=Counter();typed_positive=Counter()
    with np.load(f) as z:
        for pg,doc in bank.pages(split):
            s,e=roster(pg,z);F,names=events(pg,s,e,model);targets=case_geometry(pg,doc)
            if cal is None:
                for j,t in enumerate(CUT):sums[doc['paper']][j]+=score_events(pg,doc,decode(F,names,t))
            else:
                events_=decode(F,names,cal['joint']);sums[doc['paper']][0]+=score_events(pg,doc,events_)
                if not targets:nulls['true_none_paragraphs']+=1;nulls['with_alert']+=bool(events_)
                if cal['selective'] is not None:
                    marks=decode(F,names,cal['selective']);P[0].append(score_events(pg,doc,marks))
                    if not targets:nulls['selective_with_alert']+=bool(marks)
                true=observation(pg,doc);est=e.argmax(1)
                for y,pred in zip(true,est):typed_prediction[(int(y),int(pred))]+=1
    if cal is None:
        a=np.sum(list(sums.values()),axis=0);choice=max(range(len(CUT)),key=lambda i:quality(a[i]))
        return dict(joint=float(CUT[choice]),selective=utility(list(sums.values())),grid=CUT.tolist(),development_counts=a.tolist())
    out=summary({p:x[0] for p,x in sums.items()})
    return dict(**out,true_none=dict(nulls),selective_counts=np.sum(P[0],axis=0).tolist() if P[0] else None,typed_confusion={f'{a}-{b}':c for (a,b),c in typed_prediction.items()})


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('action',choices=('learn','calibrate','score'));p.add_argument('folder',type=Path);args=p.parse_args();bank=Bank(args.folder)
    if args.action=='learn':learn(bank)
    else:
        r=read_json(DIR/'relation-fit.json');f=DIR/'relation.joblib'
        if digest(f)!=r['classifier']:raise ValueError('Typed relation parameters changed')
        model=joblib.load(f)
        if args.action=='calibrate':write_json(DIR/'thresholds.json',score(bank,'calibration',model))
        else:
            cal=read_json(DIR/'thresholds.json')
            for s in ['held','published_dev','external']:
                result=score(bank,s,model,cal);write_json(DIR/(s+'.json'),dict(weights=digest(f),thresholds=digest(DIR/'thresholds.json'),**result));print('Evaluated typed causal spans',s,flush=True)
