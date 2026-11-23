"""Train independent typed token evidence and symbol-span readouts from complete original annotations."""
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
import zlib
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from research.backbone_transfer.extract import digest,read_json,write_json
from research.reading_validation.symbol_geometry import OUT
from research.reading_validation.symbol_model import Bank
from research.reading_validation.symbol_span_geometry import gold,mentions,observation,tokens,unit_features

DIR=OUT/'typed';CACHE=Path('.cache/cogllm/reading-validation/symbols/typed');M=8192
GRID=np.r_[.005,.01,.02,.03,.05,.075,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.65,.75,.85,.92,.97,.995]
PLAN=Path(__file__).with_name('symbol-span-readout-protocol.md')


def token_arrays(page):
    cat,x=tokens(page);v=np.array([[zlib.crc32(w.encode())%M for w in row] for row in cat],dtype=np.int32)
    return v,x


def tables(cat,y):
    k=cat.shape[1]
    z=np.stack([np.bincount(cat[:,j]*5+y,minlength=M*5).reshape(M,5) for j in range(k)]).astype('int32')
    return z


def encoded(C,stats,prior):
    groups=np.moveaxis(np.stack([stats[j,c] for j,c in enumerate(C.T)]),0,1);smooth=12
    p=(groups+prior[None,None,:]*smooth)/(groups.sum(2,keepdims=True)+smooth)
    # Direct category probabilities estimate both beginnings/continuations of a true release and independent descriptions.
    return np.asarray(np.log(p[:,:,1:]/p[:,:,:1]).reshape(len(C),-1),'float32')


def sampling(page,doc):
    ms=mentions(page.text);known=gold(doc)[0];hits=[];negative=[]
    for m in ms:
        if (m.start,m.end) in known:hits.append(m)
        else:negative.append(m)
    rng=np.random.default_rng(int(doc['id'][:8],16));selected=set(m for m in negative if m.kind in ('delimited','expression','atom','decorated'))
    if len(negative):
        sel=rng.choice(len(negative),min(80,len(negative)),replace=False);selected.update(negative[i] for i in sel)
    for m in hits:
        hard=[a for a in negative if min(a.end,m.end)>max(a.start,m.start) and (m.start==a.start or m.end==a.end)]
        if hard:
            pick=rng.choice(len(hard),min(8,len(hard)),replace=False);selected.update(hard[i] for i in pick)
    return hits,sorted(selected,key=lambda m:(m.committed,m.start,m.end)),len(negative),len(known)


def create_head(x,y,weights,nclasses):
    model=HistGradientBoostingClassifier(loss='log_loss',learning_rate=.11,max_iter=110,max_leaf_nodes=23,min_samples_leaf=30,l2_regularization=1.5,early_stopping=False,random_state=71)
    model.fit(x,y,sample_weight=weights/np.mean(weights));return model


def file_model(name,obj):
    p=DIR/(name+'.joblib')
    if p.exists():raise FileExistsError(p)
    joblib.dump(obj,p,compress=3);return digest(p)


def fit_source(bank):
    X=[];Y=[];weight=[];papers=[];total=Counter()
    for page,doc in bank.pages('fit'):
        hits,neg,nneg,raw=sampling(page,doc);ms=hits+neg
        total.update({'annotated':raw,'reachable':len(hits),'available_negative':nneg,'observed_negative':len(neg)})
        if not ms:continue
        X.append(unit_features(page,ms));Y.extend([1]*len(hits)+[0]*len(neg));papers.extend([doc['paper']]*len(ms));weight.extend([1.]*len(hits)+[np.sqrt(nneg/max(1,len(neg)))]*len(neg))
    a=np.vstack(X);y=np.array(Y);w=np.asarray(weight)
    model=create_head(a,y,w,2);h=file_model('source',model)
    write_json(DIR/'source-fit.json',dict(plan=digest(PLAN),checkpoint=h,samples=len(a),counts=total,feature_dimension=a.shape[1],papers=len(set(papers)),iterations=model.n_iter_))
    print('Fitted source-entity roster',len(a),total,flush=True)


def fit_evidence(bank):
    cats=[];X=[];Y=[];papers=[]
    for pg,doc in bank.pages('fit'):
        c,x=token_arrays(pg);cats.append(c);X.append(x);Y.append(observation(pg,doc));papers.extend([doc['paper']]*len(c))
    c=np.vstack(cats);x=np.vstack(X);y=np.concatenate(Y);people=np.asarray(papers);stats=tables(c,y);prior=(np.bincount(y,minlength=5)+1)/(len(y)+5)
    Z=np.zeros((len(c),c.shape[1]*4),dtype='float32')
    for group in np.unique(people):
        ix=people==group;stub=tables(c[ix],y[ix]);num=np.bincount(y[~ix],minlength=5);pr=(num+1)/(num.sum()+5);Z[ix]=encoded(c[ix],stats-stub,pr)
    features=np.column_stack((x,Z));cnt=Counter(people)
    w=np.array([1/np.sqrt(cnt[k]) for k in people])
    model=create_head(features,y,w,5);h=file_model('evidence',model);p=DIR/'emission-vocab.npz';np.savez_compressed(p,counts=stats,prior=prior)
    write_json(DIR/'evidence-fit.json',dict(plan=digest(PLAN),checkpoint=h,counts=np.bincount(y,minlength=5).tolist(),vocab=digest(p),papers=len(cnt),features=features.shape[1],iterations=model.n_iter_,training_lexical_encoding='leave one scientific paper out'))
    print('Fitted typed definition observer',len(y),flush=True)


def evidence(page,head):
    forest,stats,prior=head;c,x=token_arrays(page);z=encoded(c,stats,prior);return forest.predict_proba(np.column_stack((x,z)))


def models():
    src=read_json(DIR/'source-fit.json');obs=read_json(DIR/'evidence-fit.json');checks=[('source.joblib',src['checkpoint']),('evidence.joblib',obs['checkpoint']),('emission-vocab.npz',obs['vocab'])]
    if any(digest(DIR/n)!=h for n,h in checks):raise ValueError('A fitted typed readout changed')
    with np.load(DIR/'emission-vocab.npz') as z:c,p=z['counts'],z['prior']
    return joblib.load(DIR/'source.joblib'),(joblib.load(DIR/'evidence.joblib'),c,p)


def select_source(bank):
    model=joblib.load(DIR/'source.joblib');tallies=np.zeros((len(GRID),3),int);counts=Counter()
    for page,doc in bank.pages('calibration'):
        ms=mentions(page.text);known=gold(doc)[0];counts['annotated']+=len(known);counts['candidates']+=len(ms);tallies[:,2]+=len(known)
        if not ms:continue
        v=model.predict_proba(unit_features(page,ms))[:,1];y=np.array([(m.start,m.end) in known for m in ms],int);counts['maximum_exact_coverage']+=int(y.sum())
        for i,c in enumerate(GRID):
            pick=v>=c;tallies[i,0]+=int(y[pick].sum());tallies[i,1]+=int(pick.sum())
    f=2*tallies[:,0]/np.maximum(1,tallies[:,1]+tallies[:,2]);i=int(np.argmax(f))
    write_json(DIR/'source-threshold.json',dict(scores=tallies.tolist(),grid=GRID.tolist(),threshold=float(GRID[i]),selection='maximum calibration-paper span F1',coverage=dict(counts)))
    print('Selected source probability',GRID[i],f[i],counts,flush=True)


def trace_inputs(bank):
    source,obs=models();cut=read_json(DIR/'source-threshold.json')['threshold'];CACHE.mkdir(exist_ok=True,parents=True);digests={}
    for split in ('fit','calibration','held','published_dev','external'):
        out=CACHE/(split+'.npz');arrays={}
        for page,doc in bank.pages(split):
            k=doc['id'];ms=mentions(page.text);v=source.predict_proba(unit_features(page,ms))[:,1] if ms else np.empty(0);sel=np.where(v>=cut)[0]
            arrays[k+'_spans']=np.asarray([(ms[j].start,ms[j].end,ms[j].committed) for j in sel],dtype=np.int32).reshape(-1,3);arrays[k+'_probs']=v[sel].astype('float32');arrays[k+'_evidence']=evidence(page,obs).astype('float32')
        np.savez_compressed(out,**arrays);digests[split]=digest(out);print('Cached strictly causal typed emissions',split,flush=True)
    write_json(DIR/'input-readouts.json',dict(registered_plan=digest(PLAN),code=digest(Path(__file__)),source_threshold=digest(DIR/'source-threshold.json'),files=digests))


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('action',choices=('fit_source','fit_evidence','select_source','trace_inputs'));p.add_argument('folder',type=Path);a=p.parse_args();DIR.mkdir(exist_ok=True,parents=True);globals()[a.action](Bank(a.folder))
