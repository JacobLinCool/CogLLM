"""Estimate first and repeated source-linked explanations over a retained causal notation roster."""
from argparse import ArgumentParser
from bisect import bisect_left
from collections import Counter,defaultdict
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from research.backbone_transfer.extract import digest,read_json,write_json
from research.reading_validation.symbol_model import Bank,case_geometry
from research.reading_validation.symbol_features import canonical
from research.reading_validation.symbol_syntax import DELIM,Mention
from research.reading_validation.symbol_span_fitting import DIR as UP,CACHE
from research.reading_validation.symbol_span_relation import decode,summary,utility,CUT
from research.reading_validation.symbol_factorized import quality
from research.reading_validation.symbol_evaluate import score_events

OUT=UP.parent/'retained';PLAN=Path(__file__).with_name('symbol-retained-source-protocol.md');MOD=OUT/'observer.joblib'


def span_index(page,arrays):
    original={(m.start,m.end):m for m in page.mentions};k=page.row['id'];all_mentions=dict(original)
    for a,b,c in arrays[k+'_spans']:
        a,b,c=int(a),int(b),int(c);old=all_mentions.get((a,b))
        if old is None or c<old.committed:all_mentions[(a,b)]=Mention(a,b,c,'typed')
    markers=[]
    for m in original.values():
        if m.kind!='delimited':continue
        open_=DELIM.match(page.text,m.start);close=[d for d in DELIM.finditer(page.text,open_.end(),m.end)] if open_ else []
        if not close or close[-1].end()!=m.end:continue
        l,r=open_.end(),close[-1].start()
        while l<r and page.text[l].isspace():l+=1
        while r>l and page.text[r-1].isspace():r-=1
        markers.append((m.start,m.end,l,r,bisect_left(page.end,m.committed)))
    page.mentions=sorted(all_mentions.values(),key=lambda m:(m.committed,m.end,m.start));page.by_name=defaultdict(list)
    out=[];names=[];org=[]
    for m in page.mentions:
        name=canonical(page.text[m.start:m.end]);names.append(name);page.by_name[name].append(m);org.append(bisect_left(page.end,m.committed))
        span=min((x for x in markers if x[0]<=m.start<m.end<=x[1]),key=lambda x:x[1]-x[0],default=None)
        shape=original.get((m.start,m.end));typ=(shape.kind if shape else 'typed');v=page.text[m.start:m.end];s=v.strip('$').strip('[]').strip()
        a,b=page.token_span(m.start,m.end)
        info=[len(s)==1,s.islower(),s.isupper(),s.isalpha(),s.isdigit(),s.startswith(chr(92)),s.count('^'),s.count('_'),s.count('{')-s.count('}'),s.count(' '),np.log1p(len(s)),np.log1p(b-a),typ=='atom',typ=='expression',typ=='decorated',typ=='delimited',typ=='typed']
        out.append((span,info,a,b))
    return arrays[k+'_evidence'],out,np.array(org,dtype=int),names


def predictors(page,e,info,org,names,k,t):
    m=page.mentions[k];span,unit,a,b=info[k];surf,sem=page.features(m,t);roles=np.zeros((len(t),9),dtype='float32');age=t-org[k]
    if span:
        left,right,l,r,at=span;known=t>=at;details=[1,m.start==l,m.end==r,(m.start,m.end)==(l,r),(m.start,m.end)==(left,right),m.start>=l and m.end<=r,np.log1p(max(m.start-l,0)),np.log1p(max(r-m.end,0)),np.log1p(r-l)]
        roles[known]=details
    peers=np.array([i for i,name in enumerate(names) if name==names[k]],int);older=[i for i in peers if page.mentions[i].end<=m.start and org[i]<=org[k]]
    same=org[peers,None]<=t[None,:];roster=org[:,None]<=t[None,:];recs=np.c_[np.log1p(age),np.log1p(roster.sum(0)),np.log1p(same.sum(0)),np.full(len(t),np.log1p(len(older)))]
    # Original causal state products carry the identity of the actual earlier source exposed to each evidence token.
    return np.column_stack([surf,sem[:,:8],sem[:,8:48],sem[:,88:120],sem[:,168:208],sem[:,248:280],roles,recs,np.broadcast_to(unit,(len(t),len(unit))),e[t],e[t,3]+e[t,4]]).astype('float32')


def labeled_pairs(page,doc,z):
    e,info,org,names=span_index(page,z);by={(m.start,m.end):i for i,m in enumerate(page.mentions)};n=len(org);random=np.random.default_rng(int(doc['id'][:8],16))
    if not n:return None
    examples=defaultdict(dict);complete=defaultdict(set);exact=defaultdict(set);official=Counter()
    for issue in doc['issues']:
        left=tuple(issue['sources'][0]);all_src=[(tuple(x),by.get(tuple(x))) for x in issue['sources']];definitions=issue['definitions'];
        for j,definition in enumerate(definitions):
            t=page.interval(definition);sample=t[np.linspace(0,len(t)-1,min(5,len(t)),dtype=int)]
            for span,i in all_src:
                if i is None:continue
                for q in sample:
                    if org[i]>q or span[1]>definition[0]:continue
                    complete[int(q)].add(i)
                    if j==0 and span==left and issue['kind']=='forward':exact[int(q)].add(i)
    allzones=set(complete)
    def add(q,k):
        available=np.flatnonzero(org<=q);good=complete[q];first=exact[q];ordinary=[j for j in available if j not in good];aliases=[a for a in ordinary if any(names[a]==names[b] for b in good)];others=random.choice(ordinary,min(len(ordinary),k),replace=False) if ordinary else []
        for a in set(others)|set(aliases)|good:examples[a][q]=2 if a in first else 1 if a in good else 0
    for q in sorted(allzones):add(q,22)
    quiet=np.array([i for i in range(page.n) if i not in allzones],int)
    if len(quiet):
        high=quiet[np.argsort(e[quiet,3]+e[quiet,4])[-min(12,len(quiet)):]];rand=random.choice(quiet,min(15,len(quiet)),replace=False)
        for q in np.unique(np.r_[high,rand]):add(q,6)
    x=[];y=[]
    for k,v in examples.items():
        t=np.array(sorted(v));x.append(predictors(page,e,info,org,names,k,t));y.extend(v[l] for l in t)
    return (np.vstack(x),np.array(y,'int8')) if x else None


def learn(bank):
    X=[];Y=[];papers=[]
    with np.load(CACHE/'fit.npz') as z:
        for page,doc in bank.pages('fit'):
            res=labeled_pairs(page,doc,z)
            if res is not None:X.append(res[0]);Y.append(res[1]);papers.extend([doc['paper']]*len(res[1]))
    x=np.vstack(X);y=np.concatenate(Y);del X,Y;cnt=Counter(papers);w=np.array([1/np.sqrt(cnt[p]) for p in papers]);w/=w.mean()
    model=HistGradientBoostingClassifier(loss='log_loss',learning_rate=.105,max_iter=140,max_leaf_nodes=23,min_samples_leaf=25,l2_regularization=1.5,early_stopping=False,random_state=83).fit(x,y,sample_weight=w)
    if MOD.exists():raise FileExistsError(MOD)
    joblib.dump(model,MOD,compress=3);write_json(OUT/'fit.json',dict(plan=digest(PLAN),code=digest(Path(__file__)),head=digest(MOD),inputs=digest(UP/'input-readouts.json'),count=np.bincount(y,minlength=3).tolist(),dimensions=x.shape[1],papers=len(cnt),steps=model.n_iter_));print('Fitted retained-source observer',len(y),flush=True)


def forecast(page,z,model):
    e,info,org,names=span_index(page,z);grid=np.zeros((len(names),page.n),dtype='float32');parts=[];coords=[];n=0
    def flush():
        if not parts:return
        q=model.predict_proba(np.vstack(parts))[:,2];start=0
        for i,t in coords:grid[i,t]=q[start:start+len(t)];start+=len(t)
        parts.clear();coords.clear()
    for i,b in enumerate(org):
        t=np.arange(b,page.n)
        if len(t):parts.append(predictors(page,e,info,org,names,i,t));coords.append((i,t));n+=len(t)
        if n>=28000:flush();n=0
    flush();return grid,names


def evaluate(bank,split,model,fixed=None):
    rows=defaultdict(lambda:np.zeros((len(CUT) if fixed is None else 1,8),int));nobody=Counter();precision=[];hard=[];surv=Counter()
    with np.load(CACHE/(split+'.npz')) as z:
        for pg,doc in bank.pages(split):
            g,names=forecast(pg,z,model);targs=case_geometry(pg,doc);paper=doc['paper']
            for j,cut in enumerate(CUT if fixed is None else [fixed['joint']]):
                mark=decode(g,names,cut);rows[paper][j]+=score_events(pg,doc,mark)
            if fixed:
                if not targs:nobody['null']+=1;nobody['any_mark']+=bool(mark)
                if fixed['selective'] is not None:
                    sel=decode(g,names,fixed['selective']);precision.append(score_events(pg,doc,sel));
                    if not targs:nobody['any_selective']+=bool(sel)
    if fixed is None:
        tab=sum(rows.values());at=max(range(len(CUT)),key=lambda j:quality(tab[j]));return dict(joint=float(CUT[at]),selective=utility(list(rows.values())),grid=CUT.tolist(),calibration_counts=tab.tolist())
    return dict(**summary({k:x[0] for k,x in rows.items()}),nulls=dict(nobody),selective_counts=np.sum(precision,axis=0).tolist() if precision else None,by_paper={k:x[0].tolist() for k,x in rows.items()})


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('action',choices=('fit','calibrate','score','all'));p.add_argument('folder',type=Path);a=p.parse_args();bank=Bank(a.folder)
    if a.action in ('fit','all'):learn(bank)
    if a.action!='fit':
        fit=read_json(OUT/'fit.json');assert digest(MOD)==fit['head'] and fit['plan']==digest(PLAN);model=joblib.load(MOD)
        if a.action in ('calibrate','all'):
            fixed=evaluate(bank,'calibration',model);write_json(OUT/'calibration.json',fixed);print('Frozen retained-source policy',fixed['joint'],fixed['selective'],flush=True)
        if a.action in ('score','all'):
            fixed=read_json(OUT/'calibration.json')
            for split in ('held','published_dev','external'):
                out=evaluate(bank,split,model,fixed);write_json(OUT/(split+'.json'),dict(fit=digest(OUT/'fit.json'),calibration=digest(OUT/'calibration.json'),**out));print('Evaluated retained-source observer',split,flush=True)
