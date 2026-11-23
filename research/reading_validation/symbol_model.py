"""Paper-balanced point and identity supervision for resolving the same mathematical symbol."""
from bisect import bisect_left
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from tokenizers import Tokenizer
from research.backbone_transfer.extract import digest,read_json,stream_rows,write_json
from research.reading_validation.identity_prepare import TOKENIZER
from research.reading_validation.symbol_features import Page,canonical
from research.reading_validation.symbol_geometry import ROOT,OUT,competing
from research.reading_validation.symbol_syntax import Mention


class Bank:
    def __init__(self,folder,synthetic=False):
        self.folder=Path(folder);self.rows={r['id']:r for r in stream_rows(ROOT/'stimuli-native.jsonl')};self.map={r['id']:r for r in stream_rows(ROOT/'mapping.jsonl')};self.tok=Tokenizer.from_file(str(TOKENIZER));self.syn=synthetic
        if not synthetic:
            self.receipt=read_json(self.folder/'receipt.json');extra=read_json(OUT/'equation-key-addition.json');source=read_json(OUT/'input-seal.json');
            if self.receipt['stimuli']!=extra['stimuli'] or extra['stimuli']!=digest(ROOT/'stimuli-native.jsonl') or source['mapping']!=digest(ROOT/'mapping.jsonl') or read_json(OUT/'prefix-geometry-audit.json')['syntax_sha256']!=digest(Path(__file__).with_name('symbol_syntax.py')):raise ValueError('Text or source geometry changed')
            self.shards=self.receipt['shards'];self.check={}

    def pages(self,split,limit=None):
        keys=sorted(k for k,r in self.map.items() if r['split']==split)
        if limit:keys=keys[:limit]
        if not self.syn:keys.sort(key=lambda k:(self.shards[k],k))
        store=None;last=None
        try:
            for k in keys:
                row=self.rows[k];e=self.tok.encode(row['text'],add_special_tokens=False);n=len(e.ids)
                if self.syn:
                    rng=np.random.default_rng(int(k[:8],16));state=rng.normal(0,1,(n,80)).astype('float16');rel=np.zeros((n,len(row['keys']),2),dtype='uint8')
                else:
                    j=self.shards[k];name=f'shard-{j:03}.npz'
                    if j!=last:
                        if store:store.close()
                        if j not in self.check:
                            actual=digest(self.folder/name)
                            if self.receipt['files'][name]!=actual:raise ValueError('Corrupt symbol-prediction shard')
                            self.check[j]=actual
                        store=np.load(self.folder/name);last=j
                    state=store[k+'_s'];rel=store[k+'_a']
                yield Page(row,e,state,rel),self.map[k]
        finally:
            if store:store.close()


def anchor(page,span):
    a,b=span
    occ=next((v for v in page.mentions if v.start==a and v.end==b),None)
    return occ or Mention(a,b,b,'supplied')


def case_geometry(page,doc):
    anchors=[]
    for j in doc['issues']:
        m=anchor(page,j['sources'][0]);k=bisect_left(page.end,m.committed)
        if j['kind']=='forward':
            words=page.interval(j['definitions'][0]);valid=words[words>=k]
            if not len(valid):continue
            anchors.append((m,tuple(j['definitions'][0]),valid))
    return anchors


def sample(xs,n,rng):
    xs=np.asarray(xs,dtype=int);return np.sort(rng.choice(xs,min(n,len(xs)),replace=False)) if len(xs) else xs


def terms(page,doc):
    rng=np.random.default_rng(int(doc['id'][:8],16));true=case_geometry(page,doc);known={(m.start,m.end) for m,_,_ in true};P=[];N=[];D=[]
    saved={}
    def get(m,t):
        index=(m,tuple(t));
        if index not in saved:saved[index]=np.column_stack(page.features(m,t))
        return saved[index]
    for m,desc,zone in true:
        positive=sample(zone,4,rng);v=get(m,positive);P.extend(v)
        k=bisect_left(page.end,m.committed);before=np.arange(k,zone[0]);neg=sample(before,7,rng)
        if len(neg):N.extend(get(m,neg))
        alts=[v for v,other,tt in true if v!=m and (tt[-1]<zone[0] or tt[0]>zone[-1]) and v.committed<=min(page.end[positive]) and canonical(page.text[v.start:v.end])!=canonical(page.text[m.start:m.end])]
        for j in sample(np.arange(len(alts)),3,rng):
            w=get(alts[j],positive);N.extend(w);D.extend(v-w)
        # Choose an already visible distractor for the identical observed evidence. We exclude any literal referring to the same symbol.
        aliases={canonical(page.text[a:b]) for j in doc['issues'] if tuple(j['sources'][0])==(m.start,m.end) for a,b in j['sources']}
        decoys=[n for n in page.mentions if (n.start,n.end) not in known and n.committed<=page.end[positive[0]] and canonical(page.text[n.start:n.end]) not in aliases and not(n.start<m.end and n.end>m.start)]
        for j in sample(np.arange(len(decoys)),2,rng):
            w=get(decoys[j],positive);N.extend(w);D.extend(v-w)
    # Ordinary text, including entire paragraphs without an annotated direct definition.
    distract=[m for m in page.mentions if (m.start,m.end) not in known]
    for j in sample(np.arange(len(distract)),5,rng):
        m=distract[j];k=bisect_left(page.end,m.committed);ts=sample(np.arange(k,page.n),8,rng)
        if len(ts):N.extend(get(m,ts))
    return P,N,D


def collect(bank,split,limit=None):
    rows=[[],[],[]];papers=[[],[],[]]
    for page,doc in bank.pages(split,limit):
        for i,x in enumerate(terms(page,doc)):
            rows[i].extend(x);papers[i].extend([doc['paper']]*len(x))
    if any(not x for x in rows):raise ValueError('No observed supervision for one objective')
    return [np.asarray(v,dtype=np.float32) for v in rows],papers


def fit(data,arm):
    blocks,names=data
    # First 27 columns capture features observable directly from text (verified by synthetic schema check).
    n_surface=27
    selected=np.arange(n_surface) if arm=='surface' else np.arange(blocks[0].shape[1])
    # Source and token position only, with no actual linguistic content.
    if arm=='position':selected=np.array([0,9,10,11,12,13,14])
    # Keep one FP32 working design with FP64 statistics and loss reductions.
    # This bounds host memory when the raw relation assay contains many weak negatives.
    total=sum(len(v) for v in blocks);XX=np.empty((total,len(selected)),dtype='float32');cursor=0
    for part in blocks:
        end=cursor+len(part);XX[cursor:end]=part[:,selected];cursor=end
    N=len(blocks[0])+len(blocks[1]);mu=XX[:N].mean(0,dtype=np.float64);sd=np.maximum(XX[:N].std(0,dtype=np.float64),.05)
    XX[:N]-=mu;XX/=sd
    def balance(a):
        c=Counter(a);return np.array([1/len(c)/c[g] for g in a])
    # Equal mass for positive interval locations, false alerts, and same-cue pairwise comparisons.
    WW=np.concatenate([balance(a)/3 for a in names]);Y=np.r_[np.ones(len(blocks[0])),np.zeros(len(blocks[1])),np.ones(len(blocks[2]))]
    intercept=np.r_[np.ones(len(blocks[0])+len(blocks[1])),np.zeros(len(blocks[2]))];ridge=.03
    def objective(v):
        loss=.5*ridge*(v[1:]@v[1:]);grad=np.r_[0.,ridge*v[1:]]
        for lo in range(0,len(XX),4096):
            sl=slice(lo,lo+4096);part=XX[sl].astype(np.float64);z=part@v[1:]+intercept[sl]*v[0];e=WW[sl]*(expit(z)-Y[sl]);loss+=WW[sl]@(np.logaddexp(0,z)-Y[sl]*z);grad[0]+=intercept[sl]@e;grad[1:]+=part.T@e
        return float(loss),grad
    init=np.zeros(len(mu)+1);sol=minimize(objective,init,jac=True,method='L-BFGS-B',options={'maxiter':500,'ftol':1e-12,'gtol':2e-7});g=float(np.max(np.abs(sol.jac)))
    if not sol.success or g>2e-5:raise ArithmeticError('Symbol supervision failed to converge: '+str((sol.message,g)))
    w=np.zeros(data[0][0].shape[1]);w[selected]=sol.x[1:]/sd
    return {'arm':arm,'coefficients':w.tolist(),'intercept':float(sol.x[0]-mu@(sol.x[1:]/sd)),'convergence':{'steps':int(sol.nit),'max_gradient':g,'initial_loss':float(objective(init)[0]),'final_loss':float(sol.fun)},'components':list(map(len,blocks)),'surface_columns':n_surface,'input_dimension':len(w)}
