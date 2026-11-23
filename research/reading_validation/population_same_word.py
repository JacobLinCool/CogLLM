"""Hold the incoming printed word fixed when predicting change in population lexical disagreement."""
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
from research.backbone_transfer.extract import digest,read_json,write_json

ROOT=Path('research/results/reading-validation-cloze-population')
INPUT=Path('.cache/cogllm/reading-validation/cloze-population/external-scored-word-rows.jsonl')
SOURCE=ROOT/'form-specificity-forecasts.npz'
PLAN=Path(__file__).with_name('population-same-word-updates.md')


def audit(mask,forms,story,y,methods):
    word_groups=defaultdict(list)
    for i in np.flatnonzero(mask):word_groups[forms[i]].append(i)
    order={s:n for n,s in enumerate(sorted(set(story)))}
    norm=np.array([order[s] for s in story])
    groups=[]
    for form,indices in sorted(word_groups.items()):
        i,j=np.triu_indices(len(indices),1);left=np.array(indices)[i];right=np.array(indices)[j]
        yes=(story[left]!=story[right])&(abs(y[left]-y[right])>1e-10);left,right=left[yes],right[yes]
        if len(left):groups.append((left,right))
    n=len(groups); B=4000
    rng=np.random.default_rng(881305)
    material=sorted({idx for p in groups for c in p for idx in norm[c]});draw=np.full((B+1,len(order)),np.nan);draw[0]=1
    draw[1:,material]=rng.exponential(1,size=(B,len(material)))
    word=np.vstack((np.ones(n),rng.exponential(1,size=(B,n))))
    models=list(methods);rep=np.zeros((B+1,len(models)))
    for k,(i,j) in enumerate(groups):
        measured=np.sign(y[i]-y[j])
        val=np.column_stack([.5+.5*measured*np.where(abs(p[i]-p[j])<1e-10,0,np.sign(p[i]-p[j])) for p in methods.values()])
        for start in range(0,len(draw),60):
            weights=draw[start:start+60,norm[i]]*draw[start:start+60,norm[j]]
            rep[start:start+60]+=word[start:start+60,k,None]*(weights@val)/weights.sum(1,keepdims=True)
    rep/=word.sum(1,keepdims=True)
    point={m:dict(concordance=float(rep[0,k]),interval=np.quantile(rep[1:,k],[.025,.975]).tolist()) for k,m in enumerate(models)}
    def comparison(a,b):
        diff=rep[:,models.index(a)]-(.5 if b=='constant' else rep[:,models.index(b)])
        return dict(gain=float(diff[0]),interval=np.quantile(diff[1:],[.025,.975]).tolist())
    comparisons={a+'_over_'+b:comparison(a,b) for a,b in (
        ('context','constant'),('context','raw'),('context','calibrated'),('context','form_fit'),('form_all','form_fit'),('form_all','form_both'))}
    return dict(forms=n,paragraphs=len(material),pairs=sum(len(a) for a,_ in groups),
             unique_occurrences=len({int(v) for p in groups for side in p for v in side}),
             estimates=point,paired=comparisons)


def main():
    if digest(INPUT)!=read_json(ROOT/'word-export.json')['file'] or digest(SOURCE)!=read_json(ROOT/'form-specificity.json')['forecast']:
        raise ValueError('Registered exposure or prediction differs')
    names={(r['Text_ID'],r['target_ordinal_zero_based']):r['shown_target'] for r in map(json.loads,INPUT.open())}
    with np.load(SOURCE) as f:
        story=f['story'];y=f['human'];forms=[names[(s,int(t))] for s,t in zip(story,f['original_word'],strict=True)]
        methods={k:f[k] for k in ('raw','calibrated','context','form_fit','form_both','form_all')}
        out={k:audit(mask,forms,story,y,methods) for k,mask in (('all',np.ones(len(y),bool)),('interior',f['within_sentence']))}
    write_json(ROOT/'same-word-updates.json',dict(protocol=digest(PLAN),forecast=digest(SOURCE),material=digest(INPUT),test=out,
                stage='Observational additional analysis following aggregate Provo evaluation.'))
    print({k:(v['forms'],v['pairs'],v['estimates']['context'],v['paired']['context_over_raw']) for k,v in out.items()})

if __name__=='__main__':main()
