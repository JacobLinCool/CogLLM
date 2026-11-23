"""Evaluate registered Spanish cue specificity and four-way choices without target or reader leakage."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path
from hashlib import sha256
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from scipy.stats import spearmanr
from tokenizers import Tokenizer
from research.backbone_transfer.extract import digest, read_json, stream_rows, write_json
from research.reading_validation.meaning_score import folds
from research.reading_validation.spanish_clue_data import (
    ROOT, OUT, TOKENIZER, PUNCT, build, dictionary, logframes, norm,
)

SEED=327920


def interval(values):
    x=np.asarray(values,dtype=float)
    if len(x)==0:return None
    rng=np.random.default_rng(SEED)
    boot=np.concatenate([x[rng.integers(len(x),size=(200,len(x)))].mean(1) for _ in range(25)])
    return dict(n=len(x),mean=float(x.mean()),ci=np.quantile(boot,[.025,.975]).tolist())


def sign_p(values):
    x=np.asarray(values,dtype=float); rng=np.random.default_rng(SEED)
    hits=0
    for _ in range(100):
        null=(rng.choice([-1,1],size=(200,len(x)))*x).mean(1)
        hits+=int(np.sum(null>=x.mean()-1e-12))
    return (1+hits)/20001


def paired_items(records):
    groups=defaultdict(dict)
    for r in records:
        if r['condition'] in ('High','Low'):
            k=(norm(r['target_word']),tuple(r['first']),tuple(r['last']))
            if r['condition'] in groups[k]:raise ValueError('Ambiguous identical-prefix family')
            groups[k][r['condition']]=r
    return [d for d in groups.values() if len(d)==2]


def bank(folder):
    rec=read_json(folder/'receipt.json'); seal=read_json(OUT/'input-seal.json')
    for key in ('stimuli','vocabulary','mapping'):
        p=ROOT/('vocabulary.json' if key=='vocabulary' else key+'.jsonl')
        if digest(p)!=seal[key]:raise ValueError('The certified original word geometry changed')
    if rec['input']!=seal['stimuli'] or rec['vocabulary']!=seal['vocabulary'] or digest(folder/'signals.npz')!=rec['signals']:raise ValueError('Unverified original language-model scores')
    vocab=read_json(ROOT/'vocabulary.json')['onsets']
    return rec,np.load(folder/'signals.npz'),list(stream_rows(ROOT/'mapping.jsonl')),{v:i for i,v in enumerate(vocab)}


def cue(folder):
    rec,buf,records,V=bank(folder); categories=np.array(sorted({V[r['gold_onset']] for r in records})); effects=[]
    for pair in paired_items(records):
        h,l=pair['High'],pair['Low'];src=norm(h['target_word']);col=V[h['gold_onset']]
        if h['gold_onset']!=l['gold_onset']:raise ValueError('The named category changed')
        views={}
        for r in (h,l):
            k=r['id'];z=buf[k+'_onset'].astype(float);s=buf[k+'_static'].astype(float)
            rest=categories[categories!=col]
            views[r['condition']]=dict(
                nll=float(buf[k+'_nll'].astype(float).sum()),
                onset=z[:,col]-logsumexp(z[:,categories],axis=1),
                specific=z[:,col]-z[:,rest].mean(1),
                fixed=s[:,col]-s[:,rest].mean(1),
                traces=buf[k+'_trace'].astype(float),words=buf[k+'_wordnll'].astype(float))
        hi,lo=views['High'],views['Low']; stock=lo['traces']-hi['traces']
        effects.append(dict(word=src,high=h['id'],low=l['id'],same_length=h['article_tokens']==l['article_tokens'],
            category_nll=lo['nll']-hi['nll'],category_onset=float(hi['onset'][21]-lo['onset'][21]),
            specificity=(hi['specific']-lo['specific'])[:22].tolist(),fixed=(hi['fixed']-lo['fixed'])[:22].tolist(),
            stock=stock[:,0].tolist(), opening=stock[:,1].tolist(),resolution=(-stock[:,2]).tolist(),transition=stock[:,3].tolist(),
            word_nll=(lo['words'][1:]-hi['words'][1:]).tolist()))
    def stats(rows):
        return dict(category={s:interval([r[s] for r in rows]) for s in ('category_nll','category_onset')},
            category_p=sign_p([r['category_nll'] for r in rows]),
            category_wins=int(sum(r['category_nll']>0 for r in rows)),
            before_category={s:interval([r[s][21] for r in rows]) for s in ('specificity','fixed')},
            original_word_end={s:[interval(arr) for arr in np.asarray([r[s] for r in rows]).T] for s in ('specificity','fixed','stock','opening','resolution','transition','word_nll')},
            original_prefix_max={s:float(max(abs(q) for r in rows for q in r[s][:7])) for s in ('stock','specificity','fixed')})
    write_json(OUT/'cue-scores.json',dict(receipt=digest(folder/'receipt.json'),source=rec['input'],all=stats(effects),same_native_length=stats([r for r in effects if r['same_length']]),families=effects))
    buf.close()


def responses():
    lex=dictionary();rows=[];errors=Counter()
    for file in read_json(ROOT/'eligible-logs.json'):
        p=ROOT/'logs'/file
        person,_,matches,counts=build(p,lex); errors.update(counts)
        cards={(r['block'],r['pseudo'],tuple(r['options'])):r for r in matches}
        _,source=logframes(p)
        for block,r in source:
            if r['Procedure']!='TestProc':continue
            card=(block,r.get('TargetPseudoword',''),tuple(r['Word'+str(i)] for i in range(1,5)))
            if card not in cards:continue
            m=cards.pop(card);resp=r['DisplayChoices.RESP'];key=r['DisplayChoices.CRESP'];gold=m['slot']
            if int(key)!=gold+1:raise ValueError('Displayed answer-key code differs from the card')
            if resp not in ('1','2','3','4'):
                errors['no_valid_key']+=1;continue
            y=int(resp)-1
            if int(r['DisplayChoices.ACC'])!=int(y==gold):raise ValueError('Reconstructed human decision disagrees with the original code')
            rows.append(dict(id=m['id'],source=norm(m['pseudo']),options=m['options'],gold=gold,response=y,
                person=sha256(('CogLLM Spanish subjects '+str(person)).encode()).hexdigest()[:20]))
        if cards:raise ValueError('A frozen choice card disappeared from its original trial')
    return rows,dict(errors)


def calibrate(z,y,test):
    z=z-z.mean(1,keepdims=True)
    def loss(a):
        v=z*a
        return float((logsumexp(v,axis=1)-v[np.arange(len(y)),y]).mean())
    res=minimize_scalar(loss,bounds=(0,100.),method='bounded',options={'xatol':1e-8})
    candidates=[(res.fun,res.x),(loss(0.),0.),(loss(100.),100.)];value,a=min(candidates)
    if not res.success or not np.isfinite(value):raise ArithmeticError('Scalar likelihood did not converge')
    return softmax(a*test,axis=1),{'temperature_inverse':float(a),'training_nll':value}


def grouped(d,g):
    m=np.array([d[g==v].mean() for v in np.unique(g)])
    return interval(m)|{'sign_p':sign_p(m)}


def human(folder):
    rec,buf,records,V=bank(folder); meta={r['id']:r for r in records};tok=Tokenizer.from_file(str(TOKENIZER))
    rows,errors=responses();n=len(rows);y=np.array([r['response'] for r in rows]);gold=np.array([r['gold'] for r in rows])
    item=np.array([r['source'] for r in rows]);people=np.array([r['person'] for r in rows]);keys=np.array([r['id'] for r in rows])
    cond=np.array([meta[r['id']]['condition'] for r in rows]); words=np.array([[norm(a) for a in r['options']] for r in rows])
    wfold=folds(item,'spanish:word:');pfold=folds(people,'spanish:reader:')
    wf=np.array([wfold[v] for v in item]);pf=np.array([pfold[v] for v in people]);idx=np.arange(n)
    placements={k.split('_')[0] for k in buf.files}
    if not set(keys)<=placements:raise ValueError('Unscored material')
    arrays={k:{d:buf[k+'_'+d][-1].astype(float) for d in ('onset','static')} for k in set(keys)}
    codes=np.array([[V[tok.encode(' '+x.strip(PUNCT),add_special_tokens=False).ids[0]] for x in r['options']] for r in rows])
    z={d:np.array([arrays[k][d][v] for k,v in zip(keys,codes)]) for d in ('onset','static')};buf.close()
    distinct=np.array([len(set(v))==4 for v in codes]);prob={a:np.full((n,4),np.nan) for a in ('uniform','position','lexical','oracle','onset','static','unscaled')}
    prob['uniform'][:]=.25;prob['unscaled']=softmax(z['onset'],axis=1);details=[]
    for i in range(5):
        for j in range(5):
            tr=(wf!=i)&(pf!=j);te=(wf==i)&(pf==j)
            if set(item[tr])&set(item[te]) or set(people[tr])&set(people[te]):raise ValueError('Evaluation overlap')
            model={}
            for d in z:
                prob[d][te],model[d]=calibrate(z[d][tr],y[tr],z[d][te])
            position=np.bincount(y[tr],minlength=4)+1;prob['position'][te]=position/position.sum()
            served=Counter(words[tr].ravel());selected=Counter(words[tr,np.asarray(y[tr])])
            freq=np.array([[(selected[w]+1)/(served[w]+4) for w in card] for card in words[te]])*position
            prob['lexical'][te]=freq/freq.sum(1,keepdims=True)
            for c in np.unique(cond):
                mask=tr&(cond==c);selected_test=np.where(te&(cond==c))[0];acc=((y[mask]==gold[mask]).sum()+1)/(mask.sum()+2)
                prob['oracle'][selected_test]=(1-acc)/3;prob['oracle'][selected_test,gold[selected_test]]=acc
            details.append(dict(item_fold=i,reader_fold=j,rows=int(tr.sum()),fits=model))
    if not all(np.isfinite(a).all() and np.allclose(a.sum(1),1.) for a in prob.values()):raise ArithmeticError('Choice probabilities invalid')
    nll={s:-np.log(v[idx,y].clip(1e-15,1)) for s,v in prob.items()}
    def summarize(select):
        names={}
        for s,v in prob.items():
            data={}
            for c in np.unique(cond):
                ids=np.unique(keys[select&(cond==c)])
                true=np.array([(y[select&(keys==k)]==gold[select&(keys==k)]).mean() for k in ids]);pred=np.array([v[select&(keys==k),gold[select&(keys==k)]].mean() for k in ids])
                rho=spearmanr(true,pred).statistic if len(set(pred))>1 else np.nan
                data[c]={'families':len(ids),'human_key_accuracy':float(true.mean()),'rho_to_key_success':float(rho) if np.isfinite(rho) else None}
            best=v==v.max(1,keepdims=True);credit=best/best.sum(1,keepdims=True)
            names[s]=dict(choice_nll=float(nll[s][select].mean()),correct_key_top1=float(credit[idx[select],gold[select]].mean()),human_choice_top1=float(credit[idx[select],y[select]].mean()),condition=data)
        return names
    differences={}
    for base in ('lexical','uniform','static','oracle','position','unscaled'):
        d=nll[base]-nll['onset'];differences[base]={'word':grouped(d,item),'reader':grouped(d,people)}
    result=dict(receipt=digest(folder/'receipt.json'),words=len(set(item)),people=len(set(people)),completed_choices=n,distinct_first_piece_rows=int(sum(distinct)),
        exclusions=errors,overall=summarize(np.ones(n,dtype=bool)),distinct=summarize(distinct),human_key_accuracy=float(np.mean(y==gold)),reductions=differences,folds=details)
    primary=read_json(OUT/'cue-scores.json')['all']['category_p'];human_p=max(differences['lexical'][k]['sign_p'] for k in ('reader','word'));p=np.array([primary,human_p]);order=np.argsort(p);adj=p.copy();adj[order]=np.minimum(1,np.maximum.accumulate(p[order]*[2,1]));result['holm_directional']={'category':float(adj[0]),'human_nll_vs_lexical':float(adj[1])}
    write_json(OUT/'choice-scores.json',result)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('action',choices=('cue','human'));p.add_argument('folder',type=Path);a=p.parse_args();globals()[a.action](a.folder)
