"""Learn a genuine human four-option error distribution and validate frozen semantics on unused words."""
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from hashlib import sha256
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp,softmax
from scipy.stats import spearmanr
from tokenizers import Tokenizer
from research.backbone_transfer.extract import digest,read_json,write_json
from research.reading_validation.meaning_score import folds
from research.reading_validation.spanish_clue_data import ROOT,OUT,TOKENIZER,PUNCT,norm,logframes,build,dictionary
from research.reading_validation.spanish_clue_score import bank,responses,grouped


def external(original):
    seen={r['source'] for r in original};out=[];exclusions=Counter();lex=dictionary()
    for filename in read_json(ROOT/'eligible-logs.json'):
        p=ROOT/'logs'/filename;person,design,_,_=build(p,lex);by={}
        for r in design:
            if r['condition'] in ('High','Low'):by.setdefault((r['block'],norm(r['source'])),set()).add(r['id'])
        _,source=logframes(p)
        for block,r in source:
            if r['Procedure']!='TestProc':continue
            item=norm(r['TargetPseudoword'])
            if item in seen:continue
            ids=by.get((block,item),set())
            if len(ids)!=1:exclusions['nonunique_original_exposure']+=1;continue
            opts=[r['Word'+str(i)] for i in range(1,5)];y=r['DisplayChoices.RESP'];g=int(r['CorrectWordNumber'])-1
            if len(set(map(norm,opts)))!=4 or not 0<=g<4 or norm(opts[g])!=norm(r['CorrectWord']) or int(r['DisplayChoices.CRESP'])!=g+1:raise ValueError('Unverifiable original external card')
            if y not in ('1','2','3','4'):exclusions['missing_choice']+=1;continue
            y=int(y)-1
            if int(r['DisplayChoices.ACC'])!=int(g==y):raise ValueError('External E-Prime key code disagrees')
            out.append(dict(id=next(iter(ids)),source=item,options=opts,gold=g,response=y,person=sha256(('CogLLM Spanish subjects '+str(person)).encode()).hexdigest()[:20]))
    return out,dict(exclusions)


def material(rows,store,bank_ids,V):
    tok=Tokenizer.from_file(str(TOKENIZER));features=[];kept=[];loss=Counter();ids={r['id'] for r in rows}
    arrays={k:{s:store[k+'_'+s].astype(float) for s in ('onset','static')} for k in ids if k in bank_ids}
    for row in rows:
        k=row['id'];codes=[tok.encode(' '+o.strip(PUNCT),add_special_tokens=False).ids[0] for o in row['options']]
        if k not in arrays or any(c not in V for c in codes):loss['no_frozen_input_or_alternative_embedding']+=1;continue
        cs=[V[c] for c in codes];L=arrays[k]['onset'][:,cs];F=arrays[k]['static'][:,cs];at=np.array([6,13,16,17,20,22])
        x=np.column_stack((L[at].T,F[at].T));features.append(x-x.mean(0));kept.append(row)
    return np.array(features),kept,dict(loss)


def pref(training,test,y,words):
    loc=np.bincount(y[training],minlength=4)+1;all_count=Counter(words[training].ravel());picked=Counter(words[training,y[training]])
    rates=np.array([[(picked[w]+1)/(all_count[w]+4) for w in opts] for opts in words[test]])
    z=np.log(loc)[None,:,None]+np.zeros((len(test),4,1))
    X=np.concatenate((z,np.log(rates)[:,:,None]),axis=2);return X-X.mean(1,keepdims=True)


def train_test(train,test,semantic,y,words,wf):
    # Train each count feature on different lexical families from the choice it predicts.
    bucket=np.asarray(wf)[train];X0=np.zeros((len(train),4,2));source=np.asarray(train)
    for j in np.unique(bucket):
        tr=source[bucket!=j];te=source[bucket==j];X0[bucket==j]=pref(tr,te,y,words)
    held0=pref(train,test,y,words);probs={};fits={}
    for arm in ('surface','semantic'):
        X=X0 if arm=='surface' else np.concatenate((X0,semantic[train]),axis=2)
        Q=held0 if arm=='surface' else np.concatenate((held0,semantic[test]),axis=2)
        sd=np.maximum(np.sqrt((X**2).mean((0,1))),1e-5);x=X/sd;ll=y[train];ridge=.008;idx=np.arange(len(ll))
        def fn(a):
            z=x@a;pr=softmax(z,axis=1);pr[idx,ll]-=1
            return (logsumexp(z,axis=1)-z[idx,ll]).mean()+.5*ridge*(a@a),np.einsum('nm,nmd->d',pr,x)/len(ll)+ridge*a
        op=minimize(fn,np.zeros(X.shape[-1]),jac=True,method='L-BFGS-B',options={'maxiter':400,'ftol':1e-14,'gtol':1e-7})
        if not op.success or max(abs(op.jac))>1e-5:raise ArithmeticError('Choice likelihood failed to converge')
        probs[arm]=softmax((Q/sd)@op.x,axis=1);fits[arm]={'gradient':float(max(abs(op.jac))),'loss':float(op.fun),'coefficients':(op.x/sd).tolist(),'iterations':int(op.nit)}
    return probs,fits


def run(folder):
    receipt,store,records,V=bank(folder);meta={r['id']:r for r in records};tr,_=responses();te,excl=external(tr)
    A,tr,e0=material(tr,store,meta,V);B,te,e1=material(te,store,meta,V);store.close()
    if not len(te):raise ValueError('No additional eligible word families under fixed vocabulary')
    if set(r['source'] for r in tr)&set(r['source'] for r in te):raise ValueError('External lexical leakage')
    rows=tr+te;N=len(tr);X=np.concatenate((A,B));words=np.array([[norm(w) for w in r['options']] for r in rows]);ps=np.array([r['person'] for r in rows]);items=np.array([r['source'] for r in rows]);keys=np.array([r['id'] for r in rows]);Y=np.array([r['response'] for r in rows]);G=np.array([r['gold'] for r in rows]);color=np.array([meta[r['id']]['condition'] for r in rows]);idx=np.arange(len(Y));old=idx<N
    word_fold=folds(items[old],'spanish:word:');pers_fold=folds(ps,'spanish:reader:')
    wf=np.array([word_fold.get(v,-1) for v in items]);pf=np.array([pers_fold[v] for v in ps])
    pr={a:np.full((len(Y),4),np.nan) for a in ('surface','semantic')};log=[]
    for j in range(5):
        for i in (-1,0,1,2,3,4):
            train=np.flatnonzero(old&(wf!=i)&(pf!=j));test=np.flatnonzero((wf==i)&(pf==j))
            if not len(test):continue
            if set(items[test])&set(items[train]) or set(ps[test])&set(ps[train]):raise ValueError('Crossed heldout groups overlap')
            q,record=train_test(train,test,X,Y,words,wf)
            for a,v in q.items():pr[a][test]=v
            log.append(dict(person_fold=j,lexical_fold=i,train=len(train),test=len(test),models=record))
    if any(not np.isfinite(z).all() for z in pr.values()):raise ValueError('Incomplete four-choice forecast')
    results={}
    for name,select in [('development',old),('additional_strict_families',~old)]:
        arms={};loss={}
        for a,z in pr.items():
            l=-np.log(z[idx,Y]);loss[a]=l;summary={}
            for c in ('High','Low'):
                sub=select&(color==c);ids=np.unique(keys[sub]);true=[np.mean(Y[sub&(keys==k)]==G[sub&(keys==k)]) for k in ids];pred=[z[sub&(keys==k),G[sub&(keys==k)]].mean() for k in ids]
                summary[c]=dict(items=len(ids),rho=float(spearmanr(true,pred).statistic),human_accuracy=float(np.mean(true)))
            arms[a]=dict(nll=float(l[select].mean()),condition=summary)
        delta=(loss['surface']-loss['semantic'])[select]
        results[name]=dict(choices=int(select.sum()),words=len(set(items[select])),readers=len(set(ps[select])),models=arms,improvement={'word':grouped(delta,items[select]),'reader':grouped(delta,ps[select])})
    write_json(OUT/'choice-joint-scores.json',dict(plan=digest(Path(__file__).with_name('spanish-choice-joint-protocol.md')),result=results,program=digest(Path(__file__)),source=digest(folder/'receipt.json'),new_screen=excl,exclusions_original=e0,exclusions_additional=e1,unique_external_candidates_before_filter=len({r['source'] for r in external(tr)[0]}),folds=log))

if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);run(p.parse_args().folder)
