"""Fit an exact set-valued weak-observation likelihood to unresolved human spans."""
from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np

from research.reading_validation.transition_external_score import evaluate_one,grouped_distribution
from research.reading_validation.transition_full_movie_bounds import segment_solutions,threshold_bounds
from research.reading_validation.transition_score import dump,sha,with_interval
from research.reading_validation.transition_snac_fit import inputs
from research.reading_validation.transition_snac_prepare import RESULTS,source


ARMS=('segment_unique','word_identified','set_marginal')
OUT=Path('research/results/reading-validation-transition-partial')
PROTOCOL=Path('research/reading_validation/transition-partial-loss-protocol.md')


@dataclass
class Segment:
    doc: str
    group: str
    split: str
    ids: np.ndarray
    masks: np.ndarray


def reference_options(bounds, text, errors, whole_words):
    states={0};counts=[]
    for error in errors:
        span=' '.join(error['span'].split())
        if not span:return None,[]
        expr=re.escape(span)
        if whole_words:
            if span[0].isalnum():expr=r'(?<![A-Za-z0-9])'+expr
            if span[-1].isalnum():expr+=r'(?![A-Za-z0-9])'
        occurrences=[(m.start(),m.start()+len(span)) for m in re.finditer(f'(?={expr})',text)]
        if not occurrences:return None,[]
        choices=[]
        for a,b in occurrences:
            loc=np.flatnonzero((bounds[:,0]<b)&(bounds[:,1]>a))
            if not len(loc):return None,[]
            choices.append(sum(1<<int(j) for j in loc))
        counts.append(len(choices))
        states={p|h for p in states for h in choices}
    shifts=range(len(bounds))
    return np.asarray([[bool(mask&(1<<j)) for j in shifts] for mask in sorted(states)],dtype=bool),counts


def material(directory,candidate=None,target='CharE'):
    feats=directory/'word-features.npz';receipt=json.loads((directory/'receipt.json').read_text())
    doc,meta=inputs();raw=source();arrays=np.load(feats)
    expected=meta['packet_sha256'];packet=None
    if candidate:
        from research.reading_validation.transition_architecture_score import common,native_arms
        manifest,packet,_,_=common();expected=manifest['packet_sha256']
        if receipt['candidate']!=candidate or receipt['forwards']!=len(packet):
            raise ValueError('Wrong common frozen architecture')
    if sha(feats)!=receipt['feature_sha256'] or receipt['input_sha256']!=expected:
        raise ValueError('Original single-forward features do not match')
    n=0;segments=[];X=[];train=[];excluded=Counter()
    for item in meta['documents']:
        d=item['doc'];words=np.asarray(doc[d]['words']);x=arrays[d];fold=item['split']
        if packet is not None:
            if not np.array_equal(packet[d]['indices'],np.arange(len(words))):raise ValueError('Excluded word in cross-architecture coordinate map')
            x=native_arms(x,np.zeros((len(x),0)),receipt['local_width'],candidate)['relations']
        if len(x)!=len(words):raise ValueError('Word-state alignment changed')
        X.append(x);train.extend([fold=='train']*len(x))
        for block in item['segments']:
            k=block['index'];a,b=block['start'],block['end'];local=np.flatnonzero(words[:,4]==k)
            obs=raw[d][str(k)];rows=[e for e in obs['errors'] if e['error_type']==target]
            text=doc[d]['text'][a:b]
            if text!=' '.join(obs['text'].split()):raise ValueError('Visibility changed')
            if target=='CharE':
                masks,_=segment_solutions(words[local,:2]-a,text,rows)
            else:
                masks,_=reference_options(words[local,:2]-a,text,rows,target=='CorefE')
                if masks is None:
                    excluded[fold]+=1
                    continue
            segments.append(Segment(d,item['group'],fold,local+n,masks))
        n+=len(x)
    x=np.concatenate(X).astype(float);mu=x[train].mean(0);sd=np.maximum(x[train].std(0),1e-10)
    x-=mu;x/=sd
    return np.column_stack([np.ones(len(x)),x]),segments,mu,sd,receipt,dict(excluded)


def design(x,segments,arm):
    masses=[];blocks=[];fixed=[];groups=[];posts=[]
    for s in segments:
        z=s.masks;basis=z.min(axis=0);uncertain=z.max(axis=0)^basis
        if arm=='segment_unique' and len(z)>1:continue
        keep=~uncertain if arm=='word_identified' else np.ones(len(s.ids),dtype=bool)
        if not keep.any():continue
        ids=s.ids[keep];blocks.append(x[ids]);fixed.extend(basis[keep]);groups.extend([s.group]*len(ids))
        if arm=='set_marginal' and len(z)>1:
            # A latent state's excess positive-word sufficient statistics above the guaranteed ones.
            posts.append((s.group,(z[:,uncertain]*1.)@x[s.ids[uncertain]]))
    cnt=Counter(groups);masses=np.array([1/(len(cnt)*cnt[g]) for g in groups])
    posts=[(1/(len(cnt)*cnt[g]),v) for g,v in posts]
    return np.concatenate(blocks),masses,np.asarray(fixed,dtype=float),posts


def solve(data,start):
    x,mass,y,posts=data;beta=start.copy()
    reg=np.full(len(beta),.01);reg[0]=0.
    def objective(b,derivative):
        eta=x@b
        value=float(mass@(np.logaddexp(0.,eta)-y*eta)+.5*reg@(b*b))
        if derivative:
            p=np.exp(-np.logaddexp(0.,-eta))
            grad=x.T@(mass*(p-y))+reg*b
            # Expected complete-data Hessian majorizes the observed Hessian by the latent covariance.
            h=(x.T*(mass*p*(1-p)))@x+np.diag(reg)
        for m,v in posts:
            scores=v@b;peak=scores.max();q=np.exp(scores-peak);z=q.sum()
            value-=m*(peak+np.log(z))
            if derivative:grad-=m*(q@v)/z
        return (value,grad,h) if derivative else value
    log=[]
    for iteration in range(180):
        val,g,h=objective(beta,True);norm=float(abs(g).max());log.append([val,norm])
        if norm<=1e-8:return beta,{'iterations':iteration,'objective':val,'gradient':norm,'history':log}
        move=np.linalg.solve(h,g);step=1
        for _ in range(40):
            nxt=beta-step*move
            if objective(nxt,False)<=val-1e-4*step*(move@g):break
            step*=.5
        else:raise RuntimeError('Observed-set likelihood failed the monotone step check')
        beta=nxt
    raise RuntimeError('Source likelihood did not reach the stationary tolerance')


def checked_loss():
    x=np.eye(3);z=np.array([[1,0,1],[0,1,1]],bool);b=np.array([-1.,.8,.4])
    logits=x@b;p=np.exp(-np.logaddexp(0,-logits))
    term=np.logaddexp(0,logits).sum()-np.logaddexp.reduce(z@logits)
    np.testing.assert_allclose(term,-np.log(np.prod(np.where(z,p,1-p),axis=1).sum()),rtol=1e-13)
    q=np.exp(z@logits-np.logaddexp.reduce(z@logits));grad=x.T@(p-q@z)
    def f(v):return np.logaddexp(0,x@v).sum()-np.logaddexp.reduce(z@(x@v))
    eps=1e-6
    numer=np.array([(f(b+row*eps)-f(b-row*eps))/(2*eps) for row in np.eye(3)])
    np.testing.assert_allclose(grad,numer,rtol=1e-8)


def external(segments,ids,eta,cut,reference):
    refidx={j:i for i,j in enumerate(ids)};per={};masks=[];unique=[];words=[];groups=[];positives=[]
    for s in segments:
        local=np.array([refidx[j] for j in s.ids]);logit=eta[local];z=s.masks;v=z.min(0);unc=z.max(0)^v;p=np.exp(-np.logaddexp(0.,-logit))
        row=per.setdefault(s.group,np.zeros(6))
        nll=float(np.logaddexp(0.,logit).sum()-np.logaddexp.reduce(z@logit))
        row[0]+=nll;row[1]+=len(logit)
        row[2]+=(np.logaddexp(0.,logit[~unc])-v[~unc]*logit[~unc]).sum();row[3]+=(~unc).sum()
        if len(z)>1:row[4]+=nll;row[5]+=len(logit)
        masks.append((local,z));words.extend(logit);groups.extend([s.group]*len(local));unique.extend([len(z)==1]*len(local));positives.extend(v)
    names,idx,draws=grouped_distribution(sorted(per))
    metrics={};reps={};table=np.array([per[k] for k in sorted(per)])
    for j,k in enumerate(('all_observable_set_nll','identified_word_nll','ambiguous_observable_set_nll')):
        v=table[:,2*j];n=table[:,2*j+1];use=n>0
        if use.sum()<2:
            rr=np.full(len(draws),np.nan)
        elif not use.all():
            qgroups,_,rd=grouped_distribution(np.array(sorted(per))[use]);rr=rd@(v[use]/n[use])/len(qgroups)
        else:rr=draws@(v/n)/len(n)
        vec=v[use]/n[use];metrics[k]=with_interval(vec.mean(),rr);reps[k]=(vec,rr)
    gs=np.asarray(groups);u=np.array(unique);yy=np.array(positives,dtype=int)[u];p=np.exp(-np.logaddexp(0.,-np.asarray(words)[u]))
    g,ix,d=grouped_distribution(gs[u]); res,samp=evaluate_one(yy,np.ones_like(yy),p,cut,ix,d)
    metrics['unique_word_ap']=res['ap'];reps['unique_word_ap']=(None,samp['ap'])
    metrics['threshold_performance_on_all_words']=threshold_bounds(masks,np.exp(-np.logaddexp(0.,-eta))>cut,False)
    return metrics,reps,per


def run(directory,candidate=None,target='CharE'):
    checked_loss();destination=OUT/candidate if candidate else OUT
    if target!='CharE':destination=destination/target
    destination.mkdir(exist_ok=True,parents=True)
    active_protocol=PROTOCOL if candidate is None else Path('research/reading_validation/transition-partial-replication-protocol.md')
    if target!='CharE':active_protocol=Path('research/reading_validation/transition-reference-types-protocol.md')
    x,segments,mu,sd,receipt,excluded=material(directory,candidate,target)
    training=[s for s in segments if s.split=='train'];dev=[s for s in segments if s.split=='dev'];test=[s for s in segments if s.split=='test']
    negative=np.concatenate([s.ids[~s.masks.max(0)] for s in dev]);ids=np.concatenate([s.ids for s in test])
    weights={};events={};est={};draw={};drifts={};rawpred={};per={};fits={}
    for arm in ARMS:
        data=design(x,training,arm)
        start=weights['word_identified'] if arm=='set_marginal' else np.zeros(x.shape[1])
        w,fit=solve(data,start)
        if arm=='set_marginal':
            repeat,second=solve(data,np.zeros(x.shape[1]))
            delta=float(np.max(np.abs(x[ids]@(w-repeat))))
            if delta>1e-4 or abs(second['objective']-fit['objective'])>1e-8:
                raise ArithmeticError('Independent marginal starts failed to reproduce the readout')
            drifts={'maximum_external_logit_difference':delta,'objective_difference':second['objective']-fit['objective'],'refit_gradient':second['gradient']}
        cut=float(np.quantile(np.exp(-np.logaddexp(0.,-x[negative]@w)),.95,method='higher'))
        eta=x[ids]@w;weights[arm]=w;rawpred[arm]=eta
        fit['cutoff']=cut;fit['source_words']=len(data[1]);fit['latent_source_segments']=len(data[3]);fits[arm]=fit
        # Features and book outcomes fixed before inspecting any movie response fit or scores.
    for arm in ARMS:
        est[arm],draw[arm],per[arm]=external(test,ids,rawpred[arm],fits[arm]['cutoff'],None)
    contr={}
    for a,b in (('set_marginal','segment_unique'),('set_marginal','word_identified'),('word_identified','segment_unique')):
        d={}
        for k in draw[a]:
            v,r=draw[a][k];w,s=draw[b][k]
            d[k]=with_interval(est[a][k]['estimate']-est[b][k]['estimate'],r-s)
            if v is not None:d[k].update(movies_improved=int((v<w).sum()),evaluable_movies=len(v))
        contr[a+'_minus_'+b]=d
    pred=destination/'locked-forecasts.npz';coef=destination/'readouts.npz'
    np.savez_compressed(pred,**rawpred);np.savez_compressed(coef,mean=mu,sd=sd,**weights)
    dump(destination/'scores.json',{'protocol_sha256':sha(active_protocol),'parent_protocol_sha256':sha(PROTOCOL),'candidate':candidate or 'original_qwen',
         'target':target,'unlocatable_segments_by_fold':excluded,'feature_width':len(mu),'feature_sha256':receipt['feature_sha256'],
         'readout_sha256':sha(coef),'forecasts_sha256':sha(pred),'fits':fits,'initialization_check':drifts,
         'movie_scores':est,'paired_movie_differences':contr,'source_training_titles':len({s.group for s in training}),
         'movie_titles':len({s.group for s in test}),'reader_words':len(ids)})
    print({k:(q['all_observable_set_nll'],q['unique_word_ap']) for k,q in est.items()},flush=True)
    print(contr['set_marginal_minus_segment_unique']['all_observable_set_nll'],drifts,flush=True)


if __name__=='__main__':
    cli=ArgumentParser(description=__doc__);cli.add_argument('--features',type=Path,required=True)
    cli.add_argument('--candidate',choices=('qwen','lfm3','gemma4'));cli.add_argument('--target',choices=('CharE','RefE','CorefE'),default='CharE')
    args=cli.parse_args();run(args.features,args.candidate,args.target)
