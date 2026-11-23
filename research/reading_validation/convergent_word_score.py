"""Compare independent Spanish word ERPs with English naturalistic reading outcomes."""
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
import csv
import numpy as np
from scipy.stats import rankdata
from research.backbone_transfer.extract import digest,read_json,stream_rows,write_json
from research.reading_validation.convergent_word_prepare import OUT,ROOT,PLAN
from research.reading_validation.spanish_clue_data import ROOT as SPANISH
from research.reading_validation.meaning_score import folds

NAMES=('N400_negativity','cloze_surprisal','rated_unpredictability')
REG=.035


def fit(X,Y,w,rank=None):
    if not np.isfinite(X).all() or not np.isfinite(Y).all():raise ValueError('Unobserved value passed to dense conditional likelihood')
    w=w/w.sum();mu=w@X;sd=np.maximum(np.sqrt(w@(X-mu)**2),.025)
    center=w@Y;spread=np.maximum(np.sqrt(w@(Y-center)**2),1e-4);Z=(X-mu)/sd;T=(Y-center)/spread
    cov=(Z*w[:,None]).T@Z+REG*np.eye(Z.shape[1]);cross=(Z*w[:,None]).T@T
    b=np.linalg.solve(cov,cross);err=float(abs(cov@b-cross).max())
    if err>1e-9:raise ArithmeticError('Ridge normal equations not satisfied')
    if rank is not None:
        eig,U=np.linalg.eigh(cross.T@b);V=U[:,-rank:];b=b@V@V.T
    beta=b*spread[None,:]/sd[:,None]
    return dict(beta=beta.tolist(),bias=(center-mu@beta).tolist(),stationarity=err,rank=rank,examples=len(X))


def predict(X,model):return X@np.asarray(model['beta'])+np.asarray(model['bias'])


def assemble(folder):
    seal=read_json(OUT/'input-seal.json');rec=read_json(folder/'receipt.json');p=folder/'word-states.npz'
    if rec['input']!=seal['stimuli'] or digest(p)!=rec['word_states'] or digest(ROOT/'outcomes.csv')!=seal['english_endpoint_table']:raise ValueError('Unverified native inputs or outcome source')
    maps=list(stream_rows(ROOT/'map.jsonl'));docs={r['id']:r for r in stream_rows(ROOT/'stimuli.jsonl')};en={int(r['item_id']):r for r in csv.DictReader((ROOT/'outcomes.csv').open())}
    eitems={int(r['item_id']):r for r in csv.DictReader((ROOT/'items.csv').open())}
    spanish={r['id']:j for j,r in enumerate(v for v in maps if v['language']=='Spanish')};wave=SPANISH/'wave';ne=np.zeros((len(spanish),21));sy=np.zeros((len(spanish),21,2))
    for key in stream_rows(wave/'geometry.jsonl'):
        with np.load(wave/(key['id']+'.npz')) as zi:
            ix=[spanish[k] for k in zi['ids']];Y=zi['values'];valid=zi['valid']
            if len(set(ix))!=len(ix):raise ValueError('Repeated complete brain recording exposure')
            ne[ix]+=valid;sy[ix]+=np.where(valid[:,:,None],Y,0.)
    X=[];Y=[];C=[];ES=[];EY=[];ctx=[];pub=[];surface=[];coverage=defaultdict(int)
    with np.load(p) as z:
        for r in maps:
            k=r['id'];A=z[k].astype(float);source=docs[k]
            if r['language']=='Spanish':
                j=spanish[k];mask=(ne[j]>=3)&(np.arange(21)>0);X.extend(A[mask]);Y.extend(sy[j,mask]/ne[j,mask,None]);C.extend(ne[j,mask]);continue
            for i,item in enumerate(r['items']):
                if item is None:continue
                rr=en[item];q=eitems[item];gold=rr['item']+' '+rr['word']
                if q['sent_id']!=rr['sent_id'] or q['sent_id']!=r['article'] or source['text'][:source['words'][i][1]]!=q['sentence']+' '+q['word']:raise ValueError('Unverified English stimulus word')
                if not rr['sentence'].startswith(gold):raise ValueError('Released EEG context does not correspond to a possible original prefix')
                same=source['text'].startswith(gold) and len(gold)==source['words'][i][1];coverage['all']+=1
                if not same:coverage['prefix_mismatch']+=1;continue
                cohort=[-float(rr['N400']),float(rr['cloze_s']),-float(rr['rating_mean'])]
                if any(not np.isfinite(v) for v in cohort):raise ValueError('Missingness requires endpoint-specific fit')
                ES.append(A[i]);EY.append(cohort);ctx.append((r['article'],item,i,int(rr['is_start_end'])))
                pub.append([float(rr[c]) for c in ('s_GPT2','s_GPT2_medium','s_GPT2_large','s_GPT2_xl','s_GPTNeo_125M','s_GPTNeo','s_GPTNeo_2.7B')])
                surface.append(float(rr['Subtlex_log10']) if rr['Subtlex_log10'] else np.nan)
    return map(np.asarray,(X,Y,C,ES,EY,ctx,pub,surface)),dict(coverage)


def groups(ids):
    return np.unique(ids,return_inverse=True)[1]


def weighted_ranks(x,weight):
    values,code=np.unique(x,return_inverse=True);hist=np.bincount(code,weights=weight,minlength=len(values));return (np.cumsum(hist)-.5*hist)[code]


def corr(x,y,w=None):
    w=np.ones(len(y)) if w is None else w
    if len(y)<3 or min(len(set(x)),len(set(y)))<2:return np.nan
    a=weighted_ranks(x,w);b=weighted_ranks(y,w);w=w/w.sum();a-=w@a;b-=w@b
    d=np.sqrt((w@(a*a))*(w@(b*b)));return float(w@(a*b)/d) if d>0 else np.nan


def correlations(data,Y,sent,draws=1500):
    families,labels=np.unique(sent,return_inverse=True);rng=np.random.default_rng(941);boot=rng.multinomial(len(families),np.full(len(families),1/len(families)),size=draws);counts=np.bincount(labels);model={}
    for name,Q in data.items():
        result=[];samples=[];full=[]
        for j in range(Y.shape[1]):
            x=Q[:,j] if Q.ndim==2 else Q;y=Y[:,j]
            ok=np.isfinite(x)&np.isfinite(y);v=x[ok];u=y[ok];g=labels[ok];point=corr(v,u)
            rr=[]
            for w in boot:rr.append(corr(v,u,w[g]))
            rr=np.array(rr);samples.append(rr);full.append(point)
            within=[corr(v[g==k],u[g==k]) for k in np.unique(g) if (g==k).sum()>=4];within=np.asarray(within);within=within[np.isfinite(within)]
            center=lambda f:f-np.divide(np.bincount(g,weights=f,minlength=len(families)),np.maximum(np.bincount(g,minlength=len(families)),1))[g]
            result.append({'rho':point,'sentence_ci':np.quantile(rr,[.025,.975]).tolist(),'within_sentence_median_rho':float(np.median(within)) if len(within) else None,'demeaned_rho':corr(center(v),center(u)),'words':int(ok.sum())})
        model[name]=(result,np.stack(samples,1),np.array(full))
    return model


def table_comparisons(values,pairs):
    out={}
    for a,b in pairs:
        av,ab,ap=values[a];bv,bb,bp=values[b];change=ap-bp;ci=np.quantile(ab-bb,[.025,.975],axis=0).T
        out[a+'__over__'+b]=[{ 'rho_gain':float(x),'paired_sentence_ci':k.tolist()} for x,k in zip(change,ci)]
    return out


def main(folder):
    data,coverage=assemble(folder);X,Y,w,E,H,c,older,freq=data;sents=c[:,0];is_last=c[:,3].astype(int)>0;ids=c[:,1].astype(int);W=np.ones(len(E))
    p=read_json(OUT/'input-seal.json')
    if p['plan']!=digest(PLAN) or coverage.get('prefix_mismatch')!=180:raise ValueError('Registered source or prefix mask changed')
    cols={'surface':slice(0,5),'qwen_surface':slice(0,6),'context':slice(None)};m={};transfer={};raw={}
    for n,col in cols.items():
        m[n]=fit(X[:,col],Y,w);pred=predict(E[:,col],m[n]);transfer[n]=pred[:,0];transfer[n+'_early_control']=pred[:,1]
    transfer.update({'qwen_surprisal':E[:,5],'pending_readout':E[:,6]})
    oldnames=['GPT2','GPT2_medium','GPT2_large','GPT2_xl','GPTNeo_125M','GPTNeo','GPTNeo_2.7B']
    for j,q in enumerate(oldnames):transfer[q]=older[:,j]
    tv=correlations(transfer,H,sents)
    best={NAMES[j]:max(oldnames,key=lambda n:tv[n][0][j]['rho']) for j in range(3)}
    contrasts=[('context','surface'),('context','qwen_surface'),('context','qwen_surprisal'),('context','GPT2_xl'),('context','context_early_control')]+[('context',n) for n in set(best.values()) if n!='GPT2_xl']
    suff=lambda v:{k:x[0] for k,x in v.items()}
    external={'models':suff(tv),'increments':table_comparisons(tv,contrasts),'best_published_per_endpoint':best,'spanish_coefficients':m}
    # Missing published word frequency receives the training-fold median and its own indicator.
    designs={'surface':E[:,:5],'qwen_surface':E[:,:6],'gpt2_xl_surface':np.c_[E[:,:5],older[:,3]],'context':E,'human_offline':np.c_[E[:,:5],freq,np.isnan(freq),H[:,1:]]}
    splits=folds(sents,'UCL 2026 reading sentence:');fold=np.array([splits[s] for s in sents]);out={};fits=[]
    for name,Z in designs.items():
        for policy in ('separate','shared_rank2','joint_full'):
            if name=='human_offline' and policy!='separate':continue
            Q=np.full_like(H,np.nan)
            for v in range(5):
                tr=fold!=v;te=~tr;z=Z.copy()
                if name=='human_offline':
                    med=np.nanmedian(z[tr,5]);z[:,5]=np.where(np.isnan(z[:,5]),med,z[:,5]);F=fit(z[tr],H[tr,:1],W[tr]);Q[te,0]=predict(z[te],F)[:,0]
                elif policy=='separate':
                    for j in range(3):
                        F=fit(z[tr],H[tr,j:j+1],W[tr]);Q[te,j]=predict(z[te],F)[:,0]
                else:
                    F=fit(z[tr],H[tr],W[tr],2 if policy=='shared_rank2' else None);Q[te]=predict(z[te],F)
                fits.append({'fold':v,'arm':name,'shared':policy,'train_words':int(tr.sum()),'test_words':int(te.sum()),'normal_equation_error':F['stationarity']})
            out[name+'_'+policy]=Q
    for key in designs:
        if key!='human_offline' and np.max(np.abs(out[key+'_separate']-out[key+'_joint_full']))>1e-8:raise ArithmeticError('Unlimited three-output ridge should coincide with three separate least-squares heads')
    full={k:v for k,v in out.items() if k!='human_offline_separate' and not k.endswith('_joint_full')};cv=correlations(full,H,sents)
    comp=[('context_separate','surface_separate'),('context_separate','qwen_surface_separate'),('context_separate','gpt2_xl_surface_separate'),('context_shared_rank2','context_separate'),('context_shared_rank2','qwen_surface_shared_rank2')]
    # Proper error reporting in each response's original scale, compared within identical held word positions.
    pairs={}
    for a,b in comp:
        err=((full[b]-H)**2-(full[a]-H)**2);g=np.unique(sents);means=np.stack([err[sents==k].mean(0) for k in g]);rng=np.random.default_rng(658);ii=rng.integers(len(g),size=(4000,len(g)));q=np.quantile(means[ii].mean(1),[.025,.975],axis=0).T
        pairs[a+'__over__'+b]=[{'mean_squared_error_reduction':float(d),'sentence_ci':u.tolist()} for d,u in zip(means.mean(0),q)]
    offline={a:np.squeeze(b[:,0]) for a,b in full.items() if a in ('context_separate','qwen_surface_separate')};offline['human_offline']=out['human_offline_separate'][:,0]
    oo=correlations(offline,H[:,:1],sents,1000);cl={k:np.sqrt(((v-H)**2).mean(0)).tolist() for k,v in full.items()}
    sensitivity=correlations({k:transfer[k][~is_last] for k in ('context','qwen_surprisal','GPT2_xl')},H[~is_last],sents[~is_last],1000)
    write_json(OUT/'scores.json',{'source':digest(folder/'receipt.json'),'registered_geometry':digest(OUT/'geometry-registration.json'),'program':digest(Path(__file__)),'spanish_fit_word_cells':len(X),'spanish_trials_used':int(w.sum()),'english_sentences':len(set(sents)),'english_exact_noninitial_words':len(E),'coverage':coverage,'outcomes':NAMES,'external_spanish_no_updates':external,'english_crossvalidated':{'rho':suff(cv),'paired_rho':table_comparisons(cv,comp),'rmse':cl,'paired_MSE':pairs,'analytic_fits':fits,'offline_N400':suff(oo)},'drop_sentence_final':suff(sensitivity)})
    print('Completed independent English word-level transfer and joint English outcomes',flush=True)

if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);main(p.parse_args().folder)
