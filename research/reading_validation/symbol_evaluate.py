"""Locked paired attribution and article-only first-definition events with paper-level uncertainty."""
from argparse import ArgumentParser
from bisect import bisect_left
from collections import Counter,defaultdict
from pathlib import Path
import json
import numpy as np
from scipy.special import expit
from research.backbone_transfer.extract import digest,read_json,write_json
from research.reading_validation.symbol_features import canonical
from research.reading_validation.symbol_model import Bank,anchor,case_geometry,collect,fit
from research.reading_validation.symbol_geometry import OUT,competing

ARMS=('position','surface','native','nearest','first','literal')
THRESHOLDS=np.unique(np.r_[np.linspace(.05,.95,19),.975,.99,.995,.999,.9999]);


def probabilities(page,m,t,weights):
    a,b=page.token_span(m.start,m.end);surf,sem=page.features(m,t);features=np.column_stack((surf,sem));native=expit(features@weights[0]+weights[1]);duration=np.maximum(t-(b-1),0)
    nearest=1/(1+duration/10);first=1/(1+(a+1)/10)*np.exp(-np.abs(surf[:,1])/2);literal=np.exp(-np.expm1(surf[:,10])/5)*(.35+.65*surf[:,19]);return np.c_[native,nearest,first,literal]


def parameters(bank):
    dataset=collect(bank,'fit');return {a:fit(dataset,a) for a in ARMS[:3]}


def weights(params):return np.array([params[a]['coefficients'] for a in ARMS[:3]],dtype='float32').T,np.array([params[a]['intercept'] for a in ARMS[:3]],dtype='float32')


def pair_ranks(page,doc,w):
    vals=[];eligible_geometry=competing(doc['issues']);ties=0
    for first,second in eligible_geometry:
        A=anchor(page,first['sources'][0]);B=anchor(page,second['sources'][0]);out=[];compatible=True
        for target,other,interval in [(A,B,first['definitions'][0]),(B,A,second['definitions'][0])]:
            ids=page.interval(interval);ts=ids[np.maximum(page.end[ids],0)>=max(target.committed,other.committed)]
            if len(ts)==0:compatible=False;break
            z=probabilities(page,target,ts,w).mean(0)-probabilities(page,other,ts,w).mean(0);out.append((z>0).astype(float)+.5*(z==0))
        if compatible:vals.append(np.asarray(out))
    return np.asarray(vals),len(eligible_geometry)


def marks(page,w):
    # Only already-present proposals can release. The next source competes at the same token after closure.
    m=page.mentions;forecast=np.full((len(m),page.n,len(ARMS)),-1,dtype='float32');names=[]
    for j,x in enumerate(m):
        names.append(canonical(page.text[x.start:x.end]));first=bisect_left(page.end,x.committed);t=np.arange(first,page.n)
        if len(t):forecast[j,t]=probabilities(page,x,t,w)
    return forecast,names


def decode(forecast,names,limit):
    F=forecast.copy();out=[];closed=set();ident={k:np.asarray([i for i,name in enumerate(names) if name==k],dtype=int) for k in set(names)}
    for t in np.flatnonzero(F.max(0,initial=-1)>=limit):
        i=int(F[:,t].argmax());score=float(F[i,t]);
        if score<limit:continue
        out.append((i,int(t),score));closed.add(names[i]);F[ident[names[i]],t:]=-1
    return out


def score_events(page,doc,marks):
    if not hasattr(page,'event_geometry'):
        target=case_geometry(page,doc);gold={(m.start,m.end):set(map(int,t)) for m,d,t in target};proposals={(m.start,m.end):m.committed for m in page.mentions};page.event_geometry=(gold,proposals,competing(doc['issues']))
    g,proposals,comp=page.event_geometry;observed={};name_closing={};hits=0
    for i,t,value in marks:
        m=page.mentions[i];k=(m.start,m.end);observed[k]=t;name_closing[canonical(page.text[m.start:m.end])]=t
        if k in g and t in g[k]:hits+=1
    dual=0;kept=0;both=0;eligible=0
    for a,b in comp:
        if a['definitions'][0][0]>b['definitions'][0][0]:a,b=b,a
        ka=tuple(a['sources'][0]);kb=tuple(b['sources'][0]);
        if ka not in g or kb not in g:continue
        eligible+=1;t=observed.get(ka,-1);other=observed.get(kb,page.n+1);correct=t in g[ka]
        n_a=canonical(page.text[ka[0]:ka[1]]);n_b=canonical(page.text[kb[0]:kb[1]])
        survives=kb in proposals and n_a!=n_b and proposals[kb]<=page.end[t] and name_closing.get(n_b,page.n+1)>t if t>=0 else False
        dual+=int(correct);kept+=int(correct and survives);both+=int(correct and survives and other in g[kb]);
    return np.array([hits,len(marks),len(g),len(proposals.keys()&g.keys()),dual,kept,both,eligible],int)


def metric(v):
    h,p,n,covered,first,kept,both,pairs=np.moveaxis(np.asarray(v,dtype=float),-1,0)
    divide=lambda x,y:np.divide(x,y,out=np.full(np.broadcast_shapes(np.shape(x),np.shape(y)),np.nan),where=y>0)
    return {'precision':divide(h,p),'recall':divide(h,n),'F1':divide(2*h,p+n),'source_proposal_coverage':divide(covered,n),'correct_first_release':divide(first,pairs),'retain_other_at_correct_first':divide(kept,pairs),'both_sources_and_timings':divide(both,pairs)}


def evaluate(bank,split,params,thresholds=None,limit=None):
    w=weights(params);P=defaultdict(list);J=defaultdict(list);sweep=np.zeros((len(ARMS),len(THRESHOLDS),8),int);prediction=[]
    for i,(pg,doc) in enumerate(bank.pages(split,limit)):
        pairs,eligible=pair_ranks(pg,doc,w);forecast,names=marks(pg,w);pp=doc['paper']
        if len(pairs):P[pp].append(pairs)
        if thresholds is None:
            # Calibration only: emit one event for each source spelling and choose thresholds before opening the test papers.
            for ai in range(len(ARMS)):
                for ti,c in enumerate(THRESHOLDS):sweep[ai,ti]+=score_events(pg,doc,decode(forecast[:,:,ai],names,c))
        else:
            table=[];output={}
            for ai,arm in enumerate(ARMS):
                records=decode(forecast[:,:,ai],names,thresholds[arm]);table.append(score_events(pg,doc,records));output[arm]=[[pg.mentions[j].start,pg.mentions[j].end,t,round(s,7)] for j,t,s in records]
            J[pp].append(table);prediction.append(dict(id=doc['id'],paper=pp,events=output,paired= pairs.tolist(),eligible=eligible))
        if i and i%200==0:print(f'{split} completed {i} papers or paragraphs',flush=True)
    if thresholds is None:
        selected={}
        for ai,arm in enumerate(ARMS):
            m=metric(sweep[ai])['F1'];best=np.flatnonzero(m==np.nanmax(m))[-1];selected[arm]=float(THRESHOLDS[best])
        return selected,sweep
    return J,P,prediction


def summary(J,P):
    papers=sorted(set(J)|set(P));rng=np.random.default_rng(1763);M=np.array([np.sum(J[g],axis=0) for g in papers]);n=len(papers);sample=rng.integers(n,size=(4000,n));pooled=M.sum(0);draws=M[sample].sum(1);paired=[];count=[]
    for paper in papers:
        if P[paper]:shape=np.concatenate(P[paper]);paired.append(shape.mean(1).sum(0));count.append(len(shape))
        else:paired.append(np.zeros(len(ARMS)));count.append(0)
    paired=np.asarray(paired);count=np.array(count);total=int(count.sum());T=paired.sum(0)/total if total else np.full(len(ARMS),np.nan);den=count[sample].sum(1);D=paired[sample].sum(1)/np.maximum(den[:,None],1);D[den==0]=np.nan
    def entry(point,ds):
        q=ds[np.isfinite(ds)];return {'estimate':float(point) if np.isfinite(point) else None,'paper_bootstrap_95':np.quantile(q,[.025,.975]).tolist() if len(q) else None}
    allm=metric(pooled);alld=metric(draws);response={}
    for ai,arm in enumerate(ARMS):
        response[arm]={'counts':list(map(int,pooled[ai])),'same_evidence_identity':entry(T[ai],D[:,ai]),**{k:entry(allm[k][ai],alld[k][:,ai]) for k in allm}}
    return {'papers':n,'paired_sources':total,'papers_with_competition':int(np.count_nonzero(count)), 'counts_order':['hits','alerts','annotated_forward','proposed_gold','first_properly_released','other_retained','both_correct','dual_pairs'], 'models':response,
            'native_minus':{a:{'identity':entry(T[2]-T[j],D[:,2]-D[:,j]),'F1':entry(allm['F1'][2]-allm['F1'][j],alld['F1'][:,2]-alld['F1'][:,j])} for j,a in enumerate(ARMS) if a!='native'}}


def main(args):
    bank=Bank(args.folder,args.synthetic)
    if args.synthetic:
        pg,doc=next(bank.pages('fit'));toy=collect(bank,'fit',35);params={a:fit(toy,a) for a in ARMS[:3]};thresholds,sweep=evaluate(bank,'calibration',params,limit=3);J,P,p=evaluate(bank,'calibration',params,thresholds,limit=3);print(summary(J,P)['papers']);return
    if args.action=='fit':
        params=parameters(bank);write_json(OUT/'coefficients.json',{'protocol':digest(Path(__file__).with_name('symbol-protocol.md')),'native_code':digest(Path(__file__).with_name('symbol_features.py')),'fit_code':digest(Path(__file__).with_name('symbol_model.py')),'score_code':digest(Path(__file__)),'feature_receipt':digest(Path(args.folder)/'receipt.json'),'parameters':params});return
    record=read_json(OUT/'coefficients.json');params=record['parameters']
    if record['native_code']!=digest(Path(__file__).with_name('symbol_features.py')) or record['fit_code']!=digest(Path(__file__).with_name('symbol_model.py')) or record['score_code']!=digest(Path(__file__)):raise ValueError('Supervisor, decoder, or feature design has changed')
    if args.action=='calibrate':
        thresholds,sweep=evaluate(bank,'calibration',params);write_json(OUT/'thresholds.json',dict(coefficients=digest(OUT/'coefficients.json'),thresholds=thresholds,development_counts=sweep.tolist(),grid=THRESHOLDS.tolist()));return
    calibration=read_json(OUT/'thresholds.json');
    if calibration['coefficients']!=digest(OUT/'coefficients.json'):raise ValueError('Different validation head')
    for split in args.splits:
        J,P,preds=evaluate(bank,split,params,calibration['thresholds']);target=bank.folder.parent/(split+'-source-events.jsonl')
        with target.open('x') as f:
            for row in preds:f.write(json.dumps(row,separators=(',',':'))+'\n')
        out=summary(J,P);write_json(OUT/(split+'-scores.json'),{**out,'calibration':digest(OUT/'thresholds.json'),'forecasts':digest(target)})


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('action',choices=('fit','calibrate','score'));p.add_argument('folder',type=Path);p.add_argument('--synthetic',action='store_true');p.add_argument('--splits',nargs='+',choices=('held','published_dev','external'),default=['held','published_dev','external']);main(p.parse_args())
