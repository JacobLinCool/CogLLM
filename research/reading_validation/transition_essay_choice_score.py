"""Compare locked sentence-risk forecasts with a separate expert's blind revision choices."""
import json
from pathlib import Path
import re

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit, logsumexp
from scipy.stats import rankdata

from research.reading_validation.transition_essay_choice_data import DEST, FILES, PLAN, read
from research.reading_validation.transition_essay_rewrite import METHODS
from research.reading_validation.transition_score import dump, sha

NAMES=("directional_agreement","directional_logloss","three_way_nll","three_way_agreement")


def selection(value):
    response=re.sub(r"[\s_-]","",value.casefold())
    groups=({"1","model1","output1","system1","first"},{"0","tie","equal","both","draw"},{"2","model2","output2","system2","second"})
    idx=[i for i,s in enumerate(groups) if response in s]
    if len(idx)!=1:raise ValueError(f"Undocumented response convention {value!r}")
    return idx[0]


def uncertainty(unit,wb):
    return [float(unit),*np.quantile(wb,[.025,.975]).tolist()]


def choice_fit(slope,role,type_idx,response,training,essay,topic,weights=None):
    n=len(response);system=np.eye(3)[type_idx]*role[:,None]
    components=[np.ones(n),*system.T]
    if slope is not None: components.append(slope)
    design=np.stack(components,axis=1)
    a=design.shape[1];A=np.zeros((n,3,a+3));A[:,0,:a]=design/2;A[:,2,:a]=-design/2;A[:,1,a:]=np.eye(3)[type_idx]
    train=np.flatnonzero(training);X=A[train];y=response[train]
    if weights is None:
        weights=np.ones(len(train))
    w=weights/weights.sum()
    labels=np.eye(3)[y]
    def loss(theta):
        logits=X@theta;logp=logits-logsumexp(logits,axis=1,keepdims=True)
        reg=theta.copy();reg[0]=0
        value=-w@logp[np.arange(len(y)),y]+.005*(reg@reg)
        grad=np.einsum("n,nkf,nk->f",w,X,np.exp(logp)-labels)+.01*reg
        return float(value),grad
    f=minimize(loss,np.zeros(A.shape[-1]),method="BFGS",jac=True,options={"gtol":2e-9,"maxiter":200})
    if np.max(abs(f.jac))>1e-6:raise ArithmeticError("Expert conditional-choice calibration failed stationarity")
    pred=A[~training]@f.x;pred-=logsumexp(pred,axis=1,keepdims=True)
    return np.exp(pred),{"max_gradient":float(max(abs(f.jac))),"objective":float(f.fun),"weights":f.x.tolist()}


def main():
    seal=json.loads((DEST/"exposure-seal.json").read_text());lock=json.loads((DEST/"judgment-blind-forecast-lock.json").read_text())
    file=DEST/"judgment-blind-forecasts.npz"
    if sha(file)!=lock["predictions_sha256"] or lock["protocol_sha256"]!=sha(PLAN):raise ValueError("Expert alternatives not locked before preference scoring")
    archive=np.load(file);lookup={k:i for i,k in enumerate(archive["ids"])};blocks={k:read(k) for k in FILES}
    exposure={r["id"]:r for r in map(json.loads,Path(".cache/cogllm/reading-validation/essay-expert-choices-stimuli.jsonl").read_text().splitlines())}
    observations=[];full=[]
    for item in seal["items"].values():
        for mode,v in item["sources"].items():
            row=blocks[mode][v["row"]];a,b=v["variants"];chosen=selection(row["selection"])
            atext=exposure[a]["text"][exposure[a]["start"]:];btext=exposure[b]["text"][exposure[b]["start"]:]
            if row["model1"].split()!=atext.split() or row["model2"].split()!=btext.split():raise ValueError("Human outcomes refer to another wording")
            gpt=item["gpt_candidate"]
            role=(int(b==gpt)-int(a==gpt)) if gpt is not None else 0
            d=(item["topic"],item["essay"],mode,lookup[a],lookup[b],chosen,role,len(atext.split())-len(btext.split()))
            full.append(d)
            if not v["instruction"]:observations.append(d)
    human_wins=sum(int((row[5]==0 and row[6]>0) or (row[5]==2 and row[6]<0)) for row in full if row[2]=="human")
    if human_wins!=74:raise ValueError("Expert display order does not reproduce the published human-reference win count")
    def primary(obs):
        topic,essay,mode,a,b,y,role,dlen=map(np.array,zip(*obs,strict=True))
        a=a.astype(int);b=b.astype(int);y=y.astype(int);role=role.astype(float);dlen=dlen.astype(int)
        num=np.array([list(FILES).index(v) for v in mode]);di=(y!=1)&(a!=b);target=(y==0)
        delta={k:logit(archive[k][b])-logit(archive[k][a]) for k in METHODS}
        def summary(w):
            wd=w[:,di];den=wd.sum(axis=1)
            return {k:np.column_stack((wd@((v[di]>0)*target[di]+(v[di]<0)*(~target[di])+.5*(v[di]==0))/den,
                      wd@(np.logaddexp(0.,v[di])-target[di]*v[di])/den)) for k,v in delta.items()}
        unit=np.ones((1,len(y)));pt=summary(unit);rng=np.random.default_rng(1293)
        _,tidx=np.unique(topic,return_inverse=True);_,eidx=np.unique(essay,return_inverse=True)
        w=rng.exponential(size=(4000,tidx.max()+1))[:,tidx]*rng.exponential(size=(4000,eidx.max()+1))[:,eidx]
        boot=summary(w)
        def pair_est(a,b=None):
            A=pt[a][0]-(pt[b][0] if b else 0);B=boot[a]-(boot[b] if b else 0)
            return {n:uncertainty(x,v) for n,x,v in zip(NAMES[:2],A,B.T,strict=True)}
        methods={m:pair_est(m) for m in delta};contrasts={k:pair_est("native_joint",k) for k in delta if k!="native_joint"}
        for k,v in delta.items():
            methods[k]["per_comparison"]={}
            methods[k]["by_length"]={}
            for name,table,sub in [(n,"per_comparison",mode==n) for n in FILES]+[("same","by_length",dlen==0),("different","by_length",dlen!=0)]:
                use=di&sub
                if not use.any():continue
                val=(v[use]>0)*target[use]+(v[use]<0)*(~target[use])+.5*(v[use]==0)
                den=w[:,use].sum(axis=1)
                methods[k][table][name]={"directed_trials":int(use.sum()),"agreement":uncertainty(val.mean(),(w[:,use]@val)/den)}
        # Predict the actual three-way response after full leave-topic-out calibration.
        tracks={"provenance":None,"lexical_joint":delta["lexical_joint"],"native_frozen":delta["native_frozen"],"native_joint":delta["native_joint"]}
        learned={};stats={}
        for label,slope in tracks.items():
            p=np.zeros((len(y),3));logs={}
            for t in np.unique(topic):
                tr=topic!=t
                p[~tr],logs[t]=choice_fit(slope,role,num,y,tr,essay,topic)
            correct=p.argmax(1)==y;nll=-np.log(p[np.arange(len(y)),y]);matrix=np.column_stack((nll,correct))
            learned[label]=matrix;stats[label]={"source_fits":logs}
            for j,n in enumerate(NAMES[2:]):stats[label][n]=uncertainty(matrix[:,j].mean(),(w@matrix[:,j])/w.sum(1))
        calibrated={}
        for method in stats:
            if method=="native_joint":continue
            vv=learned["native_joint"]-learned[method]
            calibrated[method]={n:uncertainty(vv[:,j].mean(),(w@vv[:,j])/w.sum(1)) for j,n in enumerate(NAMES[2:])}
        # Conditional discrimination under randomized display order, orienting every vote toward the same source system.
        directional={}
        for method,raw in delta.items():
            cells={}
            for m in FILES:
                use=di&(mode==m)&(role!=0)
                sg=role[use];v=raw[use]*sg;yy=np.where(sg>0,target[use],~target[use])
                if len(set(yy))!=2:continue
                head=v[yy,None];tail=v[None,~yy];auc=((head>tail)+.5*(head==tail))
                w1=w[:,use][:,yy];w0=w[:,use][:,~yy];draws=((w1@auc)*w0).sum(1)/(w1.sum(1)*w0.sum(1))
                cells[m]={"directional":int(use.sum()),"challenger_preferred":int(yy.sum()),"pair_auc":uncertainty(auc.mean(),draws)}
            directional[method]=cells
        return {"judgments":len(y),"essays":len(set(essay)),"topics":len(set(topic)),"directional":int(di.sum()),
                "output1_tie_output2":np.bincount(y,minlength=3).tolist(),"unidentified_roles":int((role==0).sum()),
                "identical_wordings":int((a==b).sum()),"source_blind":methods,"paired_native_joint_minus":contrasts,
                "calibrated_three_way":stats,"paired_three_way_native_joint_minus":calibrated,"directional_by_system_type":directional}
    sel=primary(observations);all_records=primary(full)
    path=DEST/"scores.json";dump(path,{"protocol_sha256":sha(PLAN),"forecast_lock_sha256":sha(DEST/"judgment-blind-forecast-lock.json"),
        "experts":1,"original_items":len(seal["items"]),"eligible_actual_sentence_pairs":sel,"all_released_pairs":all_records})
    print(sel["output1_tie_output2"],sel["source_blind"]["native_joint"],sel["paired_native_joint_minus"]["lexical_joint"],
          sel["paired_three_way_native_joint_minus"],flush=True)


if __name__=="__main__":main()
