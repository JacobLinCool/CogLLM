"""Fit joint article-only openings and a source-bound, correlated closure process."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
import torch
from torch.nn import functional as F

from research.reading_validation.memory_data import sha
from research.reading_validation.narrative_course_score import count_outcomes, identity
from research.reading_validation.narrative_survival_fit import pages
from research.reading_validation.narrative_survival_score import observation_loss, uncued
from research.reading_validation.stock_process import Design, norm

OUT = Path("research/results/reading-validation-narrative-discharge")
PLAN = Path(__file__).with_name("narrative-discharge-protocol.md")
TEAMS = ("AA", "JE")
MODES = ("independent", "shared_clue")


def tensor(x, dtype=torch.float64):
    return torch.as_tensor(np.ascontiguousarray(x), dtype=dtype)


def change_in_setting(page):
    state = page.native
    columns = []
    for scale in (1, 4, 16):
        memory = np.zeros(state.shape[1])
        change = []
        for current in state:
            similarity = (current @ memory) / max(1e-8, np.linalg.norm(current)*np.linalg.norm(memory))
            change.append(1-similarity)
            memory += (current-memory)/scale
        columns.append(change)
    return np.column_stack(columns)


class Text:
    def __init__(self, page, geometry, center, scale):
        self.n = page.n
        self.row, self.col = torch.tril_indices(self.n,self.n,-1)
        self.lower = torch.tril(torch.ones(self.n,self.n,dtype=torch.bool))
        raw = np.column_stack((geometry.local(page),change_in_setting(page)))
        self.local = tensor(np.column_stack((np.ones(self.n),(raw-center)/scale)))
        pair = (geometry.pair(page)-geometry.hmode[0])/geometry.hmode[1]
        self.pair = tensor(np.column_stack((np.ones(len(pair)),pair)))

    def process(self, vector, mode, team):
        d = self.local.shape[1]; k = self.pair.shape[1]
        birth, site, origin, current, observer = torch.split(vector,[d,d,k,d,2])
        team_code = 1 if team == "AA" else -1
        log_birth = self.local@birth + team_code*observer[0]
        if log_birth.detach().max() > 60:
            raise ArithmeticError("Trial opening rate exceeded finite probability scale.")
        amount = log_birth.exp()
        logits = self.pair@origin + (self.local@current)[self.row] + team_code*observer[1]
        q = logits.sigmoid()
        gate = (self.local@site).sigmoid() if mode=="shared_clue" else torch.ones_like(amount)
        hazard = torch.zeros_like(self.lower,dtype=torch.float64)
        hazard[self.row,self.col] = gate[self.row]*q
        lower_survival = torch.log1p(-hazard.clamp(max=1-1e-12)).cumsum(0).exp()*self.lower
        stock = lower_survival@amount
        before = F.pad(lower_survival[:-1],(0,0,1,0))
        clearing = before*hazard*amount[None,:]
        return amount,gate,logits,hazard,stock,clearing


class TeamResponse:
    def __init__(self, p, name):
        records = [r for r in p.records if r.team == name]
        self.name = name
        self.birth = tensor(np.bincount([r.source for r in records],minlength=p.n))
        self.question_count = len(records)
        status = p.story.teams[name]
        self.low, self.high = tensor((status==1).sum(1)), tensor((status!=0).sum(1))
        self.intervals = []
        coordinates, lines, y = [], [], []
        for r in records:
            for line in range(r.source+1, r.definite+1):
                lines.append(line); coordinates.append(line*(line-1)//2+r.source); y.append(0)
            if r.upper == r.definite+1:
                line = r.upper; lines.append(line); coordinates.append(line*(line-1)//2+r.source); y.append(1)
            elif r.upper is not None:
                self.intervals.append((r.source,r.definite+1,r.upper+1))
        self.index = tensor(coordinates,torch.long)
        self.lines = tensor(lines,torch.long)
        self.y = tensor(y)
        self.active = torch.bincount(self.lines,minlength=p.n)>0
        self.confirmed = torch.bincount(self.lines,weights=self.y,minlength=p.n)>0

    def loss(self, out, mode, birth_scale, count_scale):
        amount,gate,raw,hazard,stock,_ = out
        birth_loss = (amount-self.birth*amount.log()).mean()/birth_scale
        individual = -F.binary_cross_entropy_with_logits(raw[self.index],self.y,reduction="none")
        line_ll = torch.zeros_like(amount).index_add(0,self.lines,individual)
        if mode=="shared_clue":
            log_gate = gate.clamp(1e-12,1-1e-12).log()
            no_gate = torch.log1p(-gate.clamp(max=1-1e-12))
            likelihood = torch.where(self.confirmed,log_gate+line_ll,torch.logaddexp(no_gate,log_gate+line_ll))
        else:
            likelihood = line_ll
        lifetime = -likelihood[self.active].sum()
        for start,a,b in self.intervals:
            survived = torch.log1p(-hazard[a:b,start].clamp(max=1-1e-12)).sum()
            lifetime -= (-torch.expm1(survived)).clamp_min(1e-20).log()
        excess = (stock-self.high).clamp_min(0) + (stock-self.low).clamp_max(0)
        return birth_loss + lifetime/(4*self.question_count) + .5*excess.square().mean()/count_scale**2


def fit(training, mode):
    geometry = Design(training)
    centers = np.vstack([np.column_stack((geometry.local(p),change_in_setting(p))) for p in training])
    center,scale = norm(centers)
    make = lambda p: Text(p,geometry,center,scale)
    documents = [(make(p),[TeamResponse(p,k) for k in p.story.teams]) for p in training]
    count_scale = np.mean([np.mean([(.5*(h.low+h.high)).mean().item() for h in team]) for _,team in documents])
    d,k=documents[0][0].local.shape[1],documents[0][0].pair.shape[1]
    start=np.zeros(3*d+k+2);start[0]=np.log(geometry.birth_scale);start[d]=-2. if mode=="shared_clue" else 0.
    start[2*d]=-1.2 if mode=="shared_clue" else -3.2
    prior=tensor(start)
    shrink=np.full(len(start),.055);shrink[[0,d,2*d]]=.005
    shrink[-2:]=.03
    penalty=tensor(shrink)
    def objective(raw):
        v=tensor(raw).requires_grad_()
        loss=sum(sum(h.loss(doc.process(v,mode,h.name),mode,geometry.birth_scale,count_scale)
                     for h in humans)/len(humans) for doc,humans in documents)/len(documents)
        loss=loss+.5*((v-prior).square()*penalty).sum()
        gradient,=torch.autograd.grad(loss,v)
        return loss.item(),gradient.detach().numpy()
    direction=np.sin(np.arange(len(start))+.12);direction/=np.linalg.norm(direction)
    central=(objective(start+1e-5*direction)[0]-objective(start-1e-5*direction)[0])/2e-5
    analytic=objective(start)[1]@direction
    if abs(central-analytic)>2e-5:raise ArithmeticError("Closure-burst observation gradient disagrees.")
    answer=minimize(objective,start,jac=True,method="L-BFGS-B",options={"maxiter":550,"ftol":1e-13,"gtol":1e-7})
    grad=float(abs(answer.jac).max())
    if not answer.success or grad>1e-5:raise ArithmeticError(f"Clue observation did not converge: {answer.message}, {grad}")
    return make,tensor(answer.x),dict(iterations=answer.nit,gradient=grad,objective=answer.fun,
        numerical_check=abs(analytic-central),parameters=answer.x.tolist(),count_scale=count_scale)


def group_forecast(value):
    amount,gate,_,hazard,stock,mass = value
    arrays={k:v.detach().numpy() for k,v in dict(openings=amount,gate=gate,hazard=hazard,stock=stock,mass=mass).items()}
    if not np.allclose(np.diff(arrays["stock"],prepend=0), arrays["openings"]-arrays["mass"].sum(1),atol=1e-8):
        raise ArithmeticError("Group clue failed source-specific conservation.")
    return arrays


def predict(archive,old):
    bank=pages(archive,old);folder=OUT/"forecast";folder.mkdir(parents=True,exist_ok=True)
    if (folder/"seal.json").exists():raise FileExistsError("Narrative burst forecast already complete.")
    files={}
    for name,p in bank.items():
        path=folder/(name+".npz");receipt=folder/(name+".json")
        if receipt.exists():
            info=json.loads(receipt.read_bytes())
            if info["protocol"]!=sha(PLAN) or info["hash"]!=sha(path):raise ValueError("Incompatible partial forecast.")
            files[name]=info["hash"];continue
        training=[v for key,v in bank.items() if key!=name]
        arrays={};diagnostics={}
        for mode in MODES:
            make,v,diag=fit(training,mode);doc=make(p);diagnostics[mode]=diag
            for t in TEAMS:
                for k,x in group_forecast(doc.process(v,mode,t)).items():arrays[".".join((mode,t,k))]=x
            print("Froze",name,mode,diag["iterations"],diag["gradient"],flush=True)
        np.savez_compressed(path,**arrays);files[name]=sha(path)
        receipt.write_text(json.dumps(dict(protocol=sha(PLAN),hash=files[name],fits=diagnostics),indent=2)+"\n")
    (folder/"seal.json").write_text(json.dumps(dict(protocol=sha(PLAN),archive=sha(archive/"receipt.json"),
                                     old=sha(old/"receipt.json"),files=files),indent=2)+"\n")


def score(archive,old):
    bank=pages(archive,old);folder=OUT/"forecast";lock=json.loads((folder/"seal.json").read_bytes());records={}
    for name,p in bank.items():
        file=folder/(name+".npz")
        if sha(file)!=lock["files"][name]:raise ValueError("A closed reading trajectory was changed.")
        records[name]={}
        with np.load(file) as f:
            for m in MODES:
                for cond in ("default","team"):
                    evaluations=[]
                    for t,gold in p.story.teams.items():
                        get=lambda k: f[".".join((m,t,k))] if cond=="team" else np.mean([f[".".join((m,r,k))] for r in TEAMS],axis=0)
                        predicted=get("stock");mass=get("mass");opens=get("openings")
                        # Derive a mixed cohort's hazard from surviving origin mass, preserving marginal identity.
                        survival=np.tri(p.n)*opens[None,:]-np.cumsum(mass,axis=0)
                        prev=np.vstack((np.zeros(p.n),survival[:-1]))
                        hz=np.divide(mass,prev,out=np.zeros_like(mass),where=prev>1e-15)
                        groups=p.story.teams
                        from copy import copy
                        story=copy(p.story);story.teams={t:gold}
                        story.comparisons=[r for r in story.comparisons if r[0]==t];story.events=[r for r in story.events if r[0]==t]
                        page=copy(p);page.story=story;page.records=[r for r in p.records if r.team==t]
                        region=count_outcomes(story,{"model":predicted})["model"]
                        hits=identity(story,{"hazard":hz,"uncued":mass})["methods"]
                        closures=np.bincount([r.upper for r in page.records if r.upper==r.definite+1],minlength=p.n)
                        burst=closures==closures.max()
                        evaluations.append(dict(count_interval_mse=region["interval_mse"],direction=region["direction"],
                            source_choice=hits["hazard"]["paired_accuracy"],all_earlier=hits["uncued"]["top1_any_earlier"],
                            lifetime_nll=observation_loss(page,hz),uncued_ap=uncued(page,mass)["ap"],
                            largest_human_closure=int(closures.max()),count_change_at_burst=float(np.diff(predicted,prepend=0)[burst].mean()),
                            predicted_closures_at_burst=float(mass.sum(1)[burst].mean()),
                            burst_change_magnitude_error=float(np.abs(np.diff(predicted,prepend=0)[burst]-np.diff((gold==1).sum(1),prepend=0)[burst]).mean())))
                    records[name][m+"."+cond]={k:float(np.mean([r[k] for r in evaluations])) for k in evaluations[0]}
    average={m:{k:float(np.mean([v[m][k] for v in records.values()])) for k in list(records.values())[0][m]} for m in list(records.values())[0]}
    (OUT/"scores.json").write_text(json.dumps(dict(forecast=sha(folder/"seal.json"),stories=records,equal_story=average),indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("action",choices=("fit","score"))
    ap.add_argument("archive",type=Path);ap.add_argument("old",type=Path)
    args=ap.parse_args();torch.set_num_threads(1)
    (predict if args.action=="fit" else score)(args.archive,args.old)
