"""Open new sentence judgments only after all out-of-family predictions are sealed."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from research.reading_validation.memory_data import BASE, DISPLAY, E1, OUT, dump, h, load_rows, rows_xlsx, sha
from research.reading_validation.memory_fit import MODELS


def reveal(study,rows):
    sources={(r["source_story"],r["condition"]):r for r in rows if r["study"]==study}
    workbook = BASE / f"frick{study}.xlsx"
    checks = defaultdict(list)
    for row in rows_xlsx(workbook, {"participant","QuestionKey.corr"}, require="QuestionKey.corr"):
        correct = int(row["QuestionKey.corr"])
        if correct not in (0, 1):
            raise ValueError("Unknown comprehension-check value")
        checks[row["participant"]].append(correct)
    if any(len(answers) != 24 for answers in checks.values()):
        raise ValueError("The original attention-check task has 24 trials")
    # Both original cohorts excluded readers with at least 40% incorrect checks.
    eligible = {person for person, answers in checks.items() if 5 * sum(answers) > 3 * len(answers)}
    paths=defaultdict(list)
    variants={}
    for r in rows_xlsx(workbook,{"participant","ID","Satz","LoopSentences.thisN","StoryKey.keys"}):
        if r["participant"] not in eligible:
            continue
        pos=int(r["LoopSentences.thisN"])+1
        if pos==1:
            key=DISPLAY.fullmatch(str(r["ID"]).zfill(6))
            if key is None or int(key[3])!=1:
                raise ValueError("Unmapped trial start during scoring")
            design=sources.get((int(key[1]),key[2]))
        if design is None:
            continue
        reader=h(f"{study}|{r['participant']}")
        dkey=(reader,design["story"])
        if variants.setdefault(dkey,design["id"])!=design["id"] or len(paths[dkey])+1!=pos:
            raise ValueError("Cannot reconstruct the complete response for a specific participant")
        judgment=(r["StoryKey.keys"] or "").lower()
        if judgment not in ("k","d"):
            raise ValueError("Unrecognized human keypress")
        paths[dkey].append(judgment)
    con={r["id"]:r for r in sources.values()}
    result=[dict(reader=reader,story=story,id=variants[reader,story],arm=con[variants[reader,story]]["arm"],
                 target=con[variants[reader,story]]["target"],keys=keys)
            for (reader,story),keys in paths.items()]
    persons=defaultdict(list)
    for r in result:
        persons[r["reader"]].append(r)
    if len(persons)!={2:276,3:233}[study] or any(len(rr)!=6 or len({r["arm"] for r in rr})!=1 for rr in persons.values()):
        raise ValueError("The test sample does not match the published nonillustrated, between-condition design")
    return result


def trial_observation(r):
    first=next((i for i,key in enumerate(r["keys"][1:],2) if key=="d"),len(r["keys"])+1)
    return first


def summarize(f,rows,seed,*,cohorts=("strict","historical_curve"),methods=MODELS,focus="adapted",controls=("frozen","control")):
    families=sorted({r["story"] for r in rows})
    rng=np.random.default_rng(seed)
    k=len(families)
    resamples=10000
    fw=rng.multinomial(k,np.full(k,1/k),size=resamples)
    raw={}
    for arm in ("neutral","inconsistent"):
        items=[r for r in rows if r["arm"]==arm]
        readers=sorted({r["reader"] for r in items})
        who={v:i for i,v in enumerate(readers)}
        fam={v:i for i,v in enumerate(families)}
        weights=rng.multinomial(len(readers),np.full(len(readers),1/len(readers)),size=resamples).astype(float)
        present=np.zeros((len(readers),k))
        numbers={}
        def add(name,ix,value):
            if name not in numbers:
                numbers[name]=np.zeros((len(readers),k))
            numbers[name][ix]=value
        for r in items:
            ix=(who[r["reader"]],fam[r["story"]])
            if present[ix]:
                raise ValueError("Participant entered a story twice")
            present[ix]=1
            first=trial_observation(r)
            n=len(r["keys"])
            t=r["target"]-1
            add("human.target",ix,int(first==t+1))
            add("human.early",ix,int(first<t+1))
            add("human.all_target_judgments",ix,int(r["keys"][t]=="d"))
            for cohort in cohorts:
                for method in methods:
                    name=cohort+"."+method
                    hz=f[name+"."+r["id"]]
                    if len(hz)!=n or hz[0]!=0:
                        raise ValueError("Human keys and causal predictions have unequal coverage")
                    hclip=np.clip(hz,1e-12,1-1e-12)
                    before=np.r_[1.,np.cumprod(1-hz[:-1])]
                    likelihood=-np.log1p(-hclip[1:min(n,first-1)]).sum()
                    if first<=n:
                        likelihood-=np.log(hclip[first-1])
                    add(name+".target",ix,hz[t]*before[t])
                    add(name+".early",ix,1-before[t])
                    add(name+".path_nll",ix,likelihood)
        denom=weights@present
        if np.any(denom<=0):
            raise ValueError("Empty article under crossed bootstrap")
        stats={}
        for name,values in numbers.items():
            obs=(values.sum(0)/present.sum(0)).mean()
            draws=((weights@values/denom)*fw).sum(1)/k
            stats[name]=(obs,draws)
        raw[arm]=stats
    def ci(x):
        return {"estimate":float(x[0]),"interval95":np.quantile(x[1],[.025,.975]).tolist()}
    def diff(x,y):return x[0]-y[0],x[1]-y[1]
    def abserr(a,b):return abs(a[0]-b[0]),np.abs(a[1]-b[1])
    def avg(a,b):return (a[0]+b[0])/2,(a[1]+b[1])/2
    effects={name:diff(raw["inconsistent"][name],raw["neutral"][name]) for name in raw["neutral"]}
    human=effects["human.target"]
    out={"stories":len(families),"participants":len({r["reader"] for r in rows}),"reader_trials":len(rows),
         "sentence_keys":sum(len(r["keys"]) for r in rows),"human_effect":ci(human),
         "human_all_target_judgments_effect":ci(effects["human.all_target_judgments"]),
         "arms":{arm:{k:ci(v) for k,v in stats.items()} for arm,stats in raw.items()},"models":{}}
    for c in cohorts:
        result={}
        for m in methods:
            label=c+"."+m
            effect=effects[label+".target"]
            difference=ci(diff(effect,human))
            result[m]={"condition_effect":ci(effect),"model_minus_human":difference,
                       "within_10_points":bool(min(difference["interval95"])>=-.1 and max(difference["interval95"])<=.1),
                       "path_nll":ci(avg(raw["neutral"][label+".path_nll"],raw["inconsistent"][label+".path_nll"])),
                       "early_bias":{arm:ci(diff(raw[arm][label+".early"],raw[arm]["human.early"])) for arm in raw}}
        for control in controls:
            result[focus+"_abs_error_reduction_vs_"+control]=ci(diff(abserr(effects[c+"."+control+".target"],human),
                                                                     abserr(effects[c+"."+focus+".target"],human)))
            result[focus+"_path_nll_difference_vs_"+control]=ci(avg(*[diff(raw[a][c+"."+focus+".path_nll"],
                                                                           raw[a][c+"."+control+".path_nll"]) for a in raw]))
        out["models"][c]=result
    return out


def nli(path):
    archive=np.load(path/"nli-test.npz",allow_pickle=False)
    labels={r["id"]:r for r in load_rows(BASE/"memory/cad.jsonl") if r["split"]=="test"}
    if set(labels)!=set(archive["id"]):
        raise ValueError("Out-of-sample NLI groups changed")
    rows=[labels[k] for k in archive["id"]]
    y=np.array([r["y"] for r in rows])
    rng=np.random.default_rng(59120)
    sample=rng.integers(len(y),size=(10000,len(y)))
    rep={}
    for which in ("texts","adjacent"):
        hits={}
        for name in ("frozen","adapted"):
            z=archive[name+"."+which]
            p=(z.argmax(1)==y.argmax(1)).astype(float)
            hits[name]=p
            rep[name+"."+which]={"strict_triplet_accuracy":float(p.mean()),
                                  "interval95":np.quantile(p[sample].mean(1),[.025,.975]).tolist()}
        delta=hits["adapted"]-hits["frozen"]
        rep["difference."+which]={"estimate":float(delta.mean()),
                                  "interval95":np.quantile(delta[sample].mean(1),[.025,.975]).tolist()}
    return dict(families=len(y),measures=rep)


def run(folder,extraction):
    seal=json.loads((folder/"seal.json").read_text())
    if any(sha(folder/k)!=v for k,v in seal["files"].items()) or sha(extraction/"receipt.json")!=seal["source"]:
        raise ValueError("New responses require unchanged sealed forecasts")
    rows=json.loads((BASE/"memory/study-map.json").read_text())
    scores={}
    with np.load(folder/"forecasts.npz",allow_pickle=False) as f:
        for study in (2,3):
            scores[str(study)]=summarize(f,reveal(study,rows),31780+study)
    result=dict(cohorts=scores,counterfactual_source=nli(extraction),seal=sha(folder/"seal.json"))
    dump(OUT/"scores.json",result)
    print(json.dumps({k:{"n":v["reader_trials"],"human":v["human_effect"],"adapted":v["models"]["strict"]["adapted"]["condition_effect"],
                        "reduction":v["models"]["strict"]["adapted_abs_error_reduction_vs_frozen"]} for k,v in scores.items()}))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder",type=Path,default=OUT/"forecast")
    p.add_argument("--extraction",type=Path,required=True)
    run(**vars(p.parse_args()))
