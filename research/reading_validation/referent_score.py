"""Grade sealed shared-reference forecasts against direct clicks, twins, and two reader studies."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from research.backbone_transfer.extract import digest,write_json
from research.reading_validation.referent_sketch_prepare import material,OUT,ROOT,POINTER
from research.reading_validation.transition_anchored_scores import probability,relation_scores
from research.reading_validation.transition_dpr_score import same_prefix_type,pick
from research.reading_validation.transition_jones_score import interpretation,read,ORDERS,study_controls
from research.reading_validation.transition_pointer_score import hit_k
from research.reading_validation.transition_ambicoref_score import crossed

def summary(measures,methods):
    """Equal independent families, with identical bootstrap draws for all methods."""
    output={}
    rng=np.random.default_rng(72184)
    for k,families in measures.items():
        x=np.stack([np.mean(f,0) for _,f in sorted(families.items())])
        n=len(x)
        w=rng.multinomial(n,np.full(n,1/n),size=4000)/n
        boot=w@x
        def metric(y,b):return [float(y),*np.quantile(b,[.025,.975]).tolist()]
        records={name:metric(x[:,j].mean(),boot[:,j]) for j,name in enumerate(methods)}
        for a in ("shared","private"):
            for b in ("prior","private"):
                if a==b:continue
                i=methods.index(a);j=methods.index(b)
                records[a+"_minus_"+b]=metric((x[:,i]-x[:,j]).mean(),boot[:,i]-boot[:,j])
        output[k]={"groups":n,"estimates":records}
    return output

def human_clicks(rows,archive,methods):
    f=POINTER.parent/"external-clicks.json"
    lock=json.loads((POINTER.parent/"source-seal.json").read_text())
    if digest(f)!=lock["gold_sha256"]:raise ValueError("Original click observations changed")
    labels=json.loads(f.read_text())["articles"];v=defaultdict(lambda:defaultdict(list));count=0;n=0
    for r in rows:
        if r["id"] not in labels:continue
        for key,out in labels[r["id"]].items():
            q=int(key);z=archive[r["id"]+"/"+key];span=out["spans"]
            w=np.array([e[2] for e in span]);count+=int(w.sum());n+=1
            vals=[];tops=[]
            for a,b,_,_,_ in span:
                mask=(np.arange(q)>=a)&(np.arange(q)<b)
                vals.append(np.logaddexp.reduce(z,axis=-1)-np.logaddexp.reduce(z[:,a:b],axis=-1))
                tops.append([hit_k(arm,mask,1) for arm in z])
            for k,x in (("click_region_nll",vals),("unrestricted_top_word",tops)):
                v[k][r["group"]].append(np.average(x,axis=0,weights=w))
    return {"actual_linked_clicks":count,"pronouns":n,"outcomes":summary(v,methods)}

def paired(rows,archive,methods):
    packet=Path("research/results/reading-validation-transition-dpr")
    f=packet/"authored-test-key.json"
    if digest(f)!=json.loads((packet/"source-seal.json").read_text())["external_key_sha256"]:raise ValueError("Authored keys changed")
    y=json.loads(f.read_text())
    families=defaultdict(list)
    for r in rows:
        if r["fold"]=="test":families[r["group"]].append(r)
    v=defaultdict(lambda:defaultdict(list));drifts=[];per={}
    for family,twins in families.items():
        signs=[];hits=[];ident=[];sz=[]
        for r in twins:
            g=r["measurement"];z=archive[r["id"]];target=y[r["id"]]
            signs.append((2*target-1)*relation_scores(z,g["names"]))
            ident.append(r["identities"][target]);sz.append(g["k"]-g["q"])
            mask=np.isin(np.arange(g["q"]),g["names"][target])
            hits.append([hit_k(arm,mask,1) for arm in z[:,-1]])
        end=np.stack([s[:,-1] for s in signs]);v["terminal_choice_nll"][family].append(np.logaddexp(0,-end).mean(0))
        v["both_terminal_correct"][family].append(pick(end).prod(0))
        v["all_prior_word_top_hit"][family].append(np.mean(hits,0))
        same=same_prefix_type(*twins)
        if same is None or ident[0]==ident[1] or same[1] is None:continue
        pre,change=same
        lead=.5*(signs[0][:,pre]+signs[1][:,pre]);step=.5*(signs[0][:,change]+signs[1][:,change])
        drifts.append(np.abs(probability(signs[0][:,pre])-probability(-signs[1][:,pre])))
        for tag in (["identical_prefix","equal_length"] if sz[0]==sz[1] else ["identical_prefix"]):
            v[tag+"_first_changed_direction"][family].append(pick(np.where(np.abs(step)<1e-6,0.,step)))
            increment=step-lead
            v[tag+"_first_changed_logodds_increment"][family].append(increment)
            v[tag+"_direction_of_actual_update"][family].append(pick(np.where(np.abs(increment)<1e-4,0.,increment)))
            if increment[methods.index("prior")]!=0.:
                raise ValueError("A carried-prior baseline cannot update under either continuation")
        per[family]={"first_changed_since_pronoun":change,"equal_length":sz[0]==sz[1],"signed_first_change":step.tolist()}
    drift=np.max(drifts,0)
    if drift.max()>.01:raise ValueError(f"Unexpected same-prefix probability drift {drift}")
    return {"families":len(families),"maximum_unchanged_prefix_probability_drift":dict(zip(methods,drift.tolist(),strict=True)),
            "outcomes":summary({k:a for k,a in v.items() if not k.endswith("_direction_of_actual_update")},methods),
            "actual_word_update":summary({k:a for k,a in v.items() if k.endswith("_direction_of_actual_update")},methods),"first_changes":per}

def readers(rows,archive,methods):
    indexed={r["id"]:r for r in rows}
    oldpath=Path("research/results/reading-validation-transition-anchored-reference/jones-word-posteriors.json")
    old=json.loads(oldpath.read_text());choices=study_controls()
    for j,arm in enumerate(methods):
        choices[arm]=np.array([probability(relation_scores(archive["e1-"+k][j,-1],indexed["e1-"+k]["measurement"]["names"])) for k in ORDERS])
    for m in ("native_semantic_update","source_semantic","native_prior"):
        choices["original_"+m]=np.array([old["e1-"+k][m][-1] for k in ORDERS])
    return interpretation(choices,read("anonymous-exp1-script"),False,[(a,b) for a in ("private","shared") for b in ("prior","original_native_semantic_update","published_gpt3_completion")]+[("shared","private")])

def ambiguity(rows,archive,methods):
    file=ROOT/"ambicoref-observed-choices.json"
    ext=Path("research/results/reading-validation-transition-anchored-reference")
    lock=json.loads((ext/"external-seal.json").read_text())
    if digest(file)!=lock["outcomes_sha256"]:raise ValueError("Observer outcomes changed")
    gold=json.loads(file.read_text())
    forecasts={m:{} for m in methods}
    for r in rows:
        k=r["id"];p=probability(relation_scores(archive[k][:,-1],gold[k]["names"]))
        for m,e in zip(methods,p,strict=True):forecasts[m][k]=float(e)
    # The original published scoring implementation has named paired comparisons.
    # These names serve solely as its parameter slots; export the actual fitted arm names.
    aliases={"native_semantic_update":"shared","native_prior":"prior","source_semantic":"private"}
    measured=crossed(gold,{k:forecasts[m] for k,m in aliases.items()})
    return {**{k:measured[k] for k in ("sentences","lexical_skeletons","questionnaires","judgments","ordinal_response_counts")},
            "measurements":{m:measured["measurements"][k] for k,m in aliases.items()},
            "paired":{"shared_minus_"+m:measured["paired"]["native_semantic_update_minus_"+k] for k,m in aliases.items() if m!="shared"}}

def main():
    trained=json.loads((OUT/"source-fit.json").read_text());file=OUT/"forecast.npz"
    if digest(file)!=trained["forecast_sha256"] or digest(OUT/"source-models.npz")!=trained["weights_sha256"]:
        raise ValueError("Unsealed forecasts")
    studies=material();methods=trained["arms"]
    with np.load(file) as pred:
        d={"forecast_sha256":digest(file),"clicks":human_clicks(studies["pointer"],pred,methods),
           "authored":paired(studies["dpr-pointer"],pred,methods),"jones":readers(studies["dpr-pointer"],pred,methods),
           "ambicoref":ambiguity(studies["ambicoref-reading"],pred,methods)}
    update={"forecast_sha256":d["forecast_sha256"],"actual_word_update":d["authored"].pop("actual_word_update")}
    for name,value in (("evaluations.json",d),("within_pair_changes.json",update)):
        file=OUT/name
        if file.exists():
            if json.loads(file.read_text())!=value:raise ValueError("Previous grade differs for "+name)
        else:write_json(file,value)
    print("Graded all four observation studies",flush=True)

if __name__=="__main__":main()
