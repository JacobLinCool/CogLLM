"""Score every released independent action and its actual earlier words, when linked."""
from collections import Counter,defaultdict
import json
import numpy as np
from research.backbone_transfer.extract import digest,write_json
from research.reading_validation.nominal_data import OUT,PLAN,ROOT
from research.reading_validation.referent_sketch_prepare import material
from research.reading_validation.transition_pointer_score import hit_k
from research.reading_validation.referent_estimation import normalize

def snapshot(values):
    result={};rng=np.random.default_rng(80392)
    for k,g in values.items():
        obs=np.array([np.mean(items,axis=0) for _,items in sorted(g.items())])
        n=len(obs);boot=rng.multinomial(n,np.ones(n)/n,size=4000)@obs/n
        def band(x,draw):return [float(x),*np.quantile(draw,[.025,.975]).tolist()]
        result[k]={"independent_works":n,"model":{m:band(obs[:,j].mean(),boot[:,j]) for j,m in enumerate(("form","native"))},
                   "native_minus_form":band((obs[:,1]-obs[:,0]).mean(),boot[:,1]-boot[:,0])}
    return result

def grade(obs,loc,raw,rows):
    blocks=defaultdict(lambda:defaultdict(list));counts=Counter();doccounts=Counter();word_hit=Counter()
    for r in rows:
        uid=r["id"];group=r["group"];words=[r["text"][a:b].casefold() for a,b in r["words"]]
        ix=loc[uid];target=obs[uid]
        if not set(map(int,target)).issubset(ix):raise ValueError("An eligible noun was absent from article-only inference")
        if not target:continue
        acts=raw[uid+".a"];all_scores=raw[uid+".r"];offset=0;doccounts[group]+=1
        for n,q in enumerate(ix):
            s=all_scores[:,offset:offset+q].astype(float);offset+=q
            if str(q) not in target:continue
            entry=target[str(q)];y=np.array(entry["action"]);p=acts[n];w=y/y.sum()
            if len(words[:q])!=q or abs(p.sum(1)-1).max()>1e-8:raise ValueError("Invalid action distribution")
            counts["decisions"]+=int(y.sum());counts["queries"]+=1;counts["human_links"]+=int(y[0])
            lp=np.log(p);event=-lp@w
            blocks["four_action_nll"][group].append(event);blocks["action_agreement"][group].append(w[np.argmax(p,axis=1)])
            part=[];tops=[];weights=[];z=np.logaddexp.reduce(s,axis=1);form=words[q]
            for a,b,v in entry["clicks"]:
                mask=(np.arange(q)>=a)&(np.arange(q)<b)
                nll=z-np.logaddexp.reduce(s[:,a:b],axis=1);hit=np.array([hit_k(arm,mask,1) for arm in s])
                changed=form not in words[a:b];word_hit["different_surface" if changed else "repeated_surface"]+=v
                part.append(nll);tops.append(hit);weights.append(v)
            joint=event.copy()
            if len(part):
                conditional=np.average(part,weights=weights,axis=0);top=np.average(tops,weights=weights,axis=0)
                joint+=w[0]*conditional
                blocks["linked_click_nll"][group].append(conditional);blocks["linked_unrestricted_top_word"][group].append(top)
                for tag in ("previous_same_final" if form in words[:q] else "no_identical_earlier_word",):
                    blocks[tag+"/click_nll"][group].append(conditional);blocks[tag+"/top_word"][group].append(top)
                for within in (True,False):
                    use=[i for i,(a,b,_) in enumerate(entry["clicks"]) if (form in words[a:b])==within]
                    if not use:continue
                    tag="clicked_phrase_includes_terminal_word" if within else "clicked_phrase_excludes_terminal_word"
                    for k,data in (("click_nll",part),("top_word",tops)):
                        blocks[tag+"/"+k][group].append(np.average([data[i] for i in use],weights=[weights[i] for i in use],axis=0))
            blocks["full_observed_action_and_range_nll"][group].append(joint)
            # A forced word choice misses all people who decided against a backward link.
            if tops:action_word=w[0]*np.average(tops,weights=weights,axis=0)
            else:action_word=np.zeros(2)
            blocks["unconditional_word_hit_if_always_offered"][group].append(action_word)
        if offset!=all_scores.shape[1]:raise ValueError("Prior-word candidates not completely consumed")
    return {"counts":dict(counts),"articles":sum(doccounts.values()),"works":len(doccounts),
            "actual_clicked_phrase_by_word_identity":dict(word_hit),"outcomes":snapshot(blocks)}

def main():
    fit=json.loads((OUT/"fit.json").read_text());intake=json.loads((OUT/"observations.json").read_text())
    if digest(PLAN)!=intake["protocol_sha256"] or any(digest(OUT/k)!=v for k,v in fit["files"].items()):raise ValueError("Unsealed model")
    loc=json.loads((OUT/"queried-positions.json").read_text());rows=material()["pointer"];out={}
    with np.load(OUT/"predictions.npz") as raw:
        for fold,packet in (("dev","source"),("test","external")):
            name=ROOT/("nominal-"+packet+"-observations.json")
            if digest(name)!=intake["files"][packet]:raise ValueError("Original readers changed")
            out[fold]=grade(json.loads(name.read_text()),loc,raw,[r for r in rows if r["fold"]==fold])
    write_json(OUT/"scores.json",{"fit":digest(OUT/"fit.json"),"source":digest(OUT/"observations.json"),**out})

if __name__=="__main__":main()
