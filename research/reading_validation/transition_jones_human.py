"""Reproduce the public Jones–Bergen exclusions and retain anonymous experimental responses."""

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from research.reading_validation.transition_jones_prepare import OUT, ROOT
from research.reading_validation.transition_score import dump, sha


def csv(name):
    return pd.read_csv(ROOT/f"jones-{name}.csv")


def participants(experiment):
    people = csv(f"exp{experiment}-participants")
    language = csv(f"exp{experiment}-languages")
    english = language[(language.language == "English") & (language.proficiency == "5 - Native")].participant_id
    people["native"] = people.id.isin(english)
    for k in ("start_time", "end_time"):
        people[k] = pd.to_datetime(people[k], utc=True, format="mixed")
    people["minutes"] = (people.end_time - people.start_time).dt.total_seconds()/60
    return people


def exp1():
    data = csv("exp1-official-trial")
    people = participants(1)
    details = {"input_people_all_tasks": len(people), "input_people_main": int((people.condition==0).sum())}
    keys = csv("exp1-catch")[["item_id","answer"]]
    catch = data[data.item_type=="catch"].merge(keys,on="item_id",validate="many_to_one")
    rate = (catch.response == catch.answer).groupby(catch.participant_id).mean()
    keep = people.native & (people.minutes<=60) & (people.id.map(rate)>=.34)
    details["eligible_people_main_before_trial_filters"] = int((keep & (people.condition==0)).sum())
    raw = data[data.item_type=="experimental"].copy()
    raw = raw[raw.participant_id.isin(people.id[keep])]
    raw["condition_name"] = raw.condition.map({0:"Experimental",1:"Syntax norming",2:"Physics norming"})
    length = csv("exp1-syllables").rename(columns={"condition":"condition_name"})
    raw = raw.merge(length[["item_id","condition_name","sylls"]],on=["item_id","condition_name"],validate="many_to_one")
    raw["logrt"] = np.log(raw.reaction_time)
    base = raw[(raw.reaction_time>=500) & (raw.reaction_time-191*raw.sylls<=10000)].copy()
    paper = base[base.condition == 0]
    center, spread = (raw.groupby("participant_id").logrt.transform(s) for s in ("mean","std"))
    rel = raw[(abs(raw.logrt-center)<=2.5*spread)].id
    strict = base[base.id.isin(rel)].copy()
    meet = (strict.groupby("participant_id").size()/raw.groupby("participant_id").size())>=.67
    strict = strict[strict.participant_id.isin(meet.index[meet])]
    main = strict[strict.condition == 0].copy()
    if len(paper)!=1079 or paper.participant_id.nunique()!=41:
        raise ValueError(f"Paper-only Exp1 intake differs: {len(paper)}, {paper.participant_id.nunique()}")
    details.update(paper_mentioned_trials=len(paper),paper_mentioned_people=paper.participant_id.nunique(),
                   script_trials=len(main),script_people=main.participant_id.nunique())
    return main, paper, details


NAMES=("p3","p2","p1","","sp1","sp2","sp3")


def columns(label):
    return [f"rt_{label}_{k}" if k else f"rt_{label}" for k in NAMES]


def exp2():
    raw=csv("exp2-trials")
    people=participants(2)
    asked=raw[raw.answer.notna()]
    accuracy=(asked.answer==asked.expected_answer).groupby(asked.participant_id).mean()
    registered=(people.SONA_code.notna() & (people.SONA_code!=131) & people.end_time.notna()
                & (people.start_time>pd.Timestamp("2021-07-12",tz="UTC"))
                & (people.start_time<pd.Timestamp("2021-07-29",tz="UTC")))
    keep=registered & people.native & (people.minutes<=60) & (people.vision=="yes") & ~people.dyslexia & (people.id.map(accuracy)>=.5)
    t=raw[raw.participant_id.isin(people.id[keep])].copy()
    t=t.merge(csv("exp2-syllables")[["item_id","passage_sylls"]],on="item_id",validate="many_to_one")
    details={"raw_people":len(people),"raw_trials":len(raw),"after_people_exclusions":t.participant_id.nunique(),
             "trials_from_these_people":len(t)}
    rate=t.passage_reading_time/t.passage_sylls
    t=t[(rate>50)&(rate<350)].copy()
    details["after_absolute_passage"]=len(t)
    all_regions=columns("crit")+columns("cont")
    paper=t[(t[all_regions].max(axis=1)<5000)&(t[all_regions].min(axis=1)>100)].copy()
    details["paper_described_absolute_rules_only"]=len(paper)
    t["log_rate"]=np.log(t.passage_reading_time/t.passage_sylls)
    group=t.groupby("participant_id").log_rate
    t=t[abs(t.log_rate-group.transform("mean")) <= 3*group.transform("std")].copy()
    details["after_relative_passage"]=len(t)
    t=t[(t[all_regions].max(axis=1)<5000)&(t[all_regions].min(axis=1)>100)].copy()
    details["after_absolute_regions"]=len(t)
    for label in ("cont","crit"):
        z=np.log(t[columns(label)])
        gm=z.groupby(t.item_id)
        t=t[(abs(z-gm.transform("mean")) <= 3*gm.transform("std")).all(axis=1)].copy()
        details["after_relative_"+label]=len(t)
    groups=t.groupby("participant_id").size()
    t=t[t.participant_id.isin(groups.index[(30-groups)/30<=.34])]
    details.update(final_trials=len(t),final_people=t.participant_id.nunique())
    html=(ROOT/"jones-exp2-published.html").read_text()
    if len(t)!=3777 or t.participant_id.nunique()!=135 or not re.search(r"Retained:.{0,35}3777", html):
        raise ValueError(f"Exp2 intake differs from the authors' rendered analysis: {details}")
    details["paper_reported_trials"]=3947
    return t, paper, details


def save():
    seal=json.loads((OUT/"source-seal.json").read_text())
    for name, expected in seal["source_files"].items():
        if sha(ROOT/name)!=expected:
            raise ValueError("Original source file changed: "+name)
    main,paper,one=exp1()
    second,second_paper,two=exp2()
    # Only a study-local sequential code enters uncertainty estimation. Original identifiers, times,
    # recruitment codes, demographics and language profiles stay outside exported measurements.
    for i,frame in enumerate((main,paper,second,second_paper)):
        d=frame.copy();d["reader"]=pd.factorize(d.participant_id,sort=True)[0]
        if i<2:
            d=d[["reader","item_id","response"]];d["y"]=(d.response=="NP2").astype(int);d=d.drop(columns="response")
        else:
            d=d[["reader","item_id",*columns("cont")]].copy()
        filename=("exp1-script","exp1-paper","exp2","exp2-paper-rules")[i]+".csv"
        d.to_csv(ROOT/("jones-anonymous-"+filename),index=False)
    dump(OUT/"human-intake.json",{"experiment_1":one,"experiment_2":two,
         "study_local_anonymous_outcome_sha256":{f:sha(ROOT/("jones-anonymous-"+f+".csv")) for f in ("exp1-script","exp1-paper","exp2","exp2-paper-rules")}})
    print(one,two,flush=True)


if __name__=="__main__":
    save()
