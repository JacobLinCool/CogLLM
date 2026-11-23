"""Separate all unscored, source-disjoint public game passages and an original unseen reader panel."""

from collections import Counter,defaultdict
from itertools import chain
import json
from pathlib import Path

from research.reading_validation.antecedent_inference import BUNDLE,load_tagger
from research.reading_validation.transition_architecture_prepare import load_tokenizer
from research.reading_validation.transition_direct_data import display,gold_docs
from research.reading_validation.transition_game_prepare import silver_docs,shingles,work
from research.reading_validation.transition_gate_prepare import DEST as OLD
from research.reading_validation.transition_jones_prepare import article
from research.reading_validation.transition_pointer_unconditional import labels
from research.reading_validation.transition_score import dump,sha
from research.reading_validation.transition_snac_prepare import ROOT


OUT=Path("research/results/reading-validation-transition-pointer/joint-actions")
PROTOCOL=Path("research/reading_validation/transition-pointer-null-protocol.md")


def compact(answers,panel=None):
    result={}
    for q,choices in answers.items():
        v=Counter(t for person,t in choices.items() if panel is None or person not in panel)
        if v:result[str(q)]={"actions":[[k,l,r,n] for (k,l,r),n in sorted(v.items())]}
    return result


def main():
    prior=ROOT/"gate-pointer-stimuli.jsonl";seal=json.loads((OLD/"source-seal.json").read_text())
    if sha(prior)!=seal["input_sha256"]:raise ValueError("Original coref game fold changed")
    allrows={r["id"]:r for r in map(json.loads,prior.read_text().splitlines())}
    train_works={r["group"] for r in allrows.values() if r["fold"]!="test"}
    srcgrams=set().union(*(shingles(r["text"]) for r in allrows.values() if r["fold"]!="test"))
    goldgrams=set().union(*(shingles(r["text"]) for r in allrows.values() if r["fold"]=="test"))
    goldworks={r["group"] for r in allrows.values() if r["fold"]=="test"}
    ids=set()
    for d,g,f,t,cr,we,tok in silver_docs():
        if d in allrows and allrows[d]["fold"]!="test":ids.update(a.get("user_id") for a in t.iter("anchor") if a.get("mode")=="a")
    tokenizer=load_tokenizer("qwen",json.loads(Path("research/backbone_transfer/models.json").read_text())["models"]["qwen"])
    tagger=load_tagger(BUNDLE/"averaged_perceptron_tagger_eng.zip")
    new={};panels={"earlier_gold":{},"new_articles":{}}; new_voters=defaultdict(set); seen_actions=Counter();exclusions=Counter(); note={}
    for r in allrows.values():
        r.pop("actions",None)
    for doc,genre,fold,t,cr,ele,tok in chain(silver_docs(),((d,g,"test",t,c,e,z) for d,g,t,c,e,z in gold_docs())):
        if doc in allrows and allrows[doc]["fold"]!="test":continue
        if doc in allrows:
            section="earlier_gold";row=allrows[doc]
        else:
            section="new_articles";text,_,_=display(t,ele,tok);w=work(doc,genre)
            if w in train_works:exclusions["source_work"]+=1;continue
            words=shingles(text)
            if words&srcgrams:exclusions["source_exact_20_words"]+=1;continue
            if len(tokenizer.encode(text,add_special_tokens=False).ids)>2048:exclusions["whole_document_above_2048"]+=1;continue
            row=article(doc,text,tokenizer,tagger);row["group"]=w;row["fold"]="new"
            allrows[doc]=row
            note[doc]={"shares_20_words_known_gold":bool(words&goldgrams),
                       "shares_work_known_gold":w in goldworks,"source_archive":fold}
        decisions,_=labels(row,t,cr,ele,tok,minimum_readers=1)
        for choices in decisions.values():new_voters[section].update(person for person in choices if person not in ids)
        panels[section][doc]=compact(decisions,ids)
        if section=="new_articles":
            eligible={q:v for q,v in decisions.items() if len(v)>=4}
            new[doc]=compact(eligible)
            for vs in eligible.values():seen_actions.update(x[0] for x in vs.values())
    OUT.mkdir(parents=True,exist_ok=True)
    stimuli=ROOT/"null-choice-pointer-stimuli.jsonl"
    stimuli.write_text("".join(json.dumps(r,separators=(",",":"))+"\n" for r in allrows.values()))
    extra=OUT/"independent-article-actions.json";panel=OUT/"external-reader-actions.json"
    dump(extra,{"actions":new,"material_overlap":note});dump(panel,panels)
    dump(OUT/"source-seal.json",{"protocol_sha256":sha(PROTOCOL),"stimuli_sha256":sha(stimuli),"original_action_data_sha256":sha(prior),
         "external_actions_sha256":sha(extra),"reader_panel_sha256":sha(panel),"exclusions":dict(exclusions),"original_known_works":len(goldworks),
         "new_whole_articles":len(new),"new_article_works":len({allrows[d]["group"] for d in new}),
         "new_article_measured_pronouns":sum(len(z) for z in new.values()),"new_all_actions":dict(seen_actions),
         "source_players_any_independent_response":len(ids),"novel_readers":{k:len(v) for k,v in new_voters.items()},
         "new_reader_queries":{k:sum(len(row) for row in v.values()) for k,v in panels.items()},
         "complete_passages":len(allrows),"all_visible_pronouns":sum(len(r["queries"]) for r in allrows.values())})
    print(json.loads((OUT/"source-seal.json").read_text()),flush=True)


if __name__=="__main__":main()
