"""Align the original human-authored twin sentences and seal all external answer keys."""

from collections import Counter
import json
from pathlib import Path
import re

from research.reading_validation.antecedent_inference import BUNDLE,load_tagger
from research.reading_validation.transition_architecture_prepare import load_tokenizer
from research.reading_validation.transition_jones_prepare import OUT as JONES, ROOT, article
from research.reading_validation.transition_score import dump,sha
from research.reading_validation.transition_snac_prepare import WORD


OUT=Path("research/results/reading-validation-transition-dpr")
PROTOCOL=Path("research/reading_validation/transition-dpr-protocol.md")
FILES={"train":"bc0a54b3ca1009d8d5b2ca5a221086aee2b0fc5cd03b22b9dfa9cdf44c629cec",
       "test":"cf1cf025e1d59a5b363e6dcfbd5a13aae8a9830831ac16fd7f865aca7a1559d8"}


def norm(t):
    return t.replace("’","'").casefold()


def sequence(text):
    return [norm(w.group()) for w in WORD.finditer(text)]


def shingles(text):
    words=sequence(text)
    return {tuple(words[i:i+8]) for i in range(len(words)-7)}


def align(fields,ident,tokenizer,tagger):
    text,pron,parts,truth=fields
    row=article(ident,text,tokenizer,tagger)
    forms=[norm(text[a:b]) for a,b in row["words"]]
    queries=[i for i,w in enumerate(forms) if w==norm(pron)]
    if not queries or queries[0] not in row["queries"]:return "pronoun_not_supported"
    q=queries[0]
    alternatives=[w.strip() for w in parts.split(",")]
    if len(alternatives)!=2:return "nonbinary"
    if norm(truth) not in [norm(x) for x in alternatives]:return "ambiguous_answer_key"
    regions=[]
    for option in alternatives:
        words=sequence(option)
        if not words:return "empty_option"
        # Possessive contractions are single WORD units in the production observer.
        # A source key naming the bare noun in that interval refers to the same displayed word.
        matches=[i for i in range(q-len(words)+1) if all(
                    word==forms[i+h] or (h==len(words)-1 and forms[i+h]==word+"'s")
                    for h,word in enumerate(words))]
        if len(matches)!=1:return "nonunique_or_later_named_region"
        regions.append(list(range(matches[0],matches[0]+len(words))))
    if set(regions[0])&set(regions[1]):return "overlapping_candidates"
    return row,{"q":q,"k":len(forms)-1,"names":regions,"y":[norm(x) for x in alternatives].index(norm(truth)),
                "identities":[norm(x) for x in alternatives]}


def main():
    tok=load_tokenizer("qwen",json.loads(Path("research/backbone_transfer/models.json").read_text())["models"]["qwen"])
    tagger=load_tagger(BUNDLE/"averaged_perceptron_tagger_eng.zip")
    external=ROOT/"jones-pointer-stimuli.jsonl"
    if sha(external)!="d0439250a93b77e4d83da445ccfd2313e3f0c9038952b1e1219e441aec8999aa":raise ValueError("Jones exposure changed")
    jones=list(map(json.loads,external.read_text().splitlines()))
    jgrams=set().union(*(shingles(r["text"]) for r in jones)); jtexts={tuple(sequence(r["text"])) for r in jones}
    rows=[]; eval_keys={}; audits={}; original_seen={}
    for fold,pin in FILES.items():
        file=ROOT/("dpr-"+fold+".txt")
        if sha(file)!=pin:raise ValueError("Original university file changed")
        fields=[l.strip() for l in file.read_bytes().decode("utf-8-sig").splitlines() if l.strip()]
        if len(fields)%8:raise ValueError("A source twin is incomplete")
        groups=Counter(); excluded=[]; duplicates=[]
        for pair,start in enumerate(range(0,len(fields),8)):
            group=fold+"-"+str(pair+1); entries=[]
            for side in (0,1):
                f=fields[start+4*side:start+4*side+4];ident="dpr-"+group+"-"+str(side)
                match=align(f,ident,tok,tagger);entries.append(match)
            bad=[r for r in entries if isinstance(r,str)]
            if bad:
                groups[bad[0]]+=1;excluded.append([group,bad]);continue
            if any(tuple(sequence(r["text"])) in jtexts or shingles(r["text"])&jgrams for r,g in entries):
                groups["overlap_with_Jones"]+=1;excluded.append([group,"Jones 8-word or whole string"]);continue
            forms=[tuple(sequence(r["text"])) for r,g in entries]
            if fold=="test" and any(f in original_seen for f in forms):
                groups["training_test_exact_duplicate"]+=1
                duplicates.append([group,[original_seen[f] for f in forms if f in original_seen]]);continue
            for r,g in entries:
                if fold=="train":original_seen[tuple(sequence(r["text"]))]=group
                y=g.pop("y"); identity=g.pop("identities")
                r["fold"]=fold;r["group"]=group;r["measurement"]=g;r["identities"]=identity
                if fold=="train":r["answer"]=y
                else:eval_keys[r["id"]]=y
                rows.append(r)
            groups["retained_pairs"]+=1
        audits[fold]={"counts":dict(groups),"excluded":excluded,"training_overlap":duplicates}
    geometry=json.loads((JONES/"geometry.json").read_text())
    for r in jones:
        g=geometry[r["id"]]
        if "q" not in g:continue
        r["group"]="-".join(r["id"].split("-")[:2]).split("_")[0]
        r["measurement"]={"q":g["q"],"k":g["end"],"names":g["names"]}
        r["fold"]="jones";rows.append(r)
    OUT.mkdir(exist_ok=True,parents=True)
    packet=ROOT/"dpr-pointer-stimuli.jsonl"
    packet.write_text("".join(json.dumps(r,separators=(",",":"))+"\n" for r in rows))
    dump(OUT/"authored-test-key.json",eval_keys)
    dump(OUT/"source-seal.json",{"protocol_sha256":sha(PROTOCOL),"input_sha256":sha(packet),"external_key_sha256":sha(OUT/"authored-test-key.json"),
         "university_sources":FILES,"outcome_free_external_transfer_sha256":sha(external),"audit":audits,
         "texts_by_fold":dict(Counter(r["fold"] for r in rows)),"max_words":max(len(r["words"]) for r in rows),"max_tokens":max(r["token_count"] for r in rows)})
    print({k:v["counts"] for k,v in audits.items()},flush=True)


if __name__=="__main__":
    main()
