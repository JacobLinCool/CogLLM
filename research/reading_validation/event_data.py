"""Align reported supporting sentences to the event they helped explain in TellMeWhy."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from research.reading_validation.external_input import digest_string, material_key
from research.reading_validation.joint_data import DEFAULT_CACHE, digest, lines


BASE=DEFAULT_CACHE/"reading-validation"
OUTPUT=Path("research/results/reading-validation-events")
SPLITS=("train","validation","test")
QUESTION=re.compile(r"(?:rocstories|caters)_narrative_(\d+)_sentence_(\d+)_question_(\d+)")


def source(split):
    return BASE/f"tellmewhy-{split}.json"


def align(row):
    m=QUESTION.fullmatch(row["question_meta"])
    sentences=tuple(s.strip() for s in row["original_narrative_form"])
    if m is None or len(sentences)!=5 or not all(sentences):
        return None
    anchor=int(m[2])
    if not 0<=anchor<5 or material_key(sentences[anchor])!=material_key(row["original_sentence_for_question"]):
        return None
    text=" ".join(sentences)
    if material_key(text)!=material_key(row["narrative"]):
        return None
    return digest_string(text),text,sentences,anchor


def prepare():
    docs={}
    roles=defaultdict(set)
    bad=Counter()
    for role in SPLITS:
        for r in lines(source(role)):
            aligned=align(r)
            if aligned is None:
                bad[role]+=1
                continue
            k,text,sents,_=aligned
            spans=[]
            p=0
            for s in sents:
                spans.append([p,p+len(s)])
                p+=len(s)+1
            docs[k]=dict(id=k,text=text,spans=spans)
            roles[role].add(k)
    def grams(k,n):
        r=docs[k]
        sentences=[r["text"][a:b] for a,b in r["spans"]]
        return {material_key(" ".join(sentences[i:i+n])) for i in range(6-n)}
    held={s for k in roles["test"] for s in grams(k,2)}
    dropped={p:sorted(k for k in roles[p] if held & grams(k,2)) for p in ("train","validation")}
    eligible={p:sorted(roles[p]-set(dropped.get(p,[]))) for p in SPLITS}
    eligible["train"]=sorted(set(eligible["train"])-set(eligible["validation"]))
    keys=sorted(set().union(*eligible.values()))
    packet=BASE/"event-stimuli.jsonl"
    packet.write_text("".join(json.dumps(docs[k],ensure_ascii=False)+"\n" for k in keys))
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/"splits.json").write_text(json.dumps(eligible,indent=1)+"\n")
    train_sent={s for k in eligible["train"] for s in grams(k,1)}
    result=dict(revision="662fce7ab3d2e18087973b1f15470b1dfaf81f9e",source_files={p:digest(source(p)) for p in SPLITS},
                documents={k:len(v) for k,v in eligible.items()},source_alignment_excluded_rows=dict(bad),
                pairs_overlapping_test={p:len(val) for p,val in dropped.items()},
                test_stories_with_one_training_sentence=sum(bool(grams(k,1)&train_sent) for k in eligible["test"]),
                packet_sha256=digest(packet),split_sha256=digest(OUTPUT/"splits.json"),
                protocol_sha256=digest(Path(__file__).with_name("event-protocol.md")))
    (OUTPUT/"inputs.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


def responses(split, allowed):
    counts=defaultdict(lambda:defaultdict(list))
    for r in lines(source(split)):
        aligned=align(r)
        if aligned is None or aligned[0] not in allowed:
            continue
        k,_,_,anchor=aligned
        raw=r["helpful_sentences"]
        if not isinstance(raw,list) or any(type(i) is not int or not 1<=i<=5 for i in raw):
            raise ValueError("Unexpected helpful sentence coordinates")
        counts[k][anchor].append(frozenset(i-1 for i in raw))
    return counts


if __name__=="__main__":
    prepare()
