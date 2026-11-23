"""Check published support-index conventions against independently traceable answer phrases."""

from collections import Counter
import json
import re

from research.reading_validation.event_data import OUTPUT, align, source
from research.reading_validation.joint_data import digest, lines


WORD=re.compile(r"[^\W_]+")


def trigrams(text):
    words=WORD.findall(text.casefold())
    return set(zip(words,words[1:],words[2:]))


def copied(row,anchor):
    raw=row["helpful_sentences"]
    if len(raw)!=1:
        return None
    grams=trigrams(row["answer"])
    scores=[len(grams&trigrams(t)) if j!=anchor else 0 for j,t in enumerate(row["original_narrative_form"])]
    if not max(scores) or scores.count(max(scores))!=1:
        return None
    return scores.index(max(scores))


def run():
    count=Counter()
    codes=Counter()
    table=Counter()
    for r in lines(source("train")):
        a=align(r)
        if a is None:
            continue
        codes.update(r["helpful_sentences"])
        j=copied(r,a[3])
        if j is None:
            continue
        raw=r["helpful_sentences"][0]
        count["traceable"]+=1
        count["documented_one_based"]+=int(raw-1==j)
        count["zero_based"]+=int(raw==j)
        other=[s for s in range(5) if s!=a[3]]
        count["omitting_anchor"]+=int(1<=raw<=4 and other[raw-1]==j)
        table[str(raw)+"."+str(j)]+=1
    result=dict(data="training observations only",source_sha256=digest(source("train")),code_counts=dict(codes),
                agreement_with_distinctive_answer_trigram=dict(count),code_vs_sentence_zero_based=dict(table))
    (OUTPUT/"coordinate-audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__=="__main__":
    run()
