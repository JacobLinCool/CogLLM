"""Test whether word form, lexical-semantic neighbors or general overlap explain an old-goal effect."""
from collections import Counter
from functools import cache
import json
from pathlib import Path
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from research.reading_validation.transition_goal_access_data import DEST, STOP, WORD, recover
from research.reading_validation.transition_goal_access_score import describe
from research.reading_validation.transition_score import dump, sha

RES = Path(".cache/cogllm/reading-validation/nltk")


@cache
def concepts(word):
    senses = set(wn.synsets(word))
    near = {b for a in senses for b in (*a.hypernyms(), *a.hyponyms(), *a.similar_tos())}
    return senses, senses | near


def words(text):
    return [m.group().casefold() for m in WORD.finditer(text) if m.group().casefold() not in STOP]


def main():
    nltk.data.path.insert(0, str(RES))
    source = DEST/"model-item-scores.npz"
    with np.load(source, allow_pickle=False) as data:
        ids = list(data["family"])
        native = data["all.native"]
        recency = data["all.recency"]
    get = lambda v: v[:,2,:]-v[:,3,:]
    specific = get(native)-get(recency)
    adjusted = specific[:,0]-specific[:,1:].mean(1)
    family = {}
    for i, row, t in recover():
        target_s, target_near = concepts(t)
        changed = {cond: set(words(row[key])) for cond,key in (("neutral","NEUT"),("new","NG"))}
        original = set(words(row["INTRODUCTION"]))
        relevant = set(words(" ".join(s for s in row["INTRODUCTION"].split(".") if t in words(s))))
        if not relevant:
            raise ValueError("Cannot establish the original target sentence")
        names = {}
        for c, forms in changed.items():
            names[c] = {"exact": t in forms,
                        "synonyms": sorted(w for w in forms if concepts(w)[0] & target_s),
                        "direct_semantic_neighbors": sorted(w for w in forms if concepts(w)[0] & target_near),
                        "intro_jaccard": len(forms & original)/len(forms | original),
                        "goal_sentence_jaccard": len(forms & relevant)/len(forms | relevant)}
        family[str(i)]=names
    eligible = lambda flag: np.array([all(not family[i][c][flag] for c in ("new","neutral")) for i in ids])
    matched = np.array([family[i]["neutral"]["goal_sentence_jaccard"]<=family[i]["new"]["goal_sentence_jaccard"] for i in ids])
    near = eligible("direct_semantic_neighbors")
    checks = {"no_exact": eligible("exact"),"no_synset_match":eligible("synonyms"),
              "no_one_edge_semantic_neighbor":near,"neutral_has_no_greater_goal_sentence_overlap":matched,
              "both_no_one_edge_and_neutral_has_no_greater_overlap":near & matched}
    results={}
    for k, keep in checks.items():
        rng=np.random.default_rng(218034); n=int(keep.sum())
        draws=rng.integers(n,size=(10000,n))
        results[k]={"stories":n,"neutral_minus_new_untrained":describe(get(native)[keep,0],draws),
                    "after_control_words_and_recency":describe(adjusted[keep],draws)}
    output={"scope":"Diagnostic specified after seeing the 32-item primary result. WordNet any-sense same synset or one edge over hypernyms, hyponyms and similar-to. No contextual sense disambiguation.",
            "source_sha256":sha(source),"wordnet_sha256":sha(RES/"corpora/wordnet.zip"),"wordnet":wn.get_version(),
            "measurements":family,"sensitivity":results}
    dump(DEST/"lexical-sensitivity.json",output)
    print([(k,v["stories"],v["after_control_words_and_recency"]["mean"],v["after_control_words_and_recency"]["family_interval"]) for k,v in results.items()])

if __name__=="__main__":
    main()
