"""Score an editor's suggested link even when a reader chose a nonreferring interpretation."""

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from research.reading_validation.transition_direct_data import PD, PDFINGER, PRON, display, gold_docs
from research.reading_validation.transition_direct_game import expert_entities
from research.reading_validation.transition_pointer_data import OUT, ROOT
from research.reading_validation.transition_pointer_selective import evaluate, validated_predictions
from research.reading_validation.transition_score import dump, sha


PROTOCOL=Path("research/reading_validation/transition-pointer-unconditional-protocol.md")


def labels(r, tree, crac, elems, tokens, minimum_readers=4):
    _, spans = expert_entities(crac, elems, tree)
    text, coords, _ = display(tree, elems, tokens)
    if text != r["text"]:
        raise ValueError("Rendered source and unmarked passage differ")
    words = np.asarray(r["words"])
    indices = {tuple(s): i for i, s in enumerate(r["words"])}
    markables = defaultdict(list)
    for pd in tree.iter("PDante"):
        ids = spans[pd.get("id")]
        if len(ids) == 1 and tokens[ids[0]].casefold() in PRON:
            markables[ids[0]].append(pd)
    result, audit = {}, Counter()
    for original, marks in markables.items():
        q = indices.get(coords[original])
        if q not in r["queries"] or len(marks) != 1:
            audit["uncertified_or_multiple_markables"] += 1
            continue
        mark = marks[0]
        if any(k in mark.attrib for k in ("start_chr", "end_chr", "visibility")):
            audit["edited_or_hidden"] += 1
            continue
        voters = defaultdict(set)
        invalid = False
        for annotation in mark.iter("anchor"):
            if annotation.get("mode") != "a":
                continue
            person, kind = annotation.get("user_id"), annotation.get("type")
            if not person or kind not in {"DO", "DN", "PR", "NR"}:
                raise ValueError("Unknown independent reader action")
            value = (kind, -1, -1)
            if kind == "DO":
                raw = spans.get(annotation.get("ante"))
                if not raw or max(raw) >= original:
                    invalid = True
                    break
                start, stop = coords[min(raw)][0], coords[max(raw)][1]
                inside = np.flatnonzero((words[:, 0] < stop) & (start < words[:, 1]))
                if not len(inside) or inside[-1] >= q:
                    invalid = True
                    break
                value = ("DO", int(inside[0]), int(inside[-1] + 1))
            voters[person].add(value)
        if invalid:
            audit["unmappable_prior_choice"] += 1
            continue
        valid = {v: next(iter(choices)) for v, choices in voters.items() if len(choices) == 1}
        audit["conflicting_repeated_voters"] += len(voters) - len(valid)
        if len(valid) < minimum_readers:
            audit["insufficient_distinct_voters"] += 1
            continue
        result[q] = valid
    return result, audit


def run():
    seal = json.loads((OUT/"source-seal.json").read_text())
    if sha(PD) != PDFINGER:
        raise ValueError("Original human annotations changed")
    p, forecast_sha = validated_predictions(OUT, seal, seal["protocol_sha256"])
    rows = {r["id"]: r for r in map(json.loads, (ROOT/"pointer-stimuli.jsonl").read_text().splitlines()) if r["fold"]=="test"}
    selected = json.loads((OUT/"selective/source-selection.json").read_text())["methods"]["headwise"]["selected"]
    threshold = selected["threshold"]
    cohorts, responses, audit = defaultdict(list), Counter(), Counter()
    people, checked = set(), set()
    unknown = 0
    observed_correct = 0.
    all_flags = 0
    for doc, _, tree, crac, elem, tokens in gold_docs():
        if doc not in rows:
            continue
        r = rows[doc]
        answers, omissions = labels(r, tree, crac, elem, tokens)
        audit.update(omissions)
        for q in r["queries"]:
            checked.add((doc, q))
            z = p[f"{doc}/{q}"][-1].astype(float)
            top = z == z.max()
            confidence = 1 / np.exp(z-z.max()).sum()
            flagged = bool(confidence >= threshold)
            all_flags += flagged
            if q not in answers:
                unknown += flagged
                continue
            voters = answers[q]
            size = len(voters)
            people.update(voters)
            hits = 0.
            nonprior = 0
            for typ, a, b in voters.values():
                responses[typ] += 1
                if typ == "DO":
                    hits += top[a:b].sum() / top.sum()
                else:
                    nonprior += 1
            correct = float(hits/size)
            observed_correct += flagged*correct
            fields=(confidence,correct,size)
            cohorts["all"].append((doc,*fields))
            cohorts["distinct_works"].append((r["group"],*fields))
            if nonprior:
                cohorts["any_nonprior"].append((doc,*fields))
            if 2*nonprior>size:
                cohorts["majority_nonprior"].append((doc,*fields))
                audit["majority_nonprior_flags"] += flagged
            if nonprior==size:
                audit["pure_nonprior_pronouns"] += 1
                audit["pure_nonprior_flags"] += flagged
    if len(checked)!=seal["offered_gold_query_positions"]:
        raise ValueError("A forecast pronoun disappeared from the applicability audit")
    result = {k: {"threshold":evaluate(v,threshold)[0],"always":evaluate(v,0.)[0]} for k,v in cohorts.items()}
    lo,hi=observed_correct,observed_correct+unknown
    bounds={"all_forecasts":len(checked),"all_links":all_flags,"unadjudicated_links":unknown,"observed_expected_correct_links":lo,
            "sharp_equal_pronoun_precision":[lo/all_flags,hi/all_flags],
            "sharp_equal_pronoun_wrong_per_100":[100*(all_flags-hi)/len(checked),100*(all_flags-lo)/len(checked)]}
    dump(OUT/"all-reader-actions.json",{"protocol_sha256":sha(PROTOCOL),"forecast_sha256":forecast_sha,"threshold":threshold,
         "answers":dict(responses),"distinct_voters":len(people),"exclusions":dict(audit),"cohorts":result,"full_text_bounds":bounds})
    print(result["all"]["threshold"],bounds,flush=True)


if __name__=="__main__":
    run()
