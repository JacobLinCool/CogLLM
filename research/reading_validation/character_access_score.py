"""Test preregistered old-character contrasts on every original story family."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.backbone_transfer.extract import digest, write_json
from research.reading_validation.character_access_data import INPUT, OUT, PLAN
from research.reading_validation.transition_goal_access_score import METHODS, describe

ORDER = ("first","rementioned","neutral","new")
EFFECTS = {"neutral_minus_new": (2,3), "rementioned_minus_neutral": (1,2)}


def load_bank(bank, input_path, frozen, protocol, conditions):
    input_rows = [json.loads(s) for s in input_path.read_text().splitlines()]
    spec = json.loads(frozen.read_text())
    receipt = json.loads((bank/"receipt.json").read_text())
    file = bank/"goal-accessibility.jsonl"
    if (spec["protocol"] != digest(protocol) or digest(input_path) != spec["input"] or
            receipt["source_sha256"] != spec["input"] or digest(file) != receipt["forecast_sha256"] or
            receipt["native_forwards"] != len(input_rows) or receipt["participant_outcomes_in_input"] != 0):
        raise ValueError("Article, forecast, outcomes, or protocol seal is inconsistent")
    rows = {r["id"]: r for r in map(json.loads, file.read_text().splitlines())}
    if len(rows)!=len(input_rows) or set(rows)!={q["id"] for q in input_rows}:
        raise ValueError("Every original text must be predicted exactly once")
    matrices = {}; common_error = 0.
    for family, group in spec["families"].items():
        items = [rows[group["variants"][word]["id"]] for word in conditions[1:]]
        if len({i["prefix_token_sha256"] for i in items}) != 1:
            raise ValueError("Common introduction changes with its future continuation")
        names = [group["probe"], *group["control_words"]]
        for extent in ("all","introduction"):
            for method in METHODS:
                before = np.array([[i["first"][n][extent][method] for n in names] for i in items])
                common_error = max(common_error, float(np.ptp(before,axis=0).max()))
                after = [[i["last"][n][extent][method] for n in names] for i in items]
                matrices.setdefault((extent,method),[]).append([before.mean(0),*after])
    if common_error > 1e-4:
        raise ValueError("Observed estimates violate the unchanged prefix check")
    return spec,receipt,{k:np.array(v) for k,v in matrices.items()},common_error


def compare(cubes, keep, contrasts):
    rng = np.random.default_rng(394721)
    B = rng.integers(0,len(keep), size=(10000,len(keep)))
    results = {}
    for extent in ("all","introduction"):
        for component in ("target","control_mean","target_minus_controls"):
            signal = {}
            for method in METHODS:
                z = cubes[extent,method][keep]
                control = z[:,:,1:].mean(2)
                signal[method] = {"target":z[:,:,0], "control_mean":control,
                                  "target_minus_controls":z[:,:,0]-control}[component]
            for m in ("recency","uniform"):
                signal["native_minus_"+m] = signal["native"]-signal[m]
            for m, values in signal.items():
                results[f"{extent}.{component}.{m}"] = dict(
                    means=values.mean(0).tolist(),
                    effects={k:describe(values[:,a]-values[:,b],B) for k,(a,b) in contrasts.items()})
    return results


def main(bank, study="name"):
    from research.reading_validation.character_object_access_data import ARM, INPUT as OBJECTS
    path = INPUT if study=="name" else OBJECTS
    spec, receipt, cubes, error = load_bank(bank, path, OUT/(study+"-input.json"), PLAN, ORDER)
    if study=="object" and spec["addendum"]!=digest(ARM):
        raise ValueError("Original object probe addendum changed")
    n = len(spec["families"])
    if n!=32:
        raise ValueError("Incomplete set of published stories")
    overlaps = [k for k,g in spec["families"].items() if any(
        v["shared_20_lexical_words_with_old_goal_study"] for v in g["variants"].values())]
    clean = np.array([j for j,k in enumerate(spec["families"]) if k not in overlaps],int)
    contrasts = {**EFFECTS,"rementioned_minus_new":(1,3)}
    results = compare(cubes,np.arange(n),contrasts)
    item_file = OUT/(study+"-item-forecasts.npz")
    np.savez_compressed(item_file, family=np.array(list(spec["families"])),
        **{f"{s}.{m}":v for (s,m),v in cubes.items()})
    repeated = np.array([j for j,g in enumerate(spec["families"].values()) if any(
        v["intervening_primary_mentions"]>0 for v in g["variants"].values())],int)
    main_effects = list(EFFECTS) if study=="name" else ["rementioned_minus_new"]
    clean_mentions=np.setdiff1d(np.arange(n),repeated)
    write_json(OUT/(study+"-scores.json"),dict(protocol=digest(PLAN),input=spec["input"],receipt=receipt,
        item_forecasts=digest(item_file),conditions=ORDER,families=n,same_prefix_max_abs=error,
        previous_goal_stories_with_20_word_overlap=overlaps,forecasts=results,
        families_with_any_experimental_repetition=len(repeated),
        without_experimental_repetition=compare(cubes,clean_mentions,contrasts) if len(clean_mentions) not in (0,n) else None,
        with_experimental_repetition=compare(cubes,repeated,contrasts) if len(repeated) not in (0,n) else None,
        without_overlapping_stimuli=compare(cubes,clean,contrasts) if len(clean) not in (0,n) else None,
        participant_count=80 if study=="name" else 88,
        human_mean_recognition_ms={"new":757,"neutral":654,"rementioned":556} if study=="name" else {"new":984,"rementioned":943},
        primary={k:results["all.target_minus_controls.native_minus_recency"]["effects"][k] for k in main_effects}))


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("bank",type=Path);p.add_argument("--study",choices=("name","object"),default="name")
    args=p.parse_args();main(args.bank,args.study)
