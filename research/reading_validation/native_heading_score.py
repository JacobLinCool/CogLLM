"""Compare the frozen native head, the projected observer, and literal matched-body null."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil

import numpy as np

from research.expository_lora.score import ARMS, DRAWS, SEED, interval, kerwer_effects, kerwer_readers
from research.reading_validation.kerwer_heading_l4 import PROTOCOL
from research.reading_validation.local_features import CACHE, DATA, OUT, PACKET, verified_rows
from research.reading_validation.local_forecast import digest, rows
from research.reading_validation.local_inference import DEFAULT, REVISION


def compare(observed, base, current):
    by_item, draws, _ = kerwer_effects(observed, np.column_stack((base, current)))
    names = ("human", "base", "native", "native_minus_base", "base_minus_human", "native_minus_human",
             "base_item_mae", "native_item_mae", "native_minus_base_item_mae")
    return {name: interval(by_item[:, j].mean(), draws[:, j]) for j, name in enumerate(names)}


def evaluate(source):
    source = Path(source)
    receipt = json.loads((source / "receipt.json").read_bytes())
    forecasts = source / "native-headings.jsonl"
    packet = DATA / "native-heading-stimuli.jsonl"
    if (receipt["predictions_sha256"] != digest(forecasts) or receipt["stimuli_sha256"] != digest(packet)
            or receipt["bundle_sha256"] != digest(DEFAULT / "manifest.json")
            or receipt["revision"] != REVISION or receipt["native_width"] != 2048
            or receipt["passes"] != 24 or receipt["spans"] != 362):
        raise ValueError("Native output does not replay the study")
    published = json.loads((OUT / "headings/scores.json").read_bytes())
    prior_path = OUT / "headings/predictions.json"
    if digest(prior_path) != published["receipts"]["predictions_sha256"]:
        raise ValueError("Historical projected forecasts differ")
    prior = {r["id"]: r for r in json.loads(prior_path.read_bytes())}
    native = {r["id"]: r["sentences"] for r in rows(forecasts)}
    stimuli = {r["id"]: r for r in rows(packet)}
    families = defaultdict(dict)
    for r in verified_rows("kerwer", "layout.jsonl"):
        key = r["id"]
        families[r["doi"]][r["arm"]] = key
        if (r["matched_stops"] != [s["end"] for s in native[key]]
                or r["matched_stops"] != [s["stop"] for s in prior[key]["sentences"]]
                or stimuli[key]["spans"] != [[s["start"], s["end"]] for s in native[key]]):
            raise ValueError("The aligned source boundaries changed")
    sorted_cases = [(doi, [family[a] for a in ARMS]) for doi, family in sorted(families.items())]
    ratings = CACHE / "kerwer-2021/holdout/scoring/public-study-ratings.jsonl"
    expected = json.loads((PACKET / "source-audit.json").read_bytes())["sources"]["kerwer"]["source_ratings_sha256"]
    if digest(ratings) != expected or expected != published["receipts"]["source_ratings"]:
        raise ValueError("The accepted historical responses changed")
    observed = kerwer_readers(ratings, [p for doi, p in sorted_cases])
    agreement = []
    identical_prefix_deltas = []
    for i, (doi, pair) in enumerate(sorted_cases):
        indexes = [j for j in range(len(native[pair[0]])) if all(native[k][j]["rating"] is not None for k in pair)]
        if not indexes:
            raise ValueError("A source family has no certified paired sentence")
        model = [[native[k][j]["rating"]["understanding"] for j in indexes] for k in pair]
        projected = [[prior[k]["sentences"][j]["forecasts"]["local"] for j in indexes] for k in pair]
        for j in indexes:
            prefixes = [stimuli[k]["text"][:native[k][j]["end"]] for k in pair]
            if prefixes[0] == prefixes[1]:
                identical_prefix_deltas.append(abs(native[pair[1]][j]["rating"]["understanding"]
                                                    - native[pair[0]][j]["rating"]["understanding"]))
        responses = observed[:, i]
        human = responses.sum(axis=0) / (responses > 0).sum(axis=0)
        agreement.append(dict(doi=doi, presentations=pair, pairs=len(native[pair[0]]), certified=len(indexes),
                              human=float(human[1] - human[0]),
                              native=float((np.mean(model[1]) - np.mean(model[0])) * 7 / 4),
                              projected=float((np.mean(projected[1]) - np.mean(projected[0])) * 7 / 4)))
    projected = np.array([r["projected"] for r in agreement])
    deployed = np.array([r["native"] for r in agreement])
    human = np.array([r["human"] for r in agreement])
    zero = compare(observed, np.zeros_like(deployed), deployed)
    reduced = compare(observed, projected, deployed)
    counts = dict(readers=len(observed), ratings=int((observed > 0).sum()), families=len(agreement),
                  possible_pairs=sum(d["pairs"] for d in agreement), complete_pairs=sum(d["certified"] for d in agreement),
                  human_positive=int((human > 0).sum()), native_positive=int((deployed > 0).sum()),
                  sign_matches=int((np.sign(deployed) == np.sign(human)).sum()))
    if (counts["readers"], counts["ratings"], counts["families"], counts["possible_pairs"]) != (166,1283,12,181):
        raise ValueError("Human sample changed")
    if abs(zero["human"]["estimate"] - published["estimates"]["local"]["human_effect"]["estimate"]) > 1e-10:
        raise ValueError("Failed to reproduce human scale and effect")
    if counts["complete_pairs"] == 181 and abs(reduced["base"]["estimate"] - published["estimates"]["local"]["model_effect"]["estimate"]) > 1e-10:
        raise ValueError("Failed to replay historical projected observer")
    out = OUT / "native-headings"
    out.mkdir(parents=True, exist_ok=True)
    for name in ("native-headings.jsonl", "receipt.json"):
        shutil.copyfile(source / name, out / name)
    prefix_check = dict(paired_stops=len(identical_prefix_deltas),
                        largest_difference=max(identical_prefix_deltas, default=None))
    report = dict(prefix_check=prefix_check, counts=counts, native_vs_body=zero, native_vs_projected=reduced, families=agreement,
                  draws=DRAWS, seed=SEED, protocol_sha256=digest(PROTOCOL), ratings_sha256=expected,
                  scoring_code_sha256=digest(Path(__file__)), receipt_sha256=digest(source / "receipt.json"),
                  historical_sha256=digest(prior_path))
    (out / "scores.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(counts=counts, vs_body=zero, vs_projected=reduced), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_outputs", type=Path)
    evaluate(parser.parse_args().native_outputs)
