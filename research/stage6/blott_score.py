"""Score a frozen CogLLM readout against Blott judgments with local labels."""

import argparse
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np

from .blott_audit import sha256


CANDIDATE_ID = "74c26ff02569b2b3bc28ed291bf15fd2f85db5b4c93a3ad7bfc34f6a5d333ea0"
INPUTS = {
    "stimuli": "281fb2fd37ba881ae49390c39a4204187d66a325fd722002c7e9714428c5f1fe",
    "design": "20c8c68285fcdb3bbc7ad610264e6caaf154e1e7ea299948ef4452b1c83f0fcf",
    "human_trials": "9b8a517bc8c12a5189bfb339e67659a42c203da763a9c997c94f445962b7b047",
}
CONDITIONS = ("PlausUA", "PlausAmb", "AnomUA", "AnomAmb")
TARGETS = ("understanding", "ease")
RANK_METHODS = (*TARGETS, "shorter_sentence")
EFFECT_METHODS = ("human_yes_fraction", *TARGETS)
PAIRS = tuple(combinations(range(len(CONDITIONS)), 2))
FAMILIES, READERS = 47, 96
DRAWS, SEED, BATCH = 10_000, 43, 128
QUANTILES = (0.025, 0.975)


def protocol():
    return {
        "scope": "Cross-study diagnostic of sentence reading scores against behavioral coherence judgments",
        "candidate_id": CANDIDATE_ID,
        "input_sha256": INPUTS,
        "protocol_sha256": sha256(Path(__file__).with_name("blott-coherence-holdout.md")),
        "scorer_sha256": sha256(Path(__file__)),
        "conditions": CONDITIONS,
        "primary": "understanding_minus_shorter_sentence_within_quartet_rank_agreement",
        "primary_interval": "paired two-way percentile bootstrap 95% interval",
        "readout": "expected ordinal score from complete sentence (grades one to five)",
        "secondary_readout": "ease",
        "control": "negative Unicode sentence character count, no fitting",
        "tie_credit": 0.5,
        "ranking_pairs_per_family": len(PAIRS),
        "length_matched_coherent_pairs": 15,
        "bootstrap": {
            "draws": DRAWS, "seed": SEED, "batch": BATCH, "quantiles": QUANTILES,
            "weighting": "independent reader and family multinomial weights; equal original family weight",
        },
        "normalized_effect": "(PlausUA - PlausAmb) / (PlausUA - AnomUA)",
        "per_trial_probabilities": False,
    }


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, item):
    path.write_text(json.dumps(item, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def rows(path):
    with path.open(encoding="utf-8") as stream:
        yield from map(json.loads, stream)


def signed_forecasts(stimuli_path, forecasts_path, seal_path, documents):
    seal = read_json(seal_path)
    if (seal["candidate_id"] != CANDIDATE_ID or seal["input_sha256"] != sha256(stimuli_path)
            or seal["predictions_sha256"] != sha256(forecasts_path)
            or seal["documents"] != len(documents) or seal["forwards"] != len(documents)):
        raise ValueError("Forecast identity, stimuli or prediction bytes differ from the seal")
    forecasts = {}
    for row in rows(forecasts_path):
        case = row["id"]
        if case not in documents or case in forecasts or len(row["sentences"]) != 1:
            raise ValueError(f"Expected one forecast for each isolated Blott sentence: {case}")
        sentence, = row["sentences"]
        if (sentence["start"] != 0 or sentence["end"] != len(documents[case])
                or type(sentence["token"]) is not int or sentence["token"] < 0):
            raise ValueError(f"Unexpected sentence-end coordinates: {case}")
        values = []
        for target in TARGETS:
            p = np.array(sentence[f"probability_{target}"], dtype=np.float64)
            expected = sentence[target]
            if (p.shape != (5,) or np.any((p < 0) | (p > 1)) or not np.isfinite(p).all()
                    or not math.isclose(float(p.sum()), 1, abs_tol=1e-6)
                    or not math.isclose(expected, float(p @ np.arange(1, 6)), abs_tol=1e-6)):
                raise ValueError(f"Invalid ordinal distribution or expectation for {case}/{target}")
            values.append(expected)
        forecasts[case] = values
    if forecasts.keys() != documents.keys():
        raise ValueError("Forecasts omit a sealed Blott input")
    return forecasts


def read_inputs(stimuli_path, design_path):
    documents = {}
    for row in rows(stimuli_path):
        case = row["id"]
        if set(row) != {"id", "text"} or case in documents or not row["text"]:
            raise ValueError("Expected a unique ID and isolated sentence text")
        documents[case] = row["text"]
    keys = {}
    seen = set()
    for row in rows(design_path):
        case = row["id"]
        family, condition = row["family"], row["condition"]
        if case in seen or case not in documents or (family, condition) in keys:
            raise ValueError("Each sentence must map to one distinct stimulus cell")
        seen.add(case)
        keys[family, condition] = case
    families = sorted({family for family, _ in keys})
    if (seen != documents.keys() or len(families) != FAMILIES
            or set(keys) != {(family, condition) for family in families for condition in CONDITIONS}):
        raise ValueError("Incomplete Blott experimental quartets")
    cases = np.array([[keys[family, condition] for condition in CONDITIONS] for family in families])
    words = np.array([[len(documents[key].split()) for key in group] for group in cases])
    chars = np.array([[len(documents[key]) for key in group] for group in cases])
    if np.any(words != words[:, :1]) or np.count_nonzero(chars[:, 0] == chars[:, 1]) != 15:
        raise ValueError("Word and character controls differ from the sealed materials")
    return documents, cases, chars


def human_matrices(path, cases):
    outcomes = {}
    for row in rows(path):
        key = row["reader_id"], row["id"]
        if key in outcomes or type(row["coherent"]) is not bool:
            raise ValueError("Duplicate or non-binary human response")
        outcomes[key] = row["coherent"]
    readers = sorted({reader for reader, _ in outcomes})
    if len(readers) != READERS or len(outcomes) != FAMILIES * READERS:
        raise ValueError("Unexpected human response inventory")
    if {case for _, case in outcomes} != set(cases.ravel()):
        raise ValueError("Human trial table does not match the sealed stimuli")
    recorded = np.zeros((len(readers), cases.size), dtype=np.int16)
    positive = np.zeros_like(recorded)
    for reader_index, reader in enumerate(readers):
        for case_index, case in enumerate(cases.ravel()):
            outcome = outcomes.get((reader, case))
            if outcome is not None:
                recorded[reader_index, case_index] = 1
                positive[reader_index, case_index] = outcome
    if not np.all(recorded.sum(axis=0) == 24) or not np.all(
            recorded.reshape(len(readers), FAMILIES, len(CONDITIONS)).sum(axis=-1) == 1):
        raise ValueError("Human trial counts do not reproduce counterbalanced assignment")
    return recorded, positive


def weighted_family_mean(values, family_weights):
    total = family_weights.sum(axis=1)
    if np.any(total == 0):
        raise ValueError("A fixed bootstrap draw omitted all families in an analysis")
    shape = (-1,) + (1,) * (values.ndim - 2)
    return np.einsum("nf,nf...->n...", family_weights, values) / total.reshape(shape)


def measures(recorded, positive, model, chars, reader_weights, family_weights):
    families = len(chars)
    sampled_trials = reader_weights @ recorded
    if np.any(sampled_trials == 0):
        raise ValueError("A fixed bootstrap draw omitted all readers for a variant")
    observed = ((reader_weights @ positive) / sampled_trials).reshape(-1, families, len(CONDITIONS))
    left, right = zip(*PAIRS)
    observed_order = np.sign(observed[:, :, left] - observed[:, :, right])
    model_and_control = np.concatenate((model, -chars[:, :, None]), axis=-1)
    model_order = np.sign(model_and_control[:, left, :] - model_and_control[:, right, :])
    agreement = (1 + observed_order[:, :, :, None] * model_order[None, :, :, :]) / 2
    score = weighted_family_mean(agreement, family_weights)
    same_chars = chars[:, 0] == chars[:, 1]
    matched = weighted_family_mean(agreement[:, same_chars, 0, :], family_weights[:, same_chars])
    replicated_model = np.broadcast_to((model - 1) / 4, (*observed.shape, len(TARGETS)))
    by_condition = weighted_family_mean(np.concatenate((observed[..., None], replicated_model), axis=-1), family_weights)
    penalty = by_condition[:, 0, :] - by_condition[:, 1, :]
    anchor = by_condition[:, 0, :] - by_condition[:, 2, :]
    return np.concatenate((
        score.mean(axis=1)[:, None, :], score, matched[:, None, :], by_condition,
        penalty[:, None, :], anchor[:, None, :],
    ), axis=1)


def point_and_bootstrap(recorded, positive, model, chars):
    readers = len(recorded)
    families = len(chars)
    point = measures(recorded, positive, model, chars, np.ones((1, readers), int), np.ones((1, families), int))
    reader_rng, family_rng = (np.random.default_rng(seed) for seed in np.random.SeedSequence(SEED).spawn(2))
    reader_p = np.full(readers, 1 / readers)
    family_p = np.full(families, 1 / families)
    chunks = []
    for start in range(0, DRAWS, BATCH):
        size = min(BATCH, DRAWS - start)
        chunks.append(measures(
            recorded, positive, model, chars, reader_rng.multinomial(readers, reader_p, size),
            family_rng.multinomial(families, family_p, size),
        ))
    return point[0], np.concatenate(chunks, axis=0)


def interval(value, draws):
    return {"estimate": float(value), "ci95": np.quantile(draws, QUANTILES).tolist()}


def report_metrics(point, draws):
    rank_sections = ("within_quartet", *(f"{CONDITIONS[i]}_vs_{CONDITIONS[j]}" for i, j in PAIRS),
                     "equal_character_count_coherent")
    ranking = {}
    for row, section in enumerate(rank_sections):
        metrics = {key: interval(point[row, i], draws[:, row, i]) for i, key in enumerate(RANK_METHODS)}
        for i, target in enumerate(TARGETS):
            metrics[f"{target}_minus_shorter_sentence"] = interval(
                point[row, i] - point[row, -1], draws[:, row, i] - draws[:, row, -1],
            )
        ranking[section] = metrics
    effects = {}
    by_condition = len(rank_sections)
    for i, method in enumerate(EFFECT_METHODS):
        by_condition_metrics = {
            code: interval(point[by_condition + j, i], draws[:, by_condition + j, i])
            for j, code in enumerate(CONDITIONS)
        }
        penalty = interval(point[-2, i], draws[:, -2, i])
        anchor = interval(point[-1, i], draws[:, -1, i])
        ratio = None
        if anchor["estimate"] > 0:
            ratio = {"estimate": penalty["estimate"] / anchor["estimate"], "ci95": None}
            if np.all(draws[:, -1, i] > 0):
                ratio = interval(ratio["estimate"], draws[:, -2, i] / draws[:, -1, i])
        effects[method] = {
            "condition_values": by_condition_metrics, "ambiguity_penalty": penalty,
            "coherent_vs_anomalous_unambiguous": anchor, "normalized_ambiguity_penalty": ratio,
        }
    return {"rank_agreement": ranking, "effects": effects}


def run_score(args):
    if read_json(args.protocol) != json.loads(json.dumps(protocol())):
        raise ValueError("Scoring rules or code differ from the prespecified protocol")
    if sha256(args.stimuli) != INPUTS["stimuli"]:
        raise ValueError("Unexpected Blott sentence input bytes")
    if sha256(args.design) != INPUTS["design"]:
        raise ValueError("Unexpected local Blott design bytes")
    documents, cases, chars = read_inputs(args.stimuli, args.design)
    forecasts = signed_forecasts(args.stimuli, args.forecasts, args.seal, documents)
    if sha256(args.human_trials) != INPUTS["human_trials"]:
        raise ValueError("Unexpected local Blott response bytes")
    recorded, positive = human_matrices(args.human_trials, cases)
    predictions = np.array([[forecasts[key] for key in group] for group in cases])
    point, draws = point_and_bootstrap(recorded, positive, predictions, chars)
    write_json(args.output, {
        "candidate_id": CANDIDATE_ID,
        "protocol_sha256": sha256(args.protocol),
        "forecast_sha256": sha256(args.forecasts),
        "forecast_seal_sha256": sha256(args.seal),
        "families": FAMILIES, "stimuli": len(documents), "readers": READERS, "trials": int(recorded.sum()),
        **report_metrics(point, draws),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    freeze = actions.add_parser("freeze")
    freeze.add_argument("--output", type=Path, required=True)
    score = actions.add_parser("score")
    for name in ("protocol", "stimuli", "design", "human-trials", "forecasts", "seal", "output"):
        score.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "freeze":
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(protocol(), output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
    else:
        run_score(args)


if __name__ == "__main__":
    main()
