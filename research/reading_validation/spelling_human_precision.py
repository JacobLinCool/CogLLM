"""Compare exact-spelling disagreement across disjoint whole-reader halves and frozen forecasts."""

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from research.backbone_transfer.extract import (
    digest,
    read_json,
    stream_rows,
    write_json,
)
from research.reading_validation.reader_history_external import (
    answer_normalizer,
    get_geometry,
    records,
    sources,
)
from research.reading_validation.reader_history_materials import DIR
from research.reading_validation.reader_history_materials import OUT as HISTORY
from research.reading_validation.spelling_agreement import OUT as SPELLING

OUT = Path("research/results/reading-validation-spelling-precision")
PLAN = Path(__file__).with_name("spelling-human-precision-protocol.md")
INPUT = DIR / "independent-future-items.json"
TEXT = DIR / "verified-trial-stimuli.jsonl"
PROVO = DIR.parent / "provo-predictability.csv"
METHODS = ("peer", "conditional", "direct", "constant")


def inputs():
    history = read_json(HISTORY / "external-lock.json")
    fit = read_json(SPELLING / "fit-lock.json")
    raw = sources()
    forecast = SPELLING / "peelle-forecasts.npz"
    if history["future_item_assignments"] != digest(INPUT) or raw != history["inputs"]:
        raise ValueError(
            "The original readers or chronological observation assignments changed"
        )
    if fit["studies"]["peelle"]["forecast"] != digest(forecast):
        raise ValueError("Frozen Peelle forecasts have changed")
    return {
        "source_files": raw,
        "chronology": digest(INPUT),
        "stimuli": digest(TEXT),
        "forecast": digest(forecast),
        "external_lock": digest(HISTORY / "external-lock.json"),
        "model_lock": digest(SPELLING / "fit-lock.json"),
        "protocol": digest(PLAN),
        "provo_predictability": digest(PROVO),
    }


def registered():
    OUT.mkdir(exist_ok=True)
    write_json(OUT / "input-lock.json", inputs())


def responses(ids):
    geometry, _ = get_geometry()
    by_record = defaultdict(list)
    for participant, sheet, row, field, item, stem in read_json(INPUT)["chronological"]:
        by_record[sheet, row].append((participant, field, item, stem))
    normalizer = answer_normalizer()
    identities = {}
    stem_form = {}
    form_stem = []
    observed = []
    for sheet, row, entry, _ in records():
        keys = (sheet, row)
        rid, _, _, sites = geometry[keys]
        eligible = {(field, item): uid for uid, field, item in sites}
        for participant, field, item, stem in by_record.get(keys, ()):
            if participant != rid or eligible.get((field, item)) != ids[stem]:
                raise ValueError(
                    "The authentic reader or printed stem differs from the registered row"
                )
            _, spelling = normalizer(entry[field], item)
            if spelling is None:
                continue
            if identities.setdefault(rid, entry["WorkerId"]) != entry["WorkerId"]:
                raise ValueError(
                    "A retained reader index links different source people"
                )
            key = stem, spelling
            if key not in stem_form:
                stem_form[key] = len(form_stem)
                form_stem.append(stem)
            observed.append((rid, stem, stem_form[key]))
    if len(identities) != 233 or len(set(identities.values())) != 233:
        raise ValueError("Could not recover the original 233 distinct readers")
    a = np.asarray(observed, dtype=np.int32)
    reader_ids, readers = np.unique(a[:, 0], return_inverse=True)
    if len(a) != 297927 or len(np.unique(readers * len(ids) + a[:, 1])) != len(a):
        raise ValueError("The original distinct reader–stem response count differs")
    return readers, a[:, 1], a[:, 2], np.asarray(form_stem), len(reader_ids)


def disagreement(form_counts, form_stem, stem_counts):
    same = np.bincount(
        form_stem, weights=form_counts * (form_counts - 1), minlength=len(stem_counts)
    )
    opportunities = stem_counts * (stem_counts - 1)
    return 1 - np.divide(
        same, opportunities, out=np.zeros_like(same), where=opportunities > 0
    )


def split_samples(readers, stems, forms, form_stem, n_stems, n_readers):
    size = np.bincount(stems, minlength=n_stems)
    totals = np.bincount(forms, minlength=len(form_stem))
    mask = np.zeros(n_readers, dtype=bool)
    minimum = size.copy()
    maximum = np.zeros_like(size)
    rng = np.random.default_rng(715513)
    observed = np.empty((256, 2, n_stems))
    counts = np.empty((256, 2, n_stems), dtype=np.int16)
    for i in range(len(observed)):
        mask[:] = False
        mask[rng.permutation(n_readers)[: n_readers // 2]] = True
        select = mask[readers]
        first = np.bincount(forms[select], minlength=len(totals))
        current = np.bincount(stems[select], minlength=n_stems)
        counts[i] = current, size - current
        observed[i, 0] = disagreement(first, form_stem, current)
        observed[i, 1] = disagreement(totals - first, form_stem, size - current)
        minimum = np.minimum(minimum, counts[i].min(0))
        maximum = np.maximum(maximum, counts[i].max(0))
    return (
        observed,
        counts,
        minimum,
        maximum,
        size,
        disagreement(totals, form_stem, size),
    )


def interval(items):
    return np.quantile(items, [0.025, 0.975], axis=0)


def compare(observed, forecasts):
    # The peer prediction exchanges the two independently observed halves at the same stem.
    per_stem = []
    per_partition = []
    for j in range(len(METHODS)):
        prediction = observed[:, ::-1] if j == 0 else forecasts[None, None, :, j - 1]
        delta = prediction - observed
        per_stem.append(
            np.stack(
                [abs(delta).mean((0, 1)), delta.mean((0, 1)), (delta**2).mean((0, 1))]
            )
        )
        per_partition.append(abs(delta).mean((1, 2)))
    quantities = np.stack(per_stem, axis=1)
    split_means = np.stack(per_partition, axis=1)
    size = quantities.shape[-1]
    rng = np.random.default_rng(715514)
    boot = np.empty((4000, 3, len(METHODS)))
    for start in range(0, len(boot), 50):
        weights = rng.multinomial(size, np.full(size, 1 / size), size=50)
        boot[start : start + 50] = np.einsum("bs,tms->btm", weights / size, quantities)
    means = quantities.mean(-1)
    boot[:, 2] = np.sqrt(boot[:, 2])
    means[2] = np.sqrt(means[2])
    low, high = interval(boot)
    metrics = ("mae", "signed_prediction_minus_observed", "rmse")
    rates = {}
    for m, name in enumerate(METHODS):
        scores = {
            s: {"estimate": means[k, m], "stem_ci95": [low[k, m], high[k, m]]}
            for k, s in enumerate(metrics)
        }
        scores["mae_across_partitions_middle95"] = interval(split_means[:, m]).tolist()
        rates[name] = scores
    differences = {}
    for left, right in itertools.combinations(range(len(METHODS)), 2):
        d = boot[:, 0, right] - boot[:, 0, left]
        differences[f"{METHODS[right]}_minus_{METHODS[left]}"] = {
            "estimate": means[0, right] - means[0, left],
            "stem_ci95": interval(d).tolist(),
        }
    return quantities, rates, differences


def grade():
    paths = inputs()
    if paths != read_json(OUT / "input-lock.json"):
        raise ValueError("Precision inputs changed after registration")
    with np.load(SPELLING / "peelle-forecasts.npz", allow_pickle=False) as data:
        ids = data["ids"].tolist()
        names = data["methods"].tolist()
        forecast = data["forecast"][:, [names.index(k) for k in METHODS[1:]]]
    if ids != [r["id"] for r in stream_rows(TEXT)] or len(set(ids)) != 3085:
        raise ValueError("Ordered stems differ from their signed native exposures")
    readers, stems, forms, form_stem, people = responses(ids)
    half, counts, smallest, largest, size, full = split_samples(
        readers, stems, forms, form_stem, len(ids), people
    )
    eligible = smallest >= 20
    if not eligible.any():
        raise ValueError("No stem has adequate observations in every partition")
    metrics, rates, diffs = compare(half[:, :, eligible], forecast[eligible])
    per_stem = OUT / "stem-precision.csv"
    with per_stem.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "stem_sha256",
                "reader_count",
                "minimum_half",
                "maximum_half",
                "full_panel_disagreement",
                *[name + "_half_mae" for name in METHODS],
            ]
        )
        for j, i in enumerate(np.flatnonzero(eligible)):
            writer.writerow(
                [ids[i], size[i], smallest[i], largest[i], full[i], *metrics[0, :, j]]
            )
    with PROVO.open(encoding="mac_roman") as f:
        provo_fields = csv.DictReader(f).fieldnames
    whole = abs(forecast - full[:, None]).mean(0)
    original = read_json(SPELLING / "scores.json")["studies"]["peelle"]["methods"]
    if not np.allclose(
        whole, [original[k]["mae"] for k in METHODS[1:]], atol=1e-12, rtol=0
    ):
        raise ValueError("Independent restoration did not reproduce the full-panel MAE")
    result = {
        "forecast_and_source_lock_sha256": digest(OUT / "input-lock.json"),
        "estimator_sha256": digest(Path(__file__)),
        "per_stem_sha256": digest(per_stem),
        "reader_stem_observations": len(readers),
        "people": people,
        "people_in_each_half": [people // 2, people - people // 2],
        "partitions": len(half),
        "stems": len(ids),
        "stems_eligible_in_every_partition": int(eligible.sum()),
        "observations_on_eligible_stems": int(size[eligible].sum()),
        "minimum_median_maximum_people_per_stem": np.quantile(
            size[eligible], [0, 0.5, 1]
        ).tolist(),
        "minimum_median_maximum_half_people": np.quantile(
            counts[:, :, eligible], [0, 0.5, 1]
        ).tolist(),
        "all_stems_full_panel_mae": dict(zip(METHODS[1:], whole, strict=True)),
        "half_results": rates,
        "paired_mae_differences": diffs,
        "stem_bootstrap_draws": 4000,
        "provo_source_columns": provo_fields,
    }
    write_json(OUT / "scores.json", result)
    print(json.dumps({k: v["mae"] for k, v in rates.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "score"))
    if parser.parse_args().action == "register":
        registered()
    else:
        grade()
