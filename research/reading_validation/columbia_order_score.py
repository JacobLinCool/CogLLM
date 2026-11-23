"""Score a held writer's actual continuation against every other still available news unit."""

import json
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

from research.reading_validation.coherence_order_data import read
from research.reading_validation.columbia_order_data import OUT, TOPICS, units
from research.reading_validation.local_forecast import digest
from research.reading_validation.memory_data import h


@dataclass(frozen=True)
class Choice:
    topic: str
    person: int
    predecessor: str
    available: tuple[str, ...]
    selected: int
    pair_indices: tuple[int, ...]


def source_group(event):
    return "salvador" if event in ("salvadorA", "salvadorB") else event


def design(keys, order_tables, excluded):
    data = units()
    index = {k: i for i, k in enumerate(keys)}
    trials = []
    references = {}
    omitted = {}
    for topic in TOPICS:
        items = tuple(sorted(data[topic]))
        required = {f"{topic}/{i}/{j}" for i in items for j in items if i != j}
        if topic in excluded or required - set(index):
            omitted[topic] = (
                "training material overlap"
                if topic in excluded
                else "incomplete native candidate set"
            )
            continue
        orders = order_tables[topic]
        if not len(orders) > 1 or any(sorted(o) != list(items) for o in orders):
            raise ValueError(f"Incomplete human permutation in {topic}")
        chosen = int(h(topic)[:16], 16) % len(orders)
        references[topic] = {"person": chosen, "ordering": orders[chosen]}
        for reader, order in enumerate(orders):
            if reader == chosen:
                continue
            for j in range(1, len(order) - 1):
                prev = order[j - 1]
                possible = tuple(sorted(set(items) - set(order[:j])))
                next_one = possible.index(order[j])
                ix = tuple(index[f"{topic}/{prev}/{v}"] for v in possible)
                trials.append(Choice(topic, reader, prev, possible, next_one, ix))
    if not trials:
        raise ValueError("No direct human choices")
    return trials, references, omitted


def decision(score, winner):
    arr = np.asarray(score, float)
    true = arr[winner]
    other = np.delete(arr, winner)
    ties = np.isclose(true, other, atol=1e-10, rtol=0)
    better = (true > other) & ~ties
    win = np.mean(better + 0.5 * ties)
    top = 0.0 if np.any((other > true) & ~ties) else 1 / (1 + ties.sum())
    return float(win), float(top)


def event_average(trials, values, topics):
    people = defaultdict(list)
    for t, v in zip(trials, values, strict=True):
        people[t.topic, t.person].append(v)
    event = defaultdict(list)
    for (topic, r), v in people.items():
        event[topic].append(np.asarray(v).mean(axis=0))
    return np.asarray([np.mean(event[x], axis=0) for x in topics])


def sampling(topics):
    family = np.array([source_group(k) for k in topics])
    groups, code = np.unique(family, return_inverse=True)
    g = len(groups)
    m = np.random.default_rng(425026).multinomial(g, np.full(g, 1 / g), size=10000)[
        :, code
    ]
    return family, m / m.sum(1, keepdims=True)


def estimate(a, w, number=1):
    b = w @ a
    return {
        "estimate": float(a.mean()),
        "ci95": np.quantile(b, [0.025, 0.975]).tolist(),
        "ci95_simultaneous": np.quantile(
            b, [0.025 / number, 1 - 0.025 / number]
        ).tolist(),
    }


def difference(a, w, group, number):
    q = estimate(a, w, number)
    levels = np.unique(group)
    contributions = np.array([a[group == g].sum() / len(a) for g in levels])
    flip = np.asarray(list(product((-1.0, 1.0), repeat=len(levels)))) @ contributions
    q["one_sided_exact_sign_flip"] = float(np.mean(flip >= a.mean() - 1e-12))
    q["bonferroni_p"] = min(1.0, q["one_sided_exact_sign_flip"] * number)
    return q


def load_design():
    prior = read(OUT / "forecast-seal.json")
    plan = read(OUT / "instrument-lock.json")
    answers = read(OUT / "observed-orders.json")
    if (
        digest(OUT / "forecasts.npz") != prior["forecasts"]
        or digest(OUT / "pair-geometry.json") != read(OUT / "inputs.json")["geometry"]
    ):
        raise ValueError("Forecasts or material changed")
    if any(
        digest(Path(__file__).with_name(n)) != sha
        for n, sha in {**prior["code"], **plan["registered_code"]}.items()
    ):
        raise ValueError("The model or estimators have changed since registration")
    with np.load(OUT / "forecasts.npz", allow_pickle=False) as f:
        forecast = dict(f)
    excluded = {
        topic
        for topic, _ in read(OUT / "inputs.json")[
            "unit_intersection_with_bilingual_fit"
        ]
    }
    trials, reference, omitted = design(
        forecast.pop("keys"), answers["orders"], excluded
    )
    return forecast, trials, reference, omitted


def main():
    forecast, trials, ref, omitted = load_design()
    topics = sorted({t.topic for t in trials})
    family, draw = sampling(topics)
    output = {}
    per_event = {}
    supported = {}
    rates = {}
    for arm in [*forecast, "independent_human_order", "uniform"]:
        local = []
        for t in trials:
            if arm == "uniform":
                scores = np.zeros(len(t.available))
            elif arm == "independent_human_order":
                u = ref[t.topic]["ordering"]
                start = u.index(t.predecessor) + 1
                scores = -np.array([abs(u.index(k) - start) for k in t.available])
            else:
                scores = forecast[arm][list(t.pair_indices)]
            local.append(decision(scores, t.selected))
        estimates = event_average(trials, local, topics)
        rates[arm] = estimates[:, 0]
        output[arm] = {
            metric: estimate(estimates[:, i], draw)
            for i, metric in enumerate(("pairwise_choice", "top_one"))
        }
        per_event[arm] = {k: v.tolist() for k, v in zip(topics, estimates, strict=True)}
    for key, d in {
        "adaptive_over_uniform": rates["adaptive.centered"] - 0.5,
        "adaptive_over_raw_contextual_surprisal": rates["adaptive.centered"]
        - rates["mean_surprisal.centered"],
        "augmented_over_adaptive": rates["augmented.centered"]
        - rates["adaptive.centered"],
    }.items():
        supported[key] = difference(d, draw, family, 3)
    data = {
        "topic_sets": len(topics),
        "topic_families": len(set(family)),
        "real_test_orders": len({(t.topic, t.person) for t in trials}),
        "nontrivial_human_decisions": len(trials),
        "alternative_wins_compared": sum(len(t.available) - 1 for t in trials),
        "reserved_reference_orders": len(ref),
        "reference_row_0based": {k: v["person"] for k, v in sorted(ref.items())},
        "omitted": omitted,
        "methods": output,
        "primary": supported,
        "paired_vs_independent_human": difference(
            rates["adaptive.centered"] - rates["independent_human_order"],
            draw,
            family,
            1,
        ),
        "per_event_pairwise_and_top_one": per_event,
        "forecast_seal": digest(OUT / "forecast-seal.json"),
        "source_order_archive": read(OUT / "observed-orders.json")["source_sha256"],
    }
    (OUT / "scores.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    print(data["primary"], output["independent_human_order"])


if __name__ == "__main__":
    main()
