"""Compare outcome-free target texts with the historical portable reader's supervised inputs."""

import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".cache/cogllm"
DATA = CACHE / "reading-validation"
RESULT = ROOT / "research/results"
DEST = RESULT / "reading-validation-release"
WORD = re.compile(r"[^\W_]+", flags=re.UNICODE)
WINDOW = 12


def read(path):
    return json.loads(path.read_bytes())


def rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def words(text):
    return tuple(WORD.findall(text.casefold()))


def targets():
    """Use only target material, visible stops and group IDs from the previously sealed runs."""
    result = defaultdict(dict)
    for name in ("plaba", "kerwer"):
        for row in rows(CACHE / "expository-lora/prepared" / name / "stimuli.jsonl"):
            result[name][row["id"]] = row["text"]
    for row in rows(DATA / "jargon/local-stimuli.jsonl"):
        result["guo"][row["id"]] = row["text"]
    for row in rows(
        CACHE / "semantic-tracking/frick-consistency/prepared/stimuli.jsonl"
    ):
        if len(row["stops"]) > 1:
            result["frick_e1"][row["id"]] = row["text"]

    def load_selection(file, selection):
        found = set()
        for row in rows(file):
            key = row["id"]
            for dataset in selection.get(key, ()):
                result[dataset][key] = row["text"]
                found.add(key)
        if found != selection.keys():
            raise ValueError(f"Unmatched target geometry in {file.name}")

    selection = defaultdict(set)
    for row in read(DATA / "readme/materials.json")["rows"]:
        if row["split"] == "test":
            selection[row["id"]].add("readme++")
    load_selection(DATA / "external-stimuli.jsonl", selection)
    selection = defaultdict(set)
    for row in read(RESULT / "reading-validation-prior/materials.json"):
        if row["split"] == "test":
            selection[row["id"]].add("cefr_sp_" + row["domain"].casefold())
    load_selection(DATA / "prior-stimuli.jsonl", selection)
    selection = defaultdict(set)
    for row in read(RESULT / "reading-validation-transition-external/stimuli.json")[
        "material"
    ]:
        if (row["source"], row["split"]) in (
            ("cohesentia", "holdout"),
            ("decor", "external"),
        ):
            selection[row["doc"]].add(row["source"])
    load_selection(DATA / "cohesion-stimuli.jsonl", selection)
    selection = {
        row["doc"]: {"snac_movies"}
        for row in read(RESULT / "reading-validation-transition-snac/stimuli.json")[
            "documents"
        ]
        if row["split"] == "test"
    }
    load_selection(DATA / "snac-stimuli.jsonl", selection)
    selection = {
        key: {"tellmewhy"}
        for key in read(RESULT / "reading-validation-events/splits.json")["test"]
    }
    load_selection(DATA / "event-stimuli.jsonl", selection)
    with (DATA / "revision/simpda2022.csv").open(
        newline="", encoding="utf-8-sig"
    ) as stream:
        for row in csv.DictReader(stream):
            for field in ("original", "simplified"):
                text = row["Input." + field]
                result["simpda_" + field][hashlib.sha256(text.encode()).hexdigest()] = (
                    text
                )
    return result


def learning_material():
    path = CACHE / "adaptive-joint/inputs"
    receipt = read(path / "receipt.json")
    verified = {}
    for name, expected in receipt["files_sha256"].items():
        if name.startswith("token/") or name.endswith((".jsonl", "/index.json")):
            if sha(path / name) != expected:
                raise ValueError(f"Historical learning packet member changed: {name}")
            verified[name] = expected
    if receipt["sources"] != 24 or receipt["formal_labels_in_worker"]:
        raise ValueError("Unexpected historical training packet")
    tokens = {r["id"]: r["text"] for r in rows(path / "token/stimuli.jsonl")}
    observed = defaultdict(set)
    attribution = defaultdict(set)
    for role, types in read(path / "token/learning.json").items():
        for task, cases in types.items():
            for item in cases:
                text = tokens[item["id"]]
                observed[role].add(text)
                attribution[role, text].add("token_" + task)

    # A document may supply many observations. Keep its longest visible prefix in each source and role.
    ends = defaultdict(int)
    observation_dir = path / "observations"
    union = CACHE / "union-training/observations-2/stimuli.jsonl"
    texts = {r["id"]: r["text"] for r in rows(union)}
    for spec in read(observation_dir / "index.json")["tasks"]:
        for row in rows(observation_dir / spec["file"]):
            role = {0: "train", 1: "dev"}[row["role"]]
            for key, stop in row["coordinates"]:
                target = (key, role, spec["source"])
                ends[target] = max(ends[target], stop)
    for (key, role, source), stop in ends.items():
        full = texts[key]
        if not (0 < stop <= len(full)):
            raise ValueError(f"Bad visible character stop {key}, {stop}")
        text = full[:stop]
        observed[role].add(text)
        attribution[role, text].add(source)
    return (
        observed,
        attribution,
        {
            "receipt_sha256": sha(path / "receipt.json"),
            "union_stimuli_sha256": sha(union),
            "verified_learning_members": len(verified),
            "source_count": receipt["sources"],
        },
    )


def main():
    cohorts = targets()
    historical, attribution, provenance = learning_material()
    corpus = []
    full_index = defaultdict(list)
    window_index = defaultdict(list)
    for dataset, articles in sorted(cohorts.items()):
        for key, text in sorted(articles.items()):
            article = len(corpus)
            corpus.append((dataset, key))
            sequence = words(text)
            if not sequence:
                raise ValueError("Empty normalized target")
            full_index[sequence].append(article)
            for window in {
                sequence[i : i + WINDOW] for i in range(len(sequence) - WINDOW + 1)
            }:
                window_index[window].append(article)

    counts, matches = {}, {}
    for role, texts in sorted(historical.items()):
        by_article = defaultdict(lambda: defaultdict(set))
        for text in texts:
            sequence = words(text)
            for article in full_index.get(sequence, ()):
                by_article[article]["identical"].update(attribution[role, text])
            for i in range(len(sequence) - WINDOW + 1):
                for article in window_index.get(sequence[i : i + WINDOW], ()):
                    by_article[article]["shared_twelve_words"].update(
                        attribution[role, text]
                    )
        counts[role] = {
            cohort: {
                "target_texts": len(items),
                "identical": 0,
                "shared_twelve_words": 0,
                "either": 0,
            }
            for cohort, items in sorted(cohorts.items())
        }
        matches[role] = []
        for article, sources in sorted(by_article.items()):
            dataset, key = corpus[article]
            d = counts[role][dataset]
            d["either"] += 1
            for match in sources:
                d[match] += 1
            matches[role].append(
                {
                    "cohort": dataset,
                    "target_id": key,
                    **{k: sorted(sources[k]) for k in sorted(sources)},
                }
            )
    payload = {
        "protocol_sha256": sha(Path(__file__).with_name("release-overlap-scope.md")),
        "code_sha256": sha(Path(__file__)),
        "historical": provenance,
        "historical_visible_strings": {role: len(v) for role, v in historical.items()},
        "counts": counts,
        "matches": matches,
    }
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "historical-exposure.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
