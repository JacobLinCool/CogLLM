"""Reconstruct the released Beukers story choices with a separate replication label escrow."""

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen


COMMIT = "eb0edbe51042ffaf065bd90839f4f999de5f3ffb"
PAPER = "https://doi.org/10.1038/s44271-024-00079-4"
RELEASE = "https://doi.org/10.5281/zenodo.10695055"
SOURCE = f"https://raw.githubusercontent.com/PrincetonCompMemLab/blocked_training_facilitates_learning/{COMMIT}/data/human/"
SOURCE_SHA256 = {
    "blocked_thresh90.csv": "9e813cd0ddfbe5a3bbc68169ecb7f961f7509798a58009cf08806fd1be69a7db",
    "interleaved_thresh90.csv": "c7c57447f7f0f20eeb86aa5356fa3d9f34c5d242cfb45b6214c582422ac0d642",
    "blocked_rep_thresh90.csv": "b7b9f2853a4b984b5b650bf36960e0689b14bd67e83e2de8059f2b1c505e89d8",
    "interleaved_rep_thresh90.csv": "c9047a6c6ba11a15f91729e47abae874f2381e8e40747a164ae72467cc0eb510",
}
COHORTS = {
    "blocked_thresh90.csv": ("fit", "blocked", 33),
    "interleaved_thresh90.csv": ("fit", "interleaved", 39),
    "blocked_rep_thresh90.csv": ("test", "blocked", 26),
    "interleaved_rep_thresh90.csv": ("test", "interleaved", 24),
}
SENTENCES = {
    "BEGIN": "{name} wants to go out today.",
    "jungle": "{name} decides to go to the Jungle Brew House.",
    "deep": "{name} decides to go to the Deep Ocean Cafe.",
    "NODE11": "Today, {name} waited in line patiently.",
    "NODE12": "{name} was impatient, and decided to cut the line.",
    "NODE21": "While ordering tea, {name} noticed the barista’s new mustache.",
    "NODE22": "Before ordering cake, {name} took a quarter from the tip jar.",
    "NODE31": "{name} then sat by the window, and read a book for hours.",
    "NODE32": "After ordering, {name} stole a salt shaker and left.",
}
PREFIX_NODES = {1: {"NODE11", "NODE12"}, 2: {"NODE21", "NODE22"}}
OPTIONS = {1: {"NODE21", "NODE22"}, 2: {"NODE31", "NODE32"}}
FIELDS = {
    "", "subjnum", "block", "story", "depth", "type", "qtype", "state", "fromnode", "true_tonode",
    "false_tonode", "rfc", "true_rfc", "false_rfc", "response_right", "true_on_right", "correct_response", "edge",
}


@dataclass(frozen=True)
class Observation:
    record: dict
    metadata: dict
    true_node: str
    raw: dict


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def slug(kind, *parts):
    canonical = json.dumps([kind, *parts], ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return sha256(canonical.encode("utf-8")).hexdigest()[:32]


def write_json(path, row):
    path.write_text(
        json.dumps(row, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8",
    )


def write_jsonl(path, rows, private=False):
    mode = 0o600 if private else 0o644
    descriptor = os.open(path, os.O_EXCL | os.O_WRONLY | os.O_CREAT, mode)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def fetch(output):
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".beukers-source-") as directory:
        staged = Path(directory) / "source"
        staged.mkdir()
        for filename, expected in SOURCE_SHA256.items():
            with urlopen(SOURCE + filename, timeout=15) as stream:
                data = stream.read(1_750_001)
            if len(data) > 1_750_000 or sha256(data).hexdigest() != expected:
                raise ValueError(f"Unexpected bytes for {filename}")
            descriptor = os.open(staged / filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
        staged.rename(output)


def presented_identity(row):
    if row["type"] != "question" or row["qtype"] != "transQ" or row["true_rfc"] != row["false_rfc"]:
        raise ValueError("Expected an observed binary transition probe with the same named setting in both choices")
    name, separator, cafe = row["true_rfc"].removeprefix("subject-").partition("_location-")
    if not row["true_rfc"].startswith("subject-") or not name or not separator or cafe not in {"deep", "jungle"}:
        raise ValueError("Missing fictional name or café")
    return name, cafe


def ordered_nodes(row):
    if row["true_on_right"] == "True":
        return row["false_tonode"], row["true_tonode"]
    if row["true_on_right"] == "False":
        return row["true_tonode"], row["false_tonode"]
    raise ValueError("Missing binary display order")


def response(row):
    choices = ordered_nodes(row)
    if row["response_right"] not in {"True", "False"} or row["correct_response"] not in {"0", "1"}:
        raise ValueError("Missing binary choice or author correctness code")
    selected = choices[row["response_right"] == "True"]
    if (selected == row["true_tonode"]) != (row["correct_response"] == "1"):
        raise ValueError("The selected option disagrees with the released correctness code")
    return selected


def text_layout(sentences):
    text = "\n".join(sentences)
    stops, length = [], 0
    for sentence in sentences:
        length += len(sentence)
        stops.append(length)
        length += 1
    return {"id": slug("text", text, stops), "text": text, "stops": stops}


def source_rows(path):
    if digest(path) != SOURCE_SHA256[path.name]:
        raise ValueError(f"The cohort differs from the immutable Beukers release: {path.name}")
    stories = defaultdict(dict)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        if set(rows.fieldnames or ()) != FIELDS:
            raise ValueError("Unexpected Beukers source columns")
        for row in rows:
            name, cafe = presented_identity(row)
            person, story, depth = row["subjnum"], int(row["story"]), int(row["depth"])
            sequence = stories[person, story]
            if (
                not person or not 0 <= story < 200 or int(row["block"]) != story // 40
                or depth not in PREFIX_NODES or depth in sequence or row["fromnode"] not in PREFIX_NODES[depth]
                or set(ordered_nodes(row)) != OPTIONS[depth]
            ):
                raise ValueError("Invalid story position, node or duplicate probe")
            sequence[depth] = (row, name, cafe)
    return stories


def construct_cohort(path, *, use_test_responses=False):
    split, curriculum, expected_readers = COHORTS[path.name]
    stories = source_rows(path)
    orientation = defaultdict(set)
    prior_probes, prior_correct = defaultdict(Counter), defaultdict(Counter)
    observed_training_stories = set()
    readers = {person for person, _ in stories}
    if len(readers) != expected_readers:
        raise ValueError("The released cohort no longer matches the reported reader count")
    rows, layouts = [], {}
    skipped_second = 0
    for (person, story), sequence in stories.items():
        if len(sequence) == 2:
            first, first_name, first_cafe = sequence[1]
            second, second_name, second_cafe = sequence[2]
            if (first_name, first_cafe, first["true_tonode"]) != (second_name, second_cafe, second["fromnode"]):
                raise ValueError("Two probes in a story disagree about the preceding displayed event")
        for depth, (row, name, cafe) in sequence.items():
            if story < 160:
                parity = (story // 40 if curriculum == "blocked" else story) % 2
                orientation[person].add((cafe == "jungle") ^ parity)
                observed_training_stories.add((person, story))
                prior_probes[person][cafe] += 1
                prior_correct[person][cafe] += response(row) == row["true_tonode"]
                continue
            if depth == 2 and 1 not in sequence:
                skipped_second += 1
                continue
            nodes = tuple(sequence[step][0]["fromnode"] for step in range(1, depth + 1))
            left, right = ordered_nodes(row)
            prefix = text_layout([SENTENCES[node].format(name=name) for node in ("BEGIN", cafe, *nodes)])
            left_text = text_layout([SENTENCES[left].format(name=name)])
            right_text = text_layout([SENTENCES[right].format(name=name)])
            for layout in (prefix, left_text, right_text):
                if layouts.setdefault(layout["id"], layout) != layout:
                    raise ValueError("Two different layouts share the same opaque ID")
            public = {
                "id": slug("observation", path.name, person, story, depth),
                "article_id": prefix["id"], "left_id": left_text["id"], "right_id": right_text["id"],
            }
            metadata = {
                "reader_id": slug("reader", path.name, person),
                "curriculum": curriculum,
                "story": story,
                "depth": depth,
                "family": "/".join((cafe, *nodes)),
                "cafe": cafe,
                "current_node": nodes[-1],
                "left_node": left,
                "right_node": right,
            }
            rows.append((person, Observation(public, metadata, row["true_tonode"], row)))
    if orientation.keys() != readers or any(len(v) != 1 for v in orientation.values()):
        raise ValueError("A participant's observed pre-test cafés contradict their assigned training schedule")
    if any(prior_probes[person].keys() != {"deep", "jungle"} for person in readers):
        raise ValueError("A participant is missing observed pre-test responses for one café")
    locked = []
    for person, obs in rows:
        labels = {}
        if split == "fit" or use_test_responses:
            labels = {
                "id": obs.record["id"], **obs.metadata, "correct_node": obs.true_node,
                "response_node": response(obs.raw), "prior_probes": dict(prior_probes[person]),
                "prior_correct": dict(prior_correct[person]),
            }
        locked.append((obs.record, labels))
    details = {
        "readers": len(readers), "retained_probes": len(rows),
        "retained_by_depth": dict(Counter(obs.metadata["depth"] for _, obs in rows)),
        "second_probe_without_first_omitted": skipped_second,
        "unlogged_pretest_stories": len(readers) * 160 - len(observed_training_stories),
        "prior_probe_count_per_reader_and_cafe": [
            min(count for visits in prior_probes.values() for count in visits.values()),
            max(count for visits in prior_probes.values() for count in visits.values()),
        ],
    }
    return locked, layouts, details


def score_geometry(row):
    return row["article_id"], row["left_id"], row["right_id"]


def prepare(raw, output):
    if output.exists():
        raise FileExistsError(output)
    stimulus_index = {}
    records = {"fit": [], "test": []}
    fit_truth, source_summary = [], {}
    matches = defaultdict(lambda: defaultdict(Counter))
    for filename, (split, curriculum, _) in COHORTS.items():
        cohort, layouts, source_summary[filename] = construct_cohort(raw / filename)
        for key, layout in layouts.items():
            if stimulus_index.setdefault(key, layout) != layout:
                raise ValueError("Two different source layouts share a text identifier")
        for public, truth in cohort:
            records[split].append(public)
            matches[split][curriculum][score_geometry(public)] += 1
            if split == "fit":
                fit_truth.append(truth)
            elif truth:
                raise ValueError("The grade-free intake cannot release replication choices")
    all_records = [row for rows in records.values() for row in rows]
    if len({row["id"] for row in all_records}) != len(all_records):
        raise ValueError("A repeated source observation ID would join two trials")
    if ({row["article_id"] for row in all_records} |
        {row[field] for row in all_records for field in ("left_id", "right_id")}) != stimulus_index.keys():
        raise ValueError("The grade-free mapping disagrees with the complete model-text set")
    shared = {}
    for split, curricula in matches.items():
        keys = curricula["blocked"].keys() & curricula["interleaved"].keys()
        shared[split] = {
            "exact_ordered_layouts": len(keys),
            "blocked_rows": sum(curricula["blocked"][key] for key in keys),
            "interleaved_rows": sum(curricula["interleaved"][key] for key in keys),
        }
    expected = {
        "fit": {"exact_ordered_layouts": 585, "blocked_rows": 784, "interleaved_rows": 825},
        "test": {"exact_ordered_layouts": 347, "blocked_rows": 449, "interleaved_rows": 426},
    }
    if shared != expected:
        raise ValueError("The exact same-name, same-prefix and same-choice-order cohorts changed")
    fit_keys = matches["fit"]["blocked"].keys() & matches["fit"]["interleaved"].keys()
    test_keys = matches["test"]["blocked"].keys() & matches["test"]["interleaved"].keys()
    audit = {
        "paper": PAPER,
        "release": RELEASE,
        "release_license": "CC BY 4.0",
        "source_commit": COMMIT,
        "source_sha256": SOURCE_SHA256,
        "prepare_sha256": digest(Path(__file__)),
        "cohorts": source_summary,
        "fit_vs_test": "The original blocked and interleaved samples fit the readout. The independent "
                       "replications test it. Probe stories are 160–199 in both cases.",
        "source_sentence_rule": "Use the author's Methods sentences and the original fictional name "
                                "and café encoded in both displayed option records. Join displayed sentences "
                                "with newline separators. Each actual prior node for a retained second probe "
                                "is available in the earlier probe's record.",
        "source_node_reference": f"https://github.com/PrincetonCompMemLab/blocked_training_facilitates_learning/blob/{COMMIT}/humanUtils.py#L10",
        "source_sentence_templates": SENTENCES,
        "fit_reader_background": "Café-specific correct/total prior probes use only stories 0–159. "
                                 "Counterbalanced first café and subsequent assignment are checked against "
                                 "40-story blocks or story-wise alternation in recorded trials.",
        "matched_key": ["article_id", "left_id", "right_id"],
        "within_split_exact_ordered_matches": shared,
        "matched_layouts_shared_between_splits": len(fit_keys & test_keys),
        "security": {
            "model_input": "model_input/stimuli.jsonl",
            "private_covariates_and_outcomes": ["fit-truth.jsonl", "test-truth.jsonl"],
            "replication_probe_choices_opened_during_intake": False,
        },
        "counts": {
            "stimuli": len(stimulus_index),
            "candidate_stops": sum(len(row["stops"]) for row in stimulus_index.values()),
            "fit_probes": len(records["fit"]),
            "test_probes": len(records["test"]),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".beukers-intake-") as directory:
        staged = Path(directory) / "assay"
        (staged / "model_input").mkdir(parents=True)
        files = {
            "model_input/stimuli.jsonl": stimulus_index.values(),
            "fit.jsonl": records["fit"],
            "test.jsonl": records["test"],
            "fit-truth.jsonl": fit_truth,
        }
        for name, rows in files.items():
            write_jsonl(staged / name, sorted(rows, key=lambda row: row["id"]), private=name == "fit-truth.jsonl")
        audit["file_sha256"] = {name: digest(staged / name) for name in files}
        write_json(staged / "assay.json", audit)
        staged.rename(output)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def unseal(raw, assay, fit_seal):
    assay_path = assay / "assay.json"
    audit = read_json(assay_path)
    fit = read_json(fit_seal)
    assay_sha256 = digest(assay_path)
    fit_sha256 = digest(fit_seal)
    if (
        audit["prepare_sha256"] != digest(Path(__file__)) or audit["source_sha256"] != SOURCE_SHA256
        or any(digest(assay / name) != expected for name, expected in audit["file_sha256"].items())
        or fit.get("assay_sha256") != assay_sha256
        or fit.get("fit_truth_sha256") != audit["file_sha256"]["fit-truth.jsonl"]
        or not isinstance(fit.get("params"), dict) or not fit["params"]
    ):
        raise ValueError("A valid fit seal for the unchanged assay and original-cohort labels is required")
    for name in ("protocol_sha256", "scorer_sha256", "extraction_sha256"):
        checksum = fit.get(name)
        if type(checksum) is not str or len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
            raise ValueError(f"An immutable fit receipt must bind {name}")
    truth_path = assay / "test-truth.jsonl"
    seal_path = assay / "test-truth-seal.json"
    if seal_path.exists():
        raise FileExistsError(seal_path)
    predictions_design = read_jsonl(assay / "test.jsonl")
    reconstructed, truth = [], []
    for filename, (split, _, _) in COHORTS.items():
        if split == "test":
            cohort, _, _ = construct_cohort(raw / filename, use_test_responses=True)
            reconstructed.extend(public for public, _ in cohort)
            truth.extend(labels for _, labels in cohort)
    if sorted(reconstructed, key=lambda row: row["id"]) != predictions_design:
        raise ValueError("Unsealed replica outcomes do not refer to the registered grade-free layout")
    with TemporaryDirectory(dir=assay, prefix=".beukers-unseal-") as directory:
        staged = Path(directory)
        temporary_truth = staged / "test-truth.jsonl"
        write_jsonl(temporary_truth, sorted(truth, key=lambda row: row["id"]), private=True)
        truth_sha256 = digest(temporary_truth)
        if truth_path.exists():
            if digest(truth_path) != truth_sha256:
                raise FileExistsError(truth_path)
        else:
            os.link(temporary_truth, truth_path)
        temporary_seal = staged / "test-truth-seal.json"
        write_json(temporary_seal, {
            "assay_sha256": assay_sha256, "fit_sha256": fit_sha256, "test_truth_sha256": truth_sha256,
        })
        os.link(temporary_seal, seal_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    download = modes.add_parser("fetch")
    download.add_argument("--output", type=Path, required=True)
    geometry = modes.add_parser("prepare")
    geometry.add_argument("--raw", type=Path, required=True)
    geometry.add_argument("--output", type=Path, required=True)
    labels = modes.add_parser("truth")
    labels.add_argument("--raw", type=Path, required=True)
    labels.add_argument("--assay", type=Path, required=True)
    labels.add_argument("--fit-seal", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "fetch":
        fetch(args.output)
    elif args.mode == "prepare":
        prepare(args.raw, args.output)
    else:
        unseal(args.raw, args.assay, args.fit_seal)


if __name__ == "__main__":
    main()
