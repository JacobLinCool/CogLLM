"""Prepare an exact cached-state transfer assay for the Beukers inserted-block experiment."""

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
from math import isfinite
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

from research.semantic_tracking import beukers_prepare as base


ASSAY_ID = "beukers-inserted-cached-v1"
BASE_ASSAY_SHA256 = "62a33c41914671f252490861205577a284ef9435f86337bdc90cf4a531c0923a"
BASE_EXTRACTION_SHA256 = "1f370a9d19f24b44745a6c2b5e0a08b662c05d9c1c5ab0179733d2adc8bda83f"
BASE_FIT_SHA256 = "8fa16fc44b70d5675d7338ee21611edf2d058dcd83c26920231034dd93c87c4e"
PROTOCOL = Path(__file__).with_name("beukers-inserted-protocol.md")
FORECAST_CODE = Path(__file__).with_name("beukers_inserted_forecast.py")
FORECAST_METHODS = {"initial_block", "empty_context", "text"}
SCHEDULES = {"early": "ABII", "middle": "IABI", "late": "IIAB"}
ORIENTATION = {"Deep": "deep", "Jungle": "jungle"}
COHORTS = {
    "inserted_early_thresh90.csv": (
        "original", "early", 21, "9016d2072aca03308c0ac8dd769b1e4040d19d8d01870f16cb34eda2a4d710c9",
    ),
    "inserted_middle_thresh90.csv": (
        "original", "middle", 20, "105c6e24459e8920fc1bff9d87d7e13edc1ea4b198f14e50295be529c88987d8",
    ),
    "inserted_late_thresh90.csv": (
        "original", "late", 24, "fd5b43a3bdcd6cdacdc1d763181d82838703e316e116e003b1704f0045d8efd8",
    ),
    "inserted_early_rep_thresh90.csv": (
        "replication", "early", 22, "6a884968b142133c4276d63d5da6d7688f9285397e75c43d8fe104c5bbe2a65e",
    ),
    "inserted_middle_rep_thresh90.csv": (
        "replication", "middle", 24, "672e5d585defa27f8c6b91834f5dbd3e13a976b84911d4324dafc72613b92470",
    ),
    "inserted_late_rep_thresh90.csv": (
        "replication", "late", 12, "6e1cefab216e06e3565c0a75a5e0d70c65f44346e46b502477302bf787c157d4",
    ),
}
SOURCE_SHA256 = {name: source[3] for name, source in COHORTS.items()}
OUTCOMES = {"response_right", "correct_response"}


def fetch(output):
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".inserted-source-") as directory:
        staged = Path(directory) / "raw"
        staged.mkdir(mode=0o700)
        for name, checksum in SOURCE_SHA256.items():
            with urlopen(base.SOURCE + name, timeout=15) as stream:
                data = stream.read(1_100_001)
            if len(data) > 1_100_000 or sha256(data).hexdigest() != checksum:
                raise ValueError(f"Unexpected bytes for {name}")
            descriptor = os.open(staged / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
        staged.rename(output)


def cached_support(assay, extraction):
    assay_path = assay / "assay.json"
    stimuli_path = assay / "model_input/stimuli.jsonl"
    extraction_path = extraction / "seal.json"
    if base.digest(assay_path) != BASE_ASSAY_SHA256 or base.digest(extraction_path) != BASE_EXTRACTION_SHA256:
        raise ValueError("The original assay and certified extraction must match the frozen receipts")
    audit = base.read_json(assay_path)
    seal = base.read_json(extraction_path)
    stimuli_sha256 = base.digest(stimuli_path)
    if (
        seal["assay_sha256"] != BASE_ASSAY_SHA256
        or audit["file_sha256"]["model_input/stimuli.jsonl"] != stimuli_sha256
        or seal["stimuli_sha256"] != stimuli_sha256
        or any(base.digest(extraction / name) != seal["file_sha256"][name] for name in (
            "coordinates.jsonl", "vectors.npy",
        ))
    ):
        raise ValueError("The archived literals or states have changed")
    stimuli = base.read_jsonl(stimuli_path)
    stimuli_by_id = {}
    for row in stimuli:
        expected = {"id", "text", "stops"}
        if row.keys() != expected or row["id"] in stimuli_by_id or (
            row["id"] != base.slug("text", row["text"], row["stops"])
        ):
            raise ValueError("Invalid or duplicate archival stimulus")
        stimuli_by_id[row["id"]] = row
    coordinates = base.read_jsonl(extraction / "coordinates.jsonl")
    certified = {}
    for coordinate in coordinates:
        stimulus = stimuli_by_id[coordinate["id"]]
        if (
            coordinate.keys() != {"id", "stops"} or coordinate["id"] in certified
            or coordinate["stops"] != stimulus["stops"]
        ):
            raise ValueError("An archival stimulus lacks its complete certified stop sequence")
        certified[coordinate["id"]] = stimulus
    if len(certified) != seal["encoded_documents"] or sum(len(row["stops"]) for row in certified.values()) != (
        seal["certified_stops"]
    ):
        raise ValueError("The certified stimulus and stop counts disagree with the archived extraction")
    return certified, stimuli_sha256


def source_rows(path, outcomes=False):
    if base.digest(path) != SOURCE_SHA256[path.name]:
        raise ValueError(f"Inserted source differs from the pinned release: {path.name}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        if set(rows.fieldnames or ()) != base.FIELDS or len(rows.fieldnames) != len(base.FIELDS):
            raise ValueError("Unexpected inserted-source probe columns")
        for row in rows:
            if not outcomes:
                for key in OUTCOMES:
                    del row[key]
            yield row


def presented_identity(row):
    fields = ("true_rfc", "false_rfc")
    normalized = {}
    for field in fields:
        name, separator, cafe = row[field].rpartition("_location-")
        if not separator or cafe not in ORIENTATION:
            raise ValueError("Inserted-source café must be declared in both displayed options")
        normalized[field] = f"{name}{separator}{ORIENTATION[cafe]}"
    return base.presented_identity({**row, **normalized})


def geometry(row):
    return row["article_id"], row["left_id"], row["right_id"]


def verify_schedule(stories, schedule, reader_count):
    observed = defaultdict(set)
    for (reader, story), probes in stories.items():
        names_and_cafes = {pair for _, pair in probes.values()}
        if len(names_and_cafes) != 1:
            raise ValueError("A story's probes disagree on the visible fictional name or café")
        if story < 160:
            _, cafe = next(iter(names_and_cafes))
            block = story // 40
            parity = schedule[block] == "I" and story % 2 == 1
            observed[reader, block].add((cafe == "jungle") ^ parity)
    if len(observed) != reader_count * 4 or any(len(values) != 1 for values in observed.values()):
        raise ValueError("Observed cafés disagree with one or more assigned 40-story blocks")
    a, b = schedule.index("A"), schedule.index("B")
    for reader in {reader for reader, _ in stories}:
        if observed[reader, a] == observed[reader, b]:
            raise ValueError("The two single-café blocks have the same café")


def construct_cohort(path, support):
    cohort, arm, expected_readers, _ = COHORTS[path.name]
    stories = defaultdict(dict)
    for row in source_rows(path):
        person, story, depth = row["subjnum"], int(row["story"]), int(row["depth"])
        probes = stories[person, story]
        if (
            not person or not 0 <= story < 200 or int(row["block"]) != story // 40
            or depth not in base.PREFIX_NODES or depth in probes
            or row["fromnode"] not in base.PREFIX_NODES[depth] or set(base.ordered_nodes(row)) != base.OPTIONS[depth]
        ):
            raise ValueError("Unexpected story position, depth or display alternatives in inserted source")
        probes[depth] = row, presented_identity(row)
    reader_count = len({person for person, _ in stories})
    if reader_count != expected_readers:
        raise ValueError("Inserted cohort differs from the paper's retained reader count")
    verify_schedule(stories, SCHEDULES[arm], reader_count)
    rows = []
    post160 = Counter()
    missing_predecessor, missing_cached_state = 0, 0
    for (person, story), probes in stories.items():
        if 1 in probes and 2 in probes and probes[1][0]["true_tonode"] != probes[2][0]["fromnode"]:
            raise ValueError("The first probe's released continuation disagrees with the next displayed node")
        if story < 160:
            continue
        for depth, (row, (name, cafe)) in probes.items():
            post160[depth] += 1
            if depth == 2 and 1 not in probes:
                missing_predecessor += 1
                continue
            prefix = ("BEGIN", cafe, *(probes[index][0]["fromnode"] for index in range(1, depth + 1)))
            left, right = base.ordered_nodes(row)
            layouts = [
                base.text_layout([base.SENTENCES[node].format(name=name) for node in nodes])
                for nodes in (prefix, (left,), (right,))
            ]
            for layout in layouts:
                if layout["id"] in support and layout != support[layout["id"]]:
                    raise ValueError("A recreated literal or stop differs from its certified archival stimulus")
            if any(layout["id"] not in support for layout in layouts):
                missing_cached_state += 1
                continue
            article_id, left_id, right_id = (layout["id"] for layout in layouts)
            rows.append({
                "id": base.slug("inserted-probe", path.name, person, story, depth),
                "cohort": cohort,
                "arm": arm,
                "reader_id": base.slug("inserted-reader", path.name, person),
                "article_id": article_id,
                "left_id": left_id,
                "right_id": right_id,
                "story": story,
                "depth": depth,
            })
    retained_readers = len({row["reader_id"] for row in rows})
    if retained_readers != expected_readers:
        raise ValueError("Cached literal support removed a reader from the inserted cohort")
    return rows, {
        "readers": reader_count,
        "post160_probes": sum(post160.values()),
        "post160_by_depth": dict(post160),
        "missing_first_probe": missing_predecessor,
        "missing_certified_state": missing_cached_state,
        "retained_probes": len(rows),
        "retained_by_depth": dict(Counter(row["depth"] for row in rows)),
    }


def matched_inventory(rows):
    result = {}
    for cohort in ("original", "replication"):
        arms = {arm: [row for row in rows if row["cohort"] == cohort and row["arm"] == arm] for arm in SCHEDULES}
        keys = {arm: {geometry(row) for row in subset} for arm, subset in arms.items()}
        result[cohort] = {}
        for other in ("middle", "late"):
            shared = keys["early"] & keys[other]
            matched = {
                arm: [row for row in arms[arm] if geometry(row) in shared]
                for arm in ("early", other)
            }
            result[cohort][f"early_vs_{other}"] = {
                "ordered_layouts": len(shared),
                "probes": {arm: len(subset) for arm, subset in matched.items()},
                "readers": {arm: len({row["reader_id"] for row in subset}) for arm, subset in matched.items()},
            }
    return result


def prepare(raw, output, base_assay, extraction):
    if output.exists():
        raise FileExistsError(output)
    support, stimuli_sha256 = cached_support(base_assay, extraction)
    rows, inventory = [], {}
    for filename in COHORTS:
        cohort_rows, inventory[filename] = construct_cohort(raw / filename, support)
        rows.extend(cohort_rows)
    if len(rows) != len({row["id"] for row in rows}):
        raise ValueError("A trial identity would join more than one published probe")
    audit = {
        "assay_id": ASSAY_ID,
        "paper": base.PAPER,
        "source_commit": base.COMMIT,
        "source_release": base.RELEASE,
        "source_license": "CC BY 4.0",
        "source_sha256": SOURCE_SHA256,
        "source_code_sha256": base.digest(Path(__file__)),
        "base_prepare_sha256": base.digest(Path(base.__file__)),
        "base_assay_sha256": BASE_ASSAY_SHA256,
        "base_extraction_sha256": BASE_EXTRACTION_SHA256,
        "base_fit_sha256": BASE_FIT_SHA256,
        "base_stimuli_sha256": stimuli_sha256,
        "cohort_inventory": inventory,
        "matched_inventory": matched_inventory(rows),
        "test_count": len(rows),
        "schedules": SCHEDULES,
        "eligibility": (
            "Story 160–199. The earlier displayed event of a second probe must be observed in the same story. "
            "Retain only exact prefix and option texts with their full certified stops in the prior extraction."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".inserted-assay-") as directory:
        staged = Path(directory) / "assay"
        staged.mkdir(mode=0o700)
        base.write_jsonl(staged / "test.jsonl", sorted(rows, key=lambda row: row["id"]))
        audit["file_sha256"] = {"test.jsonl": base.digest(staged / "test.jsonl")}
        base.write_json(staged / "assay.json", audit)
        staged.rename(output)


def verify_forecasts(assay, forecasts):
    assay_path = assay / "assay.json"
    mapping_path = assay / "test.jsonl"
    assay_sha256, test_sha256 = base.digest(assay_path), base.digest(mapping_path)
    audit = base.read_json(assay_path)
    if (
        audit["assay_id"] != ASSAY_ID or audit["file_sha256"] != {"test.jsonl": test_sha256}
        or audit["source_sha256"] != SOURCE_SHA256 or audit["source_code_sha256"] != base.digest(Path(__file__))
        or audit["base_prepare_sha256"] != base.digest(Path(base.__file__))
        or audit["base_extraction_sha256"] != BASE_EXTRACTION_SHA256 or audit["base_fit_sha256"] != BASE_FIT_SHA256
    ):
        raise ValueError("The grade-free assay or source preparation no longer matches its receipt")
    forecast_path, seal_path = forecasts / "forecasts.jsonl", forecasts / "forecast-seal.json"
    seal_sha256 = base.digest(seal_path)
    expected = {
        "assay_sha256": assay_sha256,
        "test_sha256": test_sha256,
        "forecasts_sha256": base.digest(forecast_path),
        "protocol_sha256": base.digest(PROTOCOL),
        "base_fit_sha256": BASE_FIT_SHA256,
        "base_extraction_sha256": BASE_EXTRACTION_SHA256,
        "forecast_code_sha256": base.digest(FORECAST_CODE),
    }
    if base.read_json(seal_path) != expected:
        raise ValueError("The complete test and frozen forecaster must be bound by the forecast seal")
    mapping = base.read_jsonl(mapping_path)
    predictions = base.read_jsonl(forecast_path)
    ids = [row["id"] for row in mapping]
    if len(ids) != audit["test_count"] or len(set(ids)) != len(ids) or [row["id"] for row in predictions] != ids:
        raise ValueError("Forecasts must cover all registered trials once and in order")
    for row in predictions:
        if row.keys() != {"id", "methods"} or row["methods"].keys() != FORECAST_METHODS or any(
            type(value) not in (int, float) or not isfinite(value) or not 0 < value < 1
            for value in row["methods"].values()
        ):
            raise ValueError("Each forecast must contain only the fixed methods and valid probabilities")
    return ids, {"assay_sha256": assay_sha256, "forecast_seal_sha256": seal_sha256, "test_sha256": test_sha256}


def unseal(raw, assay, forecasts):
    ids, receipt = verify_forecasts(assay, forecasts)
    truth_path = assay / "test-truth.jsonl"
    seal_path = assay / "test-truth-seal.json"
    if seal_path.exists():
        raise FileExistsError(seal_path)
    registered = set(ids)
    truth = {}
    for name in COHORTS:
        for row in source_rows(raw / name, outcomes=True):
            story, depth = int(row["story"]), int(row["depth"])
            if story < 160:
                continue
            identifier = base.slug("inserted-probe", name, row["subjnum"], story, depth)
            if identifier not in registered:
                continue
            _, right = base.ordered_nodes(row)
            response = base.response(row)
            if identifier in truth:
                raise ValueError("A source response is repeated in the registered test set")
            truth[identifier] = {
                "id": identifier,
                "correct_side": "right" if row["true_tonode"] == right else "left",
                "response_side": "right" if response == right else "left",
            }
    if set(truth) != registered:
        raise ValueError("Some registered probes lack a released response")
    with TemporaryDirectory(dir=assay, prefix=".inserted-unseal-") as directory:
        staged = Path(directory)
        temporary_truth = staged / truth_path.name
        base.write_jsonl(temporary_truth, (truth[identifier] for identifier in ids), private=True)
        receipt["test_truth_sha256"] = base.digest(temporary_truth)
        if truth_path.exists():
            if base.digest(truth_path) != receipt["test_truth_sha256"]:
                raise FileExistsError(truth_path)
        else:
            os.link(temporary_truth, truth_path)
        temporary_seal = staged / seal_path.name
        base.write_json(temporary_seal, receipt)
        os.link(temporary_seal, seal_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    download = commands.add_parser("fetch")
    download.add_argument("--output", required=True, type=Path)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--raw", required=True, type=Path)
    preparation.add_argument("--output", required=True, type=Path)
    preparation.add_argument("--base-assay", required=True, type=Path)
    preparation.add_argument("--extraction", required=True, type=Path)
    truth = commands.add_parser("truth")
    truth.add_argument("--raw", required=True, type=Path)
    truth.add_argument("--assay", required=True, type=Path)
    truth.add_argument("--forecasts", required=True, type=Path)
    args = parser.parse_args()
    if args.mode == "fetch":
        fetch(args.output)
    elif args.mode == "prepare":
        prepare(args.raw, args.output, args.base_assay, args.extraction)
    else:
        unseal(args.raw, args.assay, args.forecasts)


if __name__ == "__main__":
    main()
