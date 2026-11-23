"""Build the pinned CogLLM Stage 2 corpus and audit artifacts."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile, TemporaryDirectory
from urllib.request import urlopen

from . import onestop, plaba, zhan


CONVERTERS = {"plaba": plaba.convert, "zhan": zhan.convert, "onestop": onestop.convert}
SOURCE_LOCK = Path(__file__).with_name("sources.json")


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fetch_inputs(directory, source, offline):
    directory.mkdir(parents=True, exist_ok=True)
    for name, spec in source["files"].items():
        path = directory / name
        expected = spec["sha256"]
        if path.exists():
            if sha256(path) != expected:
                raise ValueError(f"Cached file differs from the pinned SHA-256: {path}")
            continue
        if offline:
            raise ValueError(f"Missing pinned cache input: {path}")
        temporary = None
        try:
            url = source["download_base"] + spec["path"]
            with urlopen(url, timeout=45) as response:
                with NamedTemporaryFile(dir=directory, delete=False) as file:
                    temporary = Path(file.name)
                    shutil.copyfileobj(response, file)
            if sha256(temporary) != expected:
                raise ValueError(f"Download differs from the pinned SHA-256: {url}")
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def validate_corpus(directory, source_file_names):
    stimuli = {}
    segments = {}
    for stimulus in read_jsonl(directory / "stimuli.jsonl"):
        key = stimulus["id"]
        if key in stimuli or not stimulus["text"].strip() or not stimulus["family_id"]:
            raise ValueError(f"Duplicate or incomplete stimulus: {key}")
        stimuli[key] = stimulus
        by_id = {}
        for segment in stimulus["segments"]:
            start, end = segment["start"], segment["end"]
            if (
                segment["id"] in by_id
                or type(start) is not int
                or type(end) is not int
                or not 0 <= start < end <= len(stimulus["text"])
            ):
                raise ValueError(f"Duplicate or invalid segment at {key}: {segment}")
            by_id[segment["id"]] = end
        segments[key] = by_id

    trials = read_jsonl(directory / "trials.jsonl")
    trial_ids, raw_keys = set(), set()
    for trial in trials:
        trial_id = trial["id"]
        stimulus_id = trial["stimulus_id"]
        ref = trial["source_ref"]
        raw_key = (ref["file"], ref["row"])
        if trial_id in trial_ids or raw_key in raw_keys or stimulus_id not in stimuli:
            raise ValueError(f"Duplicate or unlinked trial: {trial_id}")
        if ref["file"] not in source_file_names or type(ref["row"]) is not int or ref["row"] < 1:
            raise ValueError(f"Invalid source key at {trial_id}: {ref}")
        trial_ids.add(trial_id)
        raw_keys.add(raw_key)
        exposure = trial["exposure"]
        length = len(stimuli[stimulus_id]["text"])
        visible_end = exposure["visible_end"]
        mode = exposure["mode"]
        if mode not in {"full_text", "prefix"} or type(visible_end) is not int or not 0 < visible_end <= length:
            raise ValueError(f"Invalid visible text at {trial_id}")
        if mode == "full_text" and visible_end != length:
            raise ValueError(f"Truncated full-text trial: {trial_id}")
        target = trial["target_segment_id"]
        if target is not None and segments[stimulus_id].get(target, length + 1) > visible_end:
            raise ValueError(f"Invisible or unknown target segment at {trial_id}")
        for name, observation in trial["observations"].items():
            status, value = observation["status"], observation["value"]
            if status not in {"observed", "missing", "unobserved"} or (status == "observed") != (value is not None):
                raise ValueError(f"Invalid observation status at {trial_id}: {name}")
    return stimuli, trials


def development_folds(stimuli, trials, group_readers):
    counts = Counter()
    reader_families = defaultdict(set)
    for trial in trials:
        family = stimuli[trial["stimulus_id"]]["family_id"]
        counts[family] += 1
        if trial["participant_id"] is not None:
            reader_families[trial["participant_id"]].add(family)

    parents = {family: family for family in counts}

    def representative(family):
        while family != parents[family]:
            family = parents[family]
        return family

    if group_readers:
        for families in reader_families.values():
            first, *rest = sorted(families)
            for family in rest:
                parents[representative(family)] = representative(first)

    groups = defaultdict(list)
    for family in sorted(counts):
        groups[representative(family)].append(family)
    if len(groups) < 5:
        raise ValueError("Five development folds require at least five independent groups")

    folds = [{"fold": index, "families": [], "trials": 0} for index in range(5)]
    weighted = [(sum(counts[family] for family in group), group) for group in groups.values()]
    for weight, group in sorted(weighted, key=lambda pair: (-pair[0], pair[1])):
        fold = min(folds, key=lambda row: (row["trials"], row["fold"]))
        fold["families"].extend(group)
        fold["trials"] += weight
    for fold in folds:
        fold["families"].sort()
    family_folds = {family: fold["fold"] for fold in folds for family in fold["families"]}
    cross_fold = sum(len({family_folds[family] for family in group}) > 1 for group in reader_families.values())
    if group_readers and cross_fold:
        raise ValueError("Reader grouping failed to keep every reader in one fold")
    return {
        "purpose": "development_cross_validation",
        "grouping": "reader_connected_stimulus_families" if group_readers else "stimulus_family",
        "group_count": len(groups),
        "reader_ids_spanning_folds": cross_fold,
        "folds": folds,
    }


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def build(cache, output, offline):
    if output.exists() or output.is_symlink():
        raise ValueError(f"Choose a new output directory: {output}")
    if cache.resolve().is_relative_to(output.resolve()):
        raise ValueError("The pinned source cache must be outside the output directory")
    sources = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as workspace:
        bundle = Path(workspace) / "bundle"
        bundle.mkdir()
        report = {}
        for name, convert in CONVERTERS.items():
            source = sources[name]
            fetch_inputs(cache / name, source, offline)
            directory = bundle / name
            statistics = convert(cache / name, directory)
            stimuli, trials = validate_corpus(directory, source["files"].keys())
            splits = development_folds(stimuli, trials, group_readers=name == "plaba")
            write_json(directory / "splits.json", splits)
            report[name] = {
                "source": source,
                "stimuli": len(stimuli),
                "trials": len(trials),
                "trials_by_task": dict(sorted(Counter(trial["exposure"]["task"] for trial in trials).items())),
                "metrics": statistics,
                "splits": splits,
                "artifacts": {
                    path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                    for path in sorted(directory.iterdir())
                },
            }
        write_json(bundle / "audit.json", report)
        bundle.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True, help="Directory for checksum-pinned source files")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for the normalized corpora")
    parser.add_argument("--offline", action="store_true", help="Read only from the checksum-pinned cache")
    args = parser.parse_args()
    try:
        report = build(args.cache_dir.resolve(), args.output_dir.absolute(), args.offline)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Stage 2: {error}\n")
    for dataset, result in report.items():
        print(f"{dataset}: {result['stimuli']} stimuli, {result['trials']} trials")
    print(f"Audit: {args.output_dir / 'audit.json'}")


if __name__ == "__main__":
    main()
