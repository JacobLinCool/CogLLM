"""Prepare Paape and Vasishth's context manipulation and unseal its keypress outcomes."""

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unicodedata
from urllib.parse import unquote


ARTICLE = "https://doi.org/10.1016/j.jml.2026.104748"
PROJECT = "https://osf.io/6uv37/"
SOURCE_SHA256 = {
    "alt_stim.txt": "7e7cf3581de3f7ec988253ba6adf175390c7ca5b6583ab79597880ed19de3f22",
    "analyses.Rmd": "9fe165881881c9887f67372048a91c72397c6ae49f4a9daabbabbb1d36dd71c8",
    "fixations.Rda": "e7bde8f75569f98942671e3d1344ca5bd0b7348c23cda206d2dc7aacd6cfdbab",
    "scanpaths.R": "876cb5bc46c0f4faa36c49be0298644755426a9918fe71cdec4725e69f4eb60b",
}
PRIOR_ENGLISH_STIMULI_SHA256 = "0f7bc8575532af166b7ff53dfc03205a1757ad2a487283daf710c5bc581d1f31"
CONDITIONS = {
    "alt-a": ("oneref", "comp"), "alt-b": ("oneref", "rc"), "alt-c": ("oneref", "unamb"),
    "alt-d": ("tworef", "comp"), "alt-e": ("tworef", "rc"), "alt-f": ("tworef", "unamb"),
}
# Distinct (item, condition, context, disamb, crit) design metadata in fixations.Rda.
# The six conditions have the same crit in every family. Source analysis uses no == crit.
CRITICAL_REGIONS = {
    1: 10, 2: 10, 3: 11, 4: 10, 5: 9, 6: 10, 7: 11, 8: 12, 9: 11, 10: 9, 11: 10, 12: 11,
    13: 11, 14: 11, 15: 10, 16: 11, 17: 11, 18: 11, 19: 11, 20: 11, 21: 11, 22: 11, 23: 11, 24: 11,
    25: 10, 26: 11, 27: 11, 28: 11, 29: 11, 30: 11, 31: 11, 32: 11, 33: 11, 34: 11, 35: 11, 36: 11,
}
RECORD = re.compile(r'^\s*(\["(?:alt-[a-f]|filler)"\s*,\s*\d+\])\s*', re.MULTILINE)
OVERLAP_WORDS = 12
METHODS = {"prior", "position", "qwen"}
MODEL = "Qwen/Qwen3.5-2B"
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
CANDIDATE = "74c26ff02569b2b3bc28ed291bf15fd2f85db5b4c93a3ad7bfc34f6a5d333ea0"
HEADS_SHA256 = "029c8503fb4110f8c853478809674b68c077c078c56ce62662ad98b03387d432"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def write_json(path, record):
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def verify_sources(source):
    for name, sha256 in SOURCE_SHA256.items():
        if digest(source / name) != sha256:
            raise ValueError(f"The author source snapshot has changed: {name}")


def read_regions(path):
    text = path.read_text(encoding="utf-8-sig")
    markers = list(RECORD.finditer(text))
    if not markers or text[:markers[0].start()].strip():
        raise ValueError("Unexpected prefix in the stimulus file")
    regions = {}
    for index, marker in enumerate(markers):
        condition, item = json.loads(marker[1])
        if condition == "filler":
            continue
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        displays = tuple(json.loads(f"[{text[marker.end():end].strip()}]"))
        if (
            (item, condition) in regions
            or not all(type(display) is str and display and display.strip() and not display[0].isspace()
                       for display in displays)
        ):
            raise ValueError(f"Invalid displayed segments for {item}:{condition}")
        regions[item, condition] = displays
    if regions.keys() != {(item, condition) for item in CRITICAL_REGIONS for condition in CONDITIONS}:
        raise ValueError("The stimulus file must contain all six variants of all 36 items")
    return regions


def design(regions):
    stimuli = {}
    targets = {}
    for (item, condition), displays in regions.items():
        context, disamb = CONDITIONS[condition]
        family = f"context-6uv37-item-{item:02d}"
        key = hashlib.sha256(json.dumps(displays, ensure_ascii=False).encode()).hexdigest()[:24]
        identifier = f"context-6uv37-{key}"
        stops = []
        end = 0
        for display in displays:
            end += len(display)
            stops.append(end)
            end += 1
        text = " ".join(displays)
        critical = CRITICAL_REGIONS[item]
        if stops[-1] != len(text) or not 2 <= critical < len(stops):
            raise ValueError(f"Invalid display geometry for {item}:{condition}")
        stimuli[item, condition] = {"id": identifier, "text": text, "stops": stops}
        for region, (display, stop) in enumerate(zip(displays, stops, strict=True), start=1):
            if region == 1:
                continue
            phase = {critical: "critical", critical + 1: "spill"}.get(region, "other")
            targets[identifier, stop] = {
                "id": identifier, "stop": stop, "family": family, "context": context,
                "disamb": disamb, "region_words": len(display.split()), "phase": phase,
            }
    if len({row["id"] for row in stimuli.values()}) != len(stimuli):
        raise ValueError("Multiple text variants share a stimulus identifier")
    return stimuli, targets


def normalized_words(text):
    return tuple(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def shingles(words, width):
    return {words[index:index + width] for index in range(len(words) - width + 1)}


def document_overlap(stimuli, prior_path):
    prior_sha256 = digest(prior_path)
    if prior_sha256 != PRIOR_ENGLISH_STIMULI_SHA256:
        raise ValueError("Expected the locked 40 English texts used in the preceding readback assay")
    prior_documents = {normalized_words(row["text"]) for row in read_jsonl(prior_path)}
    prior_five = set().union(*(shingles(document, 5) for document in prior_documents))
    prior_twelve = set().union(*(shingles(document, OVERLAP_WORDS) for document in prior_documents))
    overlapping_families = set()
    matches = set()
    five_matches = set()
    twelve_matches = set()
    for (item, _), row in stimuli.items():
        words = normalized_words(row["text"])
        five_matches.update(shingles(words, 5) & prior_five)
        spans = shingles(words, OVERLAP_WORDS) & prior_twelve
        twelve_matches.update(spans)
        if words in prior_documents:
            matches.add(words)
        if words in prior_documents or spans:
            overlapping_families.add(item)
    return {
        "prior_stimuli_sha256": prior_sha256,
        "prior_documents": len(prior_documents),
        "normalization": "Unicode NFKC, casefold, then Unicode word tokens",
        "exclusion_rule": f"Exclude all six variants of any family with an exact normalized document match "
                          f"or an identical span of {OVERLAP_WORDS} consecutive word tokens shared with a prior text",
        "normalized_document_matches": len(matches),
        "five_word_shingle_matches": len(five_matches),
        "twelve_word_shingle_matches": len(twelve_matches),
        "excluded_items": sorted(overlapping_families),
    }


def create_geometry(source, prior_path, output):
    verify_sources(source)
    regions = read_regions(source / "alt_stim.txt")
    stimuli, targets = design(regions)
    overlap = document_overlap(stimuli, prior_path)
    excluded = set(overlap["excluded_items"])
    kept_stimuli = [row for (item, _), row in sorted(stimuli.items()) if item not in excluded]
    kept_ids = {row["id"] for row in kept_stimuli}
    kept_targets = [row for (identifier, _), row in sorted(targets.items()) if identifier in kept_ids]
    if not kept_targets:
        raise ValueError("No geometry remains after the independent-text check")
    audit = {
        "article": ARTICLE,
        "project": PROJECT,
        "source_sha256": SOURCE_SHA256,
        "geometry": {
            "raw_stimulus_variants": len(stimuli),
            "stimulus_variants": len(kept_stimuli),
            "families": len({row["family"] for row in kept_targets}),
            "original_stops": sum(len(row["stops"]) for row in kept_stimuli),
            "targets": len(kept_targets),
            "targets_by_phase": dict(Counter(row["phase"] for row in kept_targets)),
            "text": "Exact UTF-8 source display strings joined by one ASCII space, with offsets in Unicode characters",
            "design_fields": "Source condition, context, disamb and crit, with phase=critical at author crit, "
                             "spill at the next original display region, and other elsewhere",
            "item_4": "The source marks display 10 as critical. The complement and relative-clause versions "
                      "first differ in display 11, which is retained as spill.",
        },
        "material_overlap": overlap,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with TemporaryDirectory(dir=output.parent) as directory:
        staging = Path(directory) / "assay"
        staging.mkdir()
        write_jsonl(staging / "stimuli.jsonl", kept_stimuli)
        write_jsonl(staging / "targets.jsonl", kept_targets)
        audit["file_sha256"] = {
            name: digest(staging / name) for name in ("stimuli.jsonl", "targets.jsonl")
        }
        write_json(staging / "assay.json", audit)
        staging.rename(output)


def first_backward(positions, directions):
    """Yield eligible original display visits with their next explicit movement."""
    if (
        not positions or positions[0] != 1 or len(positions) != len(directions)
        or directions[-1] not in {1, 2}
    ):
        raise ValueError("Expected a trial beginning at display one and ending in advance or exit")
    seen = set()
    for position, following, direction in zip(positions[:-1], positions[1:], directions[:-1], strict=True):
        step = following - position
        if position == following == 1 and direction == 3:
            continue
        backward = (direction == -1 and step == -1) or (direction == 3 and following == 1 and step < 0)
        onward = direction == 1 and step == 1
        if not (backward or onward or (direction == 1 and step > 1)):
            raise ValueError(f"Unexpected direction {direction} for display {position} → {following}")
        if position not in seen:
            seen.add(position)
            if position >= 2 and all(earlier in seen for earlier in range(1, position)) and (backward or onward):
                yield position, backward
        if backward:
            break


def read_event_table(events_path):
    import rdata
    from rdata.parser._parser import DEFAULT_ALTREP_MAP, RObjectType
    from rdata.parser._xdr import ParserXDR

    class SharedStringParser(ParserXDR):
        """Share repeated, structurally equal plain character objects in this archive."""

        def __init__(self, data):
            super().__init__(data, expand_altrep=True, altrep_constructor_dict=DEFAULT_ALTREP_MAP)
            self.atoms = {}

        def parse_R_object(self, *args, **kwargs):
            value = super().parse_R_object(*args, **kwargs)
            info = value.info
            if (
                info.type == RObjectType.CHAR and not (info.object or info.attributes or info.tag or info.reference)
                and value.attributes is value.tag is value.referenced_object is None
            ):
                shared = self.atoms.setdefault((info.gp, value.value), value)
                if value != shared:
                    raise ValueError("Attempted to share different archived character objects")
                return shared
            return value

    if rdata.__version__ != "1.1.0":
        raise ValueError("Expected the verified rdata 1.1.0 XDR parser")
    with gzip.open(events_path, "rb") as stream:
        if stream.read(7) != b"RDX3\nX\n":
            raise ValueError("Expected the source R version 3 XDR workspace")
        decoder = SharedStringParser(memoryview(stream.read()))
    parsed = decoder.parse_all()
    decoder.check_complete()
    del decoder
    workspace = rdata.conversion.convert(parsed)
    expected = ["subj", "condition", "item", "no", "text", "fixno", "rt", "dir",
                "context", "disamb", "anyreg", "crit", "end"]
    if workspace.keys() != {"rl"} or list(workspace["rl"].columns) != expected:
        raise ValueError("The source archive does not contain the expected event table")
    return workspace["rl"]


def observed_counts(events_path, regions, stimuli, target_index):
    data = read_event_table(events_path)
    fields = ["subj", "item", "condition", "no", "text", "fixno", "dir", "context", "disamb", "crit"]
    data = data[fields].sort_values(["subj", "item", "fixno"])
    if data[["subj", "item", "fixno"]].duplicated().any() or data[fields].isna().any().any():
        raise ValueError("Missing or duplicated event keys")
    counts = defaultdict(Counter)
    for (_, item), trial in data.groupby(["subj", "item"], sort=False):
        item = int(item)
        first = trial.iloc[0]
        condition = str(first.condition)
        displays = regions[item, condition]
        stimulus = stimuli[item, condition]
        context, disamb = CONDITIONS[condition]
        if any(set(trial[field]) != {value} for field, value in {
            "condition": condition, "context": context, "disamb": disamb, "crit": CRITICAL_REGIONS[item],
        }.items()):
            raise ValueError("The recorded events disagree with the grade-free design metadata")
        identifier = stimulus["id"]
        positions = []
        directions = []
        for row in trial.itertuples(index=False):
            position = int(row.no)
            direction = int(row.dir)
            if not 1 <= position <= len(displays):
                raise ValueError("An event lies outside its original text")
            wording = unquote(row.text, errors="strict")
            if item == 24 and position == 1:
                wording = wording.replace("cafÃ©", "café")
            if wording != displays[position - 1]:
                raise ValueError(f"An event display differs from stimulus {item}:{condition}:{position}")
            positions.append(position)
            directions.append(direction)
        for position, is_backward in first_backward(positions, directions):
            key = identifier, stimulus["stops"][position - 1]
            if key in target_index:
                counts[key][is_backward] += 1
    return [
        {**target_index[key], "negative": counts[key][False], "positive": counts[key][True]}
        for key in sorted(counts)
    ]


def verified_forecasts(assay_root, predictions_dir, stimuli, targets):
    code = Path(__file__).resolve()
    files = {
        "assay_sha256": assay_root / "assay.json",
        "stimuli_sha256": assay_root / "stimuli.jsonl",
        "targets_sha256": assay_root / "targets.jsonl",
        "protocol_sha256": code.with_name("context-readback-transfer.md"),
        "heads_sha256": code.parent.parent / "results" / "first-readback-transfer-l4" / "heads.npz",
        "source_code_sha256": code.with_name("context_readback_forecast.py"),
        "predictions_sha256": predictions_dir / "predictions.jsonl",
        "extraction_sha256": predictions_dir / "extraction.json",
        "causality_sha256": predictions_dir / "causality.json",
    }
    hashes = {field: digest(path) for field, path in files.items()}
    receipt_path = predictions_dir / "seal.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt != {
            **hashes, "candidate_id": CANDIDATE, "model_revision": MODEL_REVISION,
            "documents": len(stimuli), "forecast_stops": len(targets),
        }
        or hashes["heads_sha256"] != HEADS_SHA256
    ):
        raise ValueError("Predictions disagree with the sealed assay, protocol, code, heads or model")
    assay = json.loads(files["assay_sha256"].read_text(encoding="utf-8"))
    if (
        assay["source_sha256"] != SOURCE_SHA256
        or assay["file_sha256"] != {
            "stimuli.jsonl": hashes["stimuli_sha256"], "targets.jsonl": hashes["targets_sha256"],
        }
    ):
        raise ValueError("The assay does not seal the geometry and author sources")
    all_stops = {(row["id"], stop) for row in stimuli for stop in row["stops"]}
    extraction = json.loads(files["extraction_sha256"].read_text(encoding="utf-8"))
    if (
        (extraction.get("model"), extraction.get("revision"), extraction.get("dtype"))
        != (MODEL, MODEL_REVISION, "float32")
        or extraction.get("input_sha256") != hashes["stimuli_sha256"]
        or extraction.get("input_texts") != len(stimuli) or extraction.get("encoded_texts") != len(stimuli)
        or extraction.get("unique_certified_stops") != len(all_stops) or extraction.get("excluded_stops") != []
    ):
        raise ValueError("The sealed extraction does not certify every original boundary")
    causality = json.loads(files["causality_sha256"].read_text(encoding="utf-8"))
    flags = defaultdict(set)
    for check in causality["checks"]:
        key = check["id"], check["stop"]
        comparison = check["comparison"]
        if key not in all_stops or check["passed"] is not True or comparison in flags[key]:
            raise ValueError("A causality check is invalid or duplicated")
        flags[key].add(comparison)
    required = {"substring", "right_padding", "suffix_replacement", "changed_suffix_positive_control"}
    if len(flags) != 3 or any(checks != required for checks in flags.values()):
        raise ValueError("The sealed extraction does not contain all registered causality checks")

    keys = set()
    for row in read_jsonl(files["predictions_sha256"]):
        if row.keys() != {"id", "stop", "methods"} or type(row["id"]) is not str or type(row["stop"]) is not int:
            raise ValueError("Invalid forecast schema")
        key = row["id"], row["stop"]
        parts = row["methods"]
        if (
            key not in targets or key in keys or not isinstance(parts, dict) or parts.keys() != METHODS
            or any(type(p) not in (int, float) or not math.isfinite(p) or not 0 < p < 1 for p in parts.values())
        ):
            raise ValueError("A forecast is missing a registered probability or uses an invalid stop")
        keys.add(key)
    if keys != targets.keys():
        raise ValueError("Predictions must cover every registered stop")
    return hashes["assay_sha256"], digest(receipt_path)


def create_truth(source, assay_root, predictions_dir, output):
    if output.exists():
        raise FileExistsError(output)
    verify_sources(source)
    regions = read_regions(source / "alt_stim.txt")
    stimuli, design_targets = design(regions)
    selected = read_jsonl(assay_root / "stimuli.jsonl")
    kept_ids = {row["id"] for row in selected}
    if len(selected) != len(kept_ids) or selected != [
        row for _, row in sorted(stimuli.items()) if row["id"] in kept_ids
    ]:
        raise ValueError("The sealed texts do not match the stimulus source")
    targets = {key: row for key, row in design_targets.items() if key[0] in kept_ids}
    if read_jsonl(assay_root / "targets.jsonl") != [targets[key] for key in sorted(targets)]:
        raise ValueError("The registered phase and material metadata do not match the stimulus source")
    assay_sha256, forecast_seal_sha256 = verified_forecasts(assay_root, predictions_dir, selected, targets)
    truth = observed_counts(source / "fixations.Rda", regions, stimuli, targets)
    if not truth:
        raise ValueError("No observed keypress outcomes align with the sealed design")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as directory:
        staging = Path(directory) / "truth"
        staging.mkdir()
        write_jsonl(staging / "truth.jsonl", truth)
        write_json(staging / "truth-seal.json", {
            "truth_sha256": digest(staging / "truth.jsonl"), "assay_sha256": assay_sha256,
            "forecast_seal_sha256": forecast_seal_sha256, "events_sha256": SOURCE_SHA256["fixations.Rda"],
            "source_code_sha256": digest(Path(__file__)),
        })
        staging.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    geometry = commands.add_parser("geometry")
    truth = commands.add_parser("truth")
    for command in (geometry, truth):
        command.add_argument("--source", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
    geometry.add_argument("--prior-english", required=True, type=Path)
    truth.add_argument("--assay", required=True, type=Path)
    truth.add_argument("--predictions-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.mode == "geometry":
        create_geometry(args.source, args.prior_english, args.output)
    else:
        create_truth(args.source, args.assay, args.predictions_dir, args.output)


if __name__ == "__main__":
    main()
