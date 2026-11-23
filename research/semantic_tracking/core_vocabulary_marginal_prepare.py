"""Sample grade-free cue paths from the author Experiment 1 randomization."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import itertools
import json
import re
import tempfile
import unicodedata
from pathlib import Path


AUTHOR_REVISION = "bf8c249a688211f4085db2905a1e8e09804a9d8a"
SOURCE_SHA256 = {
    "codewords-main-pinned.js": "0f2e5744b2c93c3caf473327175df0484b5a3b2c1040b85686bff7a9f5066c63",
    "codewords-stim.js": "a82a510df7512795946cdf416a16f2d74a05a6abdec348a0a248724e0dc209f3",
    "codewords-jspsych.js": "bfc204f530f1f353d9d5fea087f9eb4658e4abf399eaed0f815666372c7e55bb",
    "codewords-plugin-word-pred.js": "3c03ab9ef983a9cfefeebb849c8bf8609bd77a3f940feb2a787e97ddea110cfb",
    "sentences_corpus_matched.csv": "9ce7733fed38e2e1e97300621a6001c03927597aab375260374f1749ca86d3b3",
}
DEFAULT_SEED = "cogllm-core-vocabulary-marginal-v1"
ORDINALS = 6
BIN_SIZE = 6
SUPPORT_SIZE = BIN_SIZE**ORDINALS
SEPARATOR = "\n"
MASK = "█" * 5


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def target_family(target: str) -> str:
    """Use the local target as a grouping key for fitting and human scoring."""
    normalized = unicodedata.normalize("NFKC", target).casefold()
    if not re.fullmatch(r"[a-z]+", normalized):
        raise ValueError("Invalid target key")
    return hashlib.sha256(f"core-vocabulary-family-v1:{normalized}".encode()).hexdigest()[:32]


def load_cue_bins(source: Path) -> dict[str, tuple[tuple[str, ...], ...]]:
    for name, expected in SOURCE_SHA256.items():
        if sha256(source / name) != expected:
            raise ValueError(f"Unexpected source SHA-256 for {name}")
    javascript = (source / "codewords-stim.js").read_text(encoding="utf-8")
    arrays = {
        match[1]: json.JSONDecoder().raw_decode(javascript[match.end():])[0]
        for match in re.finditer(r"(?m)^var (\w+) = ", javascript)
    }
    if arrays.keys() != {"tw", "sentw", "sen"}:
        raise ValueError("Unexpected stimulus declarations")
    conditions, assignments, sentences = (arrays[k] for k in ("tw", "sentw", "sen"))
    declared_targets = list(itertools.chain.from_iterable(conditions.values()))
    if (
        sorted(map(len, conditions.values())) != [24] * 4
        or len(set(declared_targets)) != 96
        or set(declared_targets) != set(assignments)
    ):
        raise ValueError("Unexpected target randomization groups")
    with (source / "sentences_corpus_matched.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    corpus = {row["index"]: row for row in rows}
    if len(rows) != 3456 or len(corpus) != len(rows) or corpus.keys() != sentences.keys():
        raise ValueError("Sentence pool does not match the author client")

    seen = set()
    normalized_sentences = set()
    targets = {}
    for target in sorted(assignments):
        author_bins = assignments[target]
        if author_bins.keys() != {str(i) for i in range(1, ORDINALS + 1)}:
            raise ValueError("Unexpected bin ordinals")
        visible_target = re.compile(rf"(?<!\w){re.escape(target)}(?!\w)", re.IGNORECASE)
        all_bins = []
        previous_max = float("-inf")
        for ordinal in range(1, ORDINALS + 1):
            keys = tuple(map(str, author_bins[str(ordinal)]))
            if len(keys) != BIN_SIZE or len(set(keys)) != BIN_SIZE:
                raise ValueError("Expected six distinct cues per bin")
            predictability = [float(corpus[key]["Log10_Predictability"]) for key in keys]
            if min(predictability) < previous_max:
                raise ValueError("Cue bins must ascend by source predictability")
            previous_max = max(predictability)
            cues = []
            for key in keys:
                row = corpus[key]
                if key in seen or row["target"] != target:
                    raise ValueError("Cue does not have a unique target and bin")
                seen.add(key)
                (rendered_html,) = sentences[key]
                rendered = html.unescape(rendered_html)
                if (
                    rendered.count(MASK) != 1
                    or rendered != rendered.strip()
                    or any(ord(c) < 32 for c in rendered)
                    or re.search(r"<[a-z/]", rendered_html, re.IGNORECASE)
                    or visible_target.search(rendered)
                ):
                    raise ValueError("Unexpected masked cue rendering")
                prefix, suffix = rendered.split(MASK)
                original = row["sentence"]
                end = len(original) - len(suffix)
                if (
                    not original.startswith(prefix)
                    or not original.endswith(suffix)
                    or original[len(prefix):end].casefold() != target
                ):
                    raise ValueError("Rendered cue does not match its source sentence")
                normalized = " ".join(unicodedata.normalize("NFKC", rendered).casefold().split())
                if normalized in normalized_sentences:
                    raise ValueError("Repeated masked cue")
                normalized_sentences.add(normalized)
                cues.append(rendered)
            all_bins.append(tuple(cues))
        targets[target] = tuple(all_bins)
    if seen != corpus.keys():
        raise ValueError("Cue sampling design omitted source sentences")
    return targets


def path_indices(family: str, seed: str, draws: int) -> list[int]:
    """Use unbiased rejection over SHA-256 blocks and keep distinct paths."""
    if not 1 <= draws <= SUPPORT_SIZE:
        raise ValueError("Draw count must be in the finite path support")
    selected = {}
    # Decode only the first 64 digest bits. Reject the final incomplete interval.
    limit = 2**64 - (2**64 % SUPPORT_SIZE)
    for counter in itertools.count():
        message = json.dumps([seed, family, counter], separators=(",", ":")).encode()
        block = int.from_bytes(hashlib.sha256(message).digest()[:8], "big")
        if block < limit:
            selected[block % SUPPORT_SIZE] = None
            if len(selected) == draws:
                return list(selected)
    raise AssertionError("The hash block generator terminated")


def render(bins: tuple[tuple[str, ...], ...], index: int) -> tuple[str, list[int]]:
    if not 0 <= index < SUPPORT_SIZE:
        raise ValueError("Path index outside finite support")
    cues = []
    for choices in bins:
        index, choice = divmod(index, BIN_SIZE)
        cues.append(choices[choice])
    stops = list(itertools.accumulate(len(cue) + len(SEPARATOR) for cue in cues))
    return SEPARATOR.join(cues), [stop - len(SEPARATOR) for stop in stops]


def build_packet(source: Path, output: Path, draws: int, seed: str) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if not 1 <= draws <= SUPPORT_SIZE or not seed:
        raise ValueError("Expected a nonempty seed and a valid draw count")
    targets = load_cue_bins(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with tempfile.TemporaryDirectory(dir=output.parent, prefix="core-vocab-") as staging:
        packet = Path(staging) / "packet"
        model_input = packet / "model_input"
        model_input.mkdir(parents=True)
        stimuli = model_input / "stimuli.jsonl"
        mixture = packet / "mixture.jsonl"
        with (
            stimuli.open("w", encoding="utf-8", newline="\n") as stimulus_file,
            mixture.open("w", encoding="utf-8", newline="\n") as mixture_file,
        ):
            for target, bins in targets.items():
                family = target_family(target)
                for draw, index in enumerate(path_indices(family, seed, draws), start=1):
                    text, stops = render(bins, index)
                    identifier = hashlib.sha256(
                        b"core-vocabulary-stimulus-v1\x00" + text.encode()
                    ).hexdigest()[:32]
                    if identifier in seen:
                        raise ValueError("Repeated stimulus path across the packet")
                    seen.add(identifier)
                    stimulus_row = {"id": identifier, "text": text, "stops": stops}
                    mixture_row = {"id": identifier, "family": family, "draw": draw, "weight": 1 / draws}
                    stimulus_file.write(json.dumps(stimulus_row, ensure_ascii=False) + "\n")
                    mixture_file.write(json.dumps(mixture_row) + "\n")
        receipt = {
            "schema_version": 1,
            "author_git_revision": AUTHOR_REVISION,
            "source_sha256": SOURCE_SHA256,
            "prepare_code_sha256": sha256(Path(__file__)),
            "targets": len(targets),
            "bins_per_target": ORDINALS,
            "cues_per_bin": BIN_SIZE,
            "author_paths_per_target": SUPPORT_SIZE,
            "author_path_probability": 1 / SUPPORT_SIZE,
            "sampling": "SHA-256 rejection sampling of unique path indices within each target",
            "seed": seed,
            "draws_per_target": draws,
            "stimuli": len(seen),
            "stops": len(seen) * ORDINALS,
            "separator": SEPARATOR,
            "file_sha256": {
                "model_input/stimuli.jsonl": sha256(stimuli),
                "mixture.jsonl": sha256(mixture),
            },
        }
        (packet / "preflight.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        packet.rename(output)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    args = parser.parse_args()
    print(json.dumps(build_packet(args.source, args.output, args.draws, args.seed), sort_keys=True))


if __name__ == "__main__":
    main()
