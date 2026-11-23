"""Audit native Qwen token boundaries for designated CompLex spans."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import unicodedata

def token_span(text, offsets, start, end):
    selected = []
    covered = set()
    for i, (left, right) in enumerate(offsets):
        lo, hi = max(start, left), min(end, right)
        inside = [index for index in range(lo, hi) if not text[index].isspace()]
        if not inside:
            continue
        if text[left:start].strip() or text[end:right].strip():
            return "crosses_nonwhite_boundary", None
        selected.append(i)
        covered.update(inside)
    if covered != {index for index in range(start, end) if not text[index].isspace()} or not selected:
        return "incomplete_coverage", None
    return "exact_nonwhite_span", (selected[0], selected[-1])


def audit(data, tokenizer_path, output):
    import tokenizers
    from ..stage3.probe import _jsonl
    from ..stage3.run import MODEL, REVISION

    if tokenizer_path.parent.name != REVISION or tokenizer_path.name != "tokenizer.json":
        raise ValueError("Expected tokenizer.json from the pinned Qwen3.5-2B revision")
    tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
    stimuli = {row["id"]: row for row in _jsonl(data / "stimuli.jsonl")}
    encoded = {key: tokenizer.encode(row["text"], add_special_tokens=False) for key, row in stimuli.items()}
    by_split = defaultdict(Counter)
    target_context = defaultdict(lambda: defaultdict(set))
    widths = Counter()

    for obs in _jsonl(data / "observations.jsonl"):
        row = stimuli[obs["stimulus_id"]]
        text = row["text"]
        span = obs["target_span"]
        left, right = span["start"], span["end"]
        if not (0 <= left < right <= len(text)) or text[left:right] != span["text"] or obs["split"] != row["split"]:
            raise ValueError(f"Invalid prepared CompLex target span for {obs['id']}")
        stats = by_split[row["split"]]
        stats["observations"] += 1
        reading = encoded[row["id"]]
        status, location = token_span(text, reading.offsets, left, right)
        stats[status] += 1
        if location is None:
            continue
        begin, last = location
        widths[last - begin + 1] += 1
        cut = tokenizer.encode(text[:right], add_special_tokens=False)
        if reading.ids[:last + 1] == cut.ids:
            stats["exact_span_and_stable_target_end_prefix"] += 1
        else:
            stats["exact_span_but_changed_token_prefix"] += 1
        key = (obs["kind"], unicodedata.normalize("NFKC", span["text"]).casefold())
        target_context[row["split"]][key].add(row["context_group"])

    sources = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in (
            ("tokenizer.json", tokenizer_path), ("stimuli.jsonl", data / "stimuli.jsonl"),
            ("observations.jsonl", data / "observations.jsonl"),
        )
    }
    heldout = target_context["test"]
    eligible_repeat = [contexts for contexts in heldout.values() if len(contexts) > 1]
    counts = [len(item.ids) for item in encoded.values()]
    result = {
        "model": MODEL,
        "revision": REVISION,
        "tokenizers": tokenizers.__version__,
        "input_sha256": sources,
        "span_scope": (
            "Whole-sentence native tokenization; selected overlapping tokens must cover exactly the non-whitespace "
            "characters of the designated span. A stable prefix means equality to tokenizing the original text "
            "truncated immediately after the final target character. This checks token IDs only."
        ),
        "by_split": {name: dict(by_split[name]) for name in ("train", "trial", "test")},
        "exact_target_token_width_counts": dict(sorted(widths.items())),
        "native_sentence_inventory": {
            "unique_sentences": len(encoded), "tokens": sum(counts), "shortest": min(counts), "longest": max(counts),
        },
        "test_exact_span_target_types": {
            "casefold_unique": len(heldout),
            "also_in_train": len(set(heldout) & set(target_context["train"])),
            "with_multiple_distinct_context_groups": len(eligible_repeat),
            "same_type_pairs_across_contexts_before_scoring": sum(len(group) * (len(group) - 1) // 2 for group in eligible_repeat),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "tokenizer", "output"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    args = parser.parse_args()
    audit(args.data, args.tokenizer, args.output)
