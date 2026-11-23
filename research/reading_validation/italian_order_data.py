"""Certified direct reconstruction of the four source sentence positions for held Italian passages."""

import json
from collections import Counter, defaultdict
from functools import cache
from itertools import combinations
from pathlib import Path

from research.reading_validation.coherence_order_data import ITALIAN, csv_rows, read
from research.reading_validation.coherence_order_data import OUT as COH
from research.reading_validation.coherence_order_genre import lex
from research.reading_validation.local_forecast import digest
from research.reading_validation.memory_data import h

OUT = Path("research/results/reading-validation-italian-interventions")
PLAN = Path(__file__).with_name("italian-order-interventions.md")
JOIN = " "


def sentence_bank():
    words = defaultdict(lambda: defaultdict(set))
    text = defaultdict(lambda: defaultdict(set))
    redundant = 0
    for genre in ("ted", "wiki"):
        for row in csv_rows(ITALIAN / f"task1-{genre}-train.tsv"):
            fields = row["ID"].split("#")
            group = genre + "/" + fields[0]
            paragraph = int(fields[1]) if len(fields) == 3 else -1
            offset = int(fields[-1])
            parts = row["PROMPT"].split("\n")
            if len(parts) != 3 or row["CLASS"] not in {"0", "1"}:
                raise ValueError("Invalid Task1 source coordinates")
            if row["CLASS"] == "1":
                parts.append(row["TARGET"])
            for j, sentence in enumerate(parts):
                k = (paragraph, offset + j)
                if k in words[group]:
                    redundant += 1
                words[group][k].add(lex(sentence))
                text[group][k].add(sentence)
    if any(len(v) != 1 for source in words.values() for v in source.values()):
        raise ValueError("Conflicting source sentence assignments")
    return words, text, redundant


def identify(index, material):
    by_first = defaultdict(list)
    for pos, choices in index.items():
        seq = next(iter(choices))
        if seq:
            by_first[seq[0]].append((seq, pos))
    words = lex(material)

    @cache
    def parse(first, n):
        if first == len(words):
            return [()] if n == 4 else []
        if n >= 4:
            return []
        out = []
        for sequence, position in by_first[words[first]]:
            last = first + len(sequence)
            if words[first:last] == sequence:
                out.extend((position, *suffix) for suffix in parse(last, n + 1))
        return out

    choices = parse(0, 0)
    return choices[0] if len(choices) == 1 else None


def classify(pos):
    if len({p[0] for p in pos}) != 1:
        return None
    nums = [p[1] for p in pos]
    m = min(nums)
    shape = [i - m for i in nums]
    is_consecutive = sorted(shape) == [0, 1, 2, 3]
    if shape == [0, 1, 2, 3]:
        return "original"
    if is_consecutive and sum(i != j for i, j in enumerate(shape)) == 2:
        return "transposed"
    for i in range(4):
        if nums == [
            nums[(i + 1) % 4] - ((i + 1) % 4) + j + 10 * (j == i) for j in range(4)
        ]:
            return "substitution"
    return None


def spans(parts):
    pos = 0
    out = []
    for s in parts:
        out.append([pos, pos + len(s)])
        pos += len(s) + 1
    return out


def run():
    w, raw, redundant = sentence_bank()
    at = read(COH / "genre-forecasts.json")
    source = {}
    alt = {}
    diagnostic = {}
    summary = {}
    states = Counter()
    for split in ("train", "held"):
        for r in csv_rows(ITALIAN / (split + ".tsv")):
            k = r["ID"]
            g = at[k]["matched_source"]
            if not g:
                continue
            pos = identify(w[g], r["TEXT"])
            typ = classify(pos) if pos else None
            if not typ:
                continue
            native = [raw[g][v] for v in pos]
            if not all(len(s) == 1 for s in native):
                raise ValueError("Disagreement in original punctuation")
            pieces = [next(iter(s)) for s in native]
            verbatim = JOIN.join(pieces) == r["TEXT"]
            source[k] = {
                "domain": g.split("/")[0],
                "source": g,
                "coordinates": [list(v) for v in pos],
                "type": typ,
                "split": split,
                "verbatim": verbatim,
            }
            states[(split, typ, verbatim)] += 1
            if split != "held" or not verbatim or typ == "substitution":
                continue
            original = h(r["TEXT"])
            order = list(range(4))
            canonical = sorted(order, key=lambda i: pos[i])
            if typ == "original":
                parta, partb = list(combinations(range(4), 2))[
                    int(h("cogllm-italian-order-swap-v1." + original)[:16], 16) % 6
                ]
                permuted = order.copy()
                permuted[parta], permuted[partb] = permuted[partb], permuted[parta]
                alternative = permuted
            else:
                permuted = order
                alternative = canonical
            candidate = JOIN.join(pieces[j] for j in alternative)
            other = h(candidate)
            if original == other or other in alt:
                raise ValueError("The permutation must be a distinct material")
            alt[other] = candidate
            ids = [canonical.index(i) for i in permuted]
            first = next(j for j in range(1, 4) if ids[j] != ids[j - 1] + 1)
            can_pieces = [pieces[i] for i in canonical]
            sw_pieces = [pieces[i] for i in permuted]
            can_text = JOIN.join(can_pieces)
            sw_text = JOIN.join(sw_pieces)
            diagnostic[k] = {
                "observed": original,
                "alternative": other,
                "source": g,
                "observed_type": typ,
                "canonical": h(can_text),
                "permuted": h(sw_text),
                "permuted_positions": ids,
                "first_disrupted": first,
                "canonical_interval": spans(can_pieces)[ids[first]],
                "permuted_interval": spans(sw_pieces)[first],
                "canonical_has_predecessor": ids[first] > 0,
            }
        summary[split] = dict(
            Counter(u["type"] for u in source.values() if u["split"] == split)
        )
    OUT.mkdir(parents=True, exist_ok=True)
    file = ITALIAN / "counterfactuals.jsonl"
    file.write_text(
        "".join(
            json.dumps({"id": k, "text": v}, ensure_ascii=False, separators=(",", ":"))
            + "\n"
            for k, v in sorted(alt.items())
        )
    )
    (OUT / "provenance.json").write_text(json.dumps(source, indent=2) + "\n")
    (OUT / "order-pairs.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
    rec = {
        "original_public_positions": sum(len(x) for x in w.values()),
        "independently_repeated_source_slots": redundant,
        "classification": summary,
        "source_role": {"|".join(map(str, k)): v for k, v in states.items()},
        "examples": len(alt),
        "source": digest(file),
        "registered": digest(PLAN),
        "geometry": digest(OUT / "order-pairs.json"),
        "public_reconstruction": digest(OUT / "provenance.json"),
        "outcomes_accessed": False,
    }
    (OUT / "inputs.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(rec)


if __name__ == "__main__":
    run()
