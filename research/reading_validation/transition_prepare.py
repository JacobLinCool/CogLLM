"""Extract text only, keeping all four public Carter–Hoffman versions together."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import re

from openpyxl import load_workbook

ROOT = Path(".cache/cogllm/reading-validation")
OUT = Path("research/results/reading-validation-transition")
STIMULI_SHA = "170f3093cbe87f9f860a1fb47af24d7587189618df224d7d3f25024ead4b492d"
SALT = "cogllm-carter-independent-item-v1:"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prepare():
    path = ROOT / "carter/exp1_stims.xlsx"
    if sha(path.read_bytes()) != STIMULI_SHA:
        raise ValueError("Source stimulus SHA-256 mismatch")
    wb = load_workbook(path, read_only=True, data_only=True)
    items = defaultdict(dict)
    for sheet in wb:
        for trial, condition, preamble, target, *_ in sheet.values:
            if type(trial) is not int:
                continue
            value = (" ".join(preamble.split()), " ".join(target.split()))
            if items[trial].setdefault(condition, value) != value:
                raise ValueError("Repeated or reversed list changes text")
    if len(items) != 160:
        raise ValueError("Original stimulus set incomplete")
    records, metadata = [], []
    prefix_to_family = {}
    canonical = {}
    for trial in sorted(items):
        versions = items[trial]
        if set(versions) != set("abcd"):
            raise ValueError(f"Incomplete factorial for item {trial}")
        ap, at = versions["a"]
        bp, bt = versions["b"]
        cp, ct = versions["c"]
        dp, dt = versions["d"]
        if (ap, cp, at, bt) != (bp, dp, ct, dt) or ap == cp:
            raise ValueError(f"Unmatched preamble or target under coherence for {trial}")
        tw = at.split()
        delta = [k for k, (w, v) in enumerate(zip(tw, bt.split(), strict=True)) if w != v]
        if len(delta) != 1 or delta[0] < 1:
            raise ValueError(f"Critical word for {trial} could not be aligned")
        critical = delta[0]
        prefix = " ".join(tw[:critical]).casefold()
        group = prefix_to_family.setdefault(prefix, trial)
        canonical[trial] = group
        for condition, (preamble, target) in sorted(versions.items()):
            text = preamble + " " + target
            target_start = len(preamble) + 1
            identifier = sha(text.encode())[:20]
            loc = [(target_start + match.start(), target_start + match.end()) for match in re.finditer(r"\S+", target)]
            records.append({"id": identifier, "text": text, "start": target_start,
                            "target_words": loc})
            metadata.append({"id": identifier, "item": trial, "family": group, "condition": condition,
                             "critical_index": critical, "target_length": len(loc)})
    ids = sorted(set(canonical.values()), key=lambda item: sha((SALT + str(item)).encode()))
    train = set(ids[:int(0.6 * len(ids))])
    for row in metadata:
        row["split"] = "calibration" if row["family"] in train else "holdout"
    if len({x["id"] for x in records}) != len(records):
        raise ValueError("Duplicate full text")
    OUT.mkdir(parents=True, exist_ok=True)
    packet = ROOT / "bridge-stimuli.jsonl"
    packet.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    path = OUT / "stimuli.json"
    protocol = Path("research/reading_validation/transition-protocol.md")
    path.write_text(json.dumps({"source_sha256": STIMULI_SHA, "source": "https://osf.io/download/am3y6/",
                                "source_license": "CC BY 4.0", "protocol_sha256": sha(protocol.read_bytes()),
                                "packet_sha256": sha(packet.read_bytes()), "items": len(items),
                                "families": len(ids), "calibration_groups": len(train),
                                "holdout_groups": len(ids) - len(train), "rows": metadata}, indent=2))
    print(f"{len(records)} unique passages, {len(ids)} independent target-prefix groups, {len(train)} for calibration.")


if __name__ == "__main__":
    prepare()
