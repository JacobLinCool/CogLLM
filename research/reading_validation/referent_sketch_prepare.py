"""Seal text geometry and the native component of the published source-only click prior."""
import json
from pathlib import Path
import numpy as np
from research.backbone_transfer.extract import digest, write_json
from research.reading_validation.transition_pointer_features import CONTROL

ROOT = Path(".cache/cogllm/reading-validation")
OUT = Path("research/results/reading-validation-shared-referent")
PROTOCOL = Path("research/reading_validation/shared-referent-protocol.md")
POINTER = Path("research/results/reading-validation-transition-pointer/source-weights.npz")
SOURCES = ("pointer", "dpr-pointer", "ambicoref-reading")

def material():
    return {s: list(map(json.loads, (ROOT / (s + "-stimuli.jsonl")).read_text().splitlines())) for s in SOURCES}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [r for source in material().values() for r in source]
    if len(set(r["id"] for r in rows)) != len(rows):
        raise ValueError("Study article IDs must be distinct")
    with np.load(POINTER) as weights:
        slope = weights["headwise__beta"] / weights["headwise__sd"]
        native = slope[len(CONTROL):]
    if native.shape != (52,):
        raise ValueError("Native click prior changed")
    path = ROOT / "referent-sketch-exposure.jsonl"
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps({k:row[k] for k in ("id","text","words","tokens")},ensure_ascii=False,separators=(",",":"))+"\n")
    write_json(OUT/"exposure-seal.json", {"articles":len(rows),"input_sha256":digest(path),"protocol_sha256":digest(PROTOCOL),
                "native_weights":native.tolist(),"source_weights_sha256":digest(POINTER),
                "material_sha256":{s:digest(ROOT/(s+"-stimuli.jsonl")) for s in SOURCES}})

if __name__ == "__main__":
    main()
