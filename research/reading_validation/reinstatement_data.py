"""Register all intervention narratives once without the investigator's probes in the GPU input."""

import json
from pathlib import Path

from research.backbone_transfer.extract import digest, write_json
from research.reading_validation.character_access_data import INPUT as NAMES
from research.reading_validation.character_access_data import ROOT
from research.reading_validation.character_object_access_data import INPUT as OBJECTS
from research.reading_validation.transition_goal_access_data import INPUT as GOALS

PLAN = Path(__file__).with_name("event-reinstatement-protocol.md")
OUT = Path("research/results/reading-validation-reinstatement")
INPUT = ROOT / "event-reinstatement.jsonl"
FILES = {"name": NAMES, "object": OBJECTS, "goal": GOALS}


def main():
    rows = {}
    for p in FILES.values():
        records = [json.loads(v) for v in p.read_text().splitlines()]
        if len(records) != 96:
            raise ValueError("Original factorial bank is incomplete")
        for r in records:
            spec = {
                k: r[k] for k in ("id", "text", "prior_end", "last_start", "last_end")
            }
            if rows.setdefault(r["id"], spec) != spec:
                raise ValueError("Matching article has competing region definitions")
    INPUT.write_text(
        "".join(json.dumps(v, ensure_ascii=False) + "\n" for v in rows.values())
    )
    OUT.mkdir(exist_ok=True)
    write_json(
        OUT / "input.json",
        {
            "protocol": digest(PLAN),
            "files": {k: digest(p) for k, p in FILES.items()},
            "input": digest(INPUT),
            "presentations": len(rows),
            "public_banks": len(FILES),
        },
    )
    print(len(rows), digest(INPUT))


if __name__ == "__main__":
    main()
