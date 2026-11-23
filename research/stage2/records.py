"""Serialization for Stage 2 records."""

import json


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def observed(value):
    return {"status": "observed", "value": value}
