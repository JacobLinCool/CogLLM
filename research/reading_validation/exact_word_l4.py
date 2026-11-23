"""Replay two externally scored one-pass readouts on the actual original English paragraphs."""

import json
from pathlib import Path
from time import perf_counter

from research.backbone_transfer.extract import digest, stream_rows, write_json
from research.reading_validation.spelling_inference import ReadingWordForecast


def run(runtime, reader, checkpoint, source, out):
    out = Path(out)
    out.mkdir()
    begin = perf_counter()
    head = ReadingWordForecast.load()
    material = list(stream_rows(source))
    if len(material) != 54 or any(
        set(row) != {"id", "text", "ends"} for row in material
    ):
        raise ValueError("Expected the original 54 unmarked paragraph presentations")
    forwards = 0

    def counted(*_):
        nonlocal forwards
        forwards += 1

    filename = out / "word-forecasts.jsonl"
    with (
        runtime.encoder.register_forward_pre_hook(counted),
        filename.open("x", encoding="utf-8") as f,
    ):
        for row in material:
            encoded = runtime.encode(row["text"])
            words = head.analyze(row["text"], encoded, runtime)
            for w in words:
                a, b, _, t = row["ends"][w["index"]]
                if (a, b, t) != (w["start"], w["end"], w["token"]):
                    raise ValueError(
                        "Published and recomputed word-end coordinates differ"
                    )
            f.write(
                json.dumps({"id": row["id"], "words": words}, allow_nan=False) + "\n"
            )
    if forwards != len(material):
        raise ArithmeticError("One original passage must use one forward")
    write_json(
        out / "receipt.json",
        {
            "input": digest(source),
            "output": digest(filename),
            "revision": runtime.revision,
            "documents": len(material),
            "forwards": forwards,
            "seconds": perf_counter() - begin,
            "reader_labels_in_input": False,
        },
    )
