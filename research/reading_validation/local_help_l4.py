"""Single-forward, response-free feature extraction for an editor's designated English terms."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from research.reading_validation.local_forecast import digest, rows as read_rows
from research.reading_validation.local_help_article import ArticleHelpFeatures, RESULTS


def run(runtime, reader, checkpoint, input_path, output_dir):
    del reader
    if (
        runtime.candidate != "qwen"
        or runtime.revision != checkpoint["revision"]
        or runtime.width != 2048
    ):
        raise ValueError(
            "Help preferences require the pinned native Qwen3.5-2B text backbone"
        )
    start = perf_counter()
    source = Path(input_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_rows(source)
    columns = json.loads(
        (RESULTS / "reading-validation-local/help-choice/geometry.json").read_bytes()
    )["columns"]
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Source article IDs must be unique")
    by_text = {}
    for row in rows:
        if (
            set(row) != {"id", "text", "terms"}
            or not isinstance(row["text"], str)
            or not row["text"].strip()
        ):
            raise ValueError("Supply original text and target geometry only")
        terms = row["terms"]
        if any(set(t) != {"id", "start", "end", "term"} for t in terms) or len(
            {t["id"] for t in terms}
        ) != len(terms):
            raise ValueError(
                "Each term requires a distinct ID and source character bounds"
            )
        by_text.setdefault(row["text"], []).append(row)
    builder = ArticleHelpFeatures()
    features_file = out / "features.jsonl"
    all_calls = 0
    expected = 0
    total_tokens = 0
    scored = 0
    skipped = 0

    def counted(_, __):
        nonlocal all_calls
        all_calls += 1

    with (
        runtime.encoder.register_forward_pre_hook(counted),
        torch.inference_mode(),
        features_file.open("w") as stream,
    ):
        for text, presentations in by_text.items():
            ids = runtime.tokenizer.encode(text, add_special_tokens=False).ids
            n = len(ids)
            if n > 2048:
                for q in presentations:
                    stream.write(
                        json.dumps(
                            {
                                "id": q["id"],
                                "tokens": n,
                                "length_domain": "over_limit",
                                "terms": [],
                                "excluded_terms": [
                                    {"id": t["id"], "reason": "over 2048 native tokens"}
                                    for t in q["terms"]
                                ],
                            }
                        )
                        + "\n"
                    )
                    skipped += len(q["terms"])
                continue
            queries = {}
            for q in presentations:
                for t in q["terms"]:
                    queries[q["id"], t["id"]] = (t["term"], t["start"], t["end"])
            native = runtime.encode(text)
            expected += 1
            total_tokens += n
            values, excluded = builder.from_native(
                text, queries, native, runtime.tokenizer
            )
            for q in presentations:
                complete = []
                omitted = []
                for t in q["terms"]:
                    key = q["id"], t["id"]
                    if key in excluded:
                        omitted.append({"id": t["id"], "reason": excluded[key]})
                    else:
                        x = values[key]
                        if x.shape != (len(columns),) or not np.isfinite(x).all():
                            raise ArithmeticError(
                                "Invalid scientific-term feature vector"
                            )
                        complete.append({"id": t["id"], "features": x.tolist()})
                scored += len(complete)
                skipped += len(omitted)
                stream.write(
                    json.dumps(
                        {
                            "id": q["id"],
                            "tokens": n,
                            "length_domain": "source"
                            if n <= 1024
                            else "length_extrapolation",
                            "terms": complete,
                            "excluded_terms": omitted,
                        }
                    )
                    + "\n"
                )
            del native
    if all_calls != expected:
        raise ArithmeticError(
            f"Observed {all_calls} model calls for {expected} distinct eligible inputs"
        )
    seal = {
        "data_sha256": digest(source),
        "output_sha256": digest(features_file),
        "columns": columns,
        "presentations": len(rows),
        "unique_original_inputs": len(by_text),
        "backbone_forwards": all_calls,
        "encoded_tokens": total_tokens,
        "scored_terms": scored,
        "excluded_terms": skipped,
        "qwen_revision": runtime.revision,
        "worker_sha256": digest(Path(__file__)),
        "converter_sha256": digest(Path(__file__).with_name("local_help_article.py")),
        "elapsed_seconds": perf_counter() - start,
    }
    (out / "extraction.json").write_text(json.dumps(seal, indent=2) + "\n")
    return seal
