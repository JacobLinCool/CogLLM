"""Verify localhost scoring and parity with the original PLABA research probe."""

import argparse
import csv
from html import unescape
import json
from pathlib import Path
import re

import gradio as gr
from gradio_client import Client
import mlx.core as mx
from mlx_lm import __version__ as mlx_lm_version
import numpy as np

from research.train_plaba_probe import extract_features, predict

from .app import CSS, EASE, build_app
from .plaba import BASE_REVISION, PlabaReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sentences", type=Path, required=True, help="Published PLABA-EVAL sentences.csv")
    args = parser.parse_args()

    with args.sentences.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["stimulus_id"] == "text_01"]
    rows.sort(key=lambda row: int(row["sentence_index"]))
    if len(rows) != 7:
        raise ValueError("The PLABA verification abstract must have seven annotated sentences")
    sentences = [row["sentence_text"] for row in rows]
    text = "  " + "\n\n".join(sentences) + "  "

    reader = PlabaReader(args.model)
    encoded = extract_features(args.model, {"text_01": rows})
    features = np.asarray([encoded[row["sentence_uid"]] for row in rows], dtype=np.float64)
    expected = predict(features / np.linalg.norm(features, axis=1, keepdims=True), reader.head)
    analysis = reader.analyze(text)
    if [text[span.start:span.end] for span in analysis.sentences] != sentences:
        raise AssertionError("Pasted sentence boundaries differ from the verification abstract")
    scores = np.array([(span.ease, span.understanding) for span in analysis.sentences])
    np.testing.assert_allclose(scores, expected, rtol=0, atol=2e-6)

    calls = 0
    encode = reader.encoder

    def count_forward(input_ids):
        nonlocal calls
        calls += 1
        return encode(input_ids)

    reader.encoder = count_forward
    demo = build_app(reader)
    try:
        _, address, _ = demo.launch(
            server_name="127.0.0.1", share=False, css=CSS, quiet=True, prevent_thread_lock=True,
        )
        table, article, detail, status = Client(address, verbose=False).predict(text, EASE, api_name="/analyze")
        if calls != 1 or len(table["data"]) != len(rows):
            raise AssertionError(f"Expected one forward and seven results, got {calls} and {len(table['data'])}")
        np.testing.assert_allclose([row[2:] for row in table["data"]], np.round(expected, 2), rtol=0, atol=1e-9)
        source = re.search(r'<div class="source-text"[^>]*>(.*?)</div>', article, re.DOTALL)
        if (source is None or unescape(re.sub(r"<[^>]+>", "", source[1])) != text
                or "第 1 句" not in detail or "7 句" not in status):
            raise AssertionError("Gradio did not preserve the source text and sentence details")
        if (re.findall(r'data-score="([0-9.]+)"', source[1]) != [f'{value:.2f}' for value in expected[:, 0]]
                or not {'0', '4'} <= set(re.findall(r'\bclass="sentence tone-([0-4])', source[1]))):
            raise AssertionError("Gradio did not preserve visible reading scores and relative tones")
        print(json.dumps({
            "checkpoint": BASE_REVISION,
            "gradio": gr.__version__,
            "mlx_lm": mlx_lm_version,
            "mlx_device": str(mx.default_device()),
            "sentence_count": len(rows),
            "endpoint_scores": int(scores.size),
            "max_abs_score_difference_from_original": float(np.max(np.abs(scores - expected))),
            "forward_count_api_request": calls,
            "api_roundtrip": "pass",
        }))
    finally:
        demo.close()


if __name__ == "__main__":
    main()
