"""Verify the optional shipped learner view on the actual L4 using locked public texts."""

from html.parser import HTMLParser
import json
from pathlib import Path
from hashlib import sha256
from importlib.metadata import version
from time import perf_counter

from research.reading_validation.local_forecast import CHECKPOINT, digest, rows
from research.reading_validation.local_inference import LocalInference, REVISION
from research.token_readout.learner_view import (
    EXAMPLE,
    LEVELS,
    POOLED,
    STYLE,
    display,
    validate_sentence,
)

PACKET = "a6f4dcaa41c5161e661c3273f954275ab9181006b6de6581390a4be31baf4298"


class SentenceText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.parts = []
        self.scores = []

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if self.depth:
            self.depth += 1
            if tag == "span" and "learner-word" in d.get("class", "").split():
                self.scores.append(
                    (int(d["data-start"]), int(d["data-end"]), d["data-rate"])
                )
        elif tag == "div" and d.get("class") == "learner-draft":
            self.depth = 1

    def handle_endtag(self, tag):
        if self.depth:
            self.depth -= 1

    def handle_data(self, data):
        if self.depth:
            self.parts.append(data)


def run(runtime, reader, checkpoint, input_path, output_dir):
    from research.token_readout.runner import TokenRunner
    import torch

    start = perf_counter()
    packet = Path(input_path)
    examples = rows(packet)
    if (
        digest(packet) != PACKET
        or len(examples) != 4
        or examples[0]["text"] != EXAMPLE
        or runtime.revision != REVISION
        or runtime.candidate != "qwen"
        or digest(Path("curve.pt")) != CHECKPOINT
    ):
        raise ValueError(
            "Learner verification input differs from the sealed example and published sentence texts"
        )
    runner = TokenRunner(
        runtime,
        reader,
        None,
        checkpoint["priors"],
        CHECKPOINT,
        english_head=LocalInference.load(),
    )
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    count = 0

    def hook(_module, _args):
        nonlocal count
        count += 1

    responses = {}
    checks = []
    with runtime.encoder.register_forward_pre_hook(hook):
        for i, r in enumerate(examples):
            t = r["text"]
            ident = r["id"]
            if sha256(t.encode()).hexdigest() != ident or set(r) != {"id", "text"}:
                raise ValueError("This gate accepts original text only.")
            validate_sentence(t)
            data = runner.learner_sentence(t)
            if (
                data["text"] != t
                or runner._word_report.model is not runner.english_head
            ):
                raise AssertionError(
                    "Writer did not reuse the loaded English readout or lost its sentence"
                )
            for level in (POOLED, *LEVELS):
                html = display(data, level)
                p = SentenceText()
                p.feed(html)
                p.close()
                if (
                    p.depth
                    or "".join(p.parts) != t
                    or [list(z[:2]) for z in p.scores] != data["positions"]
                    or count != i + 1
                    or runner.forward_count() != count
                ):
                    raise ValueError(
                        "A serving action changed the original sentence, its words, or the encoder count"
                    )
                if i == 0 and level == POOLED:
                    (dest / "learner.html").write_text(
                        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
                        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>CogLLM learner vocabulary</title>'
                        "<style>body{max-width:54rem;margin:6vh auto;padding:0 1rem;font:16px/1.55 system-ui;color:#24372e}"
                        + STYLE
                        + "</style></head><body><main><h1>Which words might an English learner need help with?</h1>"
                        + html
                        + "</main></body></html>"
                    )
            checks.append(
                {
                    "id": ident,
                    "words_certified": len(data["positions"]),
                    "conditions": 7,
                    "one_encoder_forward": True,
                }
            )
            responses[ident] = data
        for invalid in ["", "甲乙丙丁。", "First sentence. Another sentence."]:
            try:
                runner.learner_sentence(invalid)
            except ValueError:
                continue
            raise ValueError("An out-of-scope input incorrectly ran the learner model")
    if count != len(examples):
        raise ValueError("Invalid or cached proficiency display made a model call")
    (dest / "responses.json").write_text(
        json.dumps(responses, ensure_ascii=False) + "\n"
    )
    receipt = {
        "worker_sha256": digest(Path(__file__)),
        "packet_sha256": PACKET,
        "model_revision": REVISION,
        "checkpoint": CHECKPOINT,
        "dependency_versions": {
            s: version(s) for s in ("wordfreq", "nltk", "numpy", "scipy", "torch")
        },
        "runtime_device": torch.cuda.get_device_name(),
        "encoder_forwards": count,
        "cohort_displays": count * 7,
        "invalid_inputs_checked": 3,
        "checks": checks,
        "elapsed_seconds": perf_counter() - start,
        "files": {k: digest(dest / k) for k in ("responses.json", "learner.html")},
    }
    (dest / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def compare(source):
    import numpy as np

    out = Path("research/results/reading-validation-nns/writer")
    out.mkdir(parents=True, exist_ok=True)
    target_file = Path(".cache/cogllm/reading-validation/nns-serving/reference.json")
    if (
        digest(target_file)
        != "538574ebfb75781b1be7c0c7fdb20d5fd69596810a930e5d4e3c2b877edea176"
    ):
        raise ValueError("The source word forecasts changed")
    refs = json.loads(target_file.read_bytes())
    note = json.loads((source / "verification.json").read_bytes())
    if (
        note["worker_sha256"] != digest(Path(__file__))
        or note["packet_sha256"] != PACKET
        or any(digest(source / k) != v for k, v in note["files"].items())
    ):
        raise ValueError("Invalid L4 learner app capture")
    responses = json.loads((source / "responses.json").read_bytes())
    worst = 0
    checked = 0
    for k, u in refs.items():
        x = responses[k]
        if x["positions"] != u["positions"]:
            raise ValueError("Historical and app word ends differ")
        pred = np.column_stack([x["probabilities"][l] for l in (POOLED, *LEVELS)])
        worst = max(worst, float(np.max(abs(pred - np.asarray(u["rates"])))))
        checked += pred.size
    if worst > 1e-4 or checked != 462:
        raise ValueError(
            f"Production learner readout diverged from sealed study at {worst}"
        )
    for f in ("learner.html", "verification.json"):
        (out / f).write_bytes((source / f).read_bytes())
    note = {
        "predefined_tolerance": 0.0001,
        "max_absolute_probability_error": worst,
        "native_held_probabilities": checked,
        "source_predictions_sha256": digest(target_file),
        "source_verification_sha256": digest(source / "verification.json"),
        "using_human_answers": False,
    }
    (out / "numerical-parity.json").write_text(json.dumps(note, indent=2) + "\n")
    print(note)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", type=Path, required=True)
    compare(parser.parse_args().compare)
