"""Certify independently registered author-facing outputs using one allocated Qwen runtime."""
import gc
import json
import traceback
import torch
from argparse import ArgumentParser
from pathlib import Path
from research.backbone_transfer.extract import digest, write_json
from research.reading_validation import (
    referent_sketch_l4,
    state_writer_l4,
    transition_mandarin_prior_l4,
    transition_snac_forecast,
)
from research.reading_validation.identity_run_l4 import run_phases


def run(model, candidate, stimuli, output, protocol_sha256):
    phases = [
        ("physical-writer", state_writer_l4.run, stimuli),
        ("last-movie", transition_snac_forecast.run, Path("last-movie.jsonl")),
        (
            "mandarin-lexical-prior",
            transition_mandarin_prior_l4.run,
            Path("mandarin-lexical-prior.jsonl"),
        ),
    ]
    run_phases(model, candidate, output, protocol_sha256, phases)
    original = json.loads((output / "causality.json").read_text())
    (output / "seal.json").rename(output / "writer-seal.json")
    (output / "causality.json").rename(output / "writer-status.json")
    gc.collect()
    torch.cuda.empty_cache()
    try:
        referent_sketch_l4.run(
            model,
            Path("referent-sketch.jsonl"),
            output / "referent-sketch-forecast",
            "b12b3dc1ee66b2fa25eadfbdefc3a690405378b3c4ace34abf969c9e4adc54ca",
            "cf20f50dd8da30deabd0fe2aec8d02bfef1ed563b4c1afbc24bd08003040541a",
            Path("referent-seal.json"),
        )
        original["studies"]["referent-sketch"] = {"completed": True}
    except Exception as exc:
        traceback.print_exc()
        original["studies"]["referent-sketch"] = {
            "completed": False,
            "exception": type(exc).__name__,
            "message": str(exc),
        }
    write_json(output / "causality.json", original)
    write_json(
        output / "seal.json",
        dict(
            protocol_sha256=protocol_sha256,
            all_completed=all(v["completed"] for v in original["studies"].values()),
            files_sha256={
                str(f.relative_to(output)): digest(f)
                for f in sorted(output.rglob("*"))
                if f.is_file()
            },
        ),
    )


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    for key in ("model", "stimuli", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("candidate", "protocol-sha256"):
        p.add_argument("--" + key, required=True)
    run(**vars(p.parse_args()))
