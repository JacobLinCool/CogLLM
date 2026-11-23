"""Read native weights for intervening words without giving any probe or condition to Qwen."""

import json
import re
from hashlib import sha256
from math import log
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from research.reading_validation.transition_inference import REVISION

FORM = re.compile(r"[^\W\d_]+(?:[-’'][^\W\d_]+)*", re.UNICODE)


@torch.inference_mode()
def run(runtime, reader, checkpoint, input_path, output_dir):
    src = Path(input_path)
    dest = Path(output_dir)
    dest.mkdir(exist_ok=True, parents=True)
    rows = [json.loads(x) for x in src.read_text().splitlines()]
    if (
        runtime.candidate != "qwen"
        or runtime.revision != REVISION
        or len({r["id"] for r in rows}) != len(rows)
    ):
        raise ValueError("Unregistered encoder or duplicate article")
    n = [0]
    t = perf_counter()

    def observe(*args):
        n[0] += 1

    file = dest / "cue-strengths.jsonl"
    with file.open("w") as stream, runtime.encoder.register_forward_pre_hook(observe):
        for i, r in enumerate(rows):
            text = r["text"]
            features = runtime.encode(text)
            if (
                n[0] != i + 1
                or sha256(text.encode()).hexdigest() != r["id"]
                or set(r) != {"id", "text", "prior_end", "last_start", "last_end"}
            ):
                raise ValueError("Article forwarding or outcome separation failed")
            off = np.asarray(features.offsets)
            matches = list(FORM.finditer(text))
            words = []
            for j, m in enumerate(matches):
                parts = np.flatnonzero((off[:, 0] < m.end()) & (off[:, 1] > m.start()))
                if not len(parts):
                    raise ValueError("Original term could not be aligned")
                words.append((m.start(), m.end(), parts, j))
            query = [
                w for w in words if w[0] >= r["last_start"] and w[1] <= r["last_end"]
            ]
            cues = [
                w for w in words if w[0] >= r["prior_end"] and w[1] <= r["last_start"]
            ]
            if not query or not cues:
                raise ValueError("Missing region")
            q = np.array([w[2][-1] for w in query])
            ids = torch.tensor(q, device=runtime.device)
            selected = features.relations[ids, :, 0]
            att = (
                ((selected + log(2)).expm1() / (ids[:, None] + 1))
                .double()
                .cpu()
                .numpy()
            )
            measured = []
            for a, b, parts, j in cues:
                native = float(att[:, parts].mean())
                recency = float(np.mean([1 / (1 + w[3] - j) for w in query]))
                if not np.isfinite(native) or native < 0:
                    raise ValueError("Invalid directional retrieval mass")
                measured.append([a, b, native, recency])
            stream.write(json.dumps({"id": r["id"], "cues": measured}) + "\n")
    (dest / "receipt.json").write_text(
        json.dumps(
            {
                "source_sha256": sha256(src.read_bytes()).hexdigest(),
                "forecast_sha256": sha256(file.read_bytes()).hexdigest(),
                "forwards": n[0],
                "seconds": perf_counter() - t,
                "revision": runtime.revision,
                "provided_targets": 0,
            },
            indent=2,
        )
        + "\n"
    )
