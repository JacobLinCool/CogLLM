"""Extract frozen source readouts on the exact original English reading displays."""

import json
import re
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np

from research.reading_validation.local_forecast import CHECKPOINT, WORD, digest, rows
from research.reading_validation.local_inference import (
    REVISION,
    LocalInference,
    english_sentences,
)
from research.reading_validation.transition_external_forecast import token_surprisals
from research.stage5.complex_tokenization import token_span
from research.token_readout.runner import trace_rows

PACKET = "ce982ffd7f176652f5c201d7387974e97b475ce665fbd24eff2c094d55cc2a04"
COLS = (
    "start",
    "end",
    "native_start",
    "native_end",
    "stock",
    "positive_increase",
    "opening",
    "resolution",
    "transition",
    "word_surprisal",
    "complex",
    "complex_valid",
)


def run(runtime, reader, checkpoint, input_path, output_dir):
    import torch

    start = perf_counter()
    packet = Path(input_path)
    data = rows(packet)
    local = LocalInference.load()
    scale = checkpoint["scale"]
    if (
        digest(packet) != PACKET
        or len(data) != 518
        or runtime.revision != REVISION
        or digest(Path("curve.pt")) != CHECKPOINT
    ):
        raise ValueError(
            "Unrecognized EMTeC stimulus packet or language/readout checkpoint"
        )
    ids = []
    info = []
    chunks = []
    words = []
    calls = 0

    def track(_m, _a):
        nonlocal calls
        calls += 1

    lexical_weight = torch.as_tensor(local.lexical_weight, device=runtime.device)
    with torch.inference_mode(), runtime.encoder.register_forward_pre_hook(track):
        for row in data:
            text = row["text"]
            begin = row["body_start"]
            body = text[begin:]
            key = row["id"]
            if (
                set(row) != {"id", "text", "body_start"}
                or sha256(text.encode()).hexdigest() != key
                or text[begin - 2 : begin] != "\n\n"
                or not body.strip()
            ):
                raise ValueError(
                    "Only the validated introductory screen and body are allowed"
                )
            f = runtime.encode(text)
            s = reader(f)
            s = replace(
                s,
                opening=s.opening * scale["opening_gain"],
                resolution=s.resolution * scale["resolution_gain"]
                + scale["resolution_bias"],
            )
            trace = trace_rows(
                text, f.offsets, s, checkpoint["priors"], reader.anchor_tokens
            )
            nll = token_surprisals(runtime, f)
            spans = [(a + begin, b + begin) for a, b in english_sentences(body)]
            ratings = local.score(
                text, f, runtime.tokenizer, reader, sentence_spans=spans
            )["sentences"]
            valid = [t["rating"]["ease"] for t in ratings if t["rating"] is not None]
            exact = [i for i, (a, b) in enumerate(f.offsets) if b <= begin - 2]
            previous = trace[exact[-1]]["unresolved"] if exact else 0.0
            term_score = (
                (
                    torch.nn.functional.normalize(f.hidden.double(), dim=1)
                    @ lexical_weight
                    + local.lexical_bias
                )
                .clamp(0.0, 1.0)
                .cpu()
                .numpy()
            )
            numeric = []
            for m in re.finditer(r"\S+", body):
                a, b = begin + m.start(), begin + m.end()
                _, hit = token_span(text, f.offsets, a, b)
                if hit is None or runtime.tokenizer.encode(
                    text[:b], add_special_tokens=False
                ).ids != list(f.token_ids[: hit[1] + 1]):
                    raise ValueError(
                        "A prespecified original word-end failed certification"
                    )
                start_token, end_token = hit
                h = trace[end_token]
                stock = h["unresolved"]
                found = list(WORD.finditer(text, a, b))
                cvalid = 0
                cscore = 0.0
                if found:
                    lo, hi = found[-1].span()
                    _, term = token_span(text, f.offsets, lo, hi)
                    if term and runtime.tokenizer.encode(
                        text[:hi], add_special_tokens=False
                    ).ids == list(f.token_ids[: term[1] + 1]):
                        cscore = float(term_score[term[1]])
                        cvalid = 1
                numeric.append(
                    (
                        m.start(),
                        m.end(),
                        start_token,
                        end_token,
                        stock,
                        max(0, stock - previous),
                        h["opened"],
                        h["resolved"],
                        h["transition"],
                        float(nll[start_token : end_token + 1].sum()),
                        cscore,
                        cvalid,
                    )
                )
                previous = stock
            chunk = np.asarray(numeric, np.float64)
            if not np.isfinite(chunk).all():
                raise ArithmeticError("Invalid display word scores")
            ids.append(key)
            chunks.append(chunk)
            words.append(len(numeric))
            info.append(
                {
                    "id": key,
                    "tokens": len(f.offsets),
                    "words": len(numeric),
                    "body_sentences": len(spans),
                    "rated_sentences": len(valid),
                    "mean_source_ease": float(np.mean(valid)) if valid else None,
                }
            )
    if calls != len(data) or sum(words) != 44705 or len(set(ids)) != len(ids):
        raise ArithmeticError(
            "Single forward or complete original word coverage failed"
        )
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dest / "words.npz",
        matrix=np.concatenate(chunks),
        cuts=np.r_[0, np.cumsum(words)].astype(np.int32),
    )
    (dest / "summary.json").write_text(json.dumps(info) + "\n")
    receipt = {
        "packet_sha256": PACKET,
        "worker_sha256": digest(Path(__file__)),
        "revision": runtime.revision,
        "readout_sha256": CHECKPOINT,
        "local_sha256": local.bundle_sha256,
        "forwards": calls,
        "word_ends": sum(words),
        "columns": COLS,
        "tokens": sum(r["tokens"] for r in info),
        "device": torch.cuda.get_device_name(),
        "files_sha256": {k: digest(dest / k) for k in ("words.npz", "summary.json")},
        "elapsed_seconds": perf_counter() - start,
    }
    (dest / "extraction.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
