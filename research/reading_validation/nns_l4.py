"""Encode public sentences once and retain original, prefix-certified word-end signals."""

from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from research.backbone_transfer.extract import forward
from research.reading_validation.local_forecast import WORD, digest, rows
from research.stage5.complex_tokenization import token_span

PACKETS = {
    "b7b72e89e4c0249e44e0202e92166657edbaffbd25e646c0d27126e1aa443a87": 201,
    "fc6affa5c216f1ac8f3fc949699a5bc2dea5fcfd8a46aad054b76e0d738b7d63": 8927,
}
WEIGHTS = Path("research/results/reading-validation-local/serving/weights.npz")
WEIGHT_HASH = "2f72b3b28ec5bc4f47bfe084ae76ccf2f9fdc01e4f4d61bfd4a7a38bda99bfa3"
EXTRA_WIDTH = 64
BATCH = 16


def run(runtime, reader, checkpoint, input_path, output_dir):
    import torch

    start = perf_counter()
    source = Path(input_path)
    sha = digest(source)
    documents = rows(source)
    if (
        PACKETS.get(sha) != len(documents)
        or runtime.candidate != "qwen"
        or runtime.revision != "15852e8c16360a2fea060d615a32b45270f8a8fc"
        or checkpoint["revision"] != runtime.revision
        or digest(WEIGHTS) != WEIGHT_HASH
    ):
        raise ValueError(
            "Unrecognized public stimuli, model revision, or source weights"
        )
    ids, sequences, spans, native = [], [], [], []
    for r in documents:
        t = r["text"]
        if set(r) != {"id", "text"} or sha256(t.encode()).hexdigest() != r["id"]:
            raise ValueError("Only verified source sentences may enter the extractor")
        encoded = runtime.tokenizer.encode(t, add_special_tokens=False)
        if not 0 < len(encoded.ids) <= 1024:
            raise ValueError("Sentence is outside the certified length support")
        positions = [m.span() for m in WORD.finditer(t)]
        ends = []
        for a, b in positions:
            _, hit = token_span(t, encoded.offsets, a, b)
            certified = (
                hit is not None
                and runtime.tokenizer.encode(t[:b], add_special_tokens=False).ids
                == encoded.ids[: hit[1] + 1]
            )
            ends.append(hit[1] if certified else -1)
        sequences.append(encoded.ids)
        spans.append(positions)
        native.append(ends)
        ids.append(r["id"])
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate public sentence")
    offsets = np.r_[0, np.cumsum([len(x) for x in spans])].astype(np.int32)
    word_spans = np.asarray([s for line in spans for s in line], np.int32).reshape(
        -1, 2
    )
    word_native = np.asarray([i for line in native for i in line], np.int32)
    with np.load(WEIGHTS, allow_pickle=False) as f:
        w = torch.as_tensor(f["lexical_weight"], device=runtime.device)
        bias = float(f["lexical_bias"])
        primary = f["projection"]
    extra = np.linalg.qr(
        np.random.default_rng(428371).standard_normal((2048, EXTRA_WIDTH))
    )[0].astype("float32")
    projection = torch.as_tensor(
        np.column_stack((primary, extra)), device=runtime.device
    )
    width = projection.shape[1]
    context = np.zeros((len(word_native), width), np.float32)
    lexical = np.zeros(len(word_native), np.float32)
    order = sorted(range(len(ids)), key=lambda i: (len(sequences[i]), ids[i]))
    calls = 0

    def count(_, __):
        nonlocal calls
        calls += 1

    stored = {}
    check_indices = {order[len(order) // 4], order[len(order) // 2], order[-1]}
    with runtime.encoder.register_forward_pre_hook(count), torch.inference_mode():
        for lo in range(0, len(order), BATCH):
            chosen = order[lo : lo + BATCH]
            matrix = forward(
                runtime.encoder, [sequences[i] for i in chosen], 248044, torch
            )
            selections = [
                (k, offsets[j] + q, i)
                for k, j in enumerate(chosen)
                for q, i in enumerate(native[j])
                if i >= 0
            ]
            if selections:
                batch, loc, ix = np.array(selections).T
                observed = torch.nn.functional.normalize(
                    matrix[batch.tolist(), ix.tolist()].double(), dim=1
                )
                lexical[loc] = (observed @ w + bias).clamp(0, 1).float().cpu().numpy()
                context[loc] = (observed.float() @ projection).cpu().numpy()
            for k, j in enumerate(chosen):
                if j in check_indices:
                    stop = max(1, len(sequences[j]) // 2)
                    stored[j] = matrix[k, stop - 1].detach().clone()
    if (
        calls != (len(ids) + BATCH - 1) // BATCH
        or not np.isfinite(context).all()
        or not np.isfinite(lexical).all()
    ):
        raise ArithmeticError("Single-forward or finite-value contract failed")
    checks = {}
    for j, v in stored.items():
        stop = max(1, len(sequences[j]) // 2)
        p = forward(runtime.encoder, [sequences[j][:stop]], 248044, torch)[0, -1]
        checks[ids[j]] = float((p - v).norm() / p.norm().clamp_min(1e-12))
    if max(checks.values()) > 1e-3:
        raise ArithmeticError("A public prefix differs from the batch forecast")
    dest = Path(output_dir)
    dest.mkdir(exist_ok=True, parents=True)
    (dest / "ids.json").write_text(json.dumps(ids) + "\n")
    np.savez_compressed(
        dest / "signals.npz",
        offsets=offsets,
        word_spans=word_spans,
        native=word_native,
        lexical=lexical,
        plaba=context[:, :32],
        context=context[:, 32:].astype(np.float16),
    )
    receipt = dict(
        revision=runtime.revision,
        packet_sha256=sha,
        source_weight_sha256=WEIGHT_HASH,
        worker_sha256=digest(Path(__file__)),
        sentences=len(ids),
        word_occurrences=len(lexical),
        certified_word_occurrences=int(np.count_nonzero(word_native >= 0)),
        tokens=sum(map(len, sequences)),
        single_pass_batches=calls,
        independent_validation_calls=len(checks),
        prefix_relative_errors=checks,
        files_sha256={k: digest(dest / k) for k in ("ids.json", "signals.npz")},
        device=torch.cuda.get_device_name(),
        elapsed_seconds=perf_counter() - start,
    )
    (dest / "extraction.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
