"""Read the certified replacement word once per public substituted sentence on L4."""

import json
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np

from research.backbone_transfer.extract import forward
from research.reading_validation.local_forecast import WORD, digest, rows
from research.reading_validation.nns_l4 import BATCH, EXTRA_WIDTH, WEIGHT_HASH, WEIGHTS
from research.stage5.complex_tokenization import token_span

PACKET = "9350ac9cdc944c2a4eac9b2c8a47f3f7abed8bc75d9c6f2678bebcd69b9c0974"
SIZE = 3155


def run(runtime, reader, checkpoint, input_path, output_dir):
    import torch

    start = perf_counter()
    packet = Path(input_path)
    records = rows(packet)
    if (
        digest(packet) != PACKET
        or len(records) != SIZE
        or runtime.candidate != "qwen"
        or runtime.revision != "15852e8c16360a2fea060d615a32b45270f8a8fc"
        or checkpoint["revision"] != runtime.revision
        or digest(WEIGHTS) != WEIGHT_HASH
    ):
        raise ValueError(
            "Unrecognized public substitution packet, backbone or source weights"
        )
    ids = []
    sequences = []
    native = []
    windows = []
    focus_ids = []
    indices = []
    for r in records:
        t = r["text"]
        doc = r["id"]
        if set(r) != {"id", "text", "spans"} or sha256(t.encode()).hexdigest() != doc:
            raise ValueError("Replacement packet includes an unknown original sentence")
        ordered = [m.span() for m in WORD.finditer(t)]
        printed = {span: i for i, span in enumerate(ordered)}
        enc = runtime.tokenizer.encode(t, add_special_tokens=False)
        if not 0 < len(enc.ids) <= 1024:
            raise ValueError("Sentence outside certified range")
        seq = []
        for a, b in r["spans"]:
            if (a, b) not in printed:
                raise ValueError("Unaligned target word")
            _, hit = token_span(t, enc.offsets, a, b)
            index = -1
            if (
                hit is not None
                and runtime.tokenizer.encode(t[:b], add_special_tokens=False).ids
                == enc.ids[: hit[1] + 1]
            ):
                index = hit[1]
            focus_ids.append(sha256(f"{doc}/{a}/{b}".encode()).hexdigest())
            seq.append(len(native))
            native.append(index)
            left = hit[0] if index >= 0 and hit[0] > 0 else -1
            third = ordered[min(len(ordered) - 1, printed[(a, b)] + 3)]
            _, next_hit = token_span(t, enc.offsets, *third)
            right = -1
            if (
                left >= 0
                and next_hit is not None
                and runtime.tokenizer.encode(
                    t[: third[1]], add_special_tokens=False
                ).ids
                == enc.ids[: next_hit[1] + 1]
            ):
                right = next_hit[1]
            windows.append((left, right))
        ids.append(doc)
        sequences.append(enc.ids)
        indices.append(seq)
    if len(set(ids)) != len(ids) or len(set(focus_ids)) != len(focus_ids):
        raise ValueError("Duplicated full text or word index")
    with np.load(WEIGHTS, allow_pickle=False) as f:
        w = torch.as_tensor(f["lexical_weight"], device=runtime.device)
        bias = float(f["lexical_bias"])
        raw = f["projection"]
    extra = np.linalg.qr(
        np.random.default_rng(428371).standard_normal((2048, EXTRA_WIDTH))
    )[0].astype("float32")
    proj = torch.as_tensor(np.column_stack([raw, extra]), device=runtime.device)
    features = np.zeros((len(focus_ids), proj.shape[1]), np.float32)
    lex = np.zeros(len(focus_ids), np.float32)
    surprisal = np.zeros((len(focus_ids), 2), np.float32)
    measured = np.zeros((len(focus_ids), 2), bool)
    order = sorted(range(len(ids)), key=lambda i: (len(sequences[i]), ids[i]))
    stored = {}
    checks = {order[len(order) // 4], order[len(order) // 2], order[-1]}
    calls = 0

    def count(_module, _args):
        nonlocal calls
        calls += 1

    with torch.inference_mode(), runtime.encoder.register_forward_pre_hook(count):
        for i in range(0, len(order), BATCH):
            batch = order[i : i + BATCH]
            h = forward(runtime.encoder, [sequences[j] for j in batch], 248044, torch)
            positions = [
                (pos, k, native[pos])
                for k, j in enumerate(batch)
                for pos in indices[j]
                if native[pos] >= 0
            ]
            if positions:
                j, k, loc = np.array(positions).T
                normalized = torch.nn.functional.normalize(
                    h[k.tolist(), loc.tolist()].double(), dim=1
                )
                features[j] = (normalized.float() @ proj).cpu().numpy()
                lex[j] = (normalized @ w + bias).clamp(0, 1).float().cpu().numpy()
            next_tokens = []
            for k, doc_index in enumerate(batch):
                for pos in indices[doc_index]:
                    left, right = windows[pos]
                    if left < 0:
                        continue
                    stop = native[pos]
                    measured[pos] = (True, right >= stop)
                    for token in range(left, max(stop, right) + 1):
                        next_tokens.append(
                            (pos, k, token, sequences[doc_index][token], token <= stop)
                        )
            for lo in range(0, len(next_tokens), 16):
                block = next_tokens[lo : lo + 16]
                positions_, members, tokens, targets, inside = np.asarray(block).T
                logits = torch.nn.functional.linear(
                    h[members.tolist(), (tokens - 1).tolist()],
                    runtime.encoder.embed_tokens.weight,
                )
                target_ids = torch.as_tensor(
                    targets, dtype=torch.long, device=runtime.device
                )
                losses = (
                    torch.nn.functional.cross_entropy(
                        logits, target_ids, reduction="none"
                    )
                    .cpu()
                    .numpy()
                )
                np.add.at(
                    surprisal[:, 0],
                    positions_[inside.astype(bool)],
                    losses[inside.astype(bool)],
                )
                np.add.at(surprisal[:, 1], positions_, losses)
            for k, j in enumerate(batch):
                if j in checks:
                    valid = [native[pos] for pos in indices[j] if native[pos] >= 0]
                    if valid:
                        stored[j] = (valid[0], h[k, valid[0]].detach().clone())
    if (
        calls != (len(ids) + BATCH - 1) // BATCH
        or not np.isfinite(features).all()
        or not np.isfinite(lex).all()
        or not np.isfinite(surprisal).all()
    ):
        raise ArithmeticError("Single forward or finite-value check failed")
    errors = {}
    for j, (pos, h) in stored.items():
        prefix = forward(runtime.encoder, [sequences[j][: pos + 1]], 248044, torch)[
            0, -1
        ]
        errors[ids[j]] = float((h - prefix).norm() / prefix.norm().clamp_min(1e-12))
    if not errors or max(errors.values()) > 1e-3:
        raise ArithmeticError("Prefix check failed")
    dest = Path(output_dir)
    dest.mkdir(exist_ok=True, parents=True)
    (dest / "ids.json").write_text(json.dumps(focus_ids) + "\n")
    np.savez_compressed(
        dest / "signals.npz",
        native=np.asarray(native, np.int32),
        lexical=lex,
        plaba=features[:, :32],
        context=features[:, 32:].astype(np.float16),
        surprisal=surprisal,
        measured_surprisal=measured,
    )
    receipt = {
        "packet_sha256": PACKET,
        "worker_sha256": digest(Path(__file__)),
        "source_weight_sha256": WEIGHT_HASH,
        "revision": runtime.revision,
        "sentences": len(ids),
        "words": len(focus_ids),
        "certified_words": sum(i >= 0 for i in native),
        "tokens": sum(map(len, sequences)),
        "single_pass_batches": calls,
        "prefix_relative_errors": errors,
        "files_sha256": {q: digest(dest / q) for q in ("signals.npz", "ids.json")},
        "elapsed_seconds": perf_counter() - start,
        "device": torch.cuda.get_device_name(),
    }
    (dest / "extraction.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
