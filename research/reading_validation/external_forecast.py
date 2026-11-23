"""Extract one causal sentence-end state per unique ReadMe++ sentence on Colab."""

from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from research.backbone_transfer.extract import forward


def run(runtime, reader, checkpoint, input_path, output_dir):
    if runtime.candidate != "qwen" or checkpoint["revision"] != runtime.revision:
        raise ValueError("Expected the pinned Qwen3.5-2B and matching reading checkpoint")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    input_path = Path(input_path)
    rows = [json.loads(line) for line in input_path.read_text().splitlines()]
    ids, sequences = [], []
    for row in rows:
        if set(row) != {"id", "text"} or sha256(row["text"].encode()).hexdigest() != row["id"]:
            raise ValueError("An external stimulus is invalid or contains unexpected fields")
        ids.append(row["id"])
        tokens = runtime.tokenizer.encode(row["text"], add_special_tokens=False).ids
        if not 0 < len(tokens) <= 2048:
            raise ValueError("Unexpected sentence length")
        sequences.append(tokens)
    if len(set(ids)) != len(ids):
        raise ValueError("Expected one forward per unique text")
    order = sorted(range(len(ids)), key=lambda i: (len(sequences[i]), ids[i]))
    states = np.empty((len(ids), runtime.width), dtype=np.float32)
    pad = 248044
    n_forwards = 0
    def counted(_, __):
        nonlocal n_forwards
        n_forwards += 1
    start = perf_counter()
    with runtime.encoder.register_forward_pre_hook(counted):
        for first in range(0, len(order), 16):
            indices = order[first:first + 16]
            seqs = [sequences[i] for i in indices]
            batch = forward(runtime.encoder, seqs, pad, torch)
            states[indices] = batch[torch.arange(len(indices), device=batch.device),
                                     [len(s) - 1 for s in seqs]].cpu().numpy()
    if not np.isfinite(states).all() or n_forwards != (len(order) + 15) // 16:
        raise ValueError("Invalid single-pass extraction")
    checks = {}
    for index in (order[len(order)//4], order[len(order)//2], order[-1]):
        seq = sequences[index]
        one = forward(runtime.encoder, [seq], pad, torch)[0]
        if len(seq) < 2:
            raise ValueError("Validation requires a multi-token sentence")
        stop = len(seq) // 2
        prefix = forward(runtime.encoder, [seq[:stop]], pad, torch)[0, -1]
        terminal = torch.from_numpy(states[index]).to(one.device)
        relative = lambda a,b: float((a-b).norm()/b.norm().clamp_min(1e-12))
        checks[ids[index]] = dict(batched_terminal=relative(terminal,one[-1]),
                                  shorter_prefix=relative(prefix,one[stop-1]))
    if max(v for check in checks.values() for v in check.values()) > 1e-3:
        raise ArithmeticError(f"Backbone equivalence check failed: {checks}")
    np.save(output / "states.npy", states, allow_pickle=False)
    (output / "ids.json").write_text(json.dumps(ids) + "\n")
    files = {name:sha256((output/name).read_bytes()).hexdigest() for name in ("states.npy", "ids.json")}
    receipt = dict(revision=runtime.revision, packet_sha256=sha256(input_path.read_bytes()).hexdigest(),
                   n_texts=len(ids), n_tokens=sum(map(len,sequences)), inference_batches=n_forwards,
                   validation_forwards=6, checks=checks, elapsed_seconds=perf_counter()-start,
                   device=torch.cuda.get_device_name(), files_sha256=files)
    (output / "extraction.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
