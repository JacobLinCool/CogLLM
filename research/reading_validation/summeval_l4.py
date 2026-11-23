"""Read both published-coherence heads from the identical single Qwen encoding of each news summary."""

import math
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from research.backbone_transfer.extract import digest, stream_rows, write_json
from research.reading_validation.argument_data import h
from research.reading_validation.argument_l4 import locations
from research.reading_validation.essay_hazard_inference import EssayHazardInference
from research.reading_validation.expectation_native import coordinates
from research.reading_validation.identity_causal_features import (
    encode_batch,
    project_basis,
)
from research.reading_validation.state_transport_l4 import word_values
from research.token_readout.backbone import Features


def common_words(runtime, seq, coords, states, final, mean, peak, spans):
    device = final.device
    ids = torch.tensor(seq, device=device)
    observed = torch.zeros(len(seq), device=device)
    for pos in range(1, len(seq), 64):
        stop = min(pos + 64, len(seq))
        observed[pos:stop] = F.cross_entropy(
            F.linear(final[pos - 1 : stop - 1], runtime.encoder.embed_tokens.weight),
            ids[pos:stop],
            reduction="none",
        )
    c = F.pad(observed.double().cumsum(0), (1, 0))
    lo, hi = torch.as_tensor(coords[:, 2:4].astype(np.int64), device=device).T
    surprisal = (c[hi + 1] - c[lo]).float()
    dims = torch.cat((states[:, :32], states[:, 96:128]), dim=1)
    running = F.pad(dims.cumsum(0), (0, 0, 1, 0))
    word = (running[hi + 1] - running[lo]) / (hi - lo + 1)[:, None]
    rel = torch.stack(
        [
            torch.stack((mean[hi, a:b].sum(1), peak[hi, a:b].amax(1)), dim=1)
            for a, b in spans
        ],
        dim=1,
    )
    ends = torch.tensor([b for _, b in spans], device=device)
    rel = rel.masked_fill((ends[None, :] > lo[:, None])[:, :, None], 0.0)
    return observed, surprisal, word, rel


@torch.inference_mode()
def run(runtime, reader, checkpoint, source, out):
    out.mkdir()
    head = EssayHazardInference.load()
    start = perf_counter()
    items = []
    skipped = {}
    arrays = {}
    docs = {}
    calls = 0
    parity = []
    basis = project_basis(runtime.width, runtime.device)
    for row in stream_rows(source):
        t = row["text"]
        k = row["id"]
        if set(row) != {"id", "text"} or h(t) != k:
            raise ValueError("Expected the complete unmarked displayed summary only")
        enc = runtime.tokenizer.encode(t, add_special_tokens=False)
        loc = locations(t, enc.offsets)
        if len(enc.ids) > 1024:
            skipped[k] = "over_1024_model_tokens"
            continue
        if loc is None:
            skipped[k] = "subword_crosses_sentences"
            continue
        words = coordinates(t, enc)
        if not len(words):
            skipped[k] = "no_printed_word"
            continue
        spans, ts = loc
        items.append(
            (k, t, tuple(enc.ids), tuple(map(tuple, enc.offsets)), spans, ts, words)
        )

    def count(_m, _i):
        nonlocal calls
        calls += 1

    items.sort(key=lambda a: (len(a[2]), a[0]))
    with runtime.encoder.register_forward_pre_hook(count):
        for i in range(0, len(items), 8):
            batch = items[i : i + 8]
            st, middle, final, mean, peak = encode_batch(
                runtime, [list(r[2]) for r in batch], basis
            )
            for j, (k, t, seq, offsets, ss, ts, chars) in enumerate(batch):
                n = len(seq)
                a, b = mean[j, :n, :n], peak[j, :n, :n]
                wf, ws, wc, wr = common_words(
                    runtime, seq, chars, st[j, :n], final[j, :n], a, b, ts
                )
                if not parity:
                    old = word_values(
                        runtime, seq, chars, st[j, :n], final[j, :n], a, b, ts
                    )
                    err = [
                        float(torch.max(abs(v - o)).item())
                        for v, o in zip((ws, wc, wr), old, strict=True)
                    ]
                    parity = err
                    if max(err) > 1e-5:
                        raise ArithmeticError(
                            "The new common native extractor differs from the frozen bilingual source"
                        )
                pos = torch.arange(1, n + 1, device=final.device)
                relations = torch.log1p(
                    torch.stack((a, b), dim=-1) * pos[:, None, None]
                ) - math.log(2)
                relations.masked_fill_(
                    torch.triu(
                        torch.ones(n, n, device=final.device, dtype=torch.bool),
                        diagonal=1,
                    ).unsqueeze(-1),
                    0.0,
                )
                features = Features(
                    seq, offsets, final[j, :n], middle[j, :n], relations
                )
                essay = head.score(
                    t, features, runtime, reader, checkpoint, ss, wf.cpu().numpy()
                )
                keys = ("start", "end", "position", "essay_hazard", "essay_lexical")
                essay = [
                    {
                        name: (int(row[name]) if index < 3 else float(row[name]))
                        for index, name in enumerate(keys)
                    }
                    for row in essay
                ]
                span_set = {tuple(s) for s in ss[1:]}
                if any(
                    (v["start"], v["end"]) not in span_set
                    or not all(
                        0 <= v[n] <= 1 for n in ("essay_hazard", "essay_lexical")
                    )
                    for v in essay
                ):
                    raise ValueError(
                        "Essay outputs do not map to observed noninitial sentence coordinates"
                    )
                docs[k] = {"tokens": n, "sentences": ss, "essay": essay}
                arrays[k + ".char"] = chars
                for name, data in [("surprisal", ws), ("words", wc), ("rel", wr)]:
                    if not torch.isfinite(data).all():
                        raise ArithmeticError("Nonfinite news summary representation")
                    arrays[k + "." + name] = (
                        data.cpu()
                        .numpy()
                        .astype("float32" if name == "surprisal" else "float16")
                    )
            if i and i % 400 == 0:
                print(
                    "Shared native summary encodings", i, "of", len(items), flush=True
                )
    if calls != (len(items) + 7) // 8:
        raise ArithmeticError(
            "Additional native forward inside an existing summary readout"
        )
    np.savez_compressed(out / "features.npz", **arrays)
    write_json(
        out / "receipt.json",
        {
            "source": digest(source),
            "file": digest(out / "features.npz"),
            "articles": docs,
            "skipped": skipped,
            "encoder_batches": calls,
            "single_forward_per_unique_text": True,
            "bilingual_word_parity": parity,
            "revision": runtime.revision,
            "seconds": perf_counter() - start,
            "device": torch.cuda.get_device_name(),
        },
    )
