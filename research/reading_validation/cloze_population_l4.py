"""Recover the pre-word expectation distribution from one causal forward of each article."""
from argparse import ArgumentParser
from pathlib import Path
import json
from time import perf_counter
import numpy as np
import torch
from torch.nn import functional as F
from research.backbone_transfer.extract import digest, write_json, stream_rows
from research.reading_validation.identity_run_l4 import run_phases
from research.reading_validation import nominal_external_l4
from research.reading_validation import (
    state_transport_l4,
    summeval_l4,
    writer_balanced_l4,
)
from research.reading_validation.transition_architecture_l4 import (
    native_logits,
    verify_logits,
)


def dictionary(runtime, raw, extra):
    weight = runtime.encoder.embed_tokens.weight
    token_lists = [r["tokens"] for r in raw]
    freq = np.array([r["frequency"] for r in raw])
    first = np.array([t[0] for t in token_lists])
    unique, route = np.unique(first, return_inverse=True)
    share = np.log(freq / np.bincount(route, weights=freq)[route])
    token_lists.extend(
        runtime.tokenizer.encode(" " + w, add_special_tokens=False).ids for w in extra
    )
    result = []
    for start in range(0, len(token_lists), 256):
        group = token_lists[start : start + 256]
        width = max(map(len, group))
        ids = np.zeros((len(group), width), int)
        mask = np.zeros_like(ids, bool)
        for i, g in enumerate(group):
            ids[i, : len(g)] = g
            mask[i, : len(g)] = True
        m = torch.as_tensor(mask, device=runtime.device)
        e = weight[torch.as_tensor(ids, device=runtime.device)]
        result.append(
            F.normalize((e * m[:, :, None]).sum(1) / m.sum(1)[:, None], dim=1)
        )
    matrix = torch.cat(result).float()
    matrix -= matrix[: len(raw)].mean(0)
    torch.manual_seed(725)
    _, _, V = torch.pca_lowrank(matrix[: len(raw)], q=32, center=False, niter=5)
    axis = matrix @ V[:, :16]
    axis /= axis[: len(raw)].std(0, unbiased=False).clamp(min=1e-7)
    return (
        unique,
        route,
        share,
        axis[: len(raw)].cpu().numpy(),
        axis[len(raw) :].cpu().numpy(),
    )


@torch.inference_mode()
def collect(runtime, reader, checkpoint, source, output):
    output.mkdir()
    start = perf_counter()
    lexpath = Path("dictionary.json")
    raw = json.loads(lexpath.read_bytes())
    rows = list(stream_rows(source))
    extra = sorted(
        {
            row["text"][a:b].strip('.,?!:;()[]"').strip().casefold().replace("’", "'")
            for row in rows
            for a, b, u, v in row["ends"]
        }
        - {""}
    )
    unique, route, share, words, observed = dictionary(runtime, raw, extra)
    ix = torch.as_tensor(unique, device=runtime.device)
    other = torch.ones(
        runtime.encoder.embed_tokens.weight.shape[0],
        dtype=torch.bool,
        device=runtime.device,
    )
    other[ix] = False
    rng = np.random.default_rng(725)
    R = [
        torch.as_tensor(
            rng.choice([-1.0, 1.0], (runtime.width, 32)).astype("float32")
            / np.sqrt(np.float32(runtime.width)),
            device=runtime.device,
        )
        for _ in range(2)
    ]
    official = verify_logits(runtime)
    counts = 0
    stored = 0
    states = {}
    logp = {}
    checks = []

    def hook(*args):
        nonlocal counts
        counts += 1

    def encode(row, n):
        full = row["text"][: row["ends"][n - 1][1]]
        f = runtime.encode(full)
        bound = np.asarray(row["ends"][:n])
        last = torch.as_tensor(bound[1:, 2] - 1, device=runtime.device)
        if len(f.token_ids) != bound[-1, 3] + 1 or any(
            f.offsets[v][1] != b for a, b, u, v in bound
        ):
            raise ValueError("Mismatch between measured printed word and native prefix")
        z = torch.cat(
            [
                F.layer_norm(x[last], (runtime.width,)) @ m
                for x, m in zip([f.mid, f.hidden], R)
            ],
            dim=1,
        )
        parts = []
        for i in range(0, len(last), 16):
            score = native_logits(runtime, f.hidden[last[i : i + 16]]).float()
            den = torch.logsumexp(score, 1)
            outside = torch.logsumexp(score[:, other], 1)
            p = torch.cat([score[:, ix], outside[:, None]], dim=1) - den[:, None]
            if abs(torch.logsumexp(p, 1)).max().item() > 0.0001:
                raise ValueError("Word-category mapping loses native probability mass")
            parts.append(p)
        return z.cpu().numpy(), torch.cat(parts).cpu().numpy()

    with runtime.encoder.register_forward_pre_hook(hook):
        for j, row in enumerate(rows):
            k = row["id"]
            n = len(row["ends"])
            z, p = encode(row, n)
            states[k] = z.astype("float16")
            logp[k] = p.astype("float16")
            stored += len(p)
            if j in (0, 155, 235):
                target = max(2, n // 2)
                zp, pp = encode(row, target)
                err = [
                    float(abs(zp - z[: target - 1]).max()),
                    float(abs(pp - p[: target - 1]).max()),
                ]
                if max(err) > 0.002:
                    raise ValueError(
                        f"Future text influenced a preceding-state distribution: {err}"
                    )
                checks.append(err)
    if counts != len(rows) + len(checks):
        raise ValueError("More than one study forward per original article")
    outputs = {
        "log-probabilities.npz": logp,
        "prefix-states.npz": states,
        "lexicon.npz": dict(
            unique=unique,
            route=route,
            share=share,
            coordinates=words,
            observed_forms=np.asarray(extra),
            observed_vectors=observed,
        ),
    }
    for name, contents in outputs.items():
        np.savez_compressed(output / name, **contents)
    write_json(
        output / "receipt.json",
        dict(
            source=digest(source),
            dictionary=digest(lexpath),
            reference_model=official,
            words=stored,
            documents=len(rows),
            production_forwards=len(rows),
            technical_prefix_tests=checks,
            files={n: digest(output / n) for n in outputs},
            seconds=perf_counter() - start,
        ),
    )


def run(model, candidate, stimuli, output, protocol_sha256):
    phases = [
        ("balanced-writer", writer_balanced_l4.run, Path("balanced-cases.json")),
        ("nominal-external", nominal_external_l4.run, Path("unseen-nominal.jsonl")),
        ("cloze-population", collect, stimuli),
        ("summeval-essay-parity", summeval_l4.run, Path("essay-parity.jsonl")),
        ("summeval-coherence", summeval_l4.run, Path("summeval.jsonl")),
        ("columbia-order", state_transport_l4.run, Path("columbia-order.jsonl")),
    ]
    run_phases(model, candidate, output, protocol_sha256, phases)


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    for k in ("model", "stimuli", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("candidate", "protocol-sha256"):
        p.add_argument("--" + k, required=True)
    run(**vars(p.parse_args()))
