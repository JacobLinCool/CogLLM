"""Extract contextual directions at the same last reading token as the frozen native forecast."""

from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from research.backbone_transfer.extract import digest, forward, stream_rows, write_json
from research.reading_validation.population_inference import PopulationWordInference
from research.reading_validation.transition_architecture_l4 import native_logits


@torch.inference_mode()
def run(runtime, reader, checkpoint, input_path, output_dir):
    start = perf_counter()
    out = Path(output_dir)
    out.mkdir(exist_ok=True, parents=True)
    population = PopulationWordInference.load()
    if runtime.revision != population.revision or runtime.candidate != "qwen":
        raise ValueError("Unregistered native encoder")
    source = Path(input_path)
    records = list(stream_rows(source))
    if len(records) != 3085 or any(set(r) != {"id", "text"} for r in records):
        raise ValueError("Only the original unlabeled reading stems are allowed")
    seq = [
        runtime.tokenizer.encode(r["text"], add_special_tokens=False).ids
        for r in records
    ]
    padding = runtime.tokenizer.token_to_id("<|endoftext|>")
    if padding is None:
        raise ValueError("Registered padding token unavailable")
    words = torch.as_tensor(population.weights["unique"], device=runtime.device)
    other = torch.ones(
        len(runtime.encoder.embed_tokens.weight),
        dtype=torch.bool,
        device=runtime.device,
    )
    other[words] = False
    projection = [
        torch.as_tensor(population.weights[k], device=runtime.device)
        for k in ("mid_proj", "last_proj")
    ]
    ordered = sorted(range(len(records)), key=lambda i: len(seq[i]))
    arr = np.empty((len(seq), len(words) + 1), dtype=np.float16)
    state = np.empty((len(seq), sum(v.shape[1] for v in projection)), dtype=np.float16)
    n = [0]
    middle = []

    def count(*_):
        n[0] += 1

    def midway(module, args, output):
        values = output[0] if isinstance(output, tuple) else output
        middle.append(values)

    with (
        runtime.encoder.register_forward_pre_hook(count),
        runtime.encoder.layers[runtime.mid_index].register_forward_hook(midway),
    ):
        for start_ix in range(0, len(ordered), 32):
            idx = ordered[start_ix : start_ix + 32]
            batch = [seq[k] for k in idx]
            ends = [len(s) - 1 for s in batch]
            middle.clear()
            last = forward(runtime.encoder, batch, padding, torch)
            if len(middle) != 1:
                raise ValueError("Intermediate state was not recorded once")
            ix = torch.arange(len(batch), device=runtime.device)
            hs = [h[ix, ends] for h in (middle[0], last)]
            state[idx] = (
                torch.cat(
                    [
                        F.layer_norm(h, (runtime.width,)) @ p
                        for h, p in zip(hs, projection, strict=True)
                    ],
                    1,
                )
                .half()
                .cpu()
                .numpy()
            )
            logits = native_logits(runtime, hs[-1])
            lp = (
                torch.cat(
                    [logits[:, words], torch.logsumexp(logits[:, other], 1)[:, None]], 1
                )
                - torch.logsumexp(logits, 1)[:, None]
            )
            arr[idx] = lp.half().cpu().numpy()
    if n[0] != (len(seq) + 31) // 32 or not np.isfinite(state).all():
        raise ArithmeticError("Unexpected encoder calls or nonfinite context")
    f = out / "states.npz"
    np.savez_compressed(f, ids=np.array([r["id"] for r in records]), state=state)
    write_json(
        out / "receipt.json",
        {
            "source": digest(source),
            "states": digest(f),
            "native_array_sha256": sha256(arr.tobytes()).hexdigest(),
            "texts": len(seq),
            "output_positions": len(seq),
            "batch_forwards": n[0],
            "study_exposures": len(seq),
            "seconds": perf_counter() - start,
            "revision": runtime.revision,
            "observed_responses_in_input": 0,
        },
    )
