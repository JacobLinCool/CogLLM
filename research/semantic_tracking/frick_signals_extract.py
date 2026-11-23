"""Export causal Frick sentence states and token signals from the pinned SDPA pass."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import itertools
import json
from pathlib import Path
import tempfile
from threading import Event, Thread
from time import perf_counter

from research.semantic_tracking.beukers_extract import inventory, validate_preflight
from research.semantic_tracking.gum_bridge_extract import CANDIDATE, digest


FEATURE_COLUMNS = (
    "surprisal_nats",
    "predictive_entropy_nats",
    "attention_prior_mass_excess",
    "attention_peak_prior_sentence_excess",
)
ATTENTION_LAYER = 23
ATTENTION_HEADS = 8
KV_HEADS = 2
HEAD_DIM = 256
VOCAB_SIZE = 248_320
VOCAB_CHUNK = 16
DOCUMENTS = 40
TOKENS = 9_386
STOPS = 624


def sentence_attention_features(torch, query, key, attention_mask, fences, scaling):
    """Reconstruct the final full-attention math and reduce it by earlier sentence."""
    batch, heads, length, width = query.shape
    if (batch, heads, width) != (1, ATTENTION_HEADS, HEAD_DIM) or key.shape != (1, KV_HEADS, length, HEAD_DIM):
        raise ValueError("Expected full Frick query/key geometry with a single unpadded document")
    if query.dtype != torch.float32 or key.dtype != torch.float32 or scaling != HEAD_DIM**-0.5:
        raise ValueError("Unexpected FP32 attention geometry or scaling")
    token_index = torch.arange(length, device=query.device)
    causal = token_index[:, None] >= token_index[None, :]
    grouped = query.reshape(batch, KV_HEADS, heads // KV_HEADS, length, width)
    scores = (grouped @ key.unsqueeze(2).transpose(-2, -1)).reshape(batch, heads, length, length) * scaling
    if attention_mask is None:
        scores = scores.masked_fill(~causal, -torch.inf)
    elif attention_mask.dtype == torch.bool:
        if bool((attention_mask & ~causal).any()):
            raise ValueError("SDPA attention mask permits future positions")
        scores = scores.masked_fill(~attention_mask, -torch.inf)
    else:
        scores = scores + attention_mask
        if bool((scores[..., ~causal] > -1e8).any()):
            raise ValueError("SDPA additive mask permits future positions")
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).mean(dim=1)[0]
    if not bool(torch.isfinite(weights).all()) or not torch.allclose(
        weights.sum(dim=-1), torch.ones(length, device=query.device), rtol=0, atol=2e-6
    ):
        raise RuntimeError("Reconstructed attention has invalid row mass")
    edges = torch.tensor(fences, device=query.device, dtype=torch.long)
    if edges.ndim != 1 or not len(edges) or fences[-1] != length - 1:
        raise ValueError("Sentence fences must cover every source token")
    current = torch.searchsorted(edges, token_index)
    earlier = current[:, None] > torch.arange(len(edges), device=query.device)
    earlier_mass = weights.cumsum(dim=-1).index_select(1, edges)
    zeros = torch.zeros((length, 1), dtype=torch.float32, device=query.device)
    mass = torch.diff(torch.cat((zeros, earlier_mass), dim=-1), dim=-1)
    counts = torch.diff(torch.cat((edges.new_tensor([-1]), edges)))
    excess = mass - counts[None, :] / (token_index[:, None] + 1)
    total = excess.masked_fill(~earlier, 0).sum(dim=-1)
    peak = excess.masked_fill(~earlier, -torch.inf).amax(dim=-1)
    peak = torch.where(current > 0, peak, 0)
    return torch.stack((total, peak), dim=-1)


class FinalAttentionCapture:
    def __init__(self, original, torch):
        self.original = original
        self.torch = torch
        self.fences = None
        self.features = None

    def __call__(self, module, query, key, value, attention_mask, **kwargs):
        result = self.original(module, query, key, value, attention_mask, **kwargs)
        if module.layer_idx == ATTENTION_LAYER:
            if self.fences is None or self.features is not None:
                raise RuntimeError("Expected exactly one registered final-attention call")
            if (
                kwargs.get("dropout", 0) != 0 or kwargs.get("position_bias") is not None
                or kwargs.get("is_causal", True) is False or not module.is_causal
            ):
                raise ValueError("Unexpected final layer SDPA options")
            self.features = sentence_attention_features(
                self.torch, query, key, attention_mask, self.fences, kwargs["scaling"]
            )
        return result


@contextmanager
def scoped_capture(qwen_module, torch):
    original_registry = qwen_module.ALL_ATTENTION_FUNCTIONS
    recorder = FinalAttentionCapture(original_registry["sdpa"], torch)
    registry = type(original_registry)()
    registry["sdpa"] = recorder
    qwen_module.ALL_ATTENTION_FUNCTIONS = registry
    try:
        yield recorder
    finally:
        qwen_module.ALL_ATTENTION_FUNCTIONS = original_registry


def predictive_features(torch, hidden, embedding, tokens):
    if embedding.shape != (VOCAB_SIZE, hidden.shape[-1]) or embedding.dtype != torch.float32:
        raise ValueError("Expected the original FP32 tied output embedding")
    features = hidden.new_zeros((len(tokens), 2))
    token_ids = torch.tensor(tokens, device=hidden.device, dtype=torch.long)
    for offset in range(1, len(tokens), VOCAB_CHUNK):
        end = min(offset + VOCAB_CHUNK, len(tokens))
        logits = torch.nn.functional.linear(hidden[offset - 1:end - 1], embedding)
        log_probability = torch.nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32)
        features[offset:end, 0] = -log_probability.gather(1, token_ids[offset:end, None])[:, 0]
        features[offset:end, 1] = -(log_probability.exp() * log_probability).sum(dim=-1)
    return features


def encode_with_signals(encoder, encode, recorder, tokens, fences, pad_id, torch):
    if not fences or tuple(sorted(set(fences))) != tuple(fences) or fences[-1] != len(tokens) - 1:
        raise ValueError("Feature calls require ordered sentence-end token indices")
    recorder.fences = fences
    try:
        with torch.inference_mode():
            hidden = encode(encoder, [list(tokens)], pad_id, torch)[0]
            if recorder.features is None or hidden.dtype != torch.float32:
                raise RuntimeError("Exact Qwen pass omitted expected FP32 features")
            features = torch.cat(
                (predictive_features(torch, hidden, encoder.embed_tokens.weight, tokens), recorder.features), dim=-1
            )
            if not bool(torch.isfinite(features).all()) or not bool(torch.isfinite(hidden).all()):
                raise RuntimeError("Nonfinite FP32 export")
            return hidden, features
    finally:
        recorder.fences = None
        recorder.features = None


def controls(documents):
    ordered = sorted(documents, key=lambda item: (-len(item.tokens), item.identifier))
    known = {tuple(item.tokens) for item in documents}
    for i, document in enumerate(ordered[:3]):
        fences = tuple(position for _, position in document.certified)
        cut = fences[len(fences) // 2] + 1
        suffix_length = len(document.tokens) - cut
        if suffix_length < 16:
            raise ValueError("Frick diagnostic needs a substantial changed suffix")
        prefix = tuple(document.tokens[:cut])
        donor = ordered[10 + i].tokens
        other = ordered[-(i + 1)].tokens
        base = prefix + tuple(itertools.islice(itertools.cycle(donor), suffix_length))
        changed = prefix + tuple(itertools.islice(itertools.cycle(other), suffix_length))
        if base in known or changed in known or prefix in known or base == changed:
            raise ValueError("Diagnostic controls must differ from every full assay input")
        yield (base, changed, prefix, fences)


def verify(encoder, encode, recorder, examples, reference, pad_id, torch, difference):
    checks = []
    for case, ((base, replacement, prefix, fences), plain) in enumerate(zip(examples, reference, strict=True), start=1):
        original, signals = encode_with_signals(encoder, encode, recorder, base, fences, pad_id, torch)
        length = len(prefix)
        cut_fences = tuple(f for f in fences if f < length)
        shortened, shortened_signals = encode_with_signals(
            encoder, encode, recorder, prefix, cut_fences, pad_id, torch,
        )
        changed, changed_signals = encode_with_signals(encoder, encode, recorder, replacement, fences, pad_id, torch)
        checks.append({
            "case": case, "prefix_tokens": length, "comparison": "hooked_vs_unhooked",
            "hidden_metrics": difference(plain, original), "feature_max_absolute": None,
            "passed": torch.equal(plain, original),
        })
        for label, measured, measured_signals, rms in (
            ("prefix_truncation", shortened, shortened_signals, 0.02),
            ("replacement_suffix", changed[:length], changed_signals[:length], 0.002),
        ):
            metrics = difference(original[:length], measured)
            feature_error = (signals[:length] - measured_signals).abs().amax(dim=0).tolist()
            feature_limits = (0.04, 0.04, 0.002, 0.002)
            checks.append({
                "case": case, "prefix_tokens": length, "comparison": label, "hidden_metrics": metrics,
                "feature_max_absolute": feature_error,
                "passed": metrics["max_token_relative_rms"] <= rms
                and all(err <= lim for err, lim in zip(feature_error, feature_limits, strict=True)),
            })
        metrics = difference(original[length:], changed[length:])
        feature_error = (signals[length:] - changed_signals[length:]).abs().amax(dim=0).tolist()
        checks.append({
            "case": case, "prefix_tokens": length, "comparison": "replacement_suffix_positive",
            "hidden_metrics": metrics, "feature_max_absolute": feature_error,
            "passed": metrics["max_token_relative_rms"] > 0.02 and max(feature_error) > 0.04,
        })
    if len(checks) != 12 or any(not row["passed"] for row in checks):
        raise RuntimeError(f"Frick instrumentation or prefix checks failed: {checks}")
    return checks


def extract(stimuli, model, checkpoint, preflight, assay_sha256, protocol_sha256, output):
    import numpy as np
    import torch
    import transformers
    from transformers.models.qwen3_5 import modeling_qwen3_5

    from research.gradio_app.causal_plaba import HIDDEN_WIDTH
    from research.gradio_app.checkpoint import BACKBONE, load_checkpoint
    from research.gradio_app.cuda import CudaCausalQwen
    from research.stage3.run import difference, encode, sample_gpu, save_json

    started = perf_counter()
    if output.exists():
        raise FileExistsError(output)
    if any(len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
           for value in (assay_sha256, protocol_sha256)):
        raise ValueError("Expected exact assay and protocol SHA-256 values")
    if load_checkpoint(checkpoint).identifier != CANDIDATE:
        raise ValueError("Expected the registered Gradio Qwen candidate")
    runtime = CudaCausalQwen(model)

    def tokenize(text):
        encoded = runtime.tokenizer.encode(text, add_special_tokens=False)
        return encoded.ids, encoded.offsets

    def prefixes(texts):
        return [runtime.tokenizer.encode(text, add_special_tokens=False).ids for text in texts]

    summary, rows, documents = inventory(stimuli, tokenize, prefixes)
    if summary != json.loads(preflight.read_text(encoding="utf-8")):
        raise ValueError("The exact pinned Transformers tokenization differs from Rust preflight")
    validate_preflight(summary, rows, digest(stimuli))
    if (len(documents), summary["input_tokens"], summary["certified_stops"]) != (DOCUMENTS, TOKENS, STOPS):
        raise ValueError("Expected only the registered 40 full Frick stories")
    if any(doc.certified[-1] != (len(row["text"]), len(doc.tokens) - 1)
           for row, doc in zip(rows, documents, strict=True)):
        raise ValueError("Every Frick story requires its full terminal coordinate")
    if type(runtime.encoder.layers[ATTENTION_LAYER].self_attn) is not modeling_qwen3_5.Qwen3_5Attention:
        raise ValueError("Expected the registered Qwen final full attention layer")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".frick-signals-", dir=output.parent) as temporary:
        stage = Path(temporary) / "result"
        stage.mkdir()
        with (stage / "coordinates.jsonl").open("x", encoding="utf-8") as f:
            for row, doc in zip(rows, documents, strict=True):
                f.write(json.dumps({"id": row["id"], "stops": row["stops"], "offsets": doc.offsets}) + "\n")
        shapes = (TOKENS, len(FEATURE_COLUMNS)), (STOPS, HIDDEN_WIDTH)
        features = np.empty(shapes[0], dtype=np.float32)
        states = np.empty(shapes[1], dtype=np.float32)
        torch.cuda.reset_peak_memory_stats()
        done = Event()
        telemetry = Thread(target=sample_gpu, args=(stage / "telemetry.jsonl", done))
        telemetry.start()
        try:
            examples = tuple(controls(documents))
            pad_id = runtime.tokenizer.pad_token_id
            reference = [encode(runtime.encoder, [list(base)], pad_id, torch)[0] for base, _, _, _ in examples]
            with scoped_capture(modeling_qwen3_5, torch) as capture:
                checks = verify(runtime.encoder, encode, capture, examples, reference, pad_id, torch, difference)
                del reference
                clock = perf_counter()
                token_row = state_row = 0
                for doc in documents:
                    ends = tuple(position for _, position in doc.certified)
                    hidden, matrix = encode_with_signals(
                        runtime.encoder, encode, capture, doc.tokens, ends, pad_id, torch,
                    )
                    next_token = token_row + len(doc.tokens)
                    next_state = state_row + len(ends)
                    features[token_row:next_token] = matrix.cpu().numpy()
                    states[state_row:next_state] = hidden[list(ends)].cpu().numpy()
                    token_row, state_row = next_token, next_state
                torch.cuda.synchronize()
                forward_time = perf_counter() - clock
            if (token_row, state_row) != (TOKENS, STOPS):
                raise RuntimeError("Exported arrays do not cover the sealed token and sentence-stop inventory")
        finally:
            done.set()
            telemetry.join()
        np.save(stage / "token_features.npy", features, allow_pickle=False)
        np.save(stage / "states.npy", states, allow_pickle=False)
        save_json(stage / "causality.json", {"checks": checks})
        artifact_names = ("coordinates.jsonl", "token_features.npy", "states.npy", "causality.json", "telemetry.jsonl")
        save_json(stage / "seal.json", {
            **summary, "preflight_sha256": digest(preflight),
            "assay_sha256": assay_sha256, "protocol_sha256": protocol_sha256,
            "source_code_sha256": digest(Path(__file__)), "candidate_id": CANDIDATE,
            "model": BACKBONE["model"], "revision": BACKBONE["revision"],
            "transformers": transformers.__version__, "torch_version": torch.__version__,
            "compute_dtype": "float32", "storage_dtype": "float32", "states_dtype": "float32",
            "attention_layer": ATTENTION_LAYER, "attention_heads": ATTENTION_HEADS,
            "feature_columns": list(FEATURE_COLUMNS), "token_features_shape": list(shapes[0]),
            "states_shape": list(shapes[1]), "content_forward_per_document": 1,
            "content_backbone_forward_calls": len(documents), "verification_backbone_forward_calls": len(checks),
            "independent_causal_checks": len(checks), "forward_and_readout_s": forward_time,
            "elapsed_s": perf_counter() - started, "gpu": torch.cuda.get_device_name(0),
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
            "file_sha256": {name: digest(stage / name) for name in artifact_names},
        })
        stage.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stimuli", "model", "checkpoint", "preflight", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("assay-sha256", "protocol-sha256"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    extract(
        args.stimuli, args.model, args.checkpoint, args.preflight,
        args.assay_sha256, args.protocol_sha256, args.output,
    )


if __name__ == "__main__":
    main()
