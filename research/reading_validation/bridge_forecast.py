"""Single-pass Carter–Hoffman forecasts. Inputs contain text and display word boundaries."""
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import re

import torch
from torch.nn import functional as F

from research.token_readout.runner import trace_rows


STOP = set("a an the am is are was were be been being has have had do did does can could would should "
           "may might will shall of by on at in to from and or but for as it its if this that these those "
           "he she they them their his her him hers you i we our my me us with into out".split())
FEATURES = ("log_context_words", "log_target_words", "content_overlap", "boundary_overlap",
            "mean_transition", "mean_word_surprisal", "mean_preamble_attention",
            "mid_cosine", "final_cosine")


def norm_word(word):
    return re.sub(r"[^a-z0-9]+", "", word.casefold())


def contiguous_overlap(preamble, target):
    longest = min(len(preamble), len(target))
    for width in range(longest, 0, -1):
        if preamble[-width:] == target[:width]:
            return width / len(target)
    return 0.0


def features_for_text(runtime, reader, checkpoint, row):
    text = row["text"]
    feats = runtime.encode(text)
    scores = reader(feats)
    scale = checkpoint["scale"]
    scores = replace(scores, opening=scores.opening * scale["opening_gain"],
                     resolution=scores.resolution * scale["resolution_gain"] + scale["resolution_bias"])
    trace = trace_rows(text, feats.offsets, scores, checkpoint["priors"], reader.anchor_tokens)
    context = list(re.finditer(r"\S+", text[:row["start"]]))
    ctx_words = [norm_word(m.group()) for m in context]
    words = [norm_word(text[start:end]) for start, end in row["target_words"]]
    indices = set()
    for match, word in zip(context, ctx_words, strict=True):
        if word and word not in STOP:
            indices.update(j for j, (s, e) in enumerate(feats.offsets)
                           if e > match.start() and s < match.end() and any(c.isalnum() for c in text[s:e]))
    ctx_content = {w for w in ctx_words if w and w not in STOP}
    ctx_tokens = [j for j, (s, e) in enumerate(feats.offsets) if e <= row["start"]]
    target_tokens = [[j for j, (s, e) in enumerate(feats.offsets) if s < b and e > a]
                     for a, b in row["target_words"]]
    flat = [i for span in target_tokens for i in span]
    if not indices or not ctx_tokens or not flat or min(flat) < 1 or len(flat) != len(set(flat)):
        raise ValueError("Ambiguous words or missing causal context")
    if len(set(ctx_tokens).intersection(flat)) != 0:
        raise ValueError("Context and target share a token")
    if runtime.candidate != "qwen" or not runtime.encoder.config.tie_word_embeddings:
        raise ValueError("This output projection requires Qwen's verified tied embedding weights")
    vocab = runtime.encoder.embed_tokens.weight
    idx = torch.tensor(flat, device=feats.hidden.device)
    tokens = torch.tensor(feats.token_ids, device=idx.device)
    losses = []
    for chunk in idx.split(16):
        logits = F.linear(feats.hidden[chunk - 1], vocab)
        losses.append(F.cross_entropy(logits, tokens[chunk], reduction="none"))
    surprisal = dict(zip(flat, torch.cat(losses).tolist(), strict=True))
    # Undo the documented encoding of the across-head mean attention.
    rel = feats.relations[..., 0]
    attn = (rel + math.log(2.)).expm1() / torch.arange(1, len(rel) + 1, device=rel.device)[:, None]
    prior_attn = attn[:, ctx_tokens].sum(dim=1).tolist()
    # Pool normalized contextual representations of content-bearing subwords.
    layers = [F.normalize(x.float(), dim=-1) for x in (feats.mid, feats.hidden)]
    ctx_vectors = [x[sorted(indices)].mean(0) for x in layers]
    seen_content = set()
    attended = 0.
    cum_scores = 0.
    trans_sum = 0.
    total_tokens = 0
    seen_indices = []
    features = []
    for k, (word, selected) in enumerate(zip(words, target_tokens, strict=True), start=1):
        if not selected:
            raise ValueError("Target word has no corresponding source tokens")
        cum_scores += sum(surprisal[i] for i in selected)
        attended += sum(prior_attn[i] for i in selected)
        trans_sum += sum(trace[i]["transition"] for i in selected)
        total_tokens += len(selected)
        if word and word not in STOP:
            seen_content.add(word)
            seen_indices.extend(i for i in selected if any(c.isalnum() for c in text[slice(*feats.offsets[i])]))
        cosine = ([float(F.cosine_similarity(p, x[seen_indices].mean(0), dim=0))
                   for p, x in zip(ctx_vectors, layers, strict=True)] if seen_indices else [0., 0.])
        features.append([math.log1p(len(context)), math.log1p(k),
                         len(ctx_content & seen_content) / max(1, len(seen_content)),
                         contiguous_overlap(ctx_words, words[:k]), trans_sum / total_tokens,
                         cum_scores / k, attended / total_tokens, *cosine])
    if not all(math.isfinite(v) for word in features for v in word):
        raise ValueError("Nonfinite feature")
    return {"id": row["id"], "words": features}


@torch.inference_mode()
def run(runtime, reader, checkpoint, input_path, output_dir):
    path = Path(input_path)
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    count = [0]
    def count_call(_module, _inputs):
        count[0] += 1
    with runtime.encoder.register_forward_pre_hook(count_call):
        with (output / "features.jsonl").open("w") as out:
            for n, row in enumerate(rows, 1):
                values = features_for_text(runtime, reader, checkpoint, row)
                if count[0] != n:
                    raise RuntimeError("Each text requires exactly one encoder pass")
                out.write(json.dumps(values, allow_nan=False) + "\n")
                if n % 32 == 0:
                    out.flush()
                    print(f"Carter–Hoffman forecasts {n}/{len(rows)}", flush=True)
    (output / "receipt.json").write_text(json.dumps({"backbone": runtime.candidate,
        "revision": runtime.revision, "input_sha256": hashlib.sha256(raw).hexdigest(),
        "features": FEATURES, "forward_calls": count[0], "input_rows": len(rows),
        "output_sha256": hashlib.sha256((output / "features.jsonl").read_bytes()).hexdigest()
    }, indent=2))
