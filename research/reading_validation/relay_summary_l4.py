"""Freeze public expert-written Cochrane summary representations and PLABA forecasts in one pass."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np

from research.reading_validation.local_forecast import WORD, digest, rows
from research.reading_validation.local_inference import LocalInference


def run(runtime, reader, checkpoint, input_path, output_dir):
    import torch

    if (
        runtime.candidate != "qwen"
        or runtime.width != 2048
        or runtime.revision != checkpoint["revision"]
    ):
        raise ValueError("Expected the pinned Qwen3.5-2B text backbone")
    src = Path(input_path)
    expected = "955fbebd97551c526d0149e2ccbdd8b6fd2a30dcf69d456c78fa92a74e5de681"
    if digest(src) != expected:
        raise ValueError("Only the sealed public expert summaries are eligible")
    targets = rows(src)
    if (
        len(targets) != 91
        or len({r["text"] for r in targets}) != 91
        or any(
            set(r) != {"id", "text"} or not r["id"].startswith("relay-summary/")
            for r in targets
        )
    ):
        raise ValueError("Unexpected expert-written summary packet")
    model = LocalInference.load()
    lexical_weight = torch.as_tensor(model.lexical_weight, device=runtime.device)
    dest = Path(output_dir)
    dest.mkdir(exist_ok=True, parents=True)
    calls = 0
    tokens = 0
    vectors = []
    sent_vectors = []
    index = []
    start = perf_counter()

    def count(_, __):
        nonlocal calls
        calls += 1

    with runtime.encoder.register_forward_pre_hook(count), torch.inference_mode():
        for row in targets:
            text = row["text"]
            ids = runtime.tokenizer.encode(text, add_special_tokens=False)
            if not 0 < len(ids.ids) <= 1024:
                raise ValueError("Public summary outside supported length")
            ft = runtime.encode(text)
            terms = model.score(text, ft, runtime.tokenizer, reader)
            certified = {}
            for i, (_, b) in enumerate(ft.offsets):
                certified[b] = i
            normalized = torch.nn.functional.normalize(ft.hidden.double(), dim=1)
            lexical = (
                (normalized @ lexical_weight + model.lexical_bias)
                .clamp(0, 1)
                .float()
                .cpu()
                .numpy()
            )
            sentences = []
            for sent in terms["sentences"]:
                rating = sent["rating"]
                o = {
                    "start": sent["start"],
                    "end": sent["end"],
                    "words": sent["words"],
                    "rating": rating,
                }
                if rating is not None:
                    j = certified[sent["end"]]
                    if (
                        list(ft.token_ids[: j + 1])
                        != runtime.tokenizer.encode(
                            text[: sent["end"]], add_special_tokens=False
                        ).ids
                    ):
                        raise ValueError("Uncertified sentence terminal state")
                    word_ends = [
                        certified[m.end()]
                        for m in WORD.finditer(text, sent["start"], sent["end"])
                        if m.end() in certified
                    ]
                    if not word_ends:
                        raise ArithmeticError(
                            "Certified sentence lacks an observed word stop"
                        )
                    local = lexical[word_ends]
                    o["observed_lexical_words"] = len(local)
                    o["lexical_mean_q90"] = [
                        float(local.mean()),
                        float(np.quantile(local, 0.9)),
                    ]
                    o["state_row"] = len(sent_vectors)
                    sent_vectors.append(ft.hidden[j].detach().cpu().numpy().copy())
                sentences.append(o)
            vectors.append(ft.hidden[-1].detach().cpu().numpy().copy())
            tokens += len(ids.ids)
            index.append(
                {"id": row["id"], "tokens": len(ids.ids), "sentences": sentences}
            )
    if (
        calls != len(targets)
        or not np.isfinite(vectors).all()
        or not np.isfinite(sent_vectors).all()
    ):
        raise ArithmeticError("Single-forward or finite-feature contract failed")
    (dest / "geometry.json").write_text(json.dumps(index, sort_keys=True) + "\n")
    np.save(dest / "document.npy", np.stack(vectors), allow_pickle=False)
    np.save(dest / "sentences.npy", np.stack(sent_vectors), allow_pickle=False)
    receipt = {
        "data_sha256": digest(src),
        "worker_sha256": digest(Path(__file__)),
        "qwen_revision": runtime.revision,
        "plaba_bundle_sha256": model.bundle_sha256,
        "documents": len(targets),
        "tokens": tokens,
        "forward_calls": calls,
        "sentences": sum(len(q["sentences"]) for q in index),
        "certified_sentence_ends": len(sent_vectors),
        "files_sha256": {
            p: digest(dest / p)
            for p in ["geometry.json", "document.npy", "sentences.npy"]
        },
        "elapsed_seconds": perf_counter() - start,
    }
    (dest / "extraction.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
