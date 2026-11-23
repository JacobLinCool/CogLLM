"""Export the development readouts and replay them against causal L4 feature archives."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from research.reading_validation.local_features import (
    DATA,
    OUT,
    read_gpu,
    source_lexicon,
    verified_rows,
    build,
)
from research.reading_validation.local_forecast import digest
from research.reading_validation.local_inference import (
    DEFAULT,
    REVISION,
    SCHEMA,
    LocalInference,
    english_sentences,
)
from research.reading_validation.local_ratings import full_states


def export(target):
    lex, source = source_lexicon()
    label = OUT / "localization/deployment-weights.json"
    article = OUT / "ratings/deployment-weights.json"
    rank = json.loads(label.read_bytes())["local"]
    rating = json.loads(article.read_bytes())["full_state_local"]
    target.mkdir(exist_ok=True, parents=True)
    lexicon = lex.records()
    (target / "lexicon.json.gz").write_bytes(
        gzip.compress(json.dumps(lexicon, separators=(",", ":"), sort_keys=True).encode(), mtime=0)
    )
    arrays = {}
    for group in rank["features"]:
        arrays["rank_mean_" + group] = rank["center"][group]
        arrays["rank_scale_" + group] = rank["scale"][group]
    for group, p in rating["normalization"].items():
        arrays["rating_mean_" + group] = p["mean"]
        arrays["rating_scale_" + group] = p["scale"]
    rng = np.random.default_rng(78491)
    projection, _ = np.linalg.qr(rng.standard_normal((2048, 32)))
    with np.load(DATA / "local-lexical.npz", allow_pickle=False) as lh:
        np.savez_compressed(
            target / "weights.npz",
            rank_coef=np.array(rank["coef"]),
            rating_coef=np.array(rating["coef"]),
            rating_offset=np.array(rating["offset"]),
            lexical_weight=lh["w"],
            lexical_bias=lh["b"],
            projection=projection.astype("float32"),
            **arrays,
        )
    meta = dict(
        schema=SCHEMA,
        revision=REVISION,
        rank_features=rank["features"],
        rating_features=rating["features"],
        source_weights_sha256={
            "span": digest(label),
            "sentence": digest(article),
            "lexical": digest(DATA / "local-lexical.npz"),
        },
        files_sha256={n: digest(target / n) for n in ("weights.npz", "lexicon.json.gz")},
        training_data=source,
        target="English biomedical abstract difficulty; positive-only named spans, full-text-visible sentence ratings",
        scope="Retrospective development full fit. Independent domain transfer is evaluated separately.",
    )
    (target / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def replay(gpu, target):
    fit = LocalInference.load(target)
    cache = read_gpu(gpu)
    states = full_states()
    nodes, _ = build(gpu)
    texts = {r["id"]: r["text"] for r in verified_rows("plaba", "stimuli.jsonl")}
    rank = json.loads((OUT / "localization/deployment-weights.json").read_bytes())["local"]
    rating = json.loads((OUT / "ratings/deployment-weights.json").read_bytes())["full_state_local"]
    count = sentences = overlap = discovered = certified = 0
    max_rank_delta = max_rating_delta = 0.0
    for doc in verified_rows("plaba", "ratings.jsonl"):
        key = doc["id"]
        raw = cache["plaba/" + key]
        lite = {k: raw[k] for k in ("words", "offsets", "lexical", "projection")}
        manual = [(r["start"], r["stop"]) for r in doc["ratings"]]
        canonical = {r["stop"]: states[r["anchor_id"]] for r in doc["ratings"]}
        analyzed = fit._infer_arrays(texts[key], lite, canonical, manual)
        found = set(english_sentences(texts[key]))
        overlap += len(found & set(manual))
        discovered += len(found)
        for r, output in zip(doc["ratings"], analyzed["sentences"], strict=True):
            key = r["anchor_id"]
            n = nodes[key]
            x = np.column_stack(
                [(n.features[f] - rank["center"][f]) / rank["scale"][f] for f in rank["features"]]
            )
            scores = x @ rank["coef"]
            selected = []
            for i in np.argsort(-scores, kind="stable"):
                a, b = n.coords[i]
                if all(a >= d or c >= b for c, d in selected):
                    selected.append((int(a), int(b)))
                    if len(selected) == 5:
                        break
            coords = [(z["start"], z["end"]) for z in output["phrases"]]
            if coords != selected:
                raise ValueError(f"Deployed and offline spans disagree at {key}")
            ids = {tuple(coord): j for j, coord in enumerate(n.coords)}
            max_rank_delta = max(
                max_rank_delta,
                max(
                    abs(scores[ids[a, b]] - z["score"])
                    for (a, b), z in zip(coords, output["phrases"], strict=True)
                ),
            )
            s = dict(n.sentence)
            s["surface"] = s["surface"][[0, 1, 2, 4, 5, 6]]
            s["sentence_state"] = states[key]
            zx = np.concatenate(
                [
                    (s[f] - rating["normalization"][f]["mean"]) / rating["normalization"][f]["scale"]
                    for f in rating["features"]
                ]
            )
            logits = (zx @ np.asarray(rating["coef"]))[:, None] * (np.arange(1, 6) - 3) + rating["offset"]
            prob = np.exp(logits - logits.max(axis=-1, keepdims=True))
            expected = (prob / prob.sum(axis=-1, keepdims=True)) @ np.arange(1, 6)
            max_rating_delta = max(
                max_rating_delta,
                max(
                    abs(expected[j] - output["rating"][k]) for j, k in enumerate(("ease", "understanding"))
                ),
            )
            certified += output["certified_words"]
            count += len(coords)
            sentences += 1
    if max_rank_delta > 2e-6 or max_rating_delta > 2e-6:
        raise ValueError("Exported local readouts failed numerical replay")
    receipt = dict(
        plaba_documents=len(texts),
        sentence_endpoints=sentences,
        top_ranked_offsets_checked=count,
        certified_word_ends=certified,
        max_absolute_span_score_difference=float(max_rank_delta),
        max_absolute_rating_difference=float(max_rating_delta),
        default_detected_sentences=discovered,
        default_sentences_exactly_matching_author_spans=overlap,
        verification_input="archived Qwen L4 features and certified source sentence intervals",
        manifest_sha256=fit.bundle_sha256,
        source_gpu_sha256=digest(gpu / "seal.json"),
    )
    (target / "replay.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", type=Path, required=True)
    p.add_argument("--out", type=Path, default=DEFAULT)
    a = p.parse_args()
    export(a.out)
    replay(a.gpu, a.out)
