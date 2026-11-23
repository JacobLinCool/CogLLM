"""Apply the immutable revision estimators to unlabeled source–revision pairs."""

import argparse
import json
from pathlib import Path

import numpy as np

from research.reading_validation.external_input import digest_string
from research.reading_validation.joint_data import digest, lines
from research.reading_validation.joint_fit import normalized_hidden
from research.reading_validation.revision_comprehension_prepare import (
    DATA,
    LEXICAL_HASH,
    PACKET_HASH,
    RESULT,
    versions,
)
from research.reading_validation.revision_features import (
    EMBEDDED_DIMS,
    ROOT,
    WORD,
    comparison,
    passage,
    predictions,
    word_readouts,
)
from research.reading_validation.revision_prepare import OUT as REVISION


def states_for_pairs(archive, packet, texts):
    """Validate text geometry, gather only requested passages, and apply fixed upstream heads."""
    record = json.loads((archive / "extraction.json").read_bytes())
    if (
        digest(packet) != record["packet_sha256"]
        or record["lexical_sha256"] != LEXICAL_HASH
    ):
        raise ValueError("Unrecognized native feature packet")
    if any(digest(archive / name) != h for name, h in record["files_sha256"].items()):
        raise ValueError("Native feature checksum differs from L4 receipt")
    ids = json.loads((archive / "ids.json").read_bytes())
    material = {r["id"]: r["text"] for r in lines(packet)}
    if set(material) != set(ids) or len(ids) != len(set(ids)):
        raise ValueError("Native feature IDs differ from released texts")
    selected = [i for i, key in enumerate(ids) if material[key] in texts]
    strings = [material[ids[i]] for i in selected]
    if len(selected) != len(texts):
        raise ValueError("Unencoded input")
    states = np.load(archive / "states.npy", mmap_mode="r")[selected]
    model_path = ROOT / "models.npz"
    if (
        digest(model_path)
        != json.loads((ROOT / "seal.json").read_bytes())["files"]["models.npz"]
    ):
        raise ValueError("ReadMe++ head changed")
    with np.load(model_path, allow_pickle=False) as model:
        grades = 1 + predictions(model, states, strings)["adaptive.qwen"].sum(axis=1)
        basis = model["fit.basis.projection"][:, -EMBEDDED_DIMS:]
        terminal = (normalized_hidden(states) - model["fit.basis.hidden_mean"]) @ basis
    vectors = np.asarray(states, dtype=np.float64)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    items = {}
    with np.load(archive / "words.npz", allow_pickle=False) as arrays:
        bounds, seen = arrays["range_offsets"], arrays["observed"]
        stored, lex, offsets = (
            arrays["contextual"],
            arrays["lexical"],
            arrays["word_spans"],
        )
        for local, index in enumerate(selected):
            t = strings[local]
            span = slice(bounds[index], bounds[index + 1])
            expected = np.asarray(
                [m.span() for m in WORD.finditer(t)], dtype=int
            ).reshape(-1, 2)
            if not np.array_equal(offsets[span], expected):
                raise ValueError("Invalid native word offsets")
            contextual = stored[span].astype(np.float32)
            contextual /= np.maximum(
                np.linalg.norm(contextual, axis=1, keepdims=True), 1e-9
            )
            items[t] = (
                float(grades[local]),
                vectors[local],
                terminal[local],
                seen[span],
                contextual,
                lex[span],
            )
    return items


def score_pairs(archive, packet, pairs):
    prior = REVISION / "forecast"
    seal = json.loads((prior / "forecast-seal.json").read_bytes())
    if digest(prior / "coefficients.npz") != seal["files_sha256"]["coefficients.npz"]:
        raise ValueError("Frozen revision coefficients changed")
    if (
        digest(Path(__file__).with_name("revision_features.py"))
        != seal["feature_code_sha256"]
    ):
        raise ValueError("The feature definitions changed")
    texts = {s for pair in pairs for s in pair}
    neural = states_for_pairs(archive, packet, texts)
    parsed = {t: passage(t) for t in texts}
    blocks = {k: [] for k in ("edit", "neural", "directions")}
    raw = []
    for src, draft in pairs:
        a, b = parsed[src], parsed[draft]
        form, base, ma, mb = comparison(a, b)
        sem, high = word_readouts(a, b, ma, mb, neural[src], neural[draft])
        for k, arr in (("edit", form), ("neural", sem), ("directions", high)):
            blocks[k].append(arr)
        # Retain the symmetric frozen control and also score source-only recall with frequency weights.
        base.update(
            retained_length=-abs(len(b.words) - len(a.words)),
            source_word_recall=float(np.average(~ma, weights=a.idf)),
            readme=neural[src][0] - neural[draft][0],
            contextual_coverage=float((sem[4] + sem[15]) / 2),
        )
        raw.append(base)
    with np.load(prior / "coefficients.npz", allow_pickle=False) as coeff:
        model_scores = {}
        for name in ("edit", "full"):
            x = np.column_stack(
                [
                    np.clip(
                        (np.stack(blocks[k]) - coeff[f"{name}.center.{k}"])
                        / coeff[f"{name}.scale.{k}"],
                        -8,
                        8,
                    )
                    for k in seal["models"][name]
                ]
            )
            model_scores[name] = x @ coeff[f"{name}.coef"]
        adaptive = model_scores["edit"] + coeff["mix"] * (
            model_scores["full"] - model_scores["edit"]
        )
    for i, row in enumerate(raw):
        row.update(
            meaning=float(adaptive[i, 1]),
            simplicity=float(adaptive[i, 0]),
            fluency=float(adaptive[i, 2]),
            edit_meaning=float(model_scores["edit"][i, 1]),
        )
    if not all(np.isfinite(list(r.values())).all() for r in raw):
        raise ArithmeticError("Nonfinite frozen forecast")
    return raw


def replay(archive, packet):
    prior = REVISION / "forecast"
    target = list(lines(prior / "forecasts.jsonl"))
    text = {r["id"]: r["text"] for r in lines(packet)}
    pairs = [(text[t["source_id"]], text[t["draft_id"]]) for t in target]
    fresh = score_pairs(archive, packet, pairs)
    keys = {
        "meaning": ("adaptive", 1),
        "simplicity": ("adaptive", 0),
        "fluency": ("adaptive", 2),
        "edit_meaning": ("edit", 1),
        "readme": ("readme", None),
        "length": ("length", None),
        "word_coverage": ("word_coverage", None),
        "contextual_coverage": ("contextual_coverage", None),
    }
    differences = {}
    for name, (old, ix) in keys.items():
        original = [
            row["scores"][old] if ix is None else row["scores"][old][ix]
            for row in target
        ]
        differences[name] = float(
            np.max(np.abs(np.array(original) - [row[name] for row in fresh]))
        )
    if max(differences.values()) > 1e-8:
        raise ArithmeticError(f"Frozen-head replay mismatch {differences}")
    return {
        "variant_rows": len(target),
        "max_absolute_logit_error": differences,
        "previous_seal_sha256": digest(prior / "forecast-seal.json"),
    }


def main(archive):
    parity = json.loads((RESULT / "replay.json").read_bytes())
    families = versions()
    grid = [
        (art, para, system, info["Adv"], text)
        for (art, para), info in sorted(families.items())
        for system, text in sorted(info.items())
    ]
    if digest(DATA / "stimuli.jsonl") != PACKET_HASH:
        raise ValueError("Changed input file")
    predictions_ = score_pairs(
        archive, DATA / "stimuli.jsonl", [(x[3], x[4]) for x in grid]
    )
    path = RESULT / "forecasts.jsonl"
    with path.open("w") as stream:
        for (a, p, m, s, t), row in zip(grid, predictions_, strict=True):
            stream.write(
                json.dumps(
                    {
                        "article": a,
                        "paragraph": p,
                        "condition": m,
                        "source_sha256": digest_string(s),
                        "visible_sha256": digest_string(t),
                        "scores": row,
                    }
                )
                + "\n"
            )
    seal = {
        "geometry_sha256": digest(RESULT / "input-seal.json"),
        "forecasts_sha256": digest(path),
        "native": json.loads((archive / "extraction.json").read_bytes()),
        "replay": parity,
        "frozen_score_script_sha256": digest(
            Path(__file__).with_name("revision_comprehension_score.py")
        ),
        "forecast_script_sha256": digest(Path(__file__)),
    }
    (RESULT / "forecast-seal.json").write_text(json.dumps(seal, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", required=True, type=Path)
    p.add_argument("--verify-original", type=Path, metavar="STIMULUS_PACKET")
    args = p.parse_args()
    if args.verify_original:
        result = replay(args.archive, args.verify_original)
        (RESULT / "replay.json").write_text(json.dumps(result, indent=2) + "\n")
        print(result)
    else:
        main(args.archive)
