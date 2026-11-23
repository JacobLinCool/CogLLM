"""Apply the frozen sentence-difficulty models to independently presented rewrites."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.special import expit

from research.reading_validation.joint_data import DEFAULT_CACHE, digest, lines, read, surface
from research.reading_validation.joint_fit import Basis
from research.reading_validation.local_inference import english_sentences


ROOT = Path("research/results/reading-validation-joint/forecast")


def predictions(model, hidden, texts):
    study = SimpleNamespace(hidden=hidden, surface=np.stack([surface(t, "en") for t in texts]))
    preds = {}
    for name in ("zero", "fit"):
        basis = Basis(**{key: model[f"{name}.basis.{key}"] for key in Basis.__dataclass_fields__})
        for design in ("qwen", "surface"):
            x = basis.transform(study, design == "qwen")
            if name == "zero":
                preds[f"zero.{design}"] = x @ model[f"zero.{design}.slope"]
                continue
            p = {}
            for mode in ("independent", "pooled"):
                key = f"independent.{design}.readme" if mode == "independent" else f"pooled.{design}"
                p[mode] = expit((x @ model[f"{key}.slope"])[:, None] + model[f"{key}.offset.readme"])
            beta = float(model[f"mix.{design}.readme"])
            p["adaptive"] = p["independent"] + beta * (p["pooled"] - p["independent"])
            preds.update({f"{mode}.{design}": value for mode, value in p.items()})
    return preds


def extracted(folder, packet, selected=None):
    meta = read(folder / "extraction.json")
    if meta["packet_sha256"] != digest(packet):
        raise ValueError("Feature packet identity differs")
    for file, expected in meta["files_sha256"].items():
        if digest(folder / file) != expected:
            raise ValueError("Feature checksum differs")
    ids = read(folder / "ids.json")
    idx = {key: i for i, key in enumerate(ids)}
    if len(idx) != len(ids):
        raise ValueError("Feature IDs repeat")
    texts = {r["id"]: r["text"] for r in lines(packet)}
    if set(texts) != set(idx):
        raise ValueError("Model input and hidden state IDs differ")
    keys = ids if selected is None else selected
    matrix = np.load(folder / "states.npy", mmap_mode="r")
    return keys, matrix[[idx[k] for k in keys]], [texts[k] for k in keys], meta


def run(readme, rewrite, output, cache):
    if output.exists():
        raise FileExistsError(output)
    fit_seal = read(ROOT / "seal.json")
    if digest(ROOT / "models.npz") != fit_seal["files"]["models.npz"]:
        raise ValueError("Training weights have changed")
    material = cache / "reading-validation"
    with np.load(ROOT / "models.npz", allow_pickle=False) as model:
        heldout = [row["id"] for row in read(ROOT / "heldout.json")]
        _, hidden, texts, first = extracted(readme, material / "external-stimuli.jsonl", heldout)
        replay = predictions(model, hidden, texts)
        with np.load(ROOT / "forecasts.npz", allow_pickle=False) as archived:
            discrepancies = {k: float(np.abs(v - archived[k]).max()) for k, v in replay.items()}
        if max(discrepancies.values()) > 1e-10:
            raise ArithmeticError(f"Serving implementation differs from ReadMe++ evaluation: {discrepancies}")
        keys, hidden, texts, second = extracted(rewrite, material / "rewrite-stimuli.jsonl")
        if first["revision"] != second["revision"]:
            raise ValueError("Model revisions differ")
        raw = predictions(model, hidden, texts)
    values = {k: 1 + v.sum(axis=1) if v.ndim == 2 else v for k, v in raw.items()}
    values["length"] = [len(t.split()) for t in texts]
    values["sentence_count"] = [len(english_sentences(t)) for t in texts]
    output.mkdir(parents=True)
    np.savez_compressed(output / "scores.npz", id=keys, **values)
    seal = dict(readme_parity=discrepancies, n=len(keys), revision=first["revision"],
                data_sha256=digest(Path("research/results/reading-validation-rewrite/inputs.json")),
                model_sha256=fit_seal["files"]["models.npz"], extraction_sha256=digest(rewrite/"extraction.json"),
                scores_sha256=digest(output/"scores.npz"))
    (output / "seal.json").write_text(json.dumps(seal, indent=2) + "\n")
    return seal


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    for key in ("readme", "rewrite", "output"):
        p.add_argument("--"+key, required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(run(a.readme, a.rewrite, a.output, a.cache)))
