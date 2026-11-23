"""Separate lexical expectation changes from punctuation and local form in held passages."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from wordfreq import zipf_frequency

from research.backbone_transfer.extract import digest, write_json

ROOT = Path("research/results/reading-validation-cloze-population")
INPUT = Path(
    ".cache/cogllm/reading-validation/cloze-population/external-scored-word-rows.jsonl"
)
PLAN = Path(__file__).with_name("word-change-specificity-protocol.md")
FUNCTION = {
        "a",
        "an",
        "the",
        "am",
        "are",
        "is",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "can",
        "could",
        "may",
        "might",
        "should",
        "shall",
        "will",
        "would",
        "must",
        "to",
        "of",
        "on",
        "in",
        "at",
        "as",
        "by",
        "for",
        "and",
        "but",
        "so",
        "or",
        "if",
        "from",
        "with",
        "into",
        "that",
        "this",
        "those",
        "these",
        "who",
        "whom",
        "when",
        "why",
        "how",
        "i",
        "he",
        "she",
        "it",
        "we",
        "they",
        "you",
        "me",
        "us",
        "them",
        "him",
        "her",
        "his",
        "my",
        "their",
        "our",
        "your",
        "its",
    }
INTERIOR = re.compile(r"[a-z]+(?:['’-][a-z]+)*\Z")


def form(s):
    root = s.casefold().strip('.,!?:;"()′‘’“”')
    length = np.log1p(len(s))
    frequency = zipf_frequency(root, "en")
    return [
        length,
        length**2,
        frequency,
        frequency**2,
        root in FUNCTION,
        s[:1].isupper(),
        bool(re.search(r"[.!?][\W_]*$", s)),
        bool(re.search(r"[,;:][\W_]*$", s)),
        bool(re.search(r"[^\w\s]", s)),
        any(c.isdigit() for c in s),
    ]


def data():
    if digest(INPUT) != json.loads((ROOT / "word-export.json").read_text())["file"]:
        raise ValueError("Registered word predictions changed")
    rows = {
        (r["Text_ID"], r["target_ordinal_zero_based"]): r
        for r in map(json.loads, INPUT.read_text().splitlines())
    }
    result = []
    for (story, t), r in rows.items():
        next_row = rows.get((story, t + 1))
        if (
            not next_row
            or not r["normative_denominator_verified"]
            or not next_row["normative_denominator_verified"]
        ):
            continue
        if (
            r["target_chars"] != next_row["preceding_chars"]
            or r["shown_target"] != next_row["preceding_print"]
        ):
            raise ValueError(
                "The model update is attributed to a different printed word"
            )
        a, b = r["preceding_print"], r["shown_target"]
        z = np.log1p(t)
        predictors = {"form": form(a) + form(b) + [z, z * z]}
        direct = {}
        for k in ("raw", "calibrated", "context"):
            left = r["model_category_disagreement"][k]
            right = next_row["model_category_disagreement"][k]
            direct[k] = right - left
            predictors[k] = [left, right]
        result.append(
            (
                story,
                t,
                bool(INTERIOR.fullmatch(a) and INTERIOR.fullmatch(b)),
                predictors,
                direct,
                next_row["human_category_disagreement"]
                - r["human_category_disagreement"],
            )
        )
    return result


def main():
    rows = data()
    story = np.array([r[0] for r in rows])
    titles = sorted(
        set(story),
        key=lambda s: hashlib.sha256(("word-form-check-" + s).encode()).digest(),
    )
    fold = np.array([titles.index(s) % 5 for s in story])
    _, group = np.unique(story, return_inverse=True)
    y = np.array([r[-1] for r in rows])
    interior = np.array([r[2] for r in rows])
    forecast = {
        k: np.array([r[4][k] for r in rows]) for k in ("context", "calibrated", "raw")
    }
    designs = {
        "form_fit": ("form",),
        "form_raw": ("form", "raw"),
        "form_calibrated": ("form", "calibrated"),
        "form_both": ("form", "raw", "calibrated"),
        "form_all": ("form", "raw", "calibrated", "context"),
    }
    for name, cols in designs.items():
        X = np.array([[value for k in cols for value in r[3][k]] for r in rows], float)
        pred = np.full(len(X), np.nan)
        for f in range(5):
            tr = fold != f
            test = ~tr
            mean = X[tr].mean(0)
            std = X[tr].std(0)
            std[std < 1e-7] = 1.0
            x = (X - mean) / std
            a = x[tr]
            coefficients = np.linalg.solve(
                a.T @ a + np.eye(a.shape[1]), a.T @ (y[tr] - y[tr].mean())
            )
            pred[test] = y[tr].mean() + x[test] @ coefficients
        if not np.isfinite(pred).all():
            raise ValueError("Unforecast reading position")
        forecast[name] = pred
    out = {}
    rng = np.random.default_rng(100142)
    for name, mask in (("all", np.ones_like(y, bool)), ("inside_sentence", interior)):
        ids, g = np.unique(group[mask], return_inverse=True)
        n = len(ids)
        B = np.vstack([np.ones(n), rng.multinomial(n, np.full(n, 1 / n), size=8000)])
        valid = np.abs(y[mask]) > 1e-10
        nc = np.bincount(g[valid], minlength=n)
        exact = np.bincount(g, minlength=n)
        measures = {}
        rep = {}
        for k, p in forecast.items():
            guessed = np.sign(p[mask])
            guessed[np.abs(p[mask]) < 1e-10] = 0
            agree = (1 + guessed * np.sign(y[mask])) / 2
            hit = np.bincount(g[valid], weights=agree[valid], minlength=n)
            distance = np.bincount(g, weights=np.abs(y[mask] - p[mask]), minlength=n)
            acc = (B @ hit) / (B @ nc)
            mae = (B @ distance) / (B @ exact)
            rep[k] = (acc, mae)
            measures[k] = {
                "accuracy": float(acc[0]),
                "accuracy_interval": np.quantile(acc[1:], [0.025, 0.975]).tolist(),
                "mae": float(mae[0]),
                "predicted_ties": int((guessed[valid] == 0).sum()),
            }
        deltas = {}
        for current, baseline in (
            ("context", "raw"),
            ("context", "calibrated"),
            ("form_all", "form_both"),
            ("form_all", "form_fit"),
        ):
            x = rep[current][0] - rep[baseline][0]
            m = rep[baseline][1] - rep[current][1]
            deltas[current + "_over_" + baseline] = {
                "accuracy_gain": float(x[0]),
                "accuracy_gain_ci": np.quantile(x[1:], [0.025, 0.975]).tolist(),
                "mae_reduction": float(m[0]),
                "mae_reduction_ci": np.quantile(m[1:], [0.025, 0.975]).tolist(),
            }
        out[name] = {
            "changes": int(mask.sum()),
            "determinate": int(valid.sum()),
            "passages": n,
            "estimates": measures,
            "contrasts": deltas,
        }
    file = ROOT / "form-specificity-forecasts.npz"
    np.savez_compressed(
        file,
        story=story,
        original_word=np.array([r[1] for r in rows]),
        within_sentence=interior,
        held_fold=fold,
        human=y,
        **forecast,
    )
    write_json(
        ROOT / "form-specificity.json",
        {
            "protocol": digest(PLAN),
            "inputs": digest(INPUT),
            "forecast": digest(file),
            "results": out,
        },
    )
    print(json.dumps(out["inside_sentence"]))


if __name__ == "__main__":
    main()
