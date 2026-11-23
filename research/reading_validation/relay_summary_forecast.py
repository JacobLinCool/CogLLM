"""Build outcome-free text scores from the existing public-summary Colab extraction."""

import argparse
import json
from pathlib import Path

import numpy as np

from research.reading_validation.joint_data import surface
from research.reading_validation.joint_fit import normalized_hidden
from research.reading_validation.local_forecast import WORD, digest, rows
from research.reading_validation.noun_material import MODEL
from research.reading_validation.relay_summary_material import OUT, PACKET, STIM
from research.reading_validation.rewrite_forecast import predictions

NOUN = Path("research/results/reading-validation-noun")


def run(extraction):
    lock = json.loads((OUT / "preregistration.json").read_bytes())
    for k, h in lock["files"].items():
        if digest(Path(k)) != h:
            raise ValueError("Preregistered file changed")
    m = json.loads((extraction / "extraction.json").read_bytes())
    if (
        m["data_sha256"] != PACKET
        or m["forward_calls"] != 91
        or m["worker_sha256"] != digest(Path(__file__).with_name("relay_summary_l4.py"))
    ):
        raise ValueError("Unexpected public summary forward count, packet or worker")
    for k, h in m["files_sha256"].items():
        if digest(extraction / k) != h:
            raise ValueError("Public summary features changed")
    geometry = json.loads((extraction / "geometry.json").read_bytes())
    docs = {r["id"]: r["text"] for r in rows(STIM)}
    studies = json.loads((OUT / "formula.json").read_bytes())["grade"]
    ss = []
    info = []
    lookup = []
    ids = []
    texts = []
    for doc in geometry:
        key = doc["id"]
        text = docs[key]
        ids.append(key)
        texts.append(text)
        known = [s for s in doc["sentences"] if s["rating"] is not None]
        denominator = sum(s["words"] for s in doc["sentences"])
        coverage = sum(s["words"] for s in known) / denominator
        if not known:
            raise ValueError("A study completely lacks usable sentences")
        indexes = []
        for s in known:
            if len(ss) != s["state_row"]:
                raise ValueError("Source sentence state ordering changed")
            ss.append(text[s["start"] : s["end"]])
            info.append(s)
            indexes.append(s["state_row"])
        lookup.append((indexes, coverage))
    terminal = np.load(extraction / "document.npy", mmap_mode="r")
    states = np.load(extraction / "sentences.npy", mmap_mode="r")
    with np.load(MODEL, allow_pickle=False) as source:
        pred = predictions(source, terminal, texts)
        sgrade = predictions(source, states, ss)["adaptive.qwen"].sum(1) + 1
        basis = (normalized_hidden(states) - source["fit.basis.hidden_mean"]) @ source[
            "fit.basis.projection"
        ][:, -24:]
    lex = np.asarray([s["lexical_mean_q90"] for s in info])
    ni = np.column_stack([np.stack([surface(t, "en") for t in ss]), sgrade, lex, basis])
    with np.load(NOUN / "weights.npz", allow_pickle=False) as source:
        noun = (ni - source["extended.center"]) / source["extended.scale"] @ source[
            "extended.coef"
        ] + source["extended.intercept"]
    forecasts = []
    for i, (key, text) in enumerate(zip(ids, texts, strict=True)):
        idx, coverage = lookup[i]
        article = key.removeprefix("relay-summary/")
        used = [info[j] for j in idx]
        words = WORD.findall(text)
        scores = {
            "constant": 0.0,
            "authors_formula": -studies[article]["average"],
            "flesch_kincaid": -studies[article]["flesch_kincaid"],
            "shorter": -len(words),
            "readme": -float(1 + pred["adaptive.qwen"][i].sum()),
            "readme_surface": -float(1 + pred["adaptive.surface"][i].sum()),
            "zero_qwen": -float(pred["zero.qwen"][i]),
            "plaba_ease": float(np.mean([s["rating"]["ease"] for s in used])),
            "plaba_understanding": float(
                np.mean([s["rating"]["understanding"] for s in used])
            ),
            "noun_ease": -float(noun[idx, 0].mean()),
            "noun_correct": -float(noun[idx, 1].mean()),
        }
        count = len(geometry[i]["sentences"])
        simple = [
            float(np.mean([len(w) for w in words])),
            len(words) / count,
            float(np.log1p(len(words))),
        ]
        forecasts.append(
            {
                "article": article,
                "coverage": coverage,
                "included": coverage >= 0.9,
                "features": simple,
                "sentence_count": count,
                "scores": scores,
            }
        )
    target = OUT / "text-scores.jsonl"
    target.write_text(
        "".join(
            json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in forecasts
        )
    )
    receipt = {
        "text_score_sha256": digest(target),
        "program_sha256": digest(Path(__file__)),
        "remote": m,
        "summary_count": len(forecasts),
        "summaries_meeting_coverage": sum(x["included"] for x in forecasts),
        "minimum_certified_word_fraction": min(x["coverage"] for x in forecasts),
        "source_noun_sha256": digest(NOUN / "weights.npz"),
        "source_grade_sha256": digest(MODEL),
        "target_outcomes_opened": False,
    }
    (OUT / "forecast-seal.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        {
            k: receipt[k]
            for k in [
                "summary_count",
                "summaries_meeting_coverage",
                "minimum_certified_word_fraction",
            ]
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction", required=True, type=Path)
    run(parser.parse_args().extraction)
