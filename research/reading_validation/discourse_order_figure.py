"""Plot human passage scores, external within-domain ranking, and matched changes in sentence order."""

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] / "results"
sys.path.insert(0, str(ROOT / "reading-validation-local/figure"))
from orx_figstyle import PALETTE, use_style

DATA = ROOT / "reading-validation-coherence-order/figure"
METHODS = {
    "representation": "English-only model",
    "form": "Text form",
    "surprisal": "Form + word probability",
    "adaptive": "CogLLM",
    "genre_plus_form": "Genre + form",
    "danish_lexical": "Danish lexical model",
    "official_jaccard": "Official word overlap",
    "augmented_adaptive": "CogLLM + order training",
}


def load(folder, name):
    return json.loads((ROOT / folder / name).read_bytes())


def estimates():
    english = load("reading-validation-coherence-order", "scores.json")["decor"][
        "scores"
    ]
    italian = load("reading-validation-coherence-order", "genre-scores.json")[
        "with_recovered_source_clusters"
    ]
    danish = load("reading-validation-danish-coherence", "scores.json")["domain_equal"][
        "by_model"
    ]
    order = load("reading-validation-italian-interventions", "structural-scores.json")[
        "all"
    ]["passage"]
    extra = load("reading-validation-order-training", "scores.json")
    rows = []
    for panel, armset, metric in (
        (
            "English",
            ("augmented_adaptive", "adaptive", "surprisal", "representation"),
            "logloss",
        ),
        (
            "Italian",
            ("augmented_adaptive", "adaptive", "surprisal", "genre_plus_form"),
            "mse",
        ),
        (
            "Danish",
            ("augmented_adaptive", "adaptive", "surprisal", "danish_lexical"),
            "concordance",
        ),
        (
            "Order",
            ("augmented_adaptive", "adaptive", "surprisal", "official_jaccard"),
            "primary_correct",
        ),
    ):
        for name in armset:
            if panel == "English":
                d = (
                    extra["authentic_human_ratings"]["decor"]["arms"]["new_adaptive"]
                    if name == "augmented_adaptive"
                    else english[name][metric]
                )
            elif panel == "Italian":
                d = (
                    extra["authentic_human_ratings"]["italian"]["arms"]["new_adaptive"]
                    if name == "augmented_adaptive"
                    else italian["scores"][name][metric]
                )
            elif panel == "Danish":
                d = (
                    extra["authentic_human_ratings"]["danish"]["arms"]["new_adaptive"]
                    if name == "augmented_adaptive"
                    else danish[name][metric]
                )
            else:
                d = (
                    extra["source_order"]["all"]["rights"]["arms"]["new_adaptive"]
                    if name == "augmented_adaptive"
                    else order["arms"][name][metric]
                )
            rows.append(
                {
                    "panel": panel,
                    "metric": metric,
                    "method": METHODS[name],
                    "estimate": d["estimate"],
                    "lower95": d["ci95"][0],
                    "upper95": d["ci95"][1],
                }
            )
    return rows


def main():
    rows = estimates()
    DATA.mkdir(exist_ok=True)
    with (DATA / "source-data.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    use_style()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 4.45),
        layout="constrained",
        gridspec_kw={"wspace": 0.1, "hspace": 0.1},
    )
    titles = (
        "English essay boundaries",
        "Italian passage ratings",
        "Danish passage ratings",
        "Italian sentence order",
    )
    labels = (
        "Mean log loss",
        "Mean squared error",
        "Within-domain ordering (%)",
        "Original order preferred (%)",
    )
    for ax, key, title, xlab, letter in zip(
        axes.flat,
        ("English", "Italian", "Danish", "Order"),
        titles,
        labels,
        ("a", "b", "c", "d"),
        strict=True,
    ):
        selected = [r for r in rows if r["panel"] == key]
        pct = key in ("Danish", "Order")
        ax.set_title(title, loc="left", fontweight="bold", fontsize=9, pad=11)
        ax.text(
            -0.03,
            1.21,
            letter,
            transform=ax.transAxes,
            ha="left",
            va="baseline",
            fontweight="bold",
            fontsize=11,
        )
        if pct:
            ax.axvline(50, color="#888888", ls=(0, (3, 2)), lw=0.8)
        yloc = []
        names = []
        for j, row in enumerate(selected):
            pos = len(selected) - 1 - j
            yloc.append(pos)
            names.append(
                row["method"]
                .replace("Form + word probability", "Form + surprisal")
                .replace("CogLLM + order training", "With order training")
            )
            factor = 100 if pct else 1
            score, lo, hi = [
                row[k] * factor for k in ("estimate", "lower95", "upper95")
            ]
            color = PALETTE["blue"] if row["method"].startswith("CogLLM") else "#677381"
            ax.errorbar(
                score,
                pos,
                xerr=[[max(0.0, score - lo)], [max(0.0, hi - score)]],
                fmt="o",
                markersize=4.4,
                markerfacecolor=color if row["method"] == "CogLLM" else "white",
                color=color,
                markeredgewidth=1.1,
                capsize=2,
                elinewidth=1.1,
            )
        ax.set_yticks(yloc, labels=names, fontsize=7)
        ax.set_ylim(-0.6, len(selected) - 0.4)
        ax.grid(False)
        ax.grid(axis="x", color="#e0e0e0")
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel(xlab, labelpad=6)
        if pct:
            ax.set_xlim(
                min(45, min(r["lower95"] * 100 for r in selected)) - 2,
                max(55, max(r["upper95"] * 100 for r in selected)) + 2,
            )
    plt.savefig(DATA / "human-coherence-and-order.pdf", bbox_inches="tight")
    plt.savefig(DATA / "human-coherence-and-order.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
