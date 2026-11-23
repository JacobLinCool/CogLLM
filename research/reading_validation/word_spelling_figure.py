"""Plot exact human spellings and the actual word-end forecast on a geometry-selected passage."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from research.reading_validation.reading_response_figure import (
    BLUE,
    INK,
    ORANGE,
    OUT,
    ROOT,
    clean,
    read,
    start,
)


def main():
    reference = read("reading-validation-manuscript/figure-data.json")
    earlier = Path(
        ".cache/cogllm/reading-validation/cloze-population/external-scored-word-rows.jsonl"
    )
    observations = [json.loads(line) for line in earlier.read_text().splitlines()]
    with np.load(
        ROOT / "reading-validation-spelling-agreement/provo-forecasts.npz"
    ) as f:
        indices = {s: i for i, s in enumerate(f["ids"])}
        predictions = f["forecast"][:, list(f["methods"]).index("conditional")]
    targets = reference["displayed_positions"][:12]
    selected = sorted(
        [
            o
            for o in observations
            if str(o["Text_ID"]) == reference["provo_article"]
            and o["target_ordinal_zero_based"] in targets
        ],
        key=lambda s: s["target_ordinal_zero_based"],
    )
    if [r["target_ordinal_zero_based"] for r in selected] != targets or not all(
        r["normative_denominator_verified"] for r in selected
    ):
        raise ValueError(
            "The a priori passage window differs from the measured word ends"
        )
    samples = [
        {
            "word": r["preceding_print"],
            "source_id": r["id"],
            "people": r["released_nonempty_response_mass"],
            "human": r["human_spelling_disagreement"],
            "model": float(predictions[indices[r["id"]]]),
        }
        for r in selected
    ]
    with (OUT / "spelling-figure-words.csv").open("w", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)

    plt.rcParams.update(
        {
            "font.size": 9.5,
            "font.family": "DejaVu Sans",
            "axes.labelcolor": INK,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(9.3, 6.1), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], wspace=0.12, hspace=0.17)
    top = fig.add_subplot(grid[0, :])
    for field, color, shape, label in (
        ("human", INK, "o-", "Actual spellings from readers"),
        ("model", BLUE, "s--", "One-pass forecast"),
    ):
        top.plot(
            range(len(samples)),
            [100 * s[field] for s in samples],
            shape,
            c=color,
            lw=1.8,
            markersize=4.5,
            label=label,
        )
    top.set(
        xticks=range(len(samples)),
        xticklabels=[s["word"] for s in samples],
        xlim=(-0.3, len(samples) - 0.7),
        ylim=(-3, 104),
        yticks=[0, 25, 50, 75, 100],
        xlabel="Word just read in the original Provo passage",
        ylabel="Pairs predicting different spellings (%)",
    )
    clean(top)
    top.legend(loc="lower left", frameon=False, ncols=2)
    start(
        top, "a", "After each word, how often do two readers give different next words?"
    )
    marks = read("reading-validation-word-marks/scores.json")["arms"][
        "context_certified"
    ]["both"]["ratios"]
    ax = fig.add_subplot(grid[1, 0])
    pairs = [
        ("Correct sign among marks", "directional_precision"),
        ("Sign and size among marks", "large_same_sign_precision"),
        ("Large human changes found", "large_same_sign_recall"),
    ]
    for i, (_, k) in enumerate(pairs):
        y = 2 - i
        score = 100 * marks[k]["estimate"]
        l, h = np.array(marks[k]["paragraph_ci95"]) * 100
        ax.errorbar(
            score, y, xerr=[[score - l], [h - score]], fmt="o", color=BLUE, capsize=3
        )
    ax.set(
        yticks=range(3),
        yticklabels=[a for a, _ in pairs][::-1],
        xlabel="Share of eligible cases (%)",
        xlim=(0, 100),
        ylim=(-0.55, 2.55),
    )
    clean(ax, "x")
    start(ax, "b", "What a highlighted change tells an author")
    ax = fig.add_subplot(grid[1, 1])
    amounts = read("reading-validation-spelling-precision/scores.json")["half_results"]
    labels = [
        ("Another group of people", "peer"),
        ("Source mean", "constant"),
        ("Direct calibration", "direct"),
        ("Conditional spelling", "conditional"),
    ]
    for i, (_, key) in enumerate(labels):
        row = amounts[key]["mae"]
        amount = row["estimate"] * 100
        l, h = np.array(row["stem_ci95"]) * 100
        ax.errorbar(
            amount,
            3 - i,
            xerr=[[amount - l], [h - amount]],
            fmt="D" if key == "peer" else "o",
            color=INK if key == "peer" else ORANGE,
            capsize=3,
        )
    ax.set(
        yticks=range(4),
        yticklabels=[a for a, _ in labels][::-1],
        xlim=(0, 22),
        ylim=(-0.55, 3.55),
        xlabel="Mean absolute difference (percentage points)",
    )
    clean(ax, "x")
    start(ax, "c", "How close is the absolute level on new stems?")
    for ax in fig.axes:
        for text in list(ax.texts):
            if text.get_text() in ("a", "b", "c"):
                letter = text.get_text()
                text.remove()
                ax.set_title(
                    f"{letter}    {ax.get_title(loc='left')}",
                    loc="left",
                    weight="bold",
                    fontsize=10,
                    pad=8,
                    color=INK,
                )
    for ext in ("pdf", "png", "svg"):
        fig.savefig(
            OUT / ("human-word-expectations." + ext),
            dpi=190,
            facecolor="white",
            bbox_inches="tight",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
