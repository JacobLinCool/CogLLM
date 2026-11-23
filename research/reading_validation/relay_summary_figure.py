"""Draw source-frozen within-reader comparisons on the public summary study."""

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] / "results"
OUT = ROOT / "reading-validation-relay-summary"
sys.path.insert(0, str(ROOT / "reading-validation-local/figure"))
from orx_figstyle import BASELINE, PALETTE, WIDE, save, use_style


def main():
    results = json.loads((OUT / "scores.json").read_bytes())["datasets"]["primary"]
    target = OUT / "figure"
    target.mkdir(exist_ok=True)
    use_style()
    fig = plt.figure(figsize=(WIDE, 2.85))
    left = fig.add_axes([0.195, 0.18, 0.30, 0.64])
    right = fig.add_axes([0.645, 0.18, 0.31, 0.64])
    endpoints = [
        ("simplicity", "Perceived ease"),
        ("coherence", "Coherence"),
        ("understanding", "Fit to knowledge"),
        ("explanation", "Explanation"),
        ("correct", "Quiz accuracy"),
    ]
    data = []
    for j, (key, label) in enumerate(endpoints):
        ordinal = results["frozen"][key]
        individual = results["within_same_summary"][key]
        for panel, ax, delta, model, source, col, shape, name in (
            (
                "a",
                left,
                0.14,
                "plaba_ease",
                ordinal,
                PALETTE["blue"],
                "o",
                "PLABA ease",
            ),
            (
                "a",
                left,
                -0.14,
                "authors_formula",
                ordinal,
                BASELINE,
                "s",
                "Readability formulas",
            ),
            (
                "b",
                right,
                0,
                "prior",
                individual,
                PALETTE["red"],
                "D",
                "Prior familiarity",
            ),
        ):
            score = source["scores"][model]
            val, (low, high) = score["estimate"], score["ci95"]
            ax.errorbar(
                val * 100,
                4 - j + delta,
                xerr=([[100 * (val - low)], [100 * (high - val)]]),
                fmt=shape,
                ms=3.8,
                color=col,
                markerfacecolor=col if shape != "s" else "white",
                ecolor=col,
                lw=1.05,
                capsize=1.9,
                label=name if j == 0 else None,
            )
            data.append(
                {
                    "panel": panel,
                    "endpoint": key,
                    "method": model,
                    "agreement": val,
                    "lower_95": low,
                    "upper_95": high,
                    "unequal_pairs": source["count"],
                    "reader_count": source["reader_count"],
                    "families": source["article_family_count"],
                }
            )
    for letter, ax, heading in [
        ("a", left, "Same reader, different summaries"),
        ("b", right, "Same summary, different readers"),
    ]:
        ax.axvline(50, color=BASELINE, ls="--", lw=0.7, zorder=0)
        ax.set_ylim(-0.5, 4.5)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_yticks([4, 3, 2, 1, 0], [v for _, v in endpoints] if ax is left else [])
        ax.set_xlabel("Concordant comparisons (%)")
        ax.set_ylabel("Human response" if ax is left else "Same-summary response")
        ax.set_title(heading, fontsize=8, pad=22, loc="left")
        ax.text(
            -0.07,
            1.16,
            letter,
            fontsize=9,
            weight="bold",
            va="bottom",
            transform=ax.transAxes,
        )
        ax.legend(
            loc="lower left",
            bbox_to_anchor=(-0.04, 1.005),
            frameon=False,
            ncol=2,
            fontsize=6.5,
            handlelength=1,
            columnspacing=0.6,
            handletextpad=0.3,
            borderpad=0,
        )
    with (target / "source-data.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data[0])
        writer.writeheader()
        writer.writerows(data)
    save(fig, str(target / "lay-reading"))
    plt.close(fig)


if __name__ == "__main__":
    main()
