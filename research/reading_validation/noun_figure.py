"""Plot independent human outcomes on their native scales in the noun-phrase experiment."""

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from research.reading_validation.local_forecast import digest
from research.reading_validation.noun_data import OUT

sys.path.insert(0, str(OUT.parent / "reading-validation-local/figure"))
from orx_figstyle import PALETTE, WIDE, save, use_style  # noqa: E402


def main():
    src = OUT / "scores.json"
    data = json.loads(src.read_bytes())["scores"]
    dest = OUT / "figure"
    dest.mkdir(exist_ok=True)
    use_style()
    fig = plt.figure(figsize=(WIDE, 2.7))
    axes = [
        fig.add_axes([0.15, 0.20, 0.32, 0.58]),
        fig.add_axes([0.645, 0.20, 0.32, 0.58]),
    ]
    methods = [
        ("observed", "People"),
        ("extended", "CogLLM"),
        ("surface", "Surface"),
        ("condition", "Condition"),
    ]
    colors = (PALETTE["blue"], PALETTE["red"])
    rows = []
    for ax, (letter, target, title, units, scale) in zip(
        axes,
        [
            (
                "a",
                "difficulty",
                "Perceived sentence difficulty",
                "Change on 1–5 scale",
                1,
            ),
            (
                "b",
                "error",
                "Four-noun cloze errors",
                "Change in errors (percentage points)",
                100,
            ),
        ],
        strict=True,
    ):
        for j, (condition, label) in enumerate(
            (("split", "Split phrase"), ("pseudoword", "Pseudowords"))
        ):
            for k, (method, display) in enumerate(methods):
                z = data[target]["primary"][condition][method]["signed_change"]
                x = scale * z["mean"]
                lo, hi = [scale * v for v in z["ci95"]]
                y = len(methods) - 1 - k + (0.14 if j == 0 else -0.14)
                ax.errorbar(
                    x,
                    y,
                    xerr=[[max(0, x - lo)], [max(0, hi - x)]],
                    fmt="o" if j == 0 else "s",
                    color=colors[j],
                    markerfacecolor=colors[j] if j == 0 else "white",
                    markersize=4,
                    elinewidth=1,
                    capsize=1.8,
                    markeredgewidth=0.85,
                    zorder=3,
                )
                rows.append(
                    {
                        "panel": letter,
                        "endpoint": target,
                        "condition": condition,
                        "estimator": method,
                        "change": x,
                        "lower95": lo,
                        "upper95": hi,
                        "unit": units,
                        "families": z["families"],
                        "scores_sha256": digest(src),
                    }
                )
        ax.axvline(0, color="0.4", linewidth=0.75, linestyle=(0, (3, 3)), zorder=1)
        ax.set_ylim(-0.55, len(methods) - 0.42)
        ax.set_yticks(range(len(methods)), [x[1] for x in reversed(methods)])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="0.88", lw=0.45)
        ax.set_xlabel(units, labelpad=5)
        ax.set_title(title, loc="left", fontsize=8, pad=21)
        ax.text(
            -0.05,
            1.16,
            letter,
            ha="right",
            va="bottom",
            weight="bold",
            transform=ax.transAxes,
        )
        handles = [
            Line2D(
                [0],
                [0],
                marker=marker,
                color=color,
                linewidth=0.85,
                markersize=3.5,
                markerfacecolor=color if i == 0 else "white",
                label=label,
            )
            for i, (marker, color, label) in enumerate(
                zip(("o", "s"), colors, ("Split phrase", "Pseudowords"), strict=True)
            )
        ]
        ax.legend(
            handles=handles,
            bbox_to_anchor=(-0.035, 1.11),
            loc="upper left",
            frameon=False,
            ncols=2,
            fontsize=6.5,
            handlelength=1,
            columnspacing=0.9,
            handletextpad=0.35,
            borderpad=0,
        )
    with (dest / "source-data.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)
    save(fig, str(dest / "perceived-and-cloze"))
    plt.close(fig)


if __name__ == "__main__":
    main()
