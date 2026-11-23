"""Show the observed and forecast change from adding section headings to identical body text."""

import json
from pathlib import Path
import numpy as np
from .orx_figstyle import PALETTE, WIDE, figure_grid, save, use_style


OUT = Path(__file__).resolve().parents[2] / "results/reading-validation-local/native-headings"


def main():
    scores = json.loads((OUT / "scores.json").read_text())
    use_style()
    fig, axes = figure_grid(1, 2, width=WIDE, ratio=.53)
    avg, item = np.ravel(axes)
    colors = (PALETTE["black"], PALETTE["blue"], PALETTE["orange"], "#777777")
    models = (scores["native_vs_body"]["human"], scores["native_vs_body"]["native"],
              scores["native_vs_projected"]["base"], scores["native_vs_body"]["base"])
    for i, (result, color) in enumerate(zip(models, colors)):
        v = result["estimate"]
        avg.plot(result["ci95"], [3 - i] * 2, color=color, lw=2)
        avg.scatter(v, 3-i, s=24, color=color, zorder=3)
    avg.set_yticks(range(4), ["Identical-body null", "Projected states", "Deployed head", "Human rating"])
    avg.axvline(0, color="#a0a8aa", lw=.8)
    avg.set_ylim(-.65,3.65)
    avg.set_title("a  Mean change across 12 articles", loc="left", size=9.2, weight="bold")
    avg.set_xlabel("Change in rating (1–8 scale)")
    groups = sorted(scores["families"], key=lambda x: x["human"])
    rank = np.arange(1, len(groups) + 1)
    for key, label, color, marker in (("human", "Human", colors[0], "o"),
                                      ("native", "Deployed", colors[1], "s"),
                                      ("projected", "Projected", colors[2], "^")):
        item.scatter(rank, [r[key] for r in groups], label=label, s=23, marker=marker, color=color, zorder=3)
    item.axhline(0, lw=.7, color="#818c91")
    item.set_xticks([1,4,8,12])
    item.set_xlabel("Article rank by observed benefit")
    item.set_ylabel("Change in rating (1–8 scale)")
    item.legend(fontsize=6.5, loc="upper left", frameon=False)
    item.set_title("b  Individual article contrasts", loc="left", size=9.2, weight="bold")
    for axis in (avg, item):
        axis.grid(axis="x" if axis is avg else "y", color="#e3e6e5", linewidth=.5)
        axis.set_axisbelow(True)
    save(fig, str(OUT / "heading-benefit"), formats=("pdf", "svg", "png"))


if __name__ == "__main__":
    main()
