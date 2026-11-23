"""Compare measured and predicted effects in the rotating and nonrotating Kerwer partitions."""

import json
import numpy as np
from .headings import OUT
from .orx_figstyle import PALETTE, WIDE, figure_grid, save, use_style


def main():
    use_style()
    fig, axes = figure_grid(1, 2, width=WIDE, ratio=.44)
    series = (("human", "Human rating", PALETTE["black"]),
              ("frozen.mean", "Frozen sentence head", "#6a7680"),
              ("heading_only.mean", "Heading template", PALETTE["orange"]),
              ("native_and_heading.mean", "Native + headings", PALETTE["blue"]))
    specs = (("crossed-adaptation", "a  Rotating holdout"),
             ("independent-rectangle", "b  Disjoint six-article test"))
    for i, (axis, (file, title)) in enumerate(zip(np.ravel(axes), specs, strict=True)):
        estimates = json.loads((OUT / (file + ".json")).read_text())["estimates"]
        for j,(key, _, color) in enumerate(series):
            r = estimates[key]
            axis.plot(r["ci95"], [3-j]*2, lw=1.8, color=color)
            axis.scatter(r["estimate"], 3-j, s=24, color=color, zorder=3)
        axis.set_title(title, loc="left", size=10, weight="bold")
        axis.set_ylim(-.55,3.55)
        axis.set_yticks(range(4), [s[1] for s in reversed(series)] if i == 0 else [""]*4)
        axis.set_xlim(-.2, 1.45)
        axis.set_xticks([0,.4,.8,1.2])
        axis.set_xlabel("Heading benefit (1–8 scale)")
        axis.set_ylabel("Observation or model")
        axis.axvline(0, lw=.8, color="#707b80")
        axis.grid(axis="x", color="#dce2e0", linewidth=.55)
        axis.set_axisbelow(True)
    save(fig, str(OUT/"article-adaptation"), formats=("pdf","svg","png"))


if __name__ == "__main__":
    main()
