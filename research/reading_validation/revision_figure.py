"""Plot distinct human preference and comprehension endpoints with their clustered intervals."""

import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1] / "results"
OUT = ROOT / "reading-validation-comprehension/figure"
sys.path.insert(0, str(ROOT / "reading-validation-local/figure"))
from orx_figstyle import PALETTE, WIDE, save, use_style  # noqa: E402


def source_rows():
    rev = ROOT / "reading-validation-revision/contrasts.json"
    comp = ROOT / "reading-validation-comprehension/scores.json"
    preferences = json.loads(rev.read_bytes())
    answers = json.loads(comp.read_bytes())["primary"]
    rows = []
    for panel, path, data, categories, methods in (
        ("a", rev, preferences, [("opposed_preferences", "Different drafts"),
                                 ("aligned_preferences", "Same draft")],
         [("adaptive", "CogLLM"), ("edit", "Edit model"), ("readme", "ReadMe++")]),
        ("b", comp, answers, [("correctness", "Correct answer"), ("answerability", "Answer given")],
         [("meaning", "CogLLM"), ("edit_meaning", "Edit model"), ("source_word_recall", "Word retention"),
          ("retained_length", "Length retention")]),
    ):
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        for key, condition in categories:
            datum = data[key]
            for ident, method in methods:
                if panel == "a":
                    result, bounds = datum["crossed_95"][ident], "interval95"
                    n, clusters = datum["pairs"], datum["sources"]
                else:
                    result, bounds = datum["model"][ident], "article_bootstrap_95"
                    n, clusters = datum["observed_pairs"], datum["articles"]
                rows.append({
                    "panel": panel, "outcome": condition, "method": method, "estimated_fraction": result["estimate"],
                    "lower95": result[bounds][0], "upper95": result[bounds][1], "informative_pairs": n,
                    "material_clusters": clusters, "results_sha256": checksum,
                })
    OUT.mkdir(exist_ok=True, parents=True)
    with (OUT / "source-data.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def panel(ax, rows, letter, title, label, limits=None):
    methods = list(dict.fromkeys(s["method"] for s in rows))
    outcomes = list(dict.fromkeys(s["outcome"] for s in rows))
    colors, markers = [PALETTE["blue"], PALETTE["red"]], ["o", "s"]
    for r in rows:
        ix = outcomes.index(r["outcome"])
        v, lo, hi = [100 * r[k] for k in ("estimated_fraction", "lower95", "upper95")]
        y = len(methods) - 1 - methods.index(r["method"]) + (0.13 if ix == 0 else -0.13)
        ax.errorbar(v, y, xerr=[[max(0, v - lo)], [max(0, hi - v)]],
                    fmt=markers[ix], color=colors[ix], markerfacecolor=colors[ix] if ix == 0 else "white",
                    markersize=4, markeredgewidth=0.85, elinewidth=1, capsize=1.8, zorder=3)
    if limits is None:
        lower = math.floor(min(r["lower95"] * 100 for r in rows) / 10) * 10
        upper = math.ceil(max(r["upper95"] * 100 for r in rows) / 10) * 10
        limits = min(lower, 40), max(upper, 60)
    ax.set_xlim(limits[0]-1, limits[1]+1)
    ax.set_ylim(-0.59, len(methods)-0.37)
    ax.set_yticks(range(len(methods)), reversed(methods))
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="0.88", lw=0.45, zorder=0)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(label, labelpad=5)
    ax.set_title(title, loc="left", fontsize=8, pad=24)
    ax.text(-0.03, 1.18, letter, ha="right", va="bottom", weight="bold", transform=ax.transAxes)
    handles = [Line2D([0],[0], marker=markers[i], color=colors[i], linewidth=.9, markersize=3.5,
                      markerfacecolor=colors[i] if i == 0 else "white", label=name)
               for i,name in enumerate(outcomes)]
    ax.legend(handles=handles, bbox_to_anchor=(-.04, 1.12), loc="upper left", frameon=False,
              ncols=2, fontsize=6.3, handlelength=1, columnspacing=.8, handletextpad=.3, borderpad=0)


def main():
    rows = source_rows()
    use_style()
    fig = plt.figure(figsize=(WIDE, 2.55))
    left = fig.add_axes([.122, .20, .344, .60])
    right = fig.add_axes([.655, .20, .33, .60])
    panel(left, [r for r in rows if r["panel"] == "a"], "a", "Which draft did judges prefer?",
          "Both preferences selected (%)", (0, 70))
    panel(right, [r for r in rows if r["panel"] == "b"], "b", "Which draft could readers use?",
          "Pair ordered correctly (%)")
    right.axvline(50, color="0.38", lw=.7, linestyle=(0, (3, 3)), zorder=1)
    save(fig, str(OUT / "revision-judgments-and-comprehension"))
    plt.close(fig)


if __name__ == "__main__":
    main()
