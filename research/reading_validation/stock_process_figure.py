"""Compare source-specific unresolved questions with article-only predictions on every original line."""

import csv
import json

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from research.reading_validation.memory_data import sha
from research.reading_validation.narrative_course_data import WORK
from research.reading_validation.narrative_survival_fit import OUT as SURVIVAL
from research.reading_validation.stock_process_run import OUT
from research.reading_validation.transition_figures.orx_figstyle import use_style


def run():
    use_style()
    forecasts = OUT / "forecast"
    seal = json.loads((forecasts / "seal.json").read_bytes())
    previous = SURVIVAL / "forecast"
    previous_seal = json.loads((previous / "seal.json").read_bytes())
    if sha(previous / "predictions.npz") != previous_seal["files"]["predictions.npz"]:
        raise ValueError("Duration comparator differs from the previously graded forecast.")
    paper = json.loads((OUT / "scores.json").read_bytes())
    figure, axes = plt.subplots(2, 2, figsize=(8.0, 5.1), layout="constrained")
    series = [
        ("AA", "#324f6d", "-", "First annotation team"),
        ("JE", "#58778f", ":", "Second annotation team"),
        ("duration", "#ac642a", "--", "Elapsed duration"),
        ("count", "#007361", "-", "Conserved count model"),
    ]
    records = []
    with np.load(previous / "predictions.npz") as old, np.load(WORK / "response.npz") as human:
        for letter, (name, digest), axis in zip("abcd", sorted(seal["stories"].items()), axes.flat, strict=True):
            source = forecasts / (name + ".npz")
            if sha(source) != digest:
                raise ValueError(f"Story forecast changed after scoring: {name}")
            with np.load(source) as measured:
                curves = {"duration": old[name + ".duration.stock"], "count": measured["count.stock"],
                          "openings": measured["count.birth"], "closures": measured["count.event"].sum(axis=1)}
            x = np.arange(1, len(curves["count"]) + 1)
            errors = []
            for label, color, style, _ in series:
                if label in ("count", "duration"):
                    axis.plot(x, curves[label], color=color, ls=style, lw=1.6, zorder=3)
                    continue
                if name + "." + label not in human:
                    continue
                observations = human[name + "." + label]
                low = (observations == 1).sum(axis=1)
                high = (observations != 0).sum(axis=1)
                curves[label + "_confirmed"], curves[label + "_possible"] = low, high
                error = curves["count"] - np.clip(curves["count"], low, high)
                errors.append(np.mean(error**2))
                axis.fill_between(x, low, high, step="mid", color=color, alpha=.16, lw=0)
                axis.step(x, low, where="mid", color=color, ls=style, lw=.95)
            if not np.isclose(np.mean(errors), paper["stories"][name]["count"]["count_interval_mse"]):
                raise ValueError("Plotted annotation bound differs from the primary score.")
            if not np.allclose(np.diff(curves["count"], prepend=0), curves["openings"] - curves["closures"]):
                raise ValueError("Plotted attributed closures do not conserve outstanding questions.")
            for i in range(len(x)):
                records.append(dict(story=name, line=i+1, **{k: float(v[i]) for k, v in curves.items()}))
            axis.set_title(f"{letter}   {name}", loc="left", fontweight="bold")
            axis.set_xlim(1, x[-1])
            axis.set_ylim(bottom=0)
            axis.spines[["right", "top"]].set_visible(False)
            if letter in "cd":
                axis.set_xlabel("Reading line in the original story")
            if letter in "ac":
                axis.set_ylabel("Outstanding plot questions")
    handles = [Line2D([], [], color=color, ls=style, label=caption) for _, color, style, caption in series]
    figure.legend(handles=handles, loc="outside lower center", ncols=2, frameon=False, fontsize=8)
    for ext in ("png", "pdf"):
        figure.savefig(OUT / ("human-and-estimated-questions." + ext), dpi=225, bbox_inches="tight")
    plt.close(figure)
    with (OUT / "human-and-estimated-questions.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=["story", "line", "AA_confirmed", "AA_possible",
                                "JE_confirmed", "JE_possible", "duration", "count", "openings", "closures"])
        writer.writeheader()
        writer.writerows(records)
    print(f"Verified {len(records)} original-line estimates against the graded bounds and release curves.")


if __name__ == "__main__":
    run()
