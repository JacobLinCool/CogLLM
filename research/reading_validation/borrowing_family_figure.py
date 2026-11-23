"""Plot paired material-bootstrap calibration results and supply the numerical figure data."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

BASE = Path("research/results/reading-validation-family-bootstrap")
OUT = BASE / "figure"
BLUE = "#255F86"
GREEN = "#08765D"
DARK = "#17232E"
NEUTRAL = "#66727D"
DISPLAY = {
    "ambistory": "AmbiStory",
    "blott": "Blott",
    "clear": "CLEAR",
    "complex": "CompLex",
    "context-readback": "Contextual readback",
    "currie": "Currie",
    "cwi": "Complex words",
    "czech": "Czech garden paths",
    "first-readback": "First readback",
    "frick": "Frick",
    "kerwer": "Kerwer",
    "oliver": "Oliver",
    "onestop": "OneStopQA",
    "onestop-rereading": "OneStop rereading",
    "plaba": "PLABA–EVAL",
    "race": "RACE",
    "rodger": "Rodger",
    "scarecrow": "SCARECROW",
    "studentannotations": "StudentAnnotations",
    "textcomplexityde": "TextComplexityDE",
    "turkish": "Turkish comparisons",
    "yadav": "Yadav",
    "zhan": "Zhan",
}


def load(path):
    return json.loads(path.read_bytes())


def across():
    rows = []
    for split, label in [
        ("historical", "First partition"),
        ("fresh", "Second partition"),
        ("third", "Third, original roles"),
    ]:
        audit = load(BASE / split / "audit.json")
        for ref in ("midpoint", "map"):
            result = audit["effects"]["over_" + ref]
            rows.append(
                (
                    label,
                    ref,
                    result["gain"],
                    *result["two_level_simultaneous_ci"],
                    audit["source_n"],
                    "source and material",
                )
            )
    for split, key, label in [
        ("global", "equal_sources", "Third, global roles"),
        ("global", "equal_linked_components", "Third, linked sources"),
        ("confirmation", "equal_linked_components", "Fourth, linked sources"),
    ]:
        report = load(BASE / split / "scores.json")[key]
        for ref in ("midpoint", "map"):
            result = report["source_and_material"]["over_" + ref]
            rows.append(
                (
                    label,
                    ref,
                    result["gain"],
                    *result["joint_source_material_simultaneous_ci"],
                    report["units"],
                    "linked group and material"
                    if "linked" in key
                    else "source and material",
                )
            )
    return rows


def within():
    data = load(BASE / "global" / "scores.json")["by_source"]
    rows = []
    for s, meta in data.items():
        point = (
            meta["seed_mean_nll"]["midpoint"]
            - meta["seed_mean_nll"]["family_bootstrap"]
        )
        pair = (meta["material_intervals"] or {}).get("over_midpoint") or [None, None]
        rows.append((s, point, *pair, meta["families"]))
    return sorted(rows, key=lambda q: q[1], reverse=True)


def style(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#b9c2c9")
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.tick_params(axis="x", color="#a8b2ba", labelcolor=NEUTRAL, length=3)
    ax.axvline(0, lw=0.85, color="#515e68", ls=(0, (3, 3)), zorder=0)


def make():
    OUT.mkdir(exist_ok=True)
    rows_a = across()
    rows_b = within()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelsize": 8,
            "axes.labelcolor": DARK,
            "text.color": DARK,
        }
    )
    fig = plt.figure(figsize=(9.2, 6.35), layout=None)
    grid = fig.add_gridspec(
        1,
        2,
        left=0.174,
        right=0.976,
        top=0.875,
        bottom=0.19,
        width_ratios=(1, 1),
        wspace=0.86,
    )
    ax, by = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    style(ax)
    style(by)
    categories = list(reversed(list(dict.fromkeys(row[0] for row in rows_a))))
    labels = {
        k: v
        for k, v in zip(categories, range(len(categories) - 1, -1, -1), strict=True)
    }
    for name, kind, value, lo, hi, _, _ in rows_a:
        y = labels[name] + (0.13 if kind == "midpoint" else -0.13)
        color, marker = (GREEN, "o") if kind == "midpoint" else (BLUE, "s")
        ax.plot(
            [1000 * lo, 1000 * hi], [y, y], lw=1.5, color=color, solid_capstyle="round"
        )
        ax.plot(
            1000 * value,
            y,
            marker=marker,
            ms=5.1,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.5,
        )
    ax.set_yticks(list(labels.values()), list(labels.keys()))
    ax.set_ylim(-0.75, len(labels) - 0.25)
    ax.axhline(3.51, color="#d9dfe5", lw=0.75)
    ax.axhline(0.49, color="#d9dfe5", lw=0.75)
    ax.set_xlabel(
        "Decrease in held-out NLL\n(10⁻³ nats per recorded response)", labelpad=6
    )
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color=GREEN,
                lw=1.4,
                label="Compared with equal mixing",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color=BLUE,
                lw=1.4,
                label="Compared with target MAP",
            ),
        ],
        loc="lower left",
        bbox_to_anchor=(-0.06, -0.31),
        frameon=False,
        fontsize=8,
        handlelength=1.9,
    )
    ax.text(
        -0.01,
        1.075,
        "A   Gains across material splits",
        transform=ax.transAxes,
        fontsize=10.7,
        fontweight="bold",
    )
    ax.text(
        -0.01,
        1.023,
        "Two-comparison simultaneous 95% intervals",
        transform=ax.transAxes,
        fontsize=7.8,
        color=NEUTRAL,
    )
    ys = np.arange(len(rows_b))[::-1]
    for n, (s, point, lo, hi, families) in enumerate(rows_b):
        yy = ys[n]
        if lo is not None:
            by.plot(
                [lo * 1000, hi * 1000],
                [yy, yy],
                color="#86A99E",
                lw=1.05,
                solid_capstyle="round",
            )
        by.plot(
            point * 1000,
            yy,
            marker="o",
            ms=4.7,
            markeredgecolor=GREEN,
            markeredgewidth=0.75,
            markerfacecolor=GREEN if families > 1 else "white",
        )
    by.set_yticks(
        ys, [f"{DISPLAY[s]}  ({nf})" for s, _, _, _, nf in rows_b], fontsize=7.1
    )
    by.set_ylim(-0.8, len(rows_b) - 0.1)
    by.set_xlabel(
        "Decrease versus equal mixing\n(10⁻³ nats per recorded response)", labelpad=6
    )
    by.text(
        -0.01,
        1.075,
        "B   Which sources shifted?",
        transform=by.transAxes,
        fontsize=10.7,
        fontweight="bold",
    )
    by.text(
        -0.01,
        1.023,
        "Globally separated third split · 95% material intervals",
        transform=by.transAxes,
        fontsize=7.8,
        color=NEUTRAL,
    )
    by.text(
        -0.01,
        -0.245,
        "Parentheses give held material families.\nOpen circles denote a single observed family.",
        transform=by.transAxes,
        fontsize=7.8,
        color=NEUTRAL,
        va="top",
    )
    for extension in ("pdf", "png"):
        fig.savefig(
            OUT / f"adaptive-study-borrowing.{extension}",
            dpi=220,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.16,
        )
    plt.close(fig)
    with (OUT / "adaptive-study-borrowing.csv").open("w", newline="") as stream:
        w = csv.writer(stream)
        w.writerow(
            [
                "panel",
                "partition_or_source",
                "control",
                "gain_nats",
                "low",
                "high",
                "unit_count",
                "interval_unit",
            ]
        )
        for row in rows_a:
            w.writerow(["A", *row])
        for s, p, l, h, n in rows_b:
            w.writerow(["B", s, "midpoint", p, l, h, n, "material"])
    print(
        "wrote",
        len(rows_a) + len(rows_b),
        "estimates, raster and vector figure",
        flush=True,
    )


if __name__ == "__main__":
    make()
