"""Plot the deployed learner model's fixed probability audit with its evidence counts."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

ROOT = (
    Path(__file__).resolve().parents[1] / "results/reading-validation-nns/calibration"
)
TEAL, INK, GREY = "#13766f", "#202d2a", "#8d9898"


def panel(ax, data, labels, title, max_rate, unit_label, counts):
    n = len(data)
    y = np.arange(n)[::-1]
    for at, entry in zip(y, data, strict=True):
        pred, obs, lo, hi = (
            float(entry[k])
            for k in ("predicted", "observed", "observed_lo", "observed_hi")
        )
        ax.plot([pred, obs], [at - 0.12, at + 0.12], color="#b9c2bf", lw=1.0, zorder=1)
        ax.errorbar(
            obs,
            at + 0.12,
            xerr=[[obs - lo], [hi - obs]],
            fmt="o",
            color=TEAL,
            ms=4.5,
            capsize=2.8,
            elinewidth=1.2,
            zorder=2,
        )
        ax.scatter(
            pred,
            at - 0.12,
            s=20,
            marker="D",
            facecolor="white",
            edgecolor=INK,
            lw=1,
            zorder=3,
            clip_on=False,
        )
    ax.set_yticks(y, labels)
    ax.set_ylim(-0.6, n - 0.1)
    ax.set_xlim(0, max_rate)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=18)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#dce2df", linewidth=0.65)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("Share of unknown-word reports", labelpad=8)
    for at, count in zip(y, counts, strict=True):
        ax.text(
            1.02,
            at,
            f"{count:,}",
            transform=ax.get_yaxis_transform(),
            fontsize=8,
            color=INK,
            va="center",
        )
    ax.text(
        1.02,
        y[0] + 0.55,
        unit_label,
        transform=ax.get_yaxis_transform(),
        fontsize=7.6,
        color=INK,
        va="center",
    )
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GREY)
    ax.tick_params(axis="y", length=0, pad=4)


def run():
    with (ROOT / "source-data.csv").open() as f:
        entries = list(csv.DictReader(f))
    risk = [r for r in entries if r["group"] == "pooled" and r["risk_bin"] != "overall"]
    cefr = [
        r
        for r in entries
        if r["risk_bin"] == "overall"
        and r["group"] in ("A1", "A2", "B1", "B2", "C1", "C2")
    ]
    with plt.rc_context(
        {
            "font.size": 9,
            "font.family": "DejaVu Sans",
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        fig, (a, b) = plt.subplots(1, 2, figsize=(7.45, 3.55))
        fig.subplots_adjust(left=0.10, right=0.86, top=0.78, bottom=0.28, wspace=0.58)
        panel(
            a,
            risk,
            [r["risk_bin"].replace("<", "Below ") for r in risk],
            "a   Forecast range",
            0.8,
            "Word positions",
            [int(r["judged_cells"]) for r in risk],
        )
        a.set_xticks(np.arange(0, 0.81, 0.2))
        panel(
            b,
            cefr,
            [r["group"] for r in cefr],
            "b   Reported English proficiency",
            0.18,
            "Responses",
            [int(r["responses"]) for r in cefr],
        )
        b.set_xticks(np.arange(0, 0.181, 0.05))
        handles = [
            Line2D(
                [],
                [],
                marker="D",
                markerfacecolor="white",
                markeredgecolor=INK,
                markersize=5,
                ls="none",
                label="Mean model estimate",
            ),
            Line2D(
                [],
                [],
                marker="o",
                color=TEAL,
                markersize=5,
                linewidth=1.1,
                label="Observed share and 95% interval",
            ),
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.48, 0.04),
        )
        for extension in ("png", "pdf"):
            fig.savefig(
                ROOT / ("reported-word-difficulty." + extension),
                dpi=240,
                facecolor="white",
            )
        plt.close(fig)


if __name__ == "__main__":
    run()
