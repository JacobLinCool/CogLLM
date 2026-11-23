"""Build the source-data figure for two directly reported reading outcomes."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path("research/results")
WORD = ROOT / "reading-validation-nns"
REVISION = ROOT / "reading-validation-nns-rewrite"


def run():
    x = json.loads((WORD / "scores.json").read_bytes())["reported"]
    y = json.loads((REVISION / "scores.json").read_bytes())
    choice = y["revision_preference"]["conditional_language_model_reference"][
        "non_tie_vote"
    ]["scores"]
    tasks = [
        (
            "a",
            "Which word did more readers flag?",
            "Concordance for two words in the same sentence",
            [
                ("Word frequency", x["rank"]["scores"]["rarity"]),
                ("Supervised lexical model", x["rank"]["scores"]["lexical"]),
                ("Lexical + Qwen", x["rank"]["scores"]["adaptive"]),
            ],
            (0.78, 0.91),
        ),
        (
            "b",
            "Which substitution did a reader prefer?",
            "Agreement with individual choices among non-tie votes",
            [
                ("Supervised lexical model", choice["lexical.adaptive"]),
                ("Joint Qwen readout", choice["contextual.adaptive"]),
                ("Lexical + Qwen surprisal", choice["lexical_plus_surprisal"]),
            ],
            (0.64, 0.73),
        ),
    ]
    for panel, task, title, limits in (
        ("c", "unknown_word", "Reporting an unknown word", (-0.012, 0.080)),
        ("d", "revision_preference", "Choosing a simpler substitution", (-0.40, 2.15)),
    ):
        d = y[task]["logloss"]["oriented_gains"]
        gain = d["contextual.adaptive_gain_vs_contextual.independent"]
        tasks.append(
            (
                panel,
                title,
                "Reduction in log loss from adaptive sharing (millinats)",
                [
                    (
                        "Joint vs independent",
                        {
                            "estimate": gain["estimate"] * 1000,
                            "ci95": [v * 1000 for v in gain["ci95"]],
                        },
                    )
                ],
                limits,
            )
        )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(
        2, 2, figsize=(10.5, 5.0), gridspec_kw={"height_ratios": [1.7, 1.0]}
    )
    fig.subplots_adjust(
        left=0.18, right=0.97, top=0.93, bottom=0.11, wspace=0.74, hspace=0.91
    )
    tab = []
    for a, (panel, title, xlabel, seq, lim) in zip(ax.ravel(), tasks, strict=True):
        a.set_title(title, fontsize=10.8, fontweight="bold", loc="left", pad=9)
        a.text(
            -0.07,
            1.02,
            panel,
            transform=a.transAxes,
            ha="right",
            va="bottom",
            fontweight="bold",
            fontsize=12,
        )
        a.set_xlabel(xlabel, fontsize=8, labelpad=5, loc="left")
        for i, (name, v) in enumerate(seq):
            est = v["estimate"]
            lo, hi = v["ci95"]
            color = "#1b6f73" if ("Qwen" in name or panel in "cd") else "#59687a"
            a.errorbar(
                est,
                i,
                xerr=[[est - lo], [hi - est]],
                fmt="o",
                capsize=3.2,
                color=color,
                ms=5,
                lw=1.5,
            )
            tab.append(
                [
                    panel,
                    title,
                    name,
                    est,
                    lo,
                    hi,
                    "proportion" if panel in "ab" else "millinats",
                ]
            )
        if panel in "ab":
            a.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        else:
            a.axvline(0, color="#666", ls="--", lw=0.9)
        a.set(
            yticks=range(len(seq)),
            yticklabels=[s for s, _ in seq],
            xlim=lim,
            ylim=(len(seq) - 0.45, -0.58),
        )
        a.tick_params(axis="y", width=0, labelsize=8, pad=4)
        a.spines["left"].set_visible(False)
        a.grid(axis="x", color="#dedede", lw=0.55)
        a.set_axisbelow(True)
    fig.savefig(REVISION / "direct-reader-responses.pdf", bbox_inches="tight")
    fig.savefig(REVISION / "direct-reader-responses.png", dpi=210, bbox_inches="tight")
    with (REVISION / "direct-reader-responses.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["panel", "outcome", "model", "estimate", "ci95_low", "ci95_high", "unit"]
        )
        w.writerows(tab)
    plt.close(fig)


if __name__ == "__main__":
    run()
