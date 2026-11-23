"""Spanish transfer of English reading judgments, separating within-article rank and prevalence."""

import json
from pathlib import Path

import numpy as np

from .orx_figstyle import PALETTE, WIDE, figure_grid, save, use_style

OUT=Path(__file__).resolve().parents[2]/"results/reading-validation-transition-external/seahorse"


def main():
    metrics=json.loads((OUT/"spanish-results.json").read_text())
    use_style()
    fig,axes=figure_grid(1,2,width=WIDE,ratio=.50)
    rank,rate=np.ravel(axes)
    tasks=("comprehension","repetition","grammar")
    positions=np.arange(3)[::-1]
    for method,offset,color,label in (("native_adaptive",.09,PALETTE["blue"],"Native"),("form_adaptive",-.09,PALETTE["orange"],"Form")):
        for i,(name,y) in enumerate(zip(tasks,positions,strict=True)):
            v=metrics["conditioned_ranks"][name]["comparisons"][method]["within_article"]
            low,high=100*np.array(v["interval"])
            rank.plot([low,high],[y+offset]*2,c=color,lw=1.9)
            rank.scatter(100*v["estimate"],y+offset,s=29,c=color,label=label if i==0 else None,zorder=2,edgecolors="white",lw=.45)
        values=metrics["held"]["metrics"][method]["tasks"]
        rate.scatter([100*values[name]["mean_predicted_no"] for name in tasks],positions+offset,
                     s=29,c=color,edgecolors="white",lw=.45,label=label)
    values=metrics["held"]["metrics"]["native_adaptive"]["tasks"]
    rate.scatter([100*values[name]["no"]/values[name]["observed"] for name in tasks],positions,
                 c="#282e32",s=29,marker="D",label="Observed")
    rank.axvline(50,ls="--",lw=.8,color="#657079")
    rank.set_xlim(41,100);rank.set_xlabel("Same-article concordance (%)")
    rate.set_xlim(0,20);rate.set_xlabel("Summaries receiving a No response (%)")
    rank.set_title("a  Ordering two drafts",loc="left",fontsize=10,fontweight="bold")
    rate.set_title("b  Predicting the response rate",loc="left",fontsize=10,fontweight="bold")
    for axis in (rank,rate):
        axis.set_ylim(-.58,2.55)
        axis.set_yticks(positions,["Comprehension","Repetition","Grammar"] if axis is rank else [])
        axis.set_ylabel("Reading judgment")
        axis.legend(loc="lower left",ncols=3,frameon=False,fontsize=7)
        axis.grid(axis="x",color="#dce3e6",lw=.55);axis.set_axisbelow(True)
    save(fig,str(OUT/"spanish-transfer"),formats=("pdf","svg","png"))


if __name__=="__main__":
    main()
