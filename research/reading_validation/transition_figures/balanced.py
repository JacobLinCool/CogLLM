"""Show the word-level precision gain with original span and title uncertainty."""
import json
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

from .orx_figstyle import PALETTE,WIDE,figure_grid,save,use_style

ROOT=Path(__file__).resolve().parents[2]/"results"
DEST=ROOT/"reading-validation-transition-partial/serving"


def main():
    use_style();data=json.loads((DEST/"balanced-all-words.json").read_text())
    observed=json.loads((ROOT/"reading-validation-transition-t5/type-calibration/qwen-ner-source-matched-movies.json").read_text())["scores"]["words"]
    fig,axs=figure_grid(1,2,width=WIDE,ratio=.45)
    a,b=np.ravel(axs)
    for j,metric in enumerate(("precision","recall","f1")):
        for offset,color,method in [(.10,PALETTE["blue"],"served_marginal"),(-.10,PALETTE["orange"],"ner_surprisal")]:
            y=2-j+offset;lo,hi=data["pooled"]["scores"][method][metric];p=observed["methods"][method][metric]["estimate"]
            a.plot([lo,hi],[y,y],color=color,lw=2.5,solid_capstyle="butt");a.plot(p,y,"o",ms=4.5,color=color,mec="white",mew=.5)
    a.set_yticks([0,1,2],["F1","Recall","Precision"]);a.set_xlim(.50,.87);a.set_ylim(-.5,2.45)
    a.set_xlabel("Word-level performance");a.set_title("a  All possible human span placements",loc="left",fontsize=8,fontweight="bold")
    a.legend(handles=[Line2D([0],[0],color=PALETTE["blue"],lw=2,label="CogLLM"),Line2D([0],[0],color=PALETTE["orange"],lw=2,label="NER + surprisal")],
             loc="lower left",bbox_to_anchor=(0,-.52),ncol=2,frameon=False,fontsize=7)
    bounds=np.array([v["scores"]["paired"]["f1"] for v in data["by_title"].values()])
    bounds=bounds[np.argsort(bounds.mean(1))];positions=np.arange(1,len(bounds)+1)
    for x,(lo,hi) in zip(positions,bounds,strict=True):
        color=PALETTE["blue"] if lo>1e-9 else PALETTE["orange"] if hi< -1e-9 else "#7D898E"
        b.plot([x,x],[lo,hi],color=color,lw=1.6);b.plot(x,(lo+hi)/2,marker=".",ms=1.8,c=color)
    b.axhline(0,color="#4C585E",ls="--",lw=.7);b.set_xlabel("Movie title, ordered by paired F1 difference")
    b.set_ylabel("CogLLM minus NER + surprisal");b.set_title("b  Difference on each of 39 titles",loc="left",fontsize=8,fontweight="bold")
    b.set_xlim(0,40);b.set_xticks([1,10,20,30,39]);a.grid(axis="x",color="#e2e7eb",lw=.45);b.grid(axis="y",color="#e2e7eb",lw=.45)
    save(fig,str(DEST/"balanced-narrative-word-evidence"),formats=("pdf","svg","png"))


if __name__=="__main__":main()
