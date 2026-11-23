"""Show what adding human replacement preferences does to two distinct held-topic outcomes."""
import json
from pathlib import Path

import numpy as np
from .orx_figstyle import PALETTE, WIDE, figure_grid, save, use_style

DEST=Path(__file__).resolve().parents[2]/"results/reading-validation-transition-external/essay-cross-topic"


def main():
    data=json.loads((DEST/"expert-replacement-scores.json").read_text());use_style()
    fig,axs=figure_grid(1,2,width=WIDE,ratio=.43);ax,bx=np.ravel(axs)
    labels=("Original marks only","Replacements only","Lexical, both","Native, both")
    methods=("native_absolute","native_replacement","lexical_joint","native_joint")
    for pos,(label,m) in enumerate(zip(labels,methods,strict=True)):
        d=data["models"][m];color=PALETTE["blue"] if m=="native_joint" else PALETTE["orange"] if m=="lexical_joint" else "#63717B"
        for a,k,factor in [(ax,"expert_edit_preference",100),(bx,"logloss",1)]:
            pt,lo,hi=np.array(d[k])*factor;y=len(methods)-pos-1
            a.plot([lo,hi],[y,y],color=color,lw=1.6,solid_capstyle="round")
            a.plot(pt,y,"o",color=color,ms=5.2,mec="white",mew=.6,zorder=4)
            label_value=f"{pt:.1f}" if factor==100 else f"{pt:.3f}"
            a.text(pt,y+.19,label_value,color=color,ha="center",va="bottom",fontsize=7)
    ax.set_yticks(range(4),labels[::-1]);bx.set_yticks(range(4),[])
    for a in (ax,bx):
        a.grid(axis="x",linewidth=.5,color="#D8DDE1");a.set_ylim(-.3,3.9)
    ax.axvline(50,color="#92999D",ls="--",lw=.8);ax.set_xlim(42,83);ax.set_xticks([45,55,65,75])
    ax.set_title("a  Expert replacement preferred",loc="left",fontsize=9,fontweight="bold")
    ax.set_ylabel("Source supervision");ax.set_xlabel("Agreement on 203 changed passages (%)")
    bx.set_xlim(.51,.72);bx.set_xticks([.52,.56,.60,.64,.68,.72])
    bx.set_title("b  Probability of an expert mark",loc="left",fontsize=9,fontweight="bold")
    bx.set_ylabel("Source supervision");bx.set_xlabel("Log loss on 1,352 original sentences (lower is better)")
    save(fig,str(DEST/"essay-edit-tradeoff"),formats=("pdf","svg","png"))


if __name__=="__main__":main()
