"""Compare matched English comprehensibility judgments and isolate priors about the generating system."""
import json
from pathlib import Path
import numpy as np
from .orx_figstyle import PALETTE,WIDE,figure_grid,save,use_style

OUT=Path(__file__).resolve().parents[2]/"results/reading-validation-transition-external/seahorse"


def main():
    score=json.loads((OUT/"scores.json").read_text());rank=json.loads((OUT/"ranking-sensitivity.json").read_text())
    use_style()
    fig,axes=figure_grid(1,2,width=WIDE,ratio=.50);a,b=np.ravel(axes)
    keys=["metadata","frozen_native_raw","frozen_form_raw","form","native","character_native"]
    names=["System and corpus","Earlier news states","Earlier news form","Summary form","Summary states","States + characters"]
    palette=["#657079","#a9bfc9","#c3c5c7",PALETTE["orange"],PALETTE["blue"],PALETTE["green"]]
    for i,(name,label,color) in enumerate(zip(keys,names,palette,strict=True)):
        y=len(keys)-1-i
        for axis,measure in ((a,"within_article"),(b,"within_corpus_and_generator")):
            val=score["scores"][name]["within_article"] if axis is a else rank["ranks"][name][measure]
            point=100*val["estimate"];low,hi=np.array(val["interval"])*100
            axis.plot([low,hi],[y,y],lw=1.9,color=color,solid_capstyle="round")
            axis.scatter(point,y,s=33,color=color,edgecolor="white",linewidth=.6,zorder=3)
    for axis in (a,b):
        axis.axvline(50,linestyle="--",color="#687479",linewidth=.8)
        axis.grid(axis="x",color="#dce3e6",linewidth=.6);axis.set_axisbelow(True)
        axis.set_ylim(-.58,5.65);axis.set_yticks(range(len(names)),names[::-1] if axis is a else [])
    a.set_ylabel("Observer"); b.set_ylabel("Observer")
    a.set_xlim(47,94);a.set_xticks([50,60,70,80,90]);b.set_xlim(47,81);b.set_xticks([50,60,70,80])
    a.set_title("a  Same source article",loc="left",fontsize=10,fontweight="bold")
    b.set_title("b  Same system and dataset",loc="left",fontsize=10,fontweight="bold")
    a.set_xlabel("More difficult wording ranked higher (%)")
    b.set_xlabel("Concordance within generator and corpus (%)")
    save(fig,str(OUT/"summary-understanding"),formats=("pdf","svg","png"))


if __name__=="__main__":main()
