"""Show the first disambiguating word and the change in human response forecast accuracy."""
import json
from pathlib import Path
import numpy as np
from matplotlib.lines import Line2D
from .orx_figstyle import PALETTE,WIDE,figure_grid,save,use_style


OUT=Path(__file__).resolve().parents[2]/"results/reading-validation-transition-anchored-reference"


def main():
    use_style();fig,axes=figure_grid(1,2,width=WIDE,ratio=.49)
    left,right=np.ravel(axes)
    src=json.loads((OUT/"dpr-retrospective.json").read_text())["scores"]["identical_through_query_equal_length"]
    methods=(("lexical_steps_update","Elapsed words","#777777","s"),("lexical_attention_update","Attention",PALETTE["red"],"^"),
             ("lexical_semantic_update","Semantic state",PALETTE["blue"],"o"))
    axes_names=["unchanged_signed_preference","first_changed_signed_preference","terminal_signed_preference"]
    for k,name,color,m in methods:
        y=np.array([src[k]["scores"][v] for v in axes_names]);x=np.arange(3)+{"s":-.035,"^":0.,"o":.035}[m]
        left.errorbar(x,y[:,0],yerr=np.array([y[:,0]-y[:,1],y[:,2]-y[:,0]]),fmt=m+"-",color=color,markerfacecolor="white",
                      capsize=2.5,markersize=4.2,lw=1.1,label=name)
    left.set_xticks([0,1,2],["Last\nunchanged","First\nchanged","Final\nword"])
    left.set_xlabel("Reading position");left.set_ylabel("Shift toward the intended referent\n(mean signed log odds)")
    left.axhline(0,color="#8d9397",lw=.8,ls="--",zorder=0);left.grid(axis="y",color="#e6e6e6",lw=.4)
    left.set_title("a  Equal-length authored pairs",loc="left",fontsize=8.2,fontweight="bold")
    left.legend(loc="upper left",frameon=False,fontsize=6.5)
    j=json.loads((OUT/"jones-retrospective.json").read_text())["analyses"]["region_mass"]["interpretations"]["predictors"]
    rows=[("native_prior","At the pronoun","#777777","s"),("native_steps_update","Elapsed words","#777777","s"),
          ("native_attention_update","Attention",PALETTE["red"],"^"),("native_semantic_update","Semantic state",PALETTE["blue"],"o")]
    for i,(k,n,c,m) in enumerate(rows):
        v,a,b=j[k]["choice_logloss"];right.errorbar(v,len(rows)-i,xerr=[[v-a],[b-v]],fmt=m,color=c,markerfacecolor="white",markersize=4.6,capsize=2,lw=1.1)
    right.set_yticks(np.arange(len(rows),0,-1),[x[1] for x in rows]);right.set_ylim(.55,4.5)
    right.grid(axis="x",color="#e6e6e6",lw=.4);right.set_xlabel("Human choice log loss\n(lower is better)")
    right.set_title("b  Physical-event reader choices",loc="left",fontsize=8.2,fontweight="bold")
    save(fig,str(OUT/"referent-evidence"),formats=("pdf","svg","png"))


if __name__=="__main__":main()
