"""Plot the original human probe times and same-story changes in attention to the old goal."""
import json
from pathlib import Path
import numpy as np
from .orx_figstyle import PALETTE,WIDE,figure_grid,save,use_style

OUT=Path(__file__).resolve().parents[2]/"results/reading-validation-transition-goal-access"


def main():
    data=json.loads((OUT/"scores.json").read_text())
    use_style();fig,axes=figure_grid(1,2,width=WIDE,ratio=.52);a,b=np.ravel(axes)
    y=np.arange(4)[::-1];human=data["original_reported_human_ms"]
    labels=["First paragraph","Old goal repeated","Neutral event","New goal"]
    points=[human[k] for k in ("initial","rementioned","neutral","new")]
    a.hlines(y,690,points,color="#c9d7de",linewidth=1.3);a.scatter(points,y,c=[PALETTE["blue"]]*3+[PALETTE["red"]],s=29,zorder=3)
    for xx,yy in zip(points,y,strict=True):a.text(xx+4,yy,str(xx),va="center",fontsize=7)
    a.set_yticks(y,labels);a.set_ylim(-.55,3.6);a.set_xlim(690,940);a.set_xticks([700,750,800,850,900])
    a.set_title("a  Human recognition",loc="left",fontsize=10,fontweight="bold")
    a.set_xlabel("Time to recognize the old goal word (ms)")
    meta=json.loads((OUT/"lexical-sensitivity.json").read_text())
    with np.load(OUT/"model-item-scores.npz",allow_pickle=False) as w:
        raw=w["all.native"];rec=w["all.recency"]
    delta=(raw-rec)[:,2,:]-(raw-rec)[:,3,:];e=delta[:,0]-delta[:,1:].mean(1)
    points=np.sort(e);level=np.arange(len(e));b.axvline(0,lw=.85,color="#647078",ls="--")
    b.hlines(level,0,points,color="#c6d6dc",lw=.75)
    b.scatter(points,level,c=np.where(points>0,PALETTE["blue"],PALETTE["red"]),s=10,zorder=3)
    d=data["model_family_measurements"]["all"]["probe_minus_control_words"]["native_minus_recency"]["neutral_minus_new"]
    b.plot(d["family_interval"],[-4,-4],color="#252a2d",lw=2.1);b.scatter(d["mean"],-4,c="#252a2d",marker="D",s=19)
    b.set_yticks([-4,0,15,31],["Mean","1","16","32"]);b.set_ylim(-6,33)
    b.set_xlabel("Lower old-goal access after a new goal\n(neutral minus new, adjusted log units)")
    b.set_ylabel("Story, ordered by effect")
    b.set_title("b  Frozen model retrieval",loc="left",fontsize=10,fontweight="bold")
    for axis in (a,b):axis.grid(axis="x",color="#e0e4e6",lw=.55);axis.set_axisbelow(True)
    save(fig,str(OUT/"goal-accessibility"),formats=("pdf","svg","png"))

if __name__=="__main__":main()
