"""Plot observed and predicted changes in their human measurement units."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT=Path("research/results"); OUT=ROOT/"reading-validation-manuscript"
INK="#1E2D38"; BLUE="#08766E"; ORANGE="#AD561D"; GREY="#5B6870"; GRID="#D8E1DF"


def read(path):
    return json.loads((ROOT/path).read_text())


def clean(ax,grid="y"):
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(colors=INK)
    ax.grid(axis=grid,color=GRID,linewidth=.6)
    ax.set_axisbelow(True)


def start(ax,letter,title):
    ax.set_title(title,loc="left",weight="semibold",fontsize=10.5,pad=7,color=INK)
    ax.text(-.1,1.05,letter,transform=ax.transAxes,fontsize=12,weight="bold",color=INK,ha="right")


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":9,"font.family":"DejaVu Sans","axes.labelcolor":INK,
        "axes.edgecolor":GREY,"pdf.fonttype":42,"svg.fonttype":"none"})
    z=np.load(ROOT/"reading-validation-word-dispersion/provo-word-forecasts.npz")
    article=z["article"];pos=z["pos"]
    runs=[]
    for title in sorted(set(article)):
        ii=np.flatnonzero(article==title);ii=ii[np.argsort(pos[ii])]
        for seg in np.split(ii,np.where(np.diff(pos[ii])!=1)[0]+1):
            if len(seg)>=16:runs.append((title,seg))
    med=np.median([len(seg) for _,seg in runs])
    title,idx=min(runs,key=lambda r:(abs(len(r[1])-med),str(r[0]),pos[r[1][0]]))
    s=read("reading-validation-word-dispersion/scores.json")
    human=read("reading-validation-first-cue/scores.json")
    agree=read("reading-validation-narrative-agreement/window-scores.json")["pooled"]
    fig=plt.figure(figsize=(10.9,7.7),layout="constrained")
    gs=fig.add_gridspec(2,2,height_ratios=[1,1],width_ratios=[1.13,1],hspace=.18,wspace=.15)
    ax=fig.add_subplot(gs[0,0]);clean(ax);start(ax,"a","Readers’ next-word guesses change across a passage")
    x=pos[idx];ax.plot(x,100*z["human"][idx],"-o",color=INK,markersize=3,lw=1.4,label="Actual pairs of responses")
    ax.plot(x,100*z["D_choices"][idx],"--",color=BLUE,lw=1.6,label="Calibrated population forecast")
    ax.set(ylabel="Pairs giving different answers (%)",xlabel="Next-word prompt in an original Provo passage",ylim=(-2,103))
    ax.legend(frameon=False,loc="lower left",fontsize=8)
    ax=fig.add_subplot(gs[0,1]);clean(ax,"x");start(ax,"b","Error in the size of a word-to-word change")
    levels=("raw","original","choices","joint"); labels=("Raw Qwen","Population head","Choice calibration","Choice + change calibration")
    for j,k in enumerate(levels):
        y=len(levels)-1-j;v=s["estimates"][k]["update_mae"];a,b=np.array(v["interval"])*100;m=v["mean"]*100
        ax.errorbar(m,y,xerr=[[m-a],[b-m]],color=BLUE if k in levels[-2:] else GREY,fmt="o",capsize=3)
        ax.text(b+.25,y,f"{m:.1f}",ha="left",va="center",fontsize=8)
    ax.set(yticks=range(4),yticklabels=labels[::-1],xlabel="Mean absolute error (percentage points)",ylim=(-.55,3.55),xlim=(12,25))
    ax=fig.add_subplot(gs[1,0]);clean(ax,"x");start(ax,"c","First reports after a conflicting earlier premise")
    models=human["models"]["strict"]
    entries=[("Readers",human["human_effect"]),("Position only",models["first_cue_position"]["condition_effect"]),
        ("Text + first-cue model",models["first_cue_moment"]["condition_effect"]),
        ("Known location and condition",models["oracle"]["condition_effect"])]
    for j,(label,v) in enumerate(entries):
        y=len(entries)-1-j;m=100*v["estimate"];a,b=np.array(v["interval95"])*100
        ax.errorbar(m,y,xerr=[[m-a],[b-m]],fmt="D" if j==0 else "o",capsize=3,
                    color=INK if j==0 else ORANGE if j<3 else GREY)
    ax.set(yticks=range(4),yticklabels=[i[0] for i in entries][::-1],ylim=(-.55,3.55),
           xlabel="Increase in first reports at the unchanged sentence (points)",xlim=(-2,47))
    ax=fig.add_subplot(gs[1,1]);clean(ax);start(ax,"d","Agreement between human plot annotations")
    xx=np.arange(3)
    for kind,name,style in (("rise","Questions accumulate","-o"),("decline","Questions clear","--s")):
        values=[agree[f"{kind}.{i}"]["F1"] for i in xx]
        ax.plot(xx,values,style,color=BLUE if kind=="rise" else ORANGE,lw=1.6,label=name)
    ax.set(xticks=xx,xticklabels=["Exact line","Within 1 line","Within 2 lines"],ylabel="One-to-one matching F1",
           xlabel="Allowed distance between two teams’ labels",ylim=(0,.52),xlim=(-.15,2.15))
    ax.legend(frameon=False,loc="lower right",fontsize=8)
    for ext in ("pdf","png","svg"):
        fig.savefig(OUT/("observed-reading-changes."+ext),dpi=175,facecolor="white",bbox_inches="tight")
    plt.close(fig)
    (OUT/"figure-data.json").write_text(json.dumps(dict(provo_article=str(title),eligible_runs=len(runs),median_length=float(med),
        displayed_positions=pos[idx].tolist(),human=z["human"][idx].tolist(),model=z["D_choices"][idx].tolist(),
        selection="Median-length verified run of at least 16 consecutive positions, ties settled by article and position."),indent=2)+"\n")


if __name__=="__main__":
    main()
