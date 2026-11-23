"""Draw the paired effect of human-rating transfer from the scored archives."""
import csv
import json
from pathlib import Path
import runpy

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
HELPER=HERE.parent.parent/"reading-validation-local/figure/orx_figstyle.py"
runpy.run_path(str(HELPER))["use_style"]()
score=json.loads((HERE.parent/"scores.json").read_text())["domains"]
repeats=json.loads((HERE.parent/"replications/scores.json").read_text())
fig,axes=plt.subplots(1,2,figsize=(6.75,2.45),layout="constrained",width_ratios=(1.2,1))
data=[]
for j,(study,label,marker,color) in enumerate((("SCoRE","SCoRE","o","#21645f"),("Wikiauto","Wiki-Auto","s","#a44e26"))):
    d=score[study]["paired_rps_difference"]
    sizes=sorted(int(k.split(".")[0]) for k in d if k.endswith(".qwen.matched"))
    for i,size in enumerate(sizes):
        effect=d[f"{size}.qwen.matched"]
        value=-1000*effect["estimate"]
        lo,hi=-1000*np.array(effect["interval95"])[::-1]
        axes[0].errorbar(i+.14*(j-.5),value,yerr=[[value-lo],[hi-value]],fmt=marker,color=color,
                         capsize=2,ms=4,elinewidth=1,label=label if i==0 else None)
        data.append([label,"prespecified",size,"",value,lo,hi])
    reps=repeats["corpora"][study]
    draws=-1000*np.array(reps["paired_rps_by_draw"])
    offset=np.linspace(-.13,.13,len(draws))
    axes[1].scatter(j+offset,draws,s=8,alpha=.55,c=color,marker=marker,linewidths=0)
    value=-1000*reps["mean"]
    lo,hi=-1000*np.array(reps["interval95"])[::-1]
    axes[1].errorbar(j+.22,value,yerr=[[value-lo],[hi-value]],fmt=marker,ms=5,color="#222222",capsize=3)
    for i,x in enumerate(draws,1):
        data.append([label,"new training draw",64,i,x,"",""])
    data.append([label,"two-source bootstrap mean",64,"",value,lo,hi])
for letter,ax in zip("ab",axes):
    ax.axhline(0,color="0.4",lw=.8,ls="--",zorder=0)
    ax.text(-.13,1.04,letter,transform=ax.transAxes,fontweight="bold",size=10)
    ax.spines[["top","right"]].set_visible(False)
axes[0].set(xlim=(-.38,2.37),xticks=[0,1,2],xticklabels=["64","256","All"],ylabel="Reduction in ranked probability score\n(× 0.001; higher is better)",xlabel="Target sentences used for training")
axes[0].legend(loc="upper right",fontsize=7,frameon=False)
axes[1].set(ylabel="Reduction in score (× 0.001)",xticks=[0,1],xticklabels=["SCoRE","Wiki-Auto"],xlim=(-.32,1.42),
            xlabel="32 training draws of 64 sentences each")
for ext in ("svg","pdf"):
    fig.savefig(HERE/("rating-transfer."+ext),bbox_inches="tight")
with (HERE/"source-data.csv").open("w",newline="") as f:
    writer=csv.writer(f)
    writer.writerow(["corpus","measure","training_sentences","draw","rps_gain_x1000","lower95_x1000","upper95_x1000"])
    writer.writerows(data)
