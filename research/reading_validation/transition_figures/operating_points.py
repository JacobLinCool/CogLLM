"""Plot author-visible false alarms and detection against the sealed human judgments."""
import csv
import json
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

from .orx_figstyle import PALETTE, WIDE, figure_grid, save, use_style

RESULTS=Path(__file__).resolve().parents[2]/'results/reading-validation-transition-external'
METHODS=(('integration','Article readout',PALETTE['blue'],'-','o'),
         ('lexical_surprisal','Word surprisal + lexical controls',PALETTE['red'],'--','^'))


def roc(rows, method):
    ordered=sorted(rows,key=lambda r:-float(r[method]))
    p=np.array([float(r[method]) for r in ordered]);y=np.array([float(r['y']) for r in ordered]);n=np.array([float(r['total']) for r in ordered])
    ends=np.r_[np.flatnonzero(np.diff(p)!=0),len(p)-1]
    recall=np.r_[0,np.cumsum(y)[ends]/y.sum()];fpr=np.r_[0,np.cumsum(n-y)[ends]/(n-y).sum()]
    return 100*fpr,100*recall


def main():
    use_style();scores=json.loads((RESULTS/'scores.json').read_text());rows=list(csv.DictReader((RESULTS/'heldout-scored.csv').open()))
    fig,axes=figure_grid(1,2,width=WIDE,ratio=.52,sharey=True)
    for ax,code,key,label in zip(axes,['cohesentia','decor'],['cohesentia_per_rater','decor'],['a   Short stories','b   Student essays'],strict=True):
        subset=[r for r in rows if r['source']==code];ax.plot([0,100],[0,100],color='#a2a2a2',lw=.7,ls=':',zorder=0)
        for arm,name,color,linestyle,marker in METHODS:
            x,y=roc(subset,arm);ax.step(x,y,where='post',color=color,ls=linestyle,lw=1.45)
            metrics=scores[key]['models'][arm];f=metrics['false_alarm_rate'];r=metrics['sensitivity']
            xpos,ypos=f['estimate']*100,r['estimate']*100
            ax.errorbar(xpos,ypos,xerr=np.abs(np.array(f['family_bootstrap_95_interval'])[:,None]*100-xpos),
                        yerr=np.abs(np.array(r['family_bootstrap_95_interval'])[:,None]*100-ypos),
                        marker=marker,color=color,markerfacecolor='white',markeredgewidth=1.,ms=4.4,lw=.75,capsize=2,zorder=3)
        ax.set(xlim=(0,100),ylim=(0,100),xticks=[0,25,50,75,100],yticks=[0,25,50,75,100],
               xlabel='False alerts per 100 unmarked readings')
        ax.annotate(label,xy=(0,1),xycoords='axes fraction',xytext=(0,6),textcoords='offset points',va='bottom',weight='bold')
    axes[0].set_ylabel('Detections per 100 human-marked readings')
    handles=[Line2D([],[],color=c,ls=ls,marker=mark,markerfacecolor='white',label=n,ms=4) for _,n,c,ls,mark in METHODS]
    fig.legend(handles=handles,loc='outside lower center',ncols=2,frameon=False)
    save(fig,str(RESULTS/'reader-transition-operating-points'))


if __name__=='__main__':main()
