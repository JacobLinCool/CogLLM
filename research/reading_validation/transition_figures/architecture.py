"""Paired contrasts for word-level transfer of a character-introduction observer."""
import json
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

from .orx_figstyle import PALETTE, WIDE, figure_grid, save, use_style

OUTPUT=Path(__file__).resolve().parents[2]/'results/reading-validation-transition-architecture'
ORDER=(('qwen','Qwen3.5-2B'),('lfm3','LFM3-1.7B-Base'),('gemma4','Gemma 4 E2B'))
COHORTS=(('all','All eligible words',PALETTE['blue'],'o'),('first_person_mention','First PERSON mentions',PALETTE['red'],'s'))


def main():
    scores=json.loads((OUTPUT/'scores.json').read_text())
    use_style();fig,axes=figure_grid(1,2,width=WIDE,ratio=.42,sharey=True)
    axes=np.ravel(axes)
    for ax,comparison,title in zip(axes,['ner','local'],
                                ['A  Context and antecedents vs NER','B  Added antecedent features'],strict=True):
        bounds=[]
        for row,(candidate,name) in enumerate(ORDER):
            for k,(region,label,color,marker) in enumerate(COHORTS):
                item=scores[region]['paired_ap_differences'][f'{candidate}/relations - {candidate}/{comparison}']
                y=len(ORDER)-row-1+(0.13 if k==0 else -0.13)
                lo,hi=item['family_bootstrap_95_interval'];estimate=item['estimate']
                ax.errorbar(estimate,y,xerr=[[estimate-lo],[hi-estimate]],fmt=marker,color=color,
                            markerfacecolor='white',markersize=4,linewidth=1.15,capsize=2.3,zorder=2)
                bounds.extend([lo,hi])
        ax.axvline(0,color='#74808c',linewidth=.85,ls='--',zorder=0)
        ax.set_xlim(min(min(bounds),0)-.02,max(max(bounds),0)+.02)
        ax.set_ylim(-.5,2.6);ax.set_yticks([2,1,0],[name for _,name in ORDER])
        ax.set_xlabel('Difference in average precision')
        ax.set_title(title,loc='left',fontsize=8,fontweight='bold')
    axes[0].set_ylabel('Frozen text backbone')
    handles=[Line2D([],[],color=color,ls='',marker=marker,markerfacecolor='white',label=label,markersize=4)
             for _,label,color,marker in COHORTS]
    fig.legend(handles=handles,loc='outside lower center',ncols=2,frameon=False)
    save(fig,str(OUTPUT/'common-word-effects'),formats=('pdf','svg','png'))


if __name__=='__main__':main()
