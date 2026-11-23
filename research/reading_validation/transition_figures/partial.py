"""Draw sampling uncertainty and human-location uncertainty on separate axes."""
import json
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

from .orx_figstyle import PALETTE, WIDE, figure_grid,save,use_style

OUT=Path(__file__).resolve().parents[2]/'results/reading-validation-transition-partial'
MODELS=(('qwen','Qwen3.5-2B'),('lfm3','LFM3-1.7B-Base'),('gemma4','Gemma 4 E2B'))
COMPARISONS=(('segment_unique','Discard ambiguous segments','s',PALETTE['blue']),
             ('word_identified','Use only identifiable words','o',PALETTE['red']))


def main():
    use_style();fig,axes=figure_grid(1,2,width=WIDE,ratio=.47,sharey=True)
    a,b=np.ravel(axes)
    for i,(candidate,name) in enumerate(MODELS):
        y=2-i
        r=json.loads((OUT/candidate/'scores.json').read_text())
        for j,(key,label,marker,color) in enumerate(COMPARISONS):
            k=r['paired_movie_differences']['set_marginal_minus_'+key]['all_observable_set_nll']
            val=-1000*k['estimate'];lo,hi=-1000*np.asarray(k['family_bootstrap_95_interval'])[::-1]
            a.errorbar(val,y+.115*(-1)**j,xerr=[[val-lo],[hi-val]],fmt=marker,color=color,
                       markerfacecolor='white',markersize=4.2,capsize=2,linewidth=1.25)
        exact=json.loads((OUT/candidate/'identification.json').read_text())
        low,high=exact['paired_extrema']['versus_strong_ner']['sharp_all_word_difference']
        b.hlines(y,low,high,color=PALETTE['blue'],lw=2.1)
        b.plot([low,high],[y,y],marker='|',markersize=7,ls='none',color=PALETTE['blue'])
    for ax in (a,b):
        ax.axvline(0,color='#656e75',linestyle='--',linewidth=.8,zorder=0);ax.set_ylim(-.45,2.5)
        ax.set_yticks([2,1,0],[j[1] for j in MODELS]);ax.grid(axis='x',color='#e2e7eb',linewidth=.45)
    a.set_title('a  Learning from ambiguous marks',loc='left',fontsize=8,fontweight='bold')
    b.set_title('b  Ranking against NER and surprisal',loc='left',fontsize=8,fontweight='bold')
    a.set_ylabel('Frozen text backbone')
    a.set_xlim(-.13,5.35);a.set_xlabel('Reduction in log loss per 1,000 words\n95% movie bootstrap interval')
    b.set_xlim(-.122,.22);b.set_xlabel('Difference in average precision\nExact range over allowed human locations')
    handles=[Line2D([],[],color=c,marker=m,ls='',markerfacecolor='white',label=l,markersize=4.2)
             for _,l,m,c in COMPARISONS]
    fig.legend(handles=handles,loc='outside lower center',ncols=2,frameon=False,
               title='Training comparison in panel a')
    save(fig,str(OUT/'human-annotation-ambiguity'),formats=('pdf','svg','png'))


if __name__=='__main__':main()
