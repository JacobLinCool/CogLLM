"""Render prespecified original-word human and model contrasts from their signed result files."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

OUT=Path('research/results/reading-validation-spanish-clues')


def main():
    run=json.loads((OUT/'cue-scores.json').read_text())['all']['original_word_end']
    eeg=json.loads((OUT/'eeg-scores.json').read_text())['results']['valid']['word_effect_simultaneous_band']
    def values(name):return np.array([[r['mean'],*r['ci']] for r in run[name]])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.linewidth':.65,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'xtick.labelsize':7,'ytick.labelsize':7})
    fig,axes=plt.subplots(3,1,figsize=(7.05,5.45),sharex=True,layout='constrained',height_ratios=[1,1,1.02])
    fig.suptitle('Uninformative minus informative context',x=.08,ha='left',fontsize=11,fontweight='bold')
    series=[('a','Human N400 negativity','Difference (µV)',np.asarray(eeg),np.arange(1,22),'#0072B2','95% simultaneous band'),
            ('b','Model surprisal of the observed word','Difference (nats)',values('word_nll'),np.arange(2,22),'#B14E16','95% pointwise interval'),
            ('c','Model estimate of pending material','Difference (model units)',values('stock'),np.arange(1,22),'#167565','95% pointwise interval')]
    for ax,(letter,title,unit,v,x,color,unc) in zip(axes,series):
        ax.axvspan(7.5,14.5,color='#EEE9DB',zorder=-3);ax.axhline(0,color='#777777',lw=.75,ls='--');ax.axvline(18,color='#4C4C4C',lw=.7,ls=':')
        ax.fill_between(x,v[:,1],v[:,2],color=color,alpha=.16,linewidth=0);ax.plot(x,v[:,0],color=color,linewidth=1.45,marker='o',ms=2.4)
        j=int(np.flatnonzero(x==18)[0]);ax.plot(18,v[j,0],'o',ms=5,color=color,mec='white',mew=.8)
        ax.set_title(letter+'   '+title,loc='left',fontsize=9.1,fontweight='bold',pad=4);ax.set_title(unc,loc='right',fontsize=7.3,color='#4C4C4C',pad=4)
        ax.set_ylabel(unit);ax.yaxis.set_major_locator(MaxNLocator(4));ax.set_xlim(.7,21.3);ax.tick_params(width=.6,length=3)
    axes[0].text(11,.96,'Different middle sentence',ha='center',va='top',transform=axes[0].get_xaxis_transform(),fontsize=7,color='#5D5540')
    axes[0].text(17.85,.96,'Shared category noun',ha='right',va='top',transform=axes[0].get_xaxis_transform(),fontsize=7,color='#4C4C4C')
    axes[-1].set_xticks([1,4,7,10,14,18,21]);axes[-1].set_xlabel('Position of the printed word in reading order')
    fig.savefig(OUT/'semantic-reading-course.pdf',bbox_inches='tight',pad_inches=.06)
    fig.savefig(OUT/'semantic-reading-course.png',dpi=220,bbox_inches='tight',pad_inches=.06)
    plt.close(fig)


if __name__=='__main__':main()
