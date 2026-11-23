"""Plot paired human-association changes for three independently fitted causal backbones."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

OUT=Path('research/results/reading-validation-physiology-portability')
MODELS=(('qwen','Qwen3.5-2B','#26668b'),('lfm3','LFM3-1.7B-Base','#bb631c'),('gemma4','Gemma 4 E2B','#6d518f'))


def main():
    scores=json.loads((OUT/'english-spanish-scores.json').read_bytes());plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'xtick.labelsize':8,'ytick.labelsize':8});fig,axes=plt.subplots(1,2,figsize=(7.2,3.15),gridspec_kw={'width_ratios':[1.1,1.05]},layout='constrained');targets=(('N400','N400\nnegativity'),('cloze_miss','Cloze miss\nfraction'),('unpredictability','Rated\nunpredictability'));offsets=(-.21,0,.21)
    for i,(key,name,color) in enumerate(MODELS):
        for j,(t,label) in enumerate(targets):
            d=scores['source'][t]['rho'][key+'_context'];point=d['point'];lo,hi=d['interval'];axes[0].errorbar(j+offsets[i],point,[[point-lo],[hi-point]],fmt='o',ms=4,color=color,capsize=2,lw=1.3)
        for j,d in enumerate([scores['source']['N400'],scores['spanish']]):
            v=d['paired'][key+'_context__over__'+key+'_surface'];point=v['rho_difference'];lo,hi=v['interval'];axes[1].errorbar(j+offsets[i],point,[[point-lo],[hi-point]],fmt='o',ms=4,color=color,capsize=2,lw=1.3)
    axes[0].set_title('a   Held English reading measures',loc='left',fontsize=9.2,fontweight='bold',pad=6);axes[0].set_xticks(range(len(targets)),[x[1] for x in targets]);axes[0].set_ylabel('Spearman correlation with readers');axes[0].set_ylim(.3,.88);axes[0].set_yticks([.3,.4,.5,.6,.7,.8]);axes[0].set_xlim(-.48,2.48)
    axes[1].set_title('b   Beyond a calibrated lexical baseline',loc='left',fontsize=9.2,fontweight='bold',pad=6);axes[1].set_xticks([0,1],['English held\nsentences','Spanish external\nword families']);axes[1].set_ylabel('Change in N400 correlation');axes[1].set_xlim(-.48,1.48);axes[1].set_ylim(-.038,.09);axes[1].set_yticks([-.02,0,.02,.04,.06,.08]);axes[1].axhline(0,ls='--',lw=.8,color='#646664',zorder=0)
    for ax in axes:ax.set_axisbelow(True);ax.yaxis.grid(True,color='#d7dddf',lw=.6);ax.tick_params(length=3,width=.6)
    fig.legend([Line2D([],[],color=c,marker='o',lw=0,markersize=5) for _,_,c in MODELS],[n for _,n,_ in MODELS],loc='outside lower center',ncol=3,frameon=False,columnspacing=1.2,handletextpad=.4,fontsize=8)
    fig.savefig(OUT/'human-response-transfer.pdf',bbox_inches='tight',pad_inches=.06);fig.savefig(OUT/'human-response-transfer.png',dpi=220,bbox_inches='tight',pad_inches=.06);plt.close(fig)


if __name__=='__main__':main()
