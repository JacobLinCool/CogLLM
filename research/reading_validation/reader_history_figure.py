"""Display independently observed reader-level calibration effects with crossed uncertainty."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path('research/results')
EN=ROOT/'reading-validation-reader-history'
DE=ROOT/'reading-validation-muse'

def load(path):return json.loads(path.read_text())

def main():
    a=load(EN/'mode-scores.json')['results']
    b=load(DE/'independent-reading-scores.json')['external']['all_available']
    sec=load(EN/'sequential-scores.json')['studies']['all']
    fig,ax=plt.subplots(1,2,figsize=(10.7,4),layout='constrained',gridspec_kw={'width_ratios':[1.1,1]})
    colors=['#1B716F','#A65B19'];markers=['o','s']
    for axis in ax:
        axis.spines[['top','right']].set_visible(False);axis.axvline(0,linewidth=1,color='#6D7880',zorder=0)
        axis.grid(axis='x',alpha=.25,linewidth=.6)
    for k,(data,title) in enumerate([(a,'English, chronological'),(b,'German, across stems')]):
        for j,key in enumerate(['same_over_unknown_reader','same_over_other_history','same_over_single_head']):
            z=data['paired'][key];x=1000*z['nats_per_response'];lo,hi=1000*np.array(z['reader_sentence_interval'])
            ax[0].errorbar(x,j+(k-.5)*.22,xerr=[[x-lo],[hi-x]],color=colors[k],fmt=markers[k],capsize=3,ms=5,label=title if j==0 else None)
    ax[0].set(yticks=[0,1,2],yticklabels=['Matched peer mixture','One matched peer','Static cohort model'],ylim=(2.6,-.55),
              xlabel='Personal gain per 1,000 answers (nats)',title='A   Whose prior choices help?')
    ax[0].legend(loc='upper center',bbox_to_anchor=(.5,-.22),ncol=2,fontsize=8,frameon=False)
    keys=['Named vs outside','Identity of a named word']
    for k,(d,title) in enumerate([(a,'English'),(b,'German')]):
        item=d['choice_content']['paired']['same_over_unknown_reader']
        for j,(v,p) in enumerate([('class_nats_per_answer','class_interval'),('named_identity_nats_per_named_choice','named_interval')]):
            x=1000*item[v];lo,hi=1000*np.array(item[p])
            ax[1].errorbar(x,j+(k-.5)*.22,xerr=[[x-lo],[hi-x]],fmt=markers[k],color=colors[k],capsize=3,ms=5)
    ax[1].set(yticks=[0,1],yticklabels=keys,ylim=(1.45,-.55),xlabel='Personal gain per 1,000 eligible answers (nats)',title='B   What can the profile predict?')
    for axis in ax:axis.title.set_ha('left');axis.title.set_position((0,1.04))
    fig.suptitle('Personalization has a small effect on real readers’ word choices',fontweight='bold',fontsize=14)
    for suffix in ('.pdf','.png'):fig.savefig(EN/('reader-history-cross-cohort'+suffix),dpi=180,bbox_inches='tight',facecolor='white')
    plt.close(fig)

if __name__=='__main__':main()
