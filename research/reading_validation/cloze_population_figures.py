"""Render the observed population word course and multisite reading contrast from sealed scores."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path("research/results")
OUT=ROOT/"reading-validation-cloze-population"
LAB=ROOT/"reading-validation-multilab"
TEAL="#136E69";DARK="#202E39";AMBER="#AC5B22";MUTED="#65747B"


def data(path):
    return json.loads(path.read_text())


def decorate(ax):
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y",color="#DCE4E2",linewidth=.5)
    ax.set_axisbelow(True)


def save(fig,dest):
    for ext in ("png","pdf"):
        fig.savefig(dest.with_suffix("."+ext),dpi=185,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def curves():
    s=data(OUT/"reading-updates.json");main=s["Provo_external"]
    e=s["all_Provo_released_row_mass_sensitivity"]["illustrative_article"]
    fig=plt.figure(figsize=(9.4,6.6),layout='constrained')
    gs=fig.add_gridspec(2,2,height_ratios=[1.25,1])
    ax=fig.add_subplot(gs[0,:]);decorate(ax)
    x=np.array(e['target_word_positions']);y=np.array(e['coarse_empirical'])
    bad=np.array(e['unreconciled_reported_denominators'])
    ax.plot(x,e['modeled_contextual'],color=TEAL,linewidth=2,label='Model estimate')
    ax.plot(x,y,color=DARK,linewidth=1.25,marker='o',markersize=3.0,label='Published reader choices')
    if np.any(bad):ax.scatter(x[bad],y[bad],s=20,facecolor='white',edgecolor=DARK,label='Published denominator differs',zorder=4)
    ax.set(ylim=(-.04,1.04),xlabel="Position of the word just read",ylabel="Probability of different next guesses")
    ax.set_title(f"A   One independently measured paragraph (Provo {e['article']})",loc='left')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.24),ncols=3,fontsize=8,frameon=False)
    ax=fig.add_subplot(gs[1,0]);decorate(ax)
    names=['frequency','raw','calibrated','context'];labels=['Same at every word','Native Qwen','Calibrated native','Contextual head']
    for j,name in enumerate(names):
        m=main['models'][name];v=m['correct_direction'];low,high=m['direction_interval']
        ax.errorbar(v,j,xerr=[[v-low],[high-v]],fmt='o',ms=6,color=TEAL if name=='context' else MUTED,capsize=3)
    ax.set(yticks=range(4),yticklabels=labels,xlim=(.43,.82),xticks=np.arange(.5,.81,.1),xlabel='Fraction of correct increase/decrease calls')
    ax.axvline(.5,color='#A2AEAA',lw=.8,ls='--');ax.set_title('B   Changes after one printed word',loc='left')
    ax=fig.add_subplot(gs[1,1]);decorate(ax)
    m=main["models"]["context"];v=[m['large_increases']['correct_direction'],m['large_decreases']['correct_direction']];n=[m['large_increases']['count'],m['large_decreases']['count']]
    ax.bar([0,1],v,width=.55,color=[AMBER,TEAL])
    for i,value in enumerate(v):ax.text(i,value+.02,f"{value:.1%}\n{n[i]} positions",ha='center',fontsize=9,linespacing=1.2)
    ax.set(xticks=[0,1],xticklabels=['Increase','Decrease'],yticks=[0,.25,.5,.75,1],ylim=(0,1.13),ylabel='Correct direction')
    ax.set_title('C   Larger measured changes (≥ 0.20)',loc='left')
    fig.suptitle("Readers’ next-word expectations change as a paragraph unfolds",fontsize=14.5,color=DARK,fontweight='bold')
    save(fig,OUT/"word-expectations-in-context")


def multisite():
    s=data(LAB/"scores.json");u=s['human_signed_microvolts']
    models=s['model_associations']
    fig,axes=plt.subplots(1,2,figsize=(9.6,3.75),layout='constrained')
    a,b=axes
    for ax in axes:decorate(ax)
    for i,(k,v,(lo,hi)) in enumerate(zip(("Article","Noun"),(u['article'],u['noun']),u['ci'],strict=True)):
        a.errorbar(i,v,yerr=[[v-lo],[hi-v]],fmt='o',color=[MUTED,TEAL][i],ms=8,capsize=5)
    a.axhline(0,color='#979E9F',lw=.7);a.set(xlim=(-.6,1.6),xticks=[0,1],xticklabels=['Preceding article','Following noun'],ylabel='Unexpected − expected negativity (µV)')
    a.set_title('A   Actual N400 responses',loc='left')
    names=['raw_surprisal','english_surface_EEG','english_context_EEG'];labels=['Native surprise','Calibrated surface','Contextual EEG head']
    for j,(n,l) in enumerate(zip(names,labels,strict=True)):
        m=models[n]
        for k,(v,(lo,hi)) in enumerate(zip(m['held_item_contrast_rho'],m['contrast_rho_ci'],strict=True)):
            b.errorbar(v,j+(k-.5)*.20,xerr=[[v-lo],[hi-v]],fmt=['s','o'][k],color=[MUTED,TEAL][k],ms=4.5,capsize=2,label=['Article','Noun'][k] if j==0 else None)
    b.axvline(0,color='#979E9F',lw=.7);b.set(yticks=range(3),yticklabels=labels,xlim=(-.35,.65),xlabel='Correlation with material-by-material N400 effects')
    b.legend(loc='upper center',bbox_to_anchor=(.5,-.2),ncols=2,frameon=False,fontsize=8);b.set_title('B   Predictions across 80 passage pairs',loc='left')
    fig.suptitle('Expectation has different effects at two adjacent words',fontsize=14.5,color=DARK,fontweight='bold')
    save(fig,LAB/"article-noun-location")


if __name__=='__main__':
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'axes.titleweight':'semibold',
                         'axes.labelsize':9,'axes.edgecolor':'#64727B','svg.fonttype':'none','pdf.fonttype':42})
    curves();multisite()
