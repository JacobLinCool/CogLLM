"""Plot direct human judgments of alternative news summaries and ordering decisions."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT=Path('research/results')
NEWS=ROOT/'reading-validation-summeval'
ORDER=ROOT/'reading-validation-columbia-order'
OUT=NEWS/'figure'


def load(path):
    return json.loads(path.read_bytes())


def read_figure():
    a=load(NEWS/'scores.json');b=load(NEWS/'ballot-scores.json');c=load(NEWS/'crossover-scores.json');h=load(ORDER/'scores.json');j=load(ORDER/'crossfit-scores.json');prefix=load(NEWS/'prefix-scores.json')
    same=[
        ('Form + word probability',a['arms']['expert']['form_calibrated']['pair_order'],'reference'),
        ('Bilingual reading model',a['arms']['expert']['adaptive']['pair_order'],'previous'),
        ('Shared three-study model',a['arms']['expert']['shared_native']['pair_order'],'model'),
        ('Within-story ballot model',b['ordering']['mixture_choice']['pair_order'],'model'),
        ('Article-aware GPT-4',a['arms']['expert']['geval']['pair_order'],'published'),
        ('Independent crowd panel',a['crowd_as_expert_predictor']['pair_order'],'human'),
    ]
    cross=[(name,c['actual_grades'][arm],group) for name,arm,group in [
        ('Fixed producer identity','fixed_system_identity','reference'),('Form + word probability','form_choice','reference'),('Bilingual reading model','adaptive','previous'),('Within-story ballot model','mixture_choice','model'),('Article-aware GPT-4','geval','published'),('Independent crowd panel','independent_crowd','human')]]
    seq=[
        ('Uniform choices',h['methods']['uniform']['pairwise_choice'],'reference'),
        ('Bilingual reading model',h['methods']['adaptive.centered']['pairwise_choice'],'previous'),
        ('Raw contextual word probability',h['methods']['mean_surprisal.centered']['pairwise_choice'],'reference'),
        ('Form + word probability',j['scores']['surface_and_probability']['pairwise_choice'],'reference'),
        ('Choice-trained native model',j['scores']['native_choices']['pairwise_choice'],'model'),
        ('Shared three-study model',j['scores']['native_three_observations']['pairwise_choice'],'model'),
        ('Independent human order',h['methods']['independent_human_order']['pairwise_choice'],'human'),
    ]
    local=[(label,prefix['scores'][k]['pair_order'],group) for label,k,group in [('News prefix score decline','mixture','model'),('News form score decline','form','reference'),('Relative sentence position','relative_position','reference'),('Direct English word-probability model','local_word_probability','reference'),('Bilingual reading model','local_bilingual','previous'),('Topic-held essay observer','local_essay_head','model')]]
    return [
        ('Coherence of two drafts',f"{a['source_stories']} news stories",same),
        ('Reversal for the same two systems',f"{c['observed_preference_reversals']:,} observed reversals",cross),
        ('Next unit selected by a writer',f"{h['nontrivial_human_decisions']:,} choices · {h['topic_families']} event families",seq),
        ('A local break marked by an expert',f"{prefix['evaluated_rows']:,} sentence ends · {prefix['essays']} essays",local),
    ]


def make():
    panels=read_figure();OUT.mkdir(exist_ok=True);colors={'reference':'#68717C','previous':'#347295','model':'#116754','published':'#765C8D','human':'#AB642C'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.linewidth':.55,'xtick.labelsize':7,'ytick.labelsize':8})
    fig,axes=plt.subplots(4,1,figsize=(6.9,10.05));fig.subplots_adjust(left=.42,right=.97,top=.96,bottom=.055,hspace=.72);data=[]
    for n,(ax,(title,subtitle,items)) in enumerate(zip(axes,panels,strict=True)):
        loc=np.arange(len(items))[::-1];ax.axvline(.5,lw=.9,ls='--',color='#817F7A',zorder=0)
        for yy,(label,res,group) in zip(loc,items,strict=True):
            value=res.get('mean',res.get('estimate'));lo,hi=res['ci95'];ax.plot([lo,hi],[yy,yy],color=colors[group],lw=1.8,solid_capstyle='round');ax.plot(value,yy,marker='o',ms=5.,color=colors[group],markeredgecolor='white',markeredgewidth=.45);data.append((chr(65+n),title,label,group,value,lo,hi))
        ax.tick_params(axis='y',length=0,pad=7);ax.set_yticks(loc,[v[0] for v in items]);ax.set_xlim(.36,.865);ax.set_xticks(np.arange(.4,.86,.1),[f'{t:.0%}' for t in np.arange(.4,.86,.1)]);ax.set_ylim(-.6,len(items)-.35);ax.spines['bottom'].set_color('#B4BAC0');ax.text(0,1.16,chr(65+n)+'  '+title,transform=ax.transAxes,fontsize=10,fontweight='bold',color='#17232E');ax.text(0,1.045,subtitle,transform=ax.transAxes,fontsize=8,color='#57616C')
    axes[-1].set_xlabel('Agreement with observed human ordering',labelpad=6)
    for ext in ('pdf','png'):
        fig.savefig(OUT/('human-news-order.'+ext),dpi=220,facecolor='white',bbox_inches='tight')
    plt.close(fig)
    with (OUT/'human-news-order.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['panel','endpoint','method','kind','mean','lower_95','upper_95']);w.writerows(data)
    print('Exported',len(data),'observed estimates')


if __name__=='__main__':make()
