"""Plot predicted and observed open-question counts on original story reading coordinates."""

import csv
import json

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from research.reading_validation.memory_data import sha
from research.reading_validation.narrative_course_data import WORK
from research.reading_validation.narrative_survival_fit import OUT
from research.reading_validation.transition_figures.orx_figstyle import use_style


def run():
    use_style();path=OUT/'forecast';meta=json.loads((path/'seal.json').read_bytes())
    if sha(path/'predictions.npz')!=meta['files']['predictions.npz']:
        raise ValueError('A locked whole-story prediction changed')
    figure,axes=plt.subplots(2,2,figsize=(7.3,4.6),layout='constrained')
    rows=[]
    colors={'AA':'#294c65','JE':'#5c7890','duration':'#aa632c','wording':'#16816d'}
    names=sorted(meta['geometry'])
    with np.load(path/'predictions.npz') as prediction,np.load(WORK/'response.npz') as outcomes:
        for ax,name,letter in zip(axes.flat,names,'abcd',strict=True):
            native=prediction[name+'.wording.stock'];x=np.arange(1,len(native)+1)
            curves={}
            for team in ('AA','JE'):
                k=name+'.'+team
                if k not in outcomes:continue
                state=outcomes[k];low=np.sum(state==1,axis=1);high=np.sum(state!=0,axis=1)
                curves[team+'_confirmed']=low;curves[team+'_possible']=high
                ax.fill_between(x,low,high,step='mid',color=colors[team],alpha=.16,lw=0)
                ax.step(x,low,where='mid',color=colors[team],alpha=.86,lw=.8,linestyle='-' if team=='AA' else ':')
            for arm,style in [('duration','--'),('wording','-')]:
                y=prediction[name+'.'+arm+'.stock'];curves[arm]=y
                ax.plot(x,y,color=colors[arm],linestyle=style,lw=1.45)
            rows.extend(dict(story=name,line=int(i),**{k:float(v[i-1]) for k,v in curves.items()}) for i in x)
            ax.set_title(f'{letter}   {name}',loc='left',fontweight='bold');ax.set_xlim(1,len(x));ax.set_ylim(0,1.06*max(float(np.max(v)) for v in curves.values()))
            if letter in 'cd':ax.set_xlabel('Original reading line')
            if letter in 'ac':ax.set_ylabel('Open questions (count)')
            ax.spines[['right','top']].set_visible(False)
    handles=[Line2D([0],[0],lw=1.5,color=colors[a],linestyle=s,label=l) for a,s,l in
             [('AA','-','Annotation team AA'),('JE',':','Annotation team JE'),('duration','--','Duration only'),('wording','-','Content and question supervision')]]
    figure.legend(handles=handles,loc='outside lower center',ncols=2,frameon=False,fontsize=8)
    for ext in ('pdf','png'):figure.savefig(OUT/f'open-questions.{ext}',dpi=220,bbox_inches='tight')
    plt.close(figure)
    with (OUT/'open-questions.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['story','line','AA_confirmed','AA_possible','JE_confirmed','JE_possible','duration','wording']);writer.writeheader();writer.writerows(rows)
    print('Rendered',len(rows),'original reading lines across',len(names),'complete stories')


if __name__=='__main__':run()
