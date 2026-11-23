"""Show the probability of a reader's first report on either side of an unchanged critical sentence."""
import csv
import json
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from research.reading_validation.memory_data import BASE,sha
from research.reading_validation.memory_score import reveal,trial_observation
from pathlib import Path
OUT=Path("research/results/reading-validation-premise-emission")
from research.reading_validation.transition_figures.orx_figstyle import use_style

def trajectory(rows,model,arm,offsets):
    sample=[r for r in rows if r['arm']==arm];people=sorted({r['reader'] for r in sample});stories=sorted({r['story'] for r in rows})
    # Positive multiplier draws preserve all observed reader-by-story cells at later positions.
    rng=np.random.default_rng(773202+int(arm=='neutral'));rw=rng.exponential(size=(4000,len(people)))
    sw=np.random.default_rng(119923).exponential(size=(4000,len(stories)))
    table=[]
    for d in offsets:
        z=np.zeros((len(people),len(stories),3))
        for r in sample:
            j=r['target']-1+d
            if not 0<j<len(r['keys']):continue
            idx=people.index(r['reader']),stories.index(r['story']);hz=model['strict.directional.'+r['id']]
            z[idx]=[1.,float(trial_observation(r)==j+1),float(hz[j]*np.prod(1-hz[:j]))]
        valid=z[:,:,0].sum(0)>0;data=z[:,valid];tot=data.sum(0)
        if not valid.any():continue
        for a,label in [(1,'human'),(2,'model')]:
            pred=(tot[:,a]/tot[:,0]).mean();draw=((rw@data[:,:,a])/(rw@data[:,:,0])*sw[:,valid]).sum(1)/sw[:,valid].sum(1)
            lo,hi=np.quantile(draw,[.025,.975]);table.append(dict(offset=d,condition=arm,quantity=label,value=pred,lo=lo,hi=hi,
                                               trials=int(data[:,:,0].sum()),families=int(valid.sum())))
    return table

def run():
    use_style();out=OUT/'forecast';seal=json.loads((out/'seal.json').read_bytes())
    if sha(out/'hazards.npz')!=seal['files']['hazards.npz']:raise ValueError('Forecast differs from graded first-report study')
    meta=json.loads((BASE/'memory/study-map.json').read_bytes());responses=reveal(3,meta);arm=('neutral','inconsistent')
    q=json.loads((OUT/'scores.json').read_bytes())['cohorts']['3'];offsets=np.arange(-4,3)
    with np.load(out/'hazards.npz') as f:table=sum((trajectory(responses,f,a,offsets) for a in arm),[])
    at={r['condition']:r['value'] for r in table if r['quantity']=='human' and r['offset']==0}
    if abs(at['inconsistent']-at['neutral']-q['human_effect']['estimate'])>1e-9:raise ValueError('Figure denominator differs from primary statistical table')
    with (OUT/'first-report-trajectory.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    fig,axes=plt.subplots(1,2,figsize=(8.15,3.55),gridspec_kw={'width_ratios':[1.35,1]},layout='constrained')
    blue='#285b78';orange='#b45229';green='#256b59'
    for a,col in zip(arm,(blue,orange),strict=True):
        for measure,style in [('human','-'),('model','--')]:
            r=[v for v in table if v['condition']==a and v['quantity']==measure];x=[v['offset'] for v in r];y=[100*v['value'] for v in r]
            axes[0].plot(x,y,style,color=col,marker='o' if measure=='human' else None,ms=3,lw=1.9,zorder=3)
            if measure=='human':axes[0].fill_between(x,[100*v['lo'] for v in r],[100*v['hi'] for v in r],color=col,alpha=.11,linewidth=0)
    axes[0].axvline(0,ls=':',color='#7c8584',linewidth=.8);axes[0].set(xticks=offsets,xlabel='Sentence relative to the unchanged statement',ylabel='First inconsistency reports (%)',ylim=(0,None))
    arms=[('Observed readers',q['human_effect'],orange),('Original native readout',q['models']['strict']['original']['condition_effect'],blue),
          ('State + report model',q['models']['strict']['directional']['condition_effect'],green),('Known position and condition',q['models']['strict']['oracle']['condition_effect'],'#777b7b')]
    for j,(name,v,c) in enumerate(arms):
        y=len(arms)-1-j;m=100*v['estimate'];lo,hi=np.array(v['interval95'])*100
        axes[1].plot([lo,hi],[y,y],color=c,lw=2);axes[1].scatter(m,y,s=27,color=c,zorder=3);axes[1].annotate(f'{m:.1f}',(m,y),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8,color=c)
    axes[1].set(yticks=[3,2,1,0],yticklabels=[i[0] for i in arms],xlabel='Increase in first reports (percentage points)',ylim=(-.4,3.5));axes[1].axvline(0,color='#a0a4a3',lw=.8)
    for ax,title in zip(axes,('a   Where readers first reported a conflict','b   Effect at the unchanged statement'),strict=True):
        ax.set_title(title,loc='left',fontsize=10);ax.spines[['right','top']].set_visible(False);ax.grid(axis='y' if ax is axes[0] else 'x',color='#e6e9e8',lw=.5)
    legend=[Line2D([],[],color=blue,lw=2,label='Earlier category only'),Line2D([],[],color=orange,lw=2,label='Earlier conflicting detail'),
            Line2D([],[],color='#303d3b',lw=1.6,marker='o',ms=3,label='Readers'),Line2D([],[],color='#303d3b',lw=1.6,ls='--',label='Model')]
    fig.legend(handles=legend,loc='outside lower center',ncols=4,frameon=False,fontsize=8)
    for extension in ('pdf','png'):fig.savefig(OUT/('first-report-trajectory.'+extension),dpi=220,bbox_inches='tight')
    plt.close(fig)
if __name__=='__main__':run()
