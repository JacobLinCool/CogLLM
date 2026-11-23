"""Render lexical judgment and span-retrieval estimates from checksummed study results."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from orx_figstyle import PALETTE, WIDE, save, use_style

OUT=Path(__file__).resolve().parent
BASE=OUT.parent
GREY='#50575B'


def load(filename):
    path=BASE/filename
    return json.loads(path.read_bytes()),hashlib.sha256(path.read_bytes()).hexdigest()


def source_data():
    guo,g_hash=load('jargon/scores.json')
    start,m_hash=load('jargon/matched-start.json')
    spans,s_hash=load('localization/scores.json')
    table=[]
    def row(panel,condition,name,d,n,groups,source,fmt='guo',numerator=None):
        k,ci=('mean','crossed_95') if fmt=='guo' else ('estimate','crossed_95') if fmt=='matched' else ('mean','interval')
        lo,hi=d[ci]
        table.append(dict(panel=panel,condition=condition,method=name,estimate=d[k],lower95=lo,upper95=hi,
                         eligible_observations=n,cluster_structure=groups,numerator=numerator,source_sha256=source))
    mapping=[('form','Word form'),('local','Local phrase head'),('external_lexical','Lexical head'),('prereading_rating','Reader’s prior')]
    for k,name in mapping:
        all_pairs=guo['after_with_prereading_comparator']
        row('a','All unequal pairs',name,all_pairs['estimates'][k],all_pairs['unequal_pairs'],'83 abstracts × 11 researchers',g_hash)
        if k!='prereading_rating':
            row('a','Same prereading score',name,start['estimates'][k],start['unequal_after_pairs'],'81 abstracts × 11 researchers',m_hash,'matched')
    for k,name in [('displayed_mass','Unresolved level'),('form','Word form'),('external_lexical','Lexical head'),('local','Local phrase head')]:
        for j,label in [('hit1','First suggestion'),('hit5','First five')]:
            d=spans['estimates']['sentence'][j][k]
            row('b',label,name,d,spans['positive_spans'],'12 connected reader–article groups',s_hash,'plaba',round(d['mean']*spans['positive_spans']))
    with (OUT/'source-data.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    return table


def axes_panel(ax,table,methods,conditions,limits,ticks,xlabel):
    ax.set_axisbelow(True)
    ax.grid(axis='y',visible=False)
    ax.grid(axis='x',color='#E1E3E5',linewidth=.52)
    colors=[PALETTE['blue'],PALETTE['orange']]
    markers=['o','s']
    for c,(condition,color,marker) in enumerate(zip(conditions,colors,markers,strict=True)):
        for r in table:
            if r['condition']!=condition:continue
            y=len(methods)-1-methods.index(r['method'])+(.123 if c==0 else -.123)
            mean,lo,hi=[100*r[k] for k in ('estimate','lower95','upper95')]
            ax.errorbar(mean,y,xerr=[[mean-lo],[hi-mean]],fmt=marker,markersize=4.2,
                        color=color,markerfacecolor=color if c==0 else 'white',markeredgewidth=.95,
                        elinewidth=1.05,capsize=2,capthick=.8,zorder=3)
    ax.set_yticks(list(range(len(methods))),methods[::-1])
    ax.set_xlim(*limits);ax.set_xticks(ticks)
    ax.set_ylim(-.55,len(methods)-.4)
    ax.tick_params(axis='y',length=0,pad=3)
    ax.set_xlabel(xlabel,labelpad=5)
    ax.spines[['top','right','left']].set_visible(False)
    return [Line2D([0],[0],marker=m,markerfacecolor=k if i==0 else 'white',color=k,markeredgewidth=1.0,
                   linewidth=.95,markersize=4,label=v) for i,(k,m,v) in enumerate(zip(colors,markers,conditions,strict=True))]


def main():
    values=source_data()
    use_style()
    fig=plt.figure(figsize=(WIDE,2.60))
    ax=fig.add_axes([.15,.205,.306,.595])
    bx=fig.add_axes([.69,.205,.297,.595])
    a=axes_panel(ax,[r for r in values if r['panel']=='a'],['Word form','Local phrase head','Lexical head','Reader’s prior'],
                 ['All unequal pairs','Same prereading score'],(40,95),[40,50,60,70,80,90],'Term pairs ordered correctly (%)')
    b=axes_panel(bx,[r for r in values if r['panel']=='b'],['Unresolved level','Word form','Lexical head','Local phrase head'],
                 ['First suggestion','First five'],(0,100),[0,25,50,75,100],'Named phrases recovered (%)')
    ax.axvline(50,color=GREY,lw=.65,linestyle=(0,(3,3)),zorder=1)
    for x,letter,topic in ((.014,'a','Term familiarity'),(.545,'b','Phrase localization')):
        fig.text(x,.945,letter,weight='bold',fontsize=9,va='baseline')
        fig.text(x+.023,.945,topic,fontsize=8,va='baseline')
    fig.legend(handles=a,loc='upper left',bbox_to_anchor=(.095,.909),fontsize=6.5,ncols=2,frameon=False,
               columnspacing=1.25,handlelength=1.5,handletextpad=.4,borderpad=0)
    fig.legend(handles=b,loc='upper left',bbox_to_anchor=(.654,.909),fontsize=6.5,ncols=2,frameon=False,
               columnspacing=1.25,handlelength=1.5,handletextpad=.4,borderpad=0)
    save(fig,str(OUT/'term-familiarity-and-localization'))
    plt.close(fig)


if __name__=='__main__':
    main()
