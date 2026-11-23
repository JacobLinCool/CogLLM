"""Render a calibrated count over the whole draft and named source-to-cue connections."""

from html import escape
from math import ceil,isfinite


def check(text,record):
    level=record["expected"]
    end=record["ends"]
    start=record["starts"]
    if not 0<len(end)==len(start)==len(level)==record["tokens"]<=1024:
        raise ValueError("Terminology counts do not cover the original token sequence.")
    if (any(not isfinite(x) or x<0 for x in level) or sorted(end)!=end or
        any(not 0<=a<=b<=len(text) for a,b in zip(start,end,strict=True))):
        raise ValueError("Terminology count or location is invalid.")
    for link in record["links"]:
        a,b=link["source"];c,d=link["cue"]
        if not 0<=a<b<=c<d<=len(text) or not isfinite(link["score"]) or link["score"]<record["threshold"]:
            raise ValueError("One terminology link failed its fitted source and location criteria.")


def markup(text,record):
    check(text,record)
    y=record["expected"]
    y_max=max(1,ceil(max(y)))
    paths=" ".join(f"{1000*i/max(1,len(y)-1):.2f},{100-96*v/y_max:.2f}" for i,v in enumerate(y))
    chart=(f'<div class="term-chart" role="img" aria-label="Estimated mentions awaiting a definition, '
           f'from {y[0]:.2f} to {y[-1]:.2f}, maximum {max(y):.2f}"><span><small>{y_max}</small><small>0</small></span>'
           '<svg aria-hidden="true" viewBox="0 0 1000 104" preserveAspectRatio="none">'
           '<path d="M0 4H1000 M0 100H1000" fill="none" stroke="#83978d" stroke-width="1"/>'
           f'<polyline fill="none" stroke="#176956" stroke-width="2.5" vector-effect="non-scaling-stroke" '
           f'points="{paths}"/></svg></div>')
    items=[]
    for link in record["links"]:
        a,b=link["source"];c,d=link["cue"]
        items.append(f'<li><span class="term-source">{escape(text[a:b])}</span> '
                     f'<span aria-label="may be explained at">→</span> <span class="term-cue">{escape(text[c:d])}</span></li>')
    if items:
        extra=(f'<details><summary>Show {len(items)-4} more links</summary><ul>{"".join(items[4:])}</ul></details>'
               if len(items)>4 else "")
        findings='<p>Possible term → later wording</p><ul>'+"".join(items[:4])+'</ul>'+extra
    else:
        findings="<p>No term and later wording crossed the trained link threshold in this draft.</p>"
    return ('<section class="term-review" aria-label="Technical terms"><h3>Terms awaiting an explanation</h3>'
            '<p>The line estimates how many earlier terminology mentions are still waiting for a definition.</p>'
            +chart+f'<div class="term-x"><span>Start of draft</span><span>End · {y[-1]:.2f} expected mentions</span></div>'
            +findings+'<p class="term-scope">Trained on textbook definitions. In held biology and physics passages, '
            '30% of proposed links matched both the annotated term and its later definition interval.</p></section>')


def marked_spans(record):
    marks=[]
    for link in record["links"]:
        for kind in ("source","cue"):
            a,b=link[kind]
            marks.append((a,b,kind))
    return marks
