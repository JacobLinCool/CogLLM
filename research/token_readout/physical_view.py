"""Show the two original sentences implicated by a short physical story."""

from html import escape
from math import isfinite

EXAMPLE = ("Mina filled a glass bottle with water. She sealed the bottle and placed it in the freezer. "
           "The water inside froze overnight. Mina took the bottle out and set it on the counter. "
           "Without warming it, she immediately poured the water into a cup.")


def render(text, reading):
    if reading["scope"] != "authored_physical_conflict":
        raise ValueError("The physical-story display needs the fitted consistency observer.")
    spans = [tuple(v) for v in reading["sentences"]]
    events = reading["observations"]
    if len(events)!=len(spans)-1 or len(spans)<2:
        raise ValueError("The reading does not cover every sentence transition.")
    choices=[];total=0.
    for j,r in enumerate(events,1):
        p=r["source_probabilities"]
        if (r["sentence"]!=j or (r["start"],r["end"])!=spans[j] or len(p)!=j or
                any(not isfinite(v) or not 0<=v<=1. for v in p) or
                abs(sum(p)-r["first_conflict_probability"])>1e-5):
            raise ValueError("A physical transition lost its earlier source or normalized probability.")
        total+=sum(p)
        choices.extend((v,i,j) for i,v in enumerate(p))
    if abs(total-reading["probability_any"])>1e-5:
        raise ValueError("Candidate pairs disagree with the whole-story probability.")
    _,i,j=max(choices,key=lambda row:row[0])
    if reading["review"]:
        intro=(f'<p>Compare <strong>sentence {i+1}</strong> (earlier action) with '
               f'<strong>sentence {j+1}</strong> (possible conflict). This pair ranked highest in the story.</p>')
    else:
        intro='<p>The model did not flag a pair in this story at its development threshold.</p>'
    chunks=[];last=0
    for k,(a,b) in enumerate(spans):
        if not last<=a<b<=len(text):
            raise ValueError("Sentence boundaries differ from the original story.")
        chunks.append(escape(text[last:a]))
        original=escape(text[a:b])
        if reading["review"] and k in (i,j):
            role="earlier" if k==i else "conflict"
            title="Earlier action" if k==i else "Possible conflict"
            original=f'<mark class="physical-{role}" title="{title}" data-start="{a}" data-end="{b}">{original}</mark>'
        chunks.append(original);last=b
    chunks.append(escape(text[last:]))
    return ('<section class="physical-review" aria-label="Physical continuity">'+intro+
            '<div class="physical-draft" aria-label="Original story">'+''.join(chunks)+'</div></section>')


STYLE="""
.physical-review {background:#f8faf7; border:1px solid #b4c4bb; border-radius:.6rem; padding:1rem; color:#23352f!important;}
.physical-review p,.physical-review strong,.physical-review .physical-draft {color:#23352f!important;}
.physical-draft {white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.8;}
.physical-draft mark {color:#1d302a!important;padding:.08em 0;box-decoration-break:clone;}
.physical-earlier {background:#c7e3df!important;box-shadow:inset 0 -.12em #347065;}
.physical-conflict {background:#f7d9b9!important;outline:1px dashed #8a4726;}
"""


def add_physical_check(runner):
    import gradio as gr
    with gr.Accordion("Do two events in a short story conflict?",open=False):
        gr.Markdown("Paste five to seven sentences about everyday physical actions. "
                    "The model looks for a later action that conflicts with an earlier one.")
        story=gr.Textbox(label="Short physical story",lines=5)
        with gr.Row():
            submit=gr.Button("Compare events",variant="primary")
            example=gr.Button("Load a physical story")
        result=gr.HTML()
        def review(text):
            if not isinstance(text,str) or not text.strip():
                raise gr.Error("Enter a short physical story.")
            try:
                return render(text,runner.physical_story(text))
            except ValueError as error:
                raise gr.Error(escape(str(error))) from error
        submit.click(review,inputs=story,outputs=result,api_name="physical_story",concurrency_limit=1)
        example.click(lambda:(EXAMPLE,""),outputs=[story,result],api_name=False,show_progress="hidden")
