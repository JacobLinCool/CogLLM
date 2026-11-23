"""Present English sentence ratings and local wording candidates on the original draft."""

from html import escape
from math import floor, isfinite

from research.reading_validation.local_inference import SCHEMA
from research.token_readout import expectation_view, experimental_view, physical_view, summary_view
from research.token_readout import learner_view, reference_view, terminology_view
from research.token_readout.app import CHANNELS, initial_position, parse_analysis, plot_rows, render_article, source_html


EXAMPLE = ("A sensor measures how much light a leaf reflects. "
           "The resulting hyperspectral reflectance signature varies with the concentration of pigments in the leaf. "
           "This means that two leaves can reflect different amounts of light even when they look equally green.")
NARRATIVE_EXAMPLE = ("Mara and Leo spent the evening repairing their boat. Mara checked the engine while Leo worked on the hull. "
                     "By morning they were ready to leave.\n\nNatascha met them at the dock. She handed Mara a faded photograph.")

TERMINOLOGY_EXAMPLE = ("A light wave can change direction at the boundary between air and water. "
                       "This effect is called refraction. The refractive index of a material "
                       "is the ratio of the speed of light in a vacuum to its speed in that material.")

MODEL_CARD = "https://huggingface.co/JacobLinCool/CogLLM-Qwen3.5-2B"

STYLE = """
.writing-review .term-review { border-top:1px solid #cbd7cf; padding-top:.9rem; margin-top:.65rem; }
.writing-review .term-review h3,.writing-review .term-review p,.writing-review .term-review li { color:#23352f!important; }
.writing-review .term-review h3 { font-size:1.1rem; margin:.2rem 0; }
.writing-review .term-source { border-bottom:3px solid #3d685a; }
.writing-review .term-cue { border-bottom:3px dotted #3d685a; }
.writing-review .term-chart { display:grid; grid-template-columns:1.5em minmax(0,1fr); margin-top:.6rem; }
.writing-review .term-chart svg { display:block; width:100%; height:4.1rem; }
.writing-review .term-chart span { display:flex; flex-direction:column; justify-content:space-between; font-size:.8rem; }
.writing-review .term-x { display:flex; justify-content:space-between; flex-wrap:wrap; padding-left:1.5em; font-size:.8rem; }
.writing-review .term-scope { font-size:.83rem; }
.writing-review { --ink: #182b24; color: var(--ink)!important; background: #fcfcf8;
 border: 1px solid #cbd7cf; border-radius: .65rem; padding: 1.1rem 1.25rem; }
.writing-review .summary-understanding { margin-top:.8rem; padding-top:.65rem; border-top:1px solid #c4d1cc; }
.writing-review .summary-understanding h3,.writing-review .summary-understanding p,
.writing-review .summary-understanding strong { color:#23352f!important; }
.writing-review .reading-key,.writing-review .reading-caption { color: #394c42!important; font-size: .87rem; }
.writing-review .reading-key { display: flex; flex-wrap: wrap; gap: .35rem .72rem; align-items: center; }
.writing-review .key-swatch { display: inline-block; width: 1.3em; height: .82em; border: 1px solid #b2b0a3;
 border-radius: .15em; vertical-align: middle; margin-right: .18em; }
.writing-review .draft { white-space: pre-wrap; overflow-wrap: anywhere; font: 1.16rem/2.1 Georgia, "Times New Roman", serif;
 max-width: 85ch; padding: 1.05rem 0; }
.writing-review .draft mark { color: var(--ink)!important; padding: .14em 0; box-decoration-break: clone;
 -webkit-box-decoration-break: clone; border-radius: .16rem; }
.writing-review .phrase { text-decoration: underline solid #6e4218; text-underline-offset: .23em; }
.writing-review .character { box-shadow: inset 0 3px 0 #624a8b; padding-top: .11em; }
.writing-review .connection { outline: 2px dashed #28568c; outline-offset: .08em; }
.writing-review .review-list { margin: .6rem 0 0; padding-left: 1.5rem; }
.writing-review .review-list li { margin-bottom: .4rem; }
.writing-review h2 { margin: 0 0 .65rem; font-size: 1.3rem; }
.writing-review .reading-key { margin: .3rem 0; }
.writing-review .key-description { display: inline-block; margin-inline-end: .65rem; }
.gradio-container:has(#writer-heading) { max-width: 960px!important; margin-inline: auto; }
#writer-heading p { max-width: 68ch; }
#writer-actions { align-items: center; }
#writer-result:empty { display: none; }
@media (max-width: 550px) {
  .writing-review { padding: .85rem .75rem; }
  .writing-review .draft { font-size: 1.06rem; }
}
""" + learner_view.STYLE + reference_view.STYLE
COLORS = ("#d78353", "#e9a774", "#f0c796", "#e7e1b8", "#e8f3e7")


STYLE += physical_view.STYLE + expectation_view.STYLE + experimental_view.STYLE

# Gradio assigns foreground colors to individual HTML descendants in dark mode.
# Each fixed, light reading surface therefore supplies both its own theme variables
# and a foreground for nested source words, legend labels, links and disclosures.
STYLE += """
:is(.writing-review,.physical-review,.learner-result) {
  --reading-surface-ink:#182b24;
  --body-text-color:var(--reading-surface-ink);
  --body-text-color-subdued:var(--reading-surface-ink);
  color-scheme:light;
  color:var(--reading-surface-ink)!important;
}
:is(.writing-review,.physical-review,.learner-result) :where(*) {
  color:var(--reading-surface-ink)!important;
}
:is(.writing-review,.physical-review,.learner-result) a:any-link {
  color:#183b45!important;
  text-decoration:underline;
}
:is(.writing-review,.physical-review,.learner-result) :is(a,summary):focus-visible {
  outline:2px solid #284e62; outline-offset:2px;
}
"""


def fixed_color(score):
    if not isfinite(score) or not 1 <= score <= 5:
        raise ValueError("Reading ease must be on the validated 1–5 scale.")
    lo = min(3, floor(score) - 1)
    fraction = score - (lo + 1)
    lower, upper = COLORS[lo], COLORS[lo + 1]
    channels = (round((1 - fraction) * int(lower[i:i+2], 16) + fraction * int(upper[i:i+2], 16))
                for i in (1, 3, 5))
    return "#" + "".join(f"{channel:02x}" for channel in channels)


def validate(text, local):
    if local["schema"]!=SCHEMA or local["language"]!="en" or not local["bundle_sha256"]:
        raise ValueError("The selected readout is not the English wording model.")
    previous=0
    for s in local["sentences"]:
        a,b=s["start"],s["end"]
        if type(a) is not int or type(b) is not int or not previous<=a<b<=len(text):
            raise ValueError("Sentence boundaries differ from the original passage.")
        previous=b
        if s["rating"] is not None and any(not isfinite(s["rating"][key]) or not 1<=s["rating"][key]<=5
                                          for key in ("ease","understanding")):
            raise ValueError("A sentence rating falls outside the human rating scale.")
        intervals=[]
        for i,p in enumerate(s["phrases"],1):
            lo,hi=p["start"],p["end"]
            if (p["rank"]!=i or not a<=lo<hi<=b or not isfinite(p["score"]) or
                    any(lo<d and c<hi for c,d in intervals)):
                raise ValueError("A phrase rank or interval differs from the sentence being reviewed.")
            intervals.append((lo,hi))
    if len(local["transitions"])!=max(0,len(local["sentences"])-1):
        raise ValueError("The coherence readout does not cover the expected sentence boundaries.")
    for i,(t,s) in enumerate(zip(local["transitions"],local["sentences"][1:],strict=True),2):
        if (t["start"],t["end"],t["sentence_index"])!=(s["start"],s["end"],i):
            raise ValueError("A coherence score refers to another sentence.")
        if (t["risk"] is None)!=(t["flagged"] is None) or (t["risk"] is not None and
                (not 0<=t["risk"]<=1 or t["flagged"]!=(t["risk"]>t["threshold"]))):
            raise ValueError("Invalid coherence probability or alert threshold.")


    last = 0
    for word in local.get("narrative_words", []):
        start, end = word["start"], word["end"]
        if (type(start) is not int or type(end) is not int or not last <= start < end <= len(text)
                or not isfinite(word["risk"]) or not 0 <= word["risk"] <= 1 or type(word["flagged"]) is not bool):
            raise ValueError("An unexplained-character estimate falls outside the draft.")
        last = end


def sentence_markup(text, sentence, characters, terms=(), references=()):
    start, end = sentence["start"], sentence["end"]
    phrase = sentence["phrases"][0] if sentence["phrases"] else None
    bounds = {start, end}
    if phrase:
        bounds.update((phrase["start"], phrase["end"]))
    for word in characters:
        bounds.update((word["start"], word["end"]))
    for a,b,_ in (*terms,*references):
        bounds.update((max(start,a),min(end,b)))
    parts = []
    edges = sorted(bounds)
    for left, right in zip(edges, edges[1:]):
        css, notes = [], []
        if phrase and phrase["start"] <= left < phrase["end"]:
            css.append("phrase")
            notes.append("Top wording candidate within this sentence")
        if any(w["start"] <= left < w["end"] for w in characters):
            css.append("character")
            notes.append("This person may need an introduction")
        for kind in ("source","cue"):
            if any(k==kind and a<=left<b for a,b,k in terms):
                css.append("term-"+kind)
                notes.append("Earlier technical term" if kind=="source" else "Possible explanatory wording")
        css.extend(key for a,b,key in references if a<=left<b)
        value = source_html(text[left:right])
        if css:
            value = (f'<span class="{" ".join(css)}" data-start="{left}" data-end="{right}" '
                     f'title="{". ".join(notes)}">{value}</span>')
        parts.append(value)
    return "".join(parts)


def render(text, local):
    validate(text,local)
    sentences=local["sentences"]
    if not sentences:
        raise ValueError("Enter an English passage containing at least one word.")
    term_record=local.get("terminology")
    panel=terminology_view.markup(text,term_record) if term_record is not None else ""
    marks=terminology_view.marked_spans(term_record) if term_record is not None else []
    ref_spans,ref_panel=reference_view.result(text,local["references"]) if "references" in local else ([], "")
    expectations, expectation_panel = (expectation_view.present(text, local["expectations"])
                                       if "expectations" in local else ([], ""))
    summary_panel = summary_view.panel(local["summary"]) if "summary" in local else ""
    chunks=[]
    end=0
    for i,s in enumerate(sentences,1):
        a,b=s["start"],s["end"]
        chunks.append(source_html(text[end:a]))
        rating=s["rating"]
        if rating is None:
            color="transparent"
            tip=f"Sentence {i}. No rated endpoint for this sentence."
        else:
            color=fixed_color(rating["ease"])
            tip=f"Sentence {i}. Estimated ease {rating['ease']:.2f} of 5."
        flag=i>1 and local["transitions"][i-2]["flagged"]
        if flag:
            tip+=" Check how this sentence connects to the preceding text."
        css=' class="connection"' if flag else ""
        chunks.append(f'<mark{css} style="background:{color}" data-start="{a}" data-end="{b}" title="{tip}">')
        characters = [w for w in local.get("narrative_words", []) if w["flagged"] and a <= w["start"] < w["end"] <= b]
        chunks.append(sentence_markup(text,s,characters,[m for m in marks if m[0]<b and m[1]>a],
                                      [m for m in (*ref_spans,*expectations) if m[0]<b and m[1]>a]))
        chunks.append("</mark>")
        end=b
    chunks.append(source_html(text[end:]))
    ranked=sorted(((s["rating"]["ease"],i,s) for i,s in enumerate(sentences,1) if s["rating"] is not None),
                  key=lambda row:(row[0],row[1]))
    detail=[]
    for ease,i,s in ranked[:3]:
        phrase=s["phrases"][0] if s["phrases"] else s
        label=source_html(text[phrase["start"]:phrase["end"]])
        detail.append(f'<li>Sentence {i} · <strong>{ease:.2f}/5</strong> ease · “{label}”</li>')
    short=(f'<details><summary>Lowest estimated ease ({min(3,len(ranked))} of {len(ranked)} rated sentences)</summary>'
           f'<ol class="review-list">{"".join(detail)}</ol></details>') if ranked else ""
    key=''.join(f'<span><i class="key-swatch" style="background:{COLORS[i]}" aria-hidden="true"></i>{label}</span>'
                for i,label in ((0,"Harder · 1"),(2,"3"),(4,"Easier · 5")))
    narrative_key = (' <span class="character">Purple overline</span> · a person may need an introduction.'
                     if "narrative_words" in local else "")
    term_key = (' <span class="term-source">Solid green</span> · earlier term. '
                '<span class="term-cue">Dotted green</span> · possible explanatory wording.' if marks else "")
    expectation_key = (' <span class="expectation-increase">Solid brown</span> · next guesses spread out. '
                       '<span class="expectation-decrease">Dotted blue</span> · guesses narrow. '
                       'These marks show changes of at least 20 percentage points.' if expectation_panel else "")
    return (f'<section class="writing-review" data-model="{escape(local["bundle_sha256"],quote=True)}">'
            '<h2>Reviewed draft</h2>'
            f'<div class="reading-key" aria-label="Estimated sentence ease"><strong>Sentence ease</strong>{key}</div>'
            '<p class="reading-caption"><span class="key-description"><span class="phrase">Underline</span> · wording to review</span> '
            '<span class="key-description"><span class="connection">Dashed outline</span> · check the connection '
            f'to earlier sentences</span>{narrative_key}{term_key}{expectation_key}</p>'
            f'<div class="draft" aria-label="Original passage">{"".join(chunks)}</div>{summary_panel}{expectation_panel}{ref_panel}{panel}{short}</section>')


def build_app(runner, *, study_tools=True, max_tokens=2048):
    import gradio as gr
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("The interface needs a positive model token limit.")

    with gr.Blocks(title="CogLLM Writing Review",analytics_enabled=False) as ui:
        gr.Markdown(
            "# CogLLM Writing Review\n"
            "See where an English draft may be harder to read and which sentences may be hard to connect. "
            "Review the example below or replace it with your own text.",
            elem_id="writer-heading",
        )
        article = gr.Textbox(
            label="English draft", lines=5, value=EXAMPLE, placeholder="Paste your English draft.",
            info=f"Up to {max_tokens:,} model tokens. A token can be a word, part of a word, or punctuation.",
        )
        with gr.Row(elem_id="writer-actions"):
            submit = gr.Button("Review draft", variant="primary", scale=2)
            demo = gr.Button("Load example", scale=1)
        result = gr.HTML(elem_id="writer-result")
        with gr.Accordion("How to read the marks", open=False):
            gr.Markdown(
                "**Sentence color.** Orange means lower estimated reading ease. Pale green means higher ease. "
                "The scale comes from readers’ ratings of biomedical abstracts.\n\n"
                "**Underlined words.** Consider whether your audience knows this expression. Given the sentence, "
                "the model’s first suggestion overlapped 45 of 94 expressions that readers named as difficult.\n\n"
                "**Dashed outline.** Reread this sentence with the one before it. In 100 student essays, "
                "this alert found 114 of 446 editor-marked breaks and flagged 131 of 906 unmarked boundaries.\n\n"
                f"[Model and evaluation]({MODEL_CARD})"
            )
        if not study_tools:
            def review_draft(text):
                if not isinstance(text, str) or not text.strip():
                    raise gr.Error("Paste an English passage or load the example.")
                try:
                    reviewed = runner.review_english(text)
                    analysis = parse_analysis(text, reviewed)
                    return render(text, reviewed["english"]) + experimental_view.panel(analysis)
                except ValueError as error:
                    raise gr.Error(escape(str(error))) from error

            submit.click(review_draft, inputs=article, outputs=result, api_name="review_english",
                         concurrency_limit=1)
            article.change(fn=None, inputs=None, outputs=result, js="() => ''",
                           queue=False, api_visibility="private")
            demo.click(lambda: (EXAMPLE, ""), outputs=[article, result], api_visibility="private",
                       show_progress="hidden", queue=False)
            return ui
        gr.Markdown("## Additional reading tasks\nSelect any that apply to your draft, then choose **Review draft**.")
        summary = gr.Checkbox(label="Check whether this summary is understandable", value=False,
                              info="For a short English summary. Estimates a reader’s verdict on the whole draft.")
        expectations = gr.Checkbox(label="Show where readers’ next guesses change", value=False,
                                   info="For a short English passage. See the estimates alongside the original words.")
        narrative = gr.Checkbox(label="Check character introductions in narrative summaries", value=False,
                                info="Marks English names or references that may leave readers wondering who the person is.")
        terminology = gr.Checkbox(label="Track technical terms and their explanations",value=False,
                                  info="For English passages up to 1,024 model tokens.")
        references = gr.Checkbox(label="Inspect a pronoun that refers back",value=False,
                                 info="Choose the pronoun after reviewing your draft.")
        with gr.Accordion("Experimental token signals", open=False):
            gr.Markdown("These signals track model estimates of questions opening or closing. "
                        "In an eight-passage-pair test, the current curve identified both question–answer links in zero pairs.")
            trace=gr.HTML(elem_id="cogllm-reading")
            plot=gr.LinePlot(x="token",y="score",color="signal",height=300,
                             x_title="Token position",y_title="Model score",
                             color_map={label:color for label,color in CHANNELS.values()})
        learner_view.add_learner_check(runner)
        physical_view.add_physical_check(runner)
        def review(text, show_narrative=False, show_terminology=False, show_references=False, show_expectations=False, show_summary=False):
            if not isinstance(text,str) or not text.strip():
                raise gr.Error("Paste an English passage or load the example.")
            try:
                options={}
                if show_narrative:
                    options["narrative"]=True
                if show_terminology:
                    options["terminology"]=True
                if show_references:
                    options["references"]=True
                if show_expectations:
                    options["expectations"]=True
                if show_summary:
                    options["summary"]=True
                raw=runner.review_english(text,**options)
                analysis=parse_analysis(text,raw)
                selected,_=initial_position(analysis)
                return render(text,raw["english"]),render_article(analysis,selected),plot_rows(analysis)
            except ValueError as error:
                raise gr.Error(escape(str(error))) from error
        submit.click(review,[article,narrative,terminology,references,expectations,summary],[result,trace,plot],api_name="review_english",concurrency_limit=1)
        def load_example(show_narrative, show_terminology, show_references, show_expectations, show_summary):
            if show_narrative or show_references:
                return NARRATIVE_EXAMPLE, "", "", None
            if show_expectations and not show_terminology:
                return expectation_view.EXAMPLE, "", "", None
            if show_summary and not show_terminology:
                return summary_view.EXAMPLE, "", "", None
            return TERMINOLOGY_EXAMPLE if show_terminology else EXAMPLE, "", "", None
        demo.click(load_example, inputs=[narrative,terminology,references,expectations,summary], outputs=[article,result,trace,plot], api_name=False, show_progress="hidden")
    return ui
