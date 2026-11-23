"""Gradio surface for an injected, checkpoint-backed token readout."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from html import escape
from importlib import import_module
from math import floor, isfinite
from numbers import Real
from pathlib import Path


CHANNELS = {
    'unresolved': ('Still unexplained', '#168577'),
    'opened': ('Question raised', '#8e63bc'),
    'resolved': ('Question resolved', '#357fbe'),
    'transition': ('Transition signal', '#bd622e'),
}

EXAMPLE_TEXT = (
    'The monitor displayed a warning during the second run. '
    'The team compared the two readings and checked the calibration record. '
    'They eventually found that a loose sensor cable had triggered the warning.'
)

CSS = """
.gradio-container { max-width: 1160px !important; }
#cogllm-heading { margin-bottom: .5rem; }
#cogllm-heading h1 { letter-spacing: -.04em; font-weight: 650; }
#cogllm-reading .trace-panel {
    color: #182b29; background: #fbfcf9; border: 1px solid #dce3da;
    border-radius: .9rem;
}
#cogllm-reading .trace-legend {
    padding: .85rem .95rem; border-bottom: 1px solid #e1e8df;
    font-size: .88rem;
}
#cogllm-reading .trace-legend p, #cogllm-reading .trace-legend span {
    color: #263b34 !important;
}
#cogllm-reading .trace-legend p { margin: 0 0 .55rem; line-height: 1.55; }
#cogllm-reading .trace-legend .trace-keys {
    display: flex; flex-wrap: wrap; align-items: center; gap: .45rem .8rem;
}
#cogllm-reading .trace-legend span { border-radius: .3rem; padding: .13rem .38rem; }
#cogllm-reading .trace-text {
    white-space: pre-wrap; overflow-wrap: anywhere; line-height: 2.05;
    margin: 0 auto; padding: 1rem .95rem 1.15rem; max-width: 88ch;
    color: #182b29 !important;
}
#cogllm-reading .trace-text mark {
    color: #182b29 !important; border-radius: .16em; padding-block: .13em;
    -webkit-box-decoration-break: clone; box-decoration-break: clone;
}
#cogllm-reading .tone-0 { background: transparent; }
#cogllm-reading .tone-1 { background: #f7f1e2; }
#cogllm-reading .tone-2 { background: #f8e9c8; }
#cogllm-reading .tone-3 { background: #f8dca5; }
#cogllm-reading .tone-4 { background: #f7c88c; }
#cogllm-reading .tone-5 { background: #f5b676; }
#cogllm-reading .point-mark, #cogllm-reading .point-key {
    border-bottom: 2px solid #124f45;
}
#cogllm-reading .context-mark, #cogllm-reading .context-key {
    outline: 1px dashed #31659a; outline-offset: .1em;
}
#cogllm-details .trace-detail {
    padding: .1rem 0; display: grid; gap: .28rem; overflow-wrap: anywhere;
}
#cogllm-details .trace-detail p { margin: 0; }
#cogllm-details .trace-context-help { color: var(--body-text-color-subdued); font-size: .86rem; }
#cogllm-details .trace-diagnostics summary { cursor: pointer; font-size: .86rem; }
#cogllm-details .trace-diagnostics[open] .trace-values { margin-top: .45rem; }
#cogllm-details .trace-values {
    display: flex; flex-wrap: wrap; gap: .25rem .72rem;
    color: var(--body-text-color-subdued); font-size: .85rem;
}
#cogllm-details .trace-channel { display: inline-flex; gap: .35rem; align-items: center; }
#cogllm-details .trace-swatch { width: .6em; height: .6em; border-radius: 50%; }
#cogllm-details .trace-quote { white-space: pre-wrap; }
@media (max-width: 550px) { #cogllm-reading .trace-text { padding-inline: .68rem; } }
"""


@dataclass(frozen=True, slots=True)
class Token:
    start: int
    end: int
    unresolved: float
    opened: float
    resolved: float
    transition: float
    pointer_start: int | None
    pointer_end: int | None


@dataclass(frozen=True, slots=True)
class Analysis:
    text: str
    checkpoint: str
    tokens: tuple[Token, ...]


def _score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError('Model scores must be numeric.')
    result = float(value)
    if not isfinite(result) or result < 0:
        raise ValueError('Model scores must be finite and nonnegative.')
    return result


def parse_analysis(text: str, payload: Mapping) -> Analysis:
    """Check exact input round-trip, causal pointer spans, and display coordinates."""
    if not isinstance(payload, Mapping) or payload.get('text') != text:
        raise ValueError('The analysis did not preserve the original passage.')
    checkpoint = payload.get('checkpoint')
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise ValueError('The analysis did not identify its model checkpoint.')
    rows = payload.get('tokens')
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ValueError('The passage contains no text the model can analyze.')
    previous_start = 0
    tokens = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError('The analysis is missing scores or source positions.')
        start, end = row.get('start'), row.get('end')
        if type(start) is not int or type(end) is not int or not previous_start <= start <= end <= len(text):
            raise ValueError('The analysis returned invalid or out-of-order source positions.')
        pointer_start, pointer_end = row.get('pointer_start'), row.get('pointer_end')
        if (pointer_start is not None or pointer_end is not None) and (
            type(pointer_start) is not int or type(pointer_end) is not int
            or not 0 <= pointer_start < pointer_end <= end
        ):
            raise ValueError('The analysis linked to text outside the current reading position.')
        scores = {key: _score(row.get(key)) for key in CHANNELS}
        tokens.append(Token(start, end, **scores, pointer_start=pointer_start, pointer_end=pointer_end))
        previous_start = start
    return Analysis(text, checkpoint, tuple(tokens))


def source_html(text: str) -> str:
    """Preserve literal carriage returns through HTML's newline normalization."""
    return escape(text, quote=False).replace('\r', '&#13;')


def render_article(analysis: Analysis, selected: int) -> str:
    text, tokens = analysis.text, analysis.tokens
    current = tokens[selected]
    starts, ends = defaultdict(list), defaultdict(list)
    boundaries = {0, len(text), current.start, current.end}
    if current.pointer_start is not None:
        boundaries.update((current.pointer_start, current.pointer_end))
    for index, token in enumerate(tokens):
        if token.start < token.end:
            starts[token.start].append(index)
            ends[token.end].append(index)
            boundaries.update((token.start, token.end))

    parts = [f'<section class="trace-panel" data-checkpoint="{escape(analysis.checkpoint, quote=True)}">',
             '<div class="trace-legend" role="group" aria-label="Highlighting guide">',
             '<p>Darker shading means the model estimates that more is still unexplained at that point. '
             'Lighter shading later in the text means the estimate has fallen. Compare colors within this passage.</p>',
             '<div class="trace-keys"><span class="point-key">Underline · Selected text</span>',
             '<span class="context-key">Dashed outline · Related earlier text</span></div></div>',
             '<div class="trace-text" aria-label="Analyzed passage">']
    maximum = max(token.unresolved for token in tokens)
    ordered = sorted(boundaries)
    active = set()
    for start, end in zip(ordered, ordered[1:]):
        active.difference_update(ends[start])
        active.update(starts[start])
        is_current = current.start <= start < end <= current.end
        is_context = current.pointer_start is not None and current.pointer_start <= start < end <= current.pointer_end
        content = source_html(text[start:end])
        if not active and not is_current and not is_context:
            parts.append(content)
            continue
        classes = []
        attributes = [f'data-start="{start}"', f'data-end="{end}"']
        if active:
            index = max(active, key=lambda item: (tokens[item].unresolved, item))
            score = tokens[index].unresolved
            tone = 0 if maximum == 0 else min(5, floor(5 * (score / maximum) + .5))
            classes.append(f'tone-{tone}')
            attributes.extend((f'data-token="{index + 1}"', f'data-unresolved="{score:.9g}"',
                               f'title="Still unexplained at this point {score:.3f}"'))
        if is_current:
            classes.append('point-mark')
            attributes.append('aria-current="location"')
        if is_context:
            classes.append('context-mark')
        parts.append(f'<mark class="{" ".join(classes)}" {" ".join(attributes)}>{content}</mark>')
    parts.append('</div></section>')
    return ''.join(parts)


def render_detail(analysis: Analysis, selected: int) -> str:
    token = analysis.tokens[selected]
    text = analysis.text
    context_start, context_end = max(0, token.start - 12), min(len(text), token.end + 12)
    snippet = (('…' if context_start else '') + source_html(text[context_start:token.start])
               + '<strong>' + source_html(text[token.start:token.end]) + '</strong>'
               + source_html(text[token.end:context_end]) + ('…' if context_end < len(text) else ''))
    scores = ''.join(
        f'<span class="trace-channel"><span class="trace-swatch" style="background:{color}" '
        f'aria-hidden="true"></span>{label} {getattr(token, key):.3f}</span>'
        for key, (label, color) in CHANNELS.items()
    )
    detail = [f'<div class="trace-detail" role="status" data-selected="{selected + 1}">',
              f'<p class="trace-quote">Selected text · “{snippet}”</p>']
    if token.pointer_start is not None:
        pointer = source_html(text[token.pointer_start:token.pointer_end])
        detail.append(f'<p class="trace-quote" data-pointer-start="{token.pointer_start}" '
                      f'data-pointer-end="{token.pointer_end}"><strong>Related earlier text</strong> · “{pointer}”</p>')
        detail.append('<p class="trace-context-help">The model suggests rereading the two marked parts together.</p>')
    detail.extend(('<details class="trace-diagnostics"><summary>Model scores at this position</summary>',
                   f'<p class="trace-values">{scores}</p></details></div>'))
    return ''.join(detail)


def initial_position(analysis: Analysis) -> tuple[int, bool]:
    """Start at a local rise so cumulative color alone does not select the end."""
    tokens = analysis.tokens
    candidates = [i for i, token in enumerate(tokens) if analysis.text[token.start:token.end].strip()]

    def increase(index):
        return tokens[index].unresolved - tokens[index - 1].unresolved if index else 0.0

    selected = max(candidates, key=increase, default=0)
    has_rise = bool(candidates and increase(selected) > 0)
    return (selected if has_rise else (candidates or [0])[0]), has_rise


def plot_rows(analysis: Analysis):
    import pandas as pd

    return pd.DataFrame.from_records(
        ({'token': index, 'score': getattr(token, key), 'signal': label}
         for index, token in enumerate(analysis.tokens, 1) for key, (label, _) in CHANNELS.items()),
        columns=('token', 'score', 'signal'),
    )


def build_app(analyze: Callable[[str], Mapping]):
    import gradio as gr

    with gr.Blocks(title='CogLLM Reading Review', analytics_enabled=False) as ui:
        gr.Markdown('# CogLLM Reading Review\n'
                    'Analyze a passage to see where the model estimates an explanation is still pending.',
                    elem_id='cogllm-heading')
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=270):
                article = gr.Textbox(label='Your passage', lines=9, max_lines=18,
                                     placeholder='Paste a passage to review')
                run = gr.Button('Analyze passage', variant='primary')
                example = gr.Button('Load example', variant='secondary', size='sm')
                gr.Markdown('In the example, you might wonder what caused the warning. '
                            'The last sentence explains it. Load the example, then select **Analyze passage**.')

            with gr.Column(scale=2, min_width=320):
                status = gr.HTML()
                highlighted = gr.HTML(elem_id='cogllm-reading')
                cursor = gr.Slider(1, 2, value=1, step=1, precision=0, label='Explore a position in the passage',
                                   visible=False)
                detail = gr.HTML(elem_id='cogllm-details')
                with gr.Accordion('View the model curves', open=False):
                    gr.Markdown('Green tracks what the model estimates is still unexplained as you read. '
                                'Purple and blue estimate questions raised or resolved at each position. '
                                'Orange flags possible breaks in continuity.')
                    plot = gr.LinePlot(x='token', y='score', color='signal', label='Model signals by position',
                                       x_title='Position in the passage', y_title='Model score', color_title='Signal',
                                       show_label=True, height=330, visible=False,
                                       color_map={label: color for label, color in CHANNELS.values()})
        analysis_state = gr.State()

        def submit(text):
            if not isinstance(text, str) or not text.strip():
                raise gr.Error('Paste a passage or load the example.')
            try:
                analysis = parse_analysis(text, analyze(text))
            except ValueError as error:
                raise gr.Error(escape(str(error))) from error
            selected, has_rise = initial_position(analysis)
            count = len(analysis.tokens)
            note = ('<p role="status"><strong>Start at the underline.</strong> '
                    'The model’s estimate of what remains unexplained rises most here.</p>'
                    if has_rise else '<p role="status">Move the slider to explore the passage.</p>')
            return (
                analysis, render_article(analysis, selected), gr.update(value=plot_rows(analysis), visible=True),
                gr.update(maximum=max(count, 2), value=selected + 1, visible=True, interactive=count > 1),
                render_detail(analysis, selected), note,
            )

        def load_example():
            return (EXAMPLE_TEXT, None, '', gr.update(value=None, visible=False),
                    gr.update(maximum=2, value=1, visible=False), '', '')

        def inspect(analysis, index):
            if analysis is None or index is None or not 1 <= index <= len(analysis.tokens):
                return gr.skip(), gr.skip()
            selected = int(index) - 1
            return render_article(analysis, selected), render_detail(analysis, selected)

        run.click(submit, article, [analysis_state, highlighted, plot, cursor, detail, status],
                  api_name='analyze', concurrency_id='passage', concurrency_limit=1)
        example.click(load_example, outputs=[article, analysis_state, highlighted, plot, cursor, detail, status],
                      api_name='load_example', api_visibility='undocumented', show_progress='hidden',
                      concurrency_id='passage', concurrency_limit=1)
        cursor.input(inspect, [analysis_state, cursor], [highlighted, detail], api_name='select_token',
                     api_visibility='undocumented', show_progress='hidden', trigger_mode='always_last')
    return ui


def load_runner(spec: str, model: Path, checkpoint: Path, **options):
    """Resolve an explicitly selected runner factory for a real checkpoint."""
    module, separator, name = spec.partition(':')
    if not separator or not module or not name.isidentifier():
        raise ValueError('Runner must use module:factory syntax')
    factory = getattr(import_module(module), name)
    analyze = factory(model=model, checkpoint=checkpoint, **options)
    if not callable(analyze):
        raise TypeError('Runner factory must return a callable analysis function')
    return analyze


def main():
    import gradio as gr

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runner', default='research.token_readout.portable_runner:load',
                        help='Real candidate factory, module:factory')
    parser.add_argument('--view', choices=('writing', 'signals'), default='writing',
                        help='English reader-rated wording or the multilingual research curve')
    parser.add_argument('--model', type=Path, required=True, help='Local Qwen snapshot')
    parser.add_argument('--checkpoint', type=Path, required=True, help='Sealed token readout checkpoint')
    parser.add_argument('--narrative-operating-point', choices=('balanced','high_recall'), default='balanced',
                        help='Source-calibrated screening rule for character introductions')
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    css = CSS
    if args.view == 'writing':
        from research.reading_validation.local_inference import LocalInference
        from research.reading_validation.transition_inference import TransitionInference
        from research.reading_validation.narrative_inference import MarginalNarrativeInference
        from research.reading_validation.terminology_inference import TerminologyInference
        from research.token_readout import writing_review
        runner = load_runner(args.runner, args.model, args.checkpoint, english_head=LocalInference.load(),
                             transition_head=TransitionInference.load(), narrative_head=MarginalNarrativeInference.for_writers(args.narrative_operating_point),
                             terminology_head=TerminologyInference.load())
        ui = writing_review.build_app(runner)
        css += writing_review.STYLE
    else:
        ui = build_app(load_runner(args.runner, args.model, args.checkpoint))
    ui.launch(
        server_name='127.0.0.1', server_port=args.port, share=False, css=css,
        theme=gr.themes.Soft(primary_hue='teal', neutral_hue='slate'),
    )


if __name__ == '__main__':
    main()
