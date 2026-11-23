"""Display learner word reports for the single sentence used by the observation model."""

import re
from html import escape
from math import isfinite
from unicodedata import name

from research.reading_validation.local_forecast import WORD
from research.token_readout.app import source_html

LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")
POOLED = "unspecified"
EXAMPLE = "The translucent membrane allows light to pass through."
TITLE = "Which words might an English learner need help with?"
STOP = re.compile(r"""[.!?]+['"’”)]*(?=\s+\S)""")
ABBR = re.compile(
    r"(?:\b(?:mr|mrs|ms|dr|prof|jr|sr|vs|e\.g|i\.e)|(?:\b[A-Za-z]\.)+[A-Za-z]|\b[A-Z])\.$",
    re.I,
)
STYLE = """
.learner-result {background:#f8faf7;border:1px solid #cbd7cf;border-radius:.65rem;padding:1rem 1.2rem;}
.learner-result p,.learner-result .learner-draft {color:#24372e!important;}
.learner-result p {margin:0 0 .45rem;font-size:.9rem;}
.learner-result .learner-draft {white-space:pre-wrap;overflow-wrap:anywhere;line-height:3.1;font-size:1.13rem;margin-top:.8rem}
.learner-result .learner-word {display:inline-block;position:relative;line-height:1.45;min-width:1.35em;
 background:var(--learner-word-color);border-bottom:2px solid #889c8a;border-radius:.16rem;color:#20352a!important;}
.learner-result .learner-word::before {content:attr(data-rate);position:absolute;left:0;bottom:100%;white-space:nowrap;
 color:#3c5749;font-size:.62em;line-height:1.2;font-weight:650;font-variant-numeric:tabular-nums;}
.learner-result .high-word {border-bottom:3px solid #ac5f28;}
"""


def validate_sentence(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter one English sentence to check its words.")
    letters = [c for c in text if c.isalpha()]
    if not letters or sum("LATIN" in name(c, "") for c in letters) / len(letters) < 0.8:
        raise ValueError(
            "This learner check was trained on English. Enter one English sentence."
        )
    if any(not ABBR.search(text[: m.end()]) for m in STOP.finditer(text)) or re.search(
        r"\n[ \t]*\n\s*\S", text.strip()
    ):
        raise ValueError("Enter one sentence. Use Review draft above for a passage.")


def display(record, level=POOLED):
    if level not in (POOLED, *LEVELS):
        raise ValueError("Choose a listed CEFR level or use the full learner group.")
    if record["schema"] != "nns-self-report-en-v1":
        raise ValueError("The learner result has an unexpected model identity.")
    text = record["text"]
    scores = record["probabilities"][level]
    spans = record["positions"]
    if not spans or len(spans) != len(scores) or record["word_count"] < len(spans):
        raise ValueError(
            "There are no scorable learner word results for this sentence."
        )
    pieces = []
    left = 0
    for (a, b), p in zip(spans, scores, strict=True):
        if (
            type(a) is not int
            or type(b) is not int
            or not left <= a < b <= len(text)
            or not WORD.fullmatch(text[a:b])
            or not isfinite(p)
            or not 0 <= p <= 1
        ):
            raise ValueError("A learner score differs from its printed word.")
        prob = "<1%" if 0 < p < 0.005 else f"{p:.0%}"
        label = escape(
            f"{text[a:b]} · estimated fraction who would mark this word as unknown: {prob}",
            quote=True,
        )
        shade = tuple(
            round(x + (y - x) * (p**0.5))
            for x, y in zip((240, 245, 236), (225, 140, 69), strict=True)
        )
        color = "#" + "".join(f"{x:02x}" for x in shade)
        pieces.extend(
            [
                source_html(text[left:a]),
                f'<span class="learner-word{" high-word" if p >= 0.25 else ""}" style="--learner-word-color:{color}" '
                f'data-start="{a}" data-end="{b}" data-rate="{escape(prob, quote=True)}" '
                f'title="{label}" aria-label="{label}">{source_html(text[a:b])}</span>',
            ]
        )
        left = b
    pieces.append(source_html(text[left:]))
    group = (
        "all English learners in the study"
        if level == POOLED
        else f"English learners at {level}"
    )
    missing = record["word_count"] - len(spans)
    coverage = (
        f" {missing} {'word has' if missing == 1 else 'words have'} no score."
        if missing
        else ""
    )
    return (
        f'<section class="learner-result" data-model="{escape(record["checkpoint_sha256"], quote=True)}">'
        f"<p><strong>Estimated share who would mark each word as unknown</strong> · {group}.</p>"
        f"<p>Warmer colors indicate a higher chance. For example, 20% means about one in five learners.{coverage}</p>"
        f'<div class="learner-draft" aria-label="Original learner sentence">{"".join(pieces)}</div></section>'
    )


def add_learner_check(runner):
    import gradio as gr

    with gr.Accordion(TITLE, open=False):
        gr.Markdown(
            "Check one English sentence against learners’ own reports of words they did not understand."
        )
        text = gr.Textbox(label="English sentence", placeholder=EXAMPLE, lines=2)
        level = gr.Dropdown(
            label="Learner’s measured English level (CEFR)",
            choices=[("No level supplied", POOLED), *[(k, k) for k in LEVELS]],
            value=POOLED,
        )
        button = gr.Button("Check learner vocabulary", variant="secondary")
        result = gr.HTML()
        state = gr.State()

    def submit(sentence, measured):
        try:
            if measured not in (POOLED, *LEVELS):
                raise ValueError(
                    "Choose a listed CEFR level or use the full learner group."
                )
            packet = runner.learner_sentence(sentence)
            return packet, display(packet, measured)
        except ValueError as error:
            raise gr.Error(escape(str(error))) from error

    def change(measured, sentence, record):
        return (
            display(record, measured) if record and record["text"] == sentence else ""
        )

    button.click(
        submit,
        [text, level],
        [state, result],
        api_name="learner_vocabulary",
        concurrency_limit=1,
    )
    level.input(
        change, [level, text, state], result, api_name=False, show_progress="hidden"
    )
    text.input(
        lambda: (None, ""),
        outputs=[state, result],
        api_name=False,
        show_progress="hidden",
    )
