"""Serve the CogLLM article analysis preview on localhost."""

import argparse
from collections import deque
from html import escape
from math import floor, isfinite
from pathlib import Path

import gradio as gr

from .causal_plaba import CausalArticleRatings, CausalPlabaReader
from .checkpoint import DEFAULT_CHECKPOINT
from .combined import CombinedArticleRatings, CombinedReader
from .lexical import LexicalRating
from .plaba import ArticleRatings, BASE_REVISION, PlabaReader
from .reference import ReferenceResult, phrase_locations


EASE = "閱讀容易度"
UNDERSTANDING = "理解自評"
ANCHORS = "疑問片語"
ANCHOR_LABEL = "提問指向候選"
LEXICAL = "詞彙難度"
REFERENCE = "上文指向"
CUES = "消歧線索候選"
CSS = """
.gradio-container { max-width: 1120px !important; }
#article-result .reading-overview {
    --hard: #ffc4b9; --medium-hard: #ffe2b6; --middle: #f3eee5;
    --medium-easy: #d6ecdc; --easy: #ace6d1;
    color: #172b26; background: #fffcf7;
    border: 1px solid #d9ddd5; border-radius: 1rem;
}
#article-result .heat-legend {
    display: flex; flex-wrap: wrap; align-items: center; gap: 0.45rem;
    padding: 0.85rem 1.15rem; border-bottom: 1px solid #e6e9e2;
    font-size: 0.84rem; line-height: 1.5;
}
#article-result .legend-key { padding: 0.2rem 0.52rem; border-radius: 0.35rem; font-weight: 650; }
#article-result .legend-note { color: #455953; }
#article-result .source-text {
    white-space: pre-wrap; overflow-wrap: anywhere; line-height: 2.15;
    padding: 1.1rem 1.25rem 1.25rem; max-width: 85ch; margin: 0 auto;
    font-size: 1.02rem;
}
#article-result .tone-0 { background-color: var(--hard); }
#article-result .tone-1 { background-color: var(--medium-hard); }
#article-result .tone-2 { background-color: var(--middle); }
#article-result .tone-3 { background-color: var(--medium-easy); }
#article-result .tone-4 { background-color: var(--easy); }
#article-result .sentence, #article-result .lexical {
    color: #172b26; border-radius: 0.22em; padding-block: 0.17em;
    -webkit-box-decoration-break: clone; box-decoration-break: clone;
}
#article-result .lexical { padding-inline: 0.05em; }
#article-result .lexical-underline {
    background-color: transparent;
    text-decoration-line: underline;
    text-decoration-style: solid;
    text-decoration-color: var(--lexical-ink);
    text-decoration-thickness: var(--lexical-weight);
    text-underline-offset: 0.15em;
}
#article-result mark.lexical-underline { padding: 0; }
#article-result .lexical-underline.tone-0 { --lexical-ink: #8e2d22; --lexical-weight: 0.16em; }
#article-result .lexical-underline.tone-1 { --lexical-ink: #744608; --lexical-weight: 0.125em; }
#article-result .lexical-underline.tone-2 { --lexical-ink: #526159; --lexical-weight: 0.09em; }
#article-result .lexical-underline.tone-3 { --lexical-ink: #3c6550; --lexical-weight: 0.075em; }
#article-result .lexical-underline.tone-4 { --lexical-ink: #11594a; --lexical-weight: 0.06em; }
#article-result .sentence::after {
    content: attr(data-score); display: inline-block; margin-inline: 0.35em 0.18em;
    padding-inline: 0.29em; border: 1px solid #46675c65; border-radius: 0.27em;
    background: #ffffffa8; color: #294840; vertical-align: 0.06em; line-height: 1.25;
    font: 650 0.71em/1.3 ui-monospace, monospace; white-space: nowrap;
}
#article-result .selected {
    text-decoration-line: underline;
    text-decoration-color: #255a4e;
    text-decoration-thickness: 2px;
    text-underline-offset: 0.23em;
}
#article-result .layered .sentence.selected { text-decoration: none; }
#article-result .layered .sentence.selected::after { border-color: #255a4e; background: #f4fffa; }
#article-result .anchor {
    color: #1e3048; background: #dce9fb; border-bottom: 2px solid #516d97;
    border-radius: 0.18em; padding-block: 0.16em;
    -webkit-box-decoration-break: clone; box-decoration-break: clone;
}
#article-result .anchor::after {
    content: attr(data-rank); margin-inline: 0.36em 0.15em; color: #365777;
    font: 650 0.72em/1.3 ui-monospace, monospace; white-space: nowrap;
}
#article-result .anchor.active { background: #c2d7f5; box-shadow: 0 0 0 2px #516d9780; }
#article-result .anchor-key { background: #dce9fb; color: #1e3048; }
#article-result .reference-query, #article-result .reference-query-key {
    background: #ffebbd; color: #36280d;
}
#article-result .reference-query {
    text-decoration: underline; text-underline-offset: 0.22em; text-decoration-thickness: 2px;
    -webkit-box-decoration-break: clone; box-decoration-break: clone;
}
#article-result .reference-word { border-bottom-width: 1px; }
#article-result .cue-candidate, #article-result .cue-key {
    color: #233553; border-radius: 0.18em;
}
#article-result .cue-candidate {
    padding-block: 0.12em; border-bottom: 2px solid #52688d;
    -webkit-box-decoration-break: clone; box-decoration-break: clone;
}
#article-result .cue-rank-1 { background: #cfdef9; }
#article-result .cue-rank-2 { background: #e2eafb; }
#article-result .cue-rank-3 { background: #eef2f9; }
#article-result .cue-candidate.cue-rank-2 { border-bottom-width: 1px; }
#article-result .cue-candidate.cue-rank-3 { border-bottom: 1px dashed #52688d; }
#article-result .cue-candidate::after {
    content: attr(data-rank); margin-inline: 0.24em 0.12em;
    font: 650 0.72em/1.3 ui-monospace, monospace; white-space: nowrap;
}
#article-result .cue-candidate:focus-visible { outline: 2px solid #52688d; outline-offset: 2px; }
"""


def sentence_numbers(analysis: ArticleRatings) -> tuple[int, ...]:
    return getattr(analysis, 'sentence_numbers', tuple(range(1, len(analysis.sentences) + 1)))


def source_html(text: str) -> str:
    # Preserve a literal carriage return across HTML parser line-ending normalization.
    return escape(text, quote=False).replace('\r', '&#13;')


def word_scale(words: tuple[LexicalRating, ...]) -> tuple[float, float]:
    scores = [round(word.difficulty, 2) for word in words]
    return min(scores, default=0), max(scores, default=0)


def lexical_legend(analysis: CombinedArticleRatings, scale: tuple[float, float], *, underline: bool) -> str:
    encoding = '底線' if underline else '底色'
    parts = [f'<div class="heat-legend" role="group" aria-label="詞彙難度{encoding}">']
    low, high = scale
    if analysis.lexical:
        accent = ' lexical-underline' if underline else ''
        keys = [(2, f'同分 {low:.2f}')] if low == high else [
            (4, f'詞較易 {low:.2f}'), (0, f'詞較難 {high:.2f}'),
        ]
        parts.extend(f'<span class="legend-key tone-{tone}{accent}">{label}</span>' for tone, label in keys)
        parts.append(f'<span class="legend-note">詞{encoding} · 詞末模型估計 · 0–1，1 最難 · 篇內相對色階 · 全篇標記屬實驗性</span>')
    else:
        parts.append('<span class="legend-note">沒有可標記的詞彙。</span>')
    if analysis.unavailable_words:
        parts.append(f'<span class="legend-note">{len(analysis.unavailable_words)} 個詞未精確對齊。</span>')
    parts.append('</div>')
    return ''.join(parts)


def render_word_range(text: str, start: int, end: int, words: deque[LexicalRating],
                      scale: tuple[float, float], *, underline: bool) -> str:
    """Consume the ordered word marks in one sentence or in the gap between sentences."""
    parts = []
    low, high = scale
    accent = ' lexical-underline' if underline else ''
    while words and words[0].start < end:
        word = words.popleft()
        if not start <= word.start < word.end <= end <= len(text):
            raise ValueError('Lexical marks must be disjoint and contained within a source range')
        score = round(word.difficulty, 2)
        tone = 2 if low == high else floor(4 * (high - score) / (high - low) + 0.5)
        title = f'模型估計{LEXICAL} {score:.2f}／1 · 原文 [{word.start}, {word.end}) · 詞末 token {word.token}'
        parts.append(source_html(text[start:word.start]))
        parts.append(f'<mark class="lexical tone-{tone}{accent}" data-score="{score:.2f}" '
                     f'data-start="{word.start}" data-end="{word.end}" data-token="{word.token}" '
                     f'title="{escape(title)}">{source_html(text[word.start:word.end])}</mark>')
        start = word.end
    return ''.join((*parts, source_html(text[start:end])))


def render_article(analysis: ArticleRatings, measure: str, selected: int) -> str:
    scores = [round(sentence.ease if measure == EASE else sentence.understanding, 2)
              for sentence in analysis.sentences]
    layered = isinstance(analysis, CombinedArticleRatings)
    words = deque(analysis.lexical if layered else ())
    scale = word_scale(analysis.lexical) if layered else (0., 0.)
    classes = 'reading-overview layered' if layered else 'reading-overview'
    parts = [f'<div class="{classes}">',
             '<div class="heat-legend" role="group" aria-label="篇內相對色階">']
    low, high = min(scores, default=0), max(scores, default=0)
    if not scores:
        parts.append('<span class="legend-note">沒有可對齊句末的句子。</span>')
    elif low == high:
        parts.append(f'<span class="legend-key tone-2">同分 {low:.2f}</span>')
    else:
        below, above = ('篇內較難', '篇內較易') if measure == EASE else ('自評較低', '自評較高')
        parts.extend((f'<span class="legend-key tone-0">{below} {low:.2f}</span>',
                      f'<span class="legend-key tone-4">{above} {high:.2f}</span>'))
    timing = '每句估計只使用讀到該句句末的文字' if isinstance(analysis, CausalArticleRatings) else '模型使用全文'
    parts.extend((f'<span class="legend-note">句底色 · 模型估計{escape(measure)} · 1–5 分 · 篇內相對色階</span>',
                  f'<span class="legend-note">{timing}。參照全文可見時的英語生醫摘要讀者群體自評。'
                  '跨領域與其他語言的結果屬實驗性。</span></div>'))
    if layered:
        parts.append(lexical_legend(analysis, scale, underline=True))
    parts.append('<div class="source-text" aria-label="已分析原文">')
    previous_end = 0
    for index, (number, sentence, score) in enumerate(zip(sentence_numbers(analysis), analysis.sentences, scores, strict=True)):
        if not previous_end <= sentence.start < sentence.end <= len(analysis.text):
            raise ValueError('Sentence marks must be ordered and disjoint')
        tone = 2 if low == high else floor(4 * (score - low) / (high - low) + 0.5)
        parts.append(render_word_range(analysis.text, previous_end, sentence.start, words, scale, underline=layered))
        css_class = f'sentence tone-{tone}' + (' selected' if index == selected else '')
        rendered = render_word_range(analysis.text, sentence.start, sentence.end, words, scale, underline=layered)
        title = f'第 {number} 句 · 模型估計{measure} {score:.2f}／5'
        token = f' data-token="{analysis.sentence_tokens[index]}"' if isinstance(analysis, CausalArticleRatings) else ''
        parts.append(f'<mark class="{css_class}" data-score="{score:.2f}" '
                     f'data-start="{sentence.start}" data-end="{sentence.end}"{token} '
                     f'title="{escape(title, quote=True)}">{rendered}</mark>')
        previous_end = sentence.end
    parts.extend((render_word_range(analysis.text, previous_end, len(analysis.text), words, scale, underline=layered),
                  '</div></div>'))
    if words:
        raise ValueError('Some lexical marks are outside the original text')
    return "".join(parts)


def sentence_table(analysis: ArticleRatings) -> list[list]:
    rows = []
    for number, sentence in zip(sentence_numbers(analysis), analysis.sentences, strict=True):
        preview = " ".join(analysis.text[sentence.start:sentence.end].split())
        if len(preview) > 72:
            preview = preview[:71] + "…"
        rows.append([number, preview, round(sentence.ease, 2), round(sentence.understanding, 2)])
    return rows


def sentence_detail(analysis: ArticleRatings, selected: int) -> str:
    if not analysis.sentences:
        return '沒有可對齊句末的句子。'
    sentence = analysis.sentences[selected]
    detail = (
        f"**第 {sentence_numbers(analysis)[selected]} 句** · 模型估計 · 閱讀容易度 **{sentence.ease:.2f}** · "
        f"理解自評 **{sentence.understanding:.2f}**  \n"
        f"原文字元範圍 `[{sentence.start}, {sentence.end})`（從 0 起算，不含終點）"
    )
    if isinstance(analysis, CausalArticleRatings):
        detail += f' · 句末 token `{analysis.sentence_tokens[selected]}`'
    if isinstance(analysis, CombinedArticleRatings):
        distribution = analysis.distributions[selected]
        lines = [detail, '', '**實驗性評分分布** · 估計群體給各分數的比例', '',
                 '| 評分 | 1 | 2 | 3 | 4 | 5 |', '| --- | ---: | ---: | ---: | ---: | ---: |']
        for label, probabilities in ((EASE, distribution.ease), (UNDERSTANDING, distribution.understanding)):
            lines.append(f'| {label} | ' + ' | '.join(f'{value:.1%}' for value in probabilities) + ' |')
        detail = '\n'.join(lines)
    return detail


def render_anchor_article(analysis: CombinedArticleRatings, selected: int) -> str:
    parts = ['<div class="reading-overview">',
             f'<div class="heat-legend" role="group" aria-label="{ANCHOR_LABEL}">',
             f'<span class="legend-key anchor-key">{ANCHOR_LABEL}</span>',
             '<span class="legend-note">假設已有提問，模型依段末可見文字排序本段的可能指向。'
             '證據來自 IIRC 單段提問。</span></div>',
             '<div class="source-text" aria-label="已分析原文">']
    end = 0
    for index, proposal in sorted(enumerate(analysis.anchors), key=lambda row: row[1].start):
        if not end <= proposal.start < proposal.end <= len(analysis.text):
            raise ValueError('Anchor display requires valid, disjoint source spans')
        parts.append(source_html(analysis.text[end:proposal.start]))
        tag = 'anchor active' if index == selected else 'anchor'
        title = f'第 {proposal.paragraph_number} 段 · 提問指向候選 {proposal.rank}'
        if proposal.experimental_context:
            title += ' · 跨段實驗性'
        parts.append(f'<mark class="{tag}" data-rank="{proposal.rank}" data-start="{proposal.start}" '
                     f'data-end="{proposal.end}" data-stop="{proposal.visible_end}" title="{escape(title, quote=True)}">'
                     f'{source_html(analysis.text[proposal.start:proposal.end])}</mark>')
        end = proposal.end
    parts.extend((source_html(analysis.text[end:]), '</div></div>'))
    return ''.join(parts)


def render_lexical_article(analysis: CombinedArticleRatings) -> str:
    words = deque(analysis.lexical)
    scale = word_scale(analysis.lexical)
    source = render_word_range(analysis.text, 0, len(analysis.text), words, scale, underline=False)
    if words:
        raise ValueError('Some lexical marks are outside the original text')
    return ('<div class="reading-overview">'
            f'{lexical_legend(analysis, scale, underline=False)}'
            f'<div class="source-text" aria-label="已分析原文">{source}</div></div>')


def render_cue_article(analysis: CombinedArticleRatings) -> str:
    candidates = [item for item in analysis.cues if 1 <= item.rank <= 3]
    parts = ['<div class="reading-overview">',
             f'<div class="heat-legend" role="group" aria-label="{CUES}">']
    if candidates:
        parts.extend(f'<span class="legend-key cue-key cue-rank-{rank}">第 {rank} 名</span>'
                     for rank in range(1, 4))
        parts.append('<span class="legend-note">每句最多標出三個候選詞，數字是句內名次。</span>')
    else:
        parts.append('<span class="legend-note">沒有可精確對齊的候選詞。</span>')
    parts.extend(('<span class="legend-note">依據 Blott 的英文受控句子。讀者何時察覺疑問、何時確認解開，需另有觀測。'
                  '</span></div>', '<div class="source-text" aria-label="已分析原文">'))
    end = 0
    for item in candidates:
        if not (end <= item.start < item.end <= len(analysis.text) and isfinite(item.score)):
            raise ValueError('Cue display requires disjoint source spans and finite scores')
        label = f'第 {item.sentence_number} 句 · {CUES}第 {item.rank} 名'
        title = f'{label} · 原始分數 {item.score} · 原文 [{item.start}, {item.end}) · 詞末 token {item.token}'
        word = analysis.text[item.start:item.end]
        accessible_label = escape(f'{word} · {label}')
        parts.append(source_html(analysis.text[end:item.start]))
        parts.append(f'<mark class="cue-candidate cue-rank-{item.rank}" tabindex="0" '
                     f'data-rank="{item.rank}" data-sentence="{item.sentence_number}" data-score="{item.score}" '
                     f'data-start="{item.start}" data-end="{item.end}" data-token="{item.token}" '
                     f'aria-label="{accessible_label}" title="{escape(title)}">{source_html(word)}</mark>')
        end = item.end
    parts.extend((source_html(analysis.text[end:]), '</div></div>'))
    return ''.join(parts)


def anchor_table(analysis: CombinedArticleRatings) -> list[list]:
    return [[index, span.paragraph_number, span.rank, analysis.text[span.start:span.end],
             round(span.score, 5), '跨段實驗性' if span.experimental_context else '單段']
            for index, span in enumerate(analysis.anchors, start=1)]


def anchor_detail(analysis: CombinedArticleRatings, index: int) -> str:
    span = analysis.anchors[index]
    return (f'**第 {span.paragraph_number} 段 · 候選 {span.rank}**  \n'
            f'片語 `[{span.start}, {span.end})` · 可見段末 `{span.visible_end}`'
            + (' · 跨段實驗性' if span.experimental_context else ''))


def render_reference_article(text: str, reference: ReferenceResult | None) -> str:
    parts = ['<div class="reading-overview"><div class="heat-legend" role="group" aria-label="上文指向圖例">']
    spans = []
    if reference is None:
        parts.append('<span class="legend-note">在上方輸入原文中的完整片語以查看上文候選。</span>')
    else:
        parts.extend(('<span class="legend-key reference-query-key">指定片語</span>',
                      '<span class="legend-key anchor-key">上文候選詞</span>'))
        if not reference.candidates:
            parts.append('<span class="legend-note">片語前沒有可對齊的候選詞。</span>')
        spans = [(candidate.start, candidate.end, f'<mark class="anchor reference-word" data-rank="{candidate.rank}"')
                 for candidate in reference.candidates]
        spans.append((reference.start, reference.end, '<mark class="reference-query"'))
    parts.append('</div><div class="source-text" lang="en">')
    cursor = 0
    for start, end, tag in sorted(spans):
        parts.extend((escape(text[cursor:start]), tag, f' data-start="{start}" data-end="{end}">',
                      escape(text[start:end]), '</mark>'))
        cursor = end
    parts.extend((escape(text[cursor:]), '</div></div>'))
    return ''.join(parts).replace('\r', '&#13;')


def build_app(reader: PlabaReader | CausalPlabaReader) -> gr.Blocks:
    combined = isinstance(reader, CombinedReader)
    reference_enabled = combined and reader.reference_head is not None
    cues_enabled = combined and reader.cue_head is not None
    prefix_readout = isinstance(reader, CausalPlabaReader)
    mode = ('全文閱讀標色' if combined else
            ('因果句末預覽' if prefix_readout else '全文句級預覽'))
    scope = ('模型只讀到各句句末。真人評分是在全文可見時取得。' if prefix_readout
             else '模型先看完整篇章，再讀各焦點句。')

    def render(analysis: ArticleRatings, metric: str, sentence_index: int, anchor_index: int,
               reference: ReferenceResult | None = None) -> str:
        if reference_enabled and metric == REFERENCE:
            return render_reference_article(analysis.text, reference)
        if isinstance(analysis, CombinedArticleRatings):
            if cues_enabled and metric == CUES:
                return render_cue_article(analysis)
            if metric == ANCHORS:
                return render_anchor_article(analysis, anchor_index)
            if metric == LEXICAL:
                return render_lexical_article(analysis)
        return render_article(analysis, metric, sentence_index)

    with gr.Blocks(title="CogLLM 文章分析", analytics_enabled=False) as demo:
        gr.Markdown(
            "# CogLLM 文章分析\n"
            f"**{mode}**"
        )
        article = gr.Textbox(label="文章正文", placeholder="貼上英文文章", lines=5)
        run = gr.Button("分析文章", variant="primary")
        result = gr.State()
        selected = gr.State(0)
        selected_anchor = gr.State(-1)
        status = gr.Markdown()
        choices = [EASE, UNDERSTANDING, LEXICAL, (ANCHOR_LABEL, ANCHORS)] if combined else [EASE, UNDERSTANDING]
        if reference_enabled:
            choices.append(REFERENCE)
        if cues_enabled:
            choices.append(CUES)
        measure = gr.Radio(choices, value=EASE, label="原文標記")
        if reference_enabled:
            reference_result = gr.State()
            with gr.Accordion('查指定片語的上文', open=False):
                with gr.Row():
                    reference_phrase = gr.Textbox(label='原文片語', placeholder='例如 the door', max_lines=1, scale=4)
                    search_reference = gr.Button('查前文', scale=1, min_width=110)
                occurrence = gr.Dropdown(label='片語出現位置', choices=[], visible=False)
                reference_note = gr.HTML()
        highlighted = gr.HTML(elem_id="article-result", label="已分析原文")
        with gr.Accordion("逐句明細與原文位置", open=False):
            detail = gr.Markdown("點選表格中的句子查看原文位置。")
            table = gr.Dataframe(
                headers=["句子", "原文片段", EASE, UNDERSTANDING],
                datatype=["number", "str", "number", "number"],
                interactive=False,
                max_height=360,
                label="逐句模型估計",
            )
        if combined:
            with gr.Accordion(f'{ANCHOR_LABEL}與原文位置', open=False):
                detail_anchor = gr.Markdown('選取片語查看原文位置。')
                table_anchor = gr.Dataframe(
                    headers=['項目', '段', '次序', '片語', '排序分數', '範圍'],
                    datatype=['number', 'number', 'number', 'str', 'number', 'str'],
                    interactive=False, max_height=320, label=ANCHOR_LABEL,
                )
        with gr.Accordion("分數來源與適用範圍", open=False):
            gr.Markdown(
                "分數估計 PLABA-EVAL 英語生醫摘要讀者的群體平均自評。"
                "閱讀容易度和理解自評分別以 1–5 分呈現，5 分較高。"
                f"{scope}跨領域與其他語言的輸出屬實驗性。\n\n"
                f"Qwen3.5-2B `{BASE_REVISION}` · PLABA-EVAL 句級線性讀出。 "
                "[資料與評分說明](https://aclanthology.org/2026.readi-1.4/)"
            )
            if combined:
                gr.Markdown(f'候選 checkpoint `{reader.checkpoint.identifier}`。'
                            '逐句明細的實驗性分布採用加入改寫效應損失的讀出，目前證據來自開發資料。\n\n'
                            '詞彙難度沿用 CompLex 英語指定詞的群體平均評分，1 分最難。'
                            '模型讀到詞末時給分。全篇掃描、其他語言與領域屬實驗性。 '
                            '[CompLex 原始研究](https://aclanthology.org/2021.semeval-1.1/)\n\n'
                            '提問指向候選估計已有提問時，問題可能指向本段哪一處。'
                            '驗證涵蓋 IIRC 的英語維基百科單段提問，讀者當時看得到連結。'
                            '後續段落會沿用讀到的前文，屬跨段實驗性輸出。 '
                            '[IIRC 原始研究](https://aclanthology.org/2020.emnlp-main.86/)')
            if reference_enabled:
                gr.Markdown('「上文指向」依指定片語排序先前讀到的候選詞。'
                            'ARRAU PEAR 的首選命中率為 9.5%，前五為 34.4%。'
                            '排名採實驗性顯示，證據涵蓋人工指定片語的先行詞。 '
                            '[ARRAU 原始研究](https://doi.org/10.1017/S1351324919000056)\n\n'
                            f'上文讀出 `{reader.reference_head.identifier}`。')
            if cues_enabled:
                gr.Markdown(f'「{CUES}」按句排序，每個詞的原始分數只使用截至詞末的文字。'
                            '實驗以研究者指定的消歧詞核對定位。讀者的連貫判斷只在句末取得。 '
                            '[Blott 2021](https://doi.org/10.1037/xlm0000936) · '
                            '[Blott 2023](https://doi.org/10.1080/23273798.2022.2159992)\n\n'
                            f'消歧讀出 `{reader.cue_head.identifier}`。')

        def analyze(text: str, metric: str):
            try:
                analysis = reader.analyze(text)
            except ValueError as error:
                raise gr.Error(str(error)) from error
            if reference_enabled and metric == REFERENCE:
                metric = EASE
            unavailable = len(getattr(analysis, 'unavailable_sentences', ()))
            message = f"已分析 {len(analysis.sentences)} 句。"
            if unavailable:
                message += f" {unavailable} 句的句末無法精確對齊，保留原文。"
            if combined and analysis.unavailable_paragraphs:
                message += f" {len(analysis.unavailable_paragraphs)} 段的片語讀出無法對齊。"
            outputs = (
                analysis, 0, -1, sentence_table(analysis), render(analysis, metric, 0, -1),
                sentence_detail(analysis, 0), message,
            )
            if combined:
                detail_message = ('選取片語查看原文位置。' if analysis.anchors else '沒有可精確對齊的候選片語。')
                outputs = (*outputs, anchor_table(analysis), detail_message)
            if reference_enabled:
                return (*outputs, None)
            return outputs

        def select_sentence(analysis: ArticleRatings, metric: str, event: gr.SelectData):
            row_value = event.row_value
            if analysis is None or not event.selected or not row_value:
                return (gr.skip(),) * 4
            number = row_value[0]
            numbers = sentence_numbers(analysis)
            if not isinstance(number, int) or number not in numbers:
                return (gr.skip(),) * 4
            row = numbers.index(number)
            sentence_metric = metric if metric in (EASE, UNDERSTANDING) else EASE
            return row, render_article(analysis, sentence_metric, row), sentence_detail(analysis, row), sentence_metric

        def select_anchor(analysis: CombinedArticleRatings, event: gr.SelectData):
            row = event.row_value
            if analysis is None or not event.selected or not row or not isinstance(row[0], int):
                return (gr.skip(),) * 4
            index = row[0] - 1
            if not 0 <= index < len(analysis.anchors):
                return (gr.skip(),) * 4
            return index, render_anchor_article(analysis, index), anchor_detail(analysis, index), ANCHORS

        def update_measure(analysis: ArticleRatings, metric: str, index: int, anchor_index: int, reference=None):
            return render(analysis, metric, index, anchor_index, reference) if analysis is not None else gr.skip()

        analysis_outputs = [result, selected, selected_anchor, table, highlighted, detail, status]
        if combined:
            analysis_outputs.extend((table_anchor, detail_anchor))
        if reference_enabled:
            analysis_outputs.append(reference_result)
        submitted = run.click(
            analyze, [article, measure], analysis_outputs,
            api_name="analyze", concurrency_limit=1,
        )
        table.select(
            select_sentence, [result, measure], [selected, highlighted, detail, measure],
            api_visibility="private", show_progress="hidden",
        )
        if combined:
            table_anchor.select(
                select_anchor, [result], [selected_anchor, highlighted, detail_anchor, measure],
                api_visibility='private', show_progress='hidden',
            )
        measure_inputs = [result, measure, selected, selected_anchor]
        if reference_enabled:
            measure_inputs.append(reference_result)
        measure.input(
            update_measure, measure_inputs, highlighted,
            api_visibility="private", show_progress="hidden",
        )

        if reference_enabled:
            def reference_locations(analysis, current_text, phrase):
                if analysis is None or analysis.text != current_text:
                    raise gr.Error('請先分析目前的正文。')
                if not phrase.strip():
                    raise gr.Error('請輸入原文中的片語。')
                locations = phrase_locations(analysis.text, phrase, analysis.reference)
                if not locations:
                    raise gr.Error('找不到可對齊的完整片語。請從原文複製片語，省略前後標點。')
                return locations

            def show_reference(analysis, start, end):
                reference = reader.reference_head.rank(analysis.text, analysis.reference, start, end)
                landmarks = '、'.join(escape(analysis.text[item.start:item.end]) for item in reference.candidates)
                note = (f'<p role="status">依序標出 {len(reference.candidates)} 個上文詞，包括 {landmarks}。</p>' if landmarks
                        else '<p role="status">片語前沒有可對齊的候選詞。</p>')
                return reference, render_reference_article(analysis.text, reference), note, REFERENCE

            def find_reference(analysis, current_text, phrase):
                locations = reference_locations(analysis, current_text, phrase)
                start, end = locations[0]
                options = [(f'第 {number} 處 · 字元 {left} · …{analysis.text[max(0, left - 18):min(len(analysis.text), right + 24)]}…',
                            f'{left}:{right}') for number, (left, right) in enumerate(locations, 1)]
                selection = gr.update(choices=options, value=options[0][1], visible=len(options) > 1)
                return selection, *show_reference(analysis, start, end)

            def select_reference(analysis, current_text, phrase, chosen):
                locations = reference_locations(analysis, current_text, phrase)
                for start, end in locations:
                    if chosen == f'{start}:{end}':
                        return show_reference(analysis, start, end)
                raise gr.Error('請重新查詢目前的片語。')

            reference_inputs = [result, article, reference_phrase]
            reference_outputs = [reference_result, highlighted, reference_note, measure]
            for event in (search_reference.click, reference_phrase.submit):
                event(find_reference, reference_inputs, [occurrence, *reference_outputs],
                      api_visibility='private', show_progress='hidden')
            occurrence.input(select_reference, [*reference_inputs, occurrence], reference_outputs,
                             api_visibility='private', show_progress='hidden')

            def reset_reference_view(metric):
                return (gr.update(choices=[], value=None, visible=False), '',
                        EASE if metric == REFERENCE else gr.skip())

            submitted.success(reset_reference_view, inputs=[measure], outputs=[occurrence, reference_note, measure],
                              api_visibility='private', show_progress='hidden')
    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Local Qwen snapshot directory")
    parser.add_argument("--port", type=int, default=7860, help="Local HTTP port")
    parser.add_argument("--checkpoint", type=Path, help="Sealed candidate checkpoint directory")
    readers = parser.add_mutually_exclusive_group()
    readers.add_argument("--causal-plaba", action="store_true", help="Use the plain-text causal sentence-end checkpoint")
    readers.add_argument("--full-context-plaba", action="store_true", help="Use the historical full-context PLABA reader")
    parser.add_argument("--cuda", action="store_true", help="Run the combined reader on an NVIDIA L4")
    args = parser.parse_args()
    if args.causal_plaba or args.full_context_plaba:
        if args.cuda or args.checkpoint is not None:
            parser.error('--cuda and --checkpoint apply to the default combined reader')
        reader_class = CausalPlabaReader if args.causal_plaba else PlabaReader
        reader = reader_class(args.model)
    else:
        reader = CombinedReader(args.model, checkpoint=args.checkpoint or DEFAULT_CHECKPOINT,
                                cuda=args.cuda, with_reference=True, with_cues=True)
    build_app(reader).launch(
        server_name="127.0.0.1", server_port=args.port, share=False, css=CSS,
        theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate"),
    )


if __name__ == "__main__":
    main()
