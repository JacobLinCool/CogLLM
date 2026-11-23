"""Display changes in population word expectations at their printed positions."""

from html import escape
from math import isclose, isfinite

SCOPE = "population_next_word_disagreement"
EXAMPLE = (
    "The lights went out in the middle of dinner. "
    "Maya opened a drawer and felt around until she found a match. "
    "A moment later the candle on the table was burning."
)

STYLE = """
.writing-review .expectation-panel {border-top:1px solid #c4d1cc;padding-top:.75rem;color:#23352f;}
.writing-review .expectation-panel h3,.writing-review .expectation-panel p,
.writing-review .expectation-panel summary {color:#23352f!important;}
.writing-review .expectation-increase {border-bottom:3px solid #974814;}
.writing-review .expectation-decrease {border-bottom:3px dotted #14596a;}
.writing-review .expectation-chart {display:block;width:100%;height:8.5rem;}
.writing-review .expectation-axis {display:flex;justify-content:space-between;font-size:.85rem;}
.writing-review .expectation-table {max-height:18rem;overflow:auto;}
.writing-review .expectation-table table {width:100%;border-collapse:collapse;color:#23352f;}
.writing-review .expectation-table th,.writing-review .expectation-table td {padding:.2rem .35rem;text-align:left;background:#fcfcf8;}
"""


def probability(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or not 0 <= value <= 1:
        raise ValueError("A word expectation falls outside the measured probability scale.")
    return value


def bounds(grouped, other):
    low = probability(grouped)
    high = low + probability(other) ** 2
    if high > 1 + 1e-6:
        raise ValueError("The residual vocabulary mass disagrees with the word probabilities.")
    return low, min(high, 1.)


def label_range(low, high, *, signed=False):
    fmt = "+.1f" if signed else ".0f"
    ends = [format(100*x, fmt) for x in (low, high)]
    value = ends[0] if ends[0] == ends[1] else " to ".join(ends)
    return value if signed else value+"%"


def plot(runs):
    lower, upper, bands = [], [], []
    for run in runs:
        bottom = [f"{x:.2f} {99-98*lo:.2f}" for x,lo,hi in run]
        top = [f"{x:.2f} {99-98*hi:.2f}" for x,lo,hi in run]
        lower.append("M"+" L".join(bottom))
        upper.append("M"+" L".join(top))
        bands.append("M"+" L".join(bottom+top[::-1])+" Z")
    return (
        '<svg class="expectation-chart" viewBox="0 0 400 100" preserveAspectRatio="none" '
        'role="img" aria-label="Estimated range for the chance that two readers give different next-word answers">'
        '<path d="M0 1H400 M0 50H400 M0 99H400" stroke="#c6d1ce" fill="none" vector-effect="non-scaling-stroke"/>'
        f'<path class="expectation-band" d="{" ".join(bands)}" fill="#b8d5cf"/>'
        f'<path class="expectation-upper" d="{" ".join(upper)}" stroke="#47847e" stroke-width="1" fill="none" vector-effect="non-scaling-stroke"/>'
        f'<path class="expectation-lower" d="{" ".join(lower)}" stroke="#175c59" stroke-width="2" fill="none" vector-effect="non-scaling-stroke"/></svg>'
    )


def present(text, reading):
    if reading["scope"] != SCOPE:
        raise ValueError("Unknown population or word expectation measure.")
    words = reading["words"]
    if not words:
        raise ValueError("The passage contains no scorable English word boundary.")
    prev_end = 0
    spans, shifts, cells, runs = [], [], [], []
    for i, word in enumerate(words):
        a, b = word["start"], word["end"]
        if not (isinstance(a, int) and isinstance(b, int) and prev_end <= a < b <= len(text)
                and text[a:b] == word["text"]):
            raise ValueError("A word expectation refers to a different passage.")
        prev_end = b
        lo, hi = bounds(word["after"], word["other_after"])
        before, delta = word["before"], word["change"]
        if before is None:
            if delta is not None:
                raise ValueError("The first word of a measured run cannot have a change without an earlier estimate.")
            before_label, change_label = "—", "—"
            runs.append([])
        else:
            old_lo, old_hi = bounds(before, word["other_before"])
            if delta is None or not isclose(lo - old_lo, delta, abs_tol=1e-6) or not runs:
                raise ValueError("The displayed word change disagrees with its two probabilities.")
            low_change, high_change = lo - old_hi, hi - old_lo
            before_label = label_range(old_lo, old_hi)
            change_label = label_range(low_change, high_change, signed=True)
            if low_change >= .20 or high_change <= -.20:
                guaranteed = low_change if low_change >= .20 else high_change
                spans.append((a, b, "expectation-increase" if guaranteed > 0 else "expectation-decrease"))
                shifts.append((guaranteed, before_label, label_range(lo,hi), word))
        label = escape(word["text"])
        cells.append(f"<tr><td>{i+1}</td><td>{label}</td><td>{before_label}</td><td>{label_range(lo,hi)}</td><td>{change_label}</td></tr>")
        runs[-1].append((2 + 396*i/max(len(words)-1,1), lo, hi))
    examples = []
    for increasing in (False, True):
        selected = [v for v in shifts if (v[0] > 0 if increasing else v[0] < 0)]
        if selected:
            _, before, after, word = max(selected, key=lambda v: abs(v[0]))
            verb = "spread out" if increasing else "narrow"
            top = word["top_next"][0]["word"]
            clue = (f' The leading next word is <strong>“{escape(top)}”</strong>.' if top and not increasing else "")
            examples.append(f'After <strong>“{escape(word["text"])}”</strong> the guesses {verb} ({before} → {after}).'+clue)
    return spans, (
        '<section class="expectation-panel"><h3>Where do readers’ next guesses change?</h3>'
        '<p>After each word, how often would two readers guess different next words? '
        'A narrow band gives a more precise estimate. A wide band means many possible answers '
        'fall outside the model’s 8,192 named words. The underlines show only changes of at least '
        '20 percentage points in the same direction across the whole range.</p>'
        '<div class="expectation-axis"><span>100% disagree</span></div>' + plot(runs) +
        f'<div class="expectation-axis"><span>0% · First measured word</span><span>Last measured word</span></div>'
        '<p>' + " ".join(examples) + '</p><details><summary>Values at each word</summary>'
        '<div class="expectation-table"><table><thead><tr><th>Position</th><th>Word</th>'
        '<th>Before</th><th>After</th><th>Change (points)</th></tr></thead><tbody>' +
        "".join(cells) + '</tbody></table></div></details>'
        f'<p class="reading-caption">{len(words)} of {reading["coverage"]["surface_words"]} word endings received an estimate. '
        'An estimate at the very end assumes the text continues.</p></section>'
    )
