"""Render the already-computed prefix signal beside the reviewed draft."""

from research.token_readout.app import Analysis, source_html


STYLE = """
.token-experiment { margin-top: .8rem; }
.token-experiment > summary, .token-experiment details > summary { cursor: pointer; }
.token-experiment .signal-caption { font-size: .9rem; }
.token-experiment figure { margin: .8rem 0; }
.token-experiment .signal-plot { display: block; width: 100%; height: 8rem; }
.token-experiment .signal-axis { display: flex; justify-content: space-between; font-size: .82rem; }
.token-experiment .signal-table { max-height: 17rem; overflow: auto; }
.token-experiment table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
.token-experiment :is(th, td) { text-align: left; padding: .25rem .5rem; }
.token-experiment th { position: sticky; top: 0; background: #fcfcf8; }
.token-experiment .signal-text { white-space: pre-wrap; overflow-wrap: anywhere; }
"""


def token_text(text, token):
    value = text[token.start:token.end]
    if not value:
        return "No printed characters"
    if value.isspace():
        return "Whitespace"
    return source_html(value)


def panel(analysis: Analysis):
    tokens = analysis.tokens
    peak = max(token.unresolved for token in tokens)
    points = []
    rows = []
    for index, token in enumerate(tokens):
        x = 2 + 596 * index / max(1, len(tokens) - 1)
        y = 126 - 124 * token.unresolved / peak if peak else 126
        points.append(f"{x:.2f},{y:.2f}")
        difference = token.unresolved - tokens[index - 1].unresolved if index else None
        change = f"{difference:+.4g}" if difference is not None else "—"
        rows.append(
            f'<tr><td>{index + 1}</td><td class="signal-text">{token_text(analysis.text, token)}</td>'
            f'<td>{token.unresolved:.4g}</td><td>{change}</td></tr>'
        )

    if len(points) == 1:
        x, y = points[0].split(",")
        curve = f'<circle cx="{x}" cy="{y}" r="2" fill="#166257"/>'
    else:
        curve = (
            f'<polyline points="{" ".join(points)}" fill="none" stroke="#166257" '
            'stroke-width="2.5" vector-effect="non-scaling-stroke"/>'
        )
    return (
        '<details class="writing-review token-experiment"><summary>Experimental token curve</summary>'
        '<p>The line shows the model’s estimate of how much remains unexplained after each token. '
        'Its height is in model units. The scale adjusts to this draft.</p>'
        '<figure><svg class="signal-plot" viewBox="0 0 600 128" preserveAspectRatio="none" role="img" '
        'aria-label="Experimental unresolved signal by model token">'
        '<path d="M2 2H598 M2 126H598" fill="none" stroke="#c5d0c9" vector-effect="non-scaling-stroke"/>'
        f'{curve}</svg><div class="signal-axis"><span>Token 1</span><span>Token {len(tokens)}</span></div>'
        f'<figcaption class="signal-caption">Vertical range: 0 to {peak:.4g} model units.</figcaption></figure>'
        '<details><summary>Values at each token</summary><div class="signal-table" tabindex="0">'
        '<table><thead><tr><th>Token</th><th>Text</th><th>Signal</th><th>Change</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div></details>'
        '<p class="signal-caption">A peak has no established interpretation for an individual reader. '
        'In eight constructed tests with two open questions, the original curve correctly linked both '
        'questions to their answers in zero cases.</p></details>'
    )
