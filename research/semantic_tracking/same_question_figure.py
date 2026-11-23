"""Visualize frozen curves for the held, authored two-question continuations."""

import argparse
from collections import defaultdict
import json
from math import isfinite
from pathlib import Path

from research.backbone_transfer.extract import digest, read_json, stream_rows
from research.semantic_tracking.load_curve_figure import MAX_INLINE_BYTES, indexed


IDENTITIES = ('A', 'B')
METHODS = {'question': '問題與線索交互', 'lexical': '字面重疊', 'steps': '經過步數', 'local': '當下文字表徵'}
TIMING_METHOD = 'timing_question'


def data(paths, pairs, inputs, *, timing_paths=None, decisive=None):
    methods = METHODS if timing_paths is None else {TIMING_METHOD: '明確答案停點監督', **METHODS}
    grouped = defaultdict(dict)
    feeds = [paths]
    if timing_paths is not None:
        feeds.append(({**row, 'method': TIMING_METHOD} for row in timing_paths if row['method'] == 'question'))
    for stream in feeds:
        for row in stream:
            key = (row['doc'], row['method'], row['identity'])
            group = grouped[row['group']]
            if key in group or row['focus'] is not True:
                raise ValueError('Duplicate or out-of-scope authored trajectory')
            group[key] = row

    families = {}
    decisive = decisive if decisive is not None else {}
    seen_decisive = set()
    for pair in pairs:
        family = pair['family_id']
        if pair['split'] != 'test':
            continue
        if family in families or family not in grouped or set(pair['anchors']) != set(IDENTITIES):
            raise ValueError('Invalid or incomplete held family')
        variants = {row['answer_identity']: row for row in pair['variants']}
        if len(pair['variants']) != 2 or set(variants) != set(IDENTITIES):
            raise ValueError('Expected exactly two crossed answers per held family')
        expected = {(row['id'], method, identity) for row in variants.values()
                    for method in methods for identity in IDENTITIES}
        if expected != grouped[family].keys():
            raise ValueError('The frozen readouts do not cover the held branches and questions')

        prefix_end = pair['shared_prefix_end']
        prefix = inputs[variants['A']['id']]['text'][:prefix_end]
        questions = {key: prefix[start:end] for key, (start, end) in pair['anchors'].items()
                     if 0 <= start < end <= prefix_end}
        if set(questions) != set(IDENTITIES) or not all(questions.values()):
            raise ValueError('Both question strings must be visible in the shared prefix')
        branches, geometries = {}, set()
        for answer, variant in variants.items():
            key = variant['id']
            text = inputs[key]['text']
            cue_left, cue_end = variant['cue']
            if text[:prefix_end] != prefix or not prefix_end <= cue_left < cue_end <= len(text):
                raise ValueError('The crossed versions disagree on visible text or authored cue span')
            values = {method: {} for method in methods}
            for method in methods:
                for identity in IDENTITIES:
                    row = grouped[family][key, method, identity]
                    limits = row['initial_token'], row['cue_left_token'], row['cue_end_exclusive']
                    initial, left, right = limits
                    curve = row['survival']
                    if (row['hit'] is not (answer == identity) or any(type(n) is not int for n in limits)
                            or not 0 < initial <= left < right or len(curve) != right - initial
                            or any(type(x) not in (int, float) or not isfinite(x) or not 0 <= x <= 1 for x in curve)
                            or any(new > old for old, new in zip(curve, curve[1:]))):
                        raise ValueError('Malformed authored survival curve')
                    geometries.add(limits)
                    values[method][identity] = [1.0, *curve]
            boundary = decisive.get(key)
            if boundary is not None:
                source_family, token = boundary
                reference = grouped[family][key, 'question', answer]
                if (source_family != family or type(token) is not int
                        or not reference['cue_left_token'] <= token < reference['cue_end_exclusive']):
                    raise ValueError('The marked answer stop falls outside the exact held continuation cue')
                seen_decisive.add(key)
            branches[answer] = {'cue': text[cue_left:cue_end], 'values': values,
                                'decisive': None if boundary is None else boundary[1] + 1}
        if len(geometries) != 1:
            raise ValueError('Held branches must share native-token geometry')
        initial, left, right = geometries.pop()
        families[family] = {'questions': questions, 'branches': branches, 'initial': initial, 'left': left, 'right': right}
    if not families or set(families) != grouped.keys():
        raise ValueError('The authored forecasts must cover exactly the held families')
    if seen_decisive != decisive.keys():
        raise ValueError('Exact answer marks do not belong to the displayed held cohort')
    return {'families': dict(sorted(families.items())), 'methods': methods, 'defaultMethod': next(iter(methods))}


def read_decisive(path, sources):
    timing = read_json(path)
    if timing['paths_sha256'] not in {digest(source) for source in sources if source is not None}:
        raise ValueError('Decisive-token receipt does not bind a displayed forecast file')
    result = {}
    for row in timing['details']:
        if row['method'] == 'question':
            key = row['id']
            if key in result:
                raise ValueError('Duplicate held exact answer stop')
            result[key] = row['group'], row['authored_decisive_token']
    if not result:
        raise ValueError('The supplied receipt has no exact answer stops')
    return result


HEAD = '''<section id="cogllm-same-question-curves">
  <h2>同一前文接兩種解答</h2>
  <p class="text-small text-muted">自編續文 · 模型估計</p>
  <div class="viz-controls">
    <label class="form-label">留出題材 <select class="form-select" data-family></select></label>
    <label class="form-label">讀出方式 <select class="form-select" data-method></select></label>
  </div>
  <div class="sq-legend" aria-label="曲線所對應的問題">
    <p><span class="sq-key sq-a" aria-hidden="true"></span><span data-question="A"></span></p>
    <p><span class="sq-key sq-b" aria-hidden="true"></span><span data-question="B"></span></p>
  </div>
  <div class="sq-panels">
    <section aria-labelledby="sq-answer-a">
      <h3 id="sq-answer-a">續文回答 A</h3>
      <div class="sq-plot" data-plot="A"></div>
      <p class="text-small" data-cue="A"></p>
      <p class="text-small text-muted tabular-nums" data-end="A"></p>
    </section>
    <section aria-labelledby="sq-answer-b">
      <h3 id="sq-answer-b">續文回答 B</h3>
      <div class="sq-plot" data-plot="B"></div>
      <p class="text-small" data-cue="B"></p>
      <p class="text-small text-muted tabular-nums" data-end="B"></p>
    </section>
  </div>
  <script type="application/json" data-curves>
'''


TAIL = '''
  </script>
</section>
<style>
  #cogllm-same-question-curves {display:grid; gap:calc(var(--font-size-base) * .65);}
  #cogllm-same-question-curves .sq-legend {display:grid; gap:calc(var(--font-size-base) * .25);}
  #cogllm-same-question-curves .sq-legend p {display:flex; align-items:baseline; gap:calc(var(--font-size-base) * .45);}
  #cogllm-same-question-curves .sq-key {inline-size:1.4em; flex:none; border-top:2px solid; transform:translateY(-.2em);}
  #cogllm-same-question-curves .sq-a {color:var(--viz-series-1);}
  #cogllm-same-question-curves .sq-b {color:var(--viz-series-2);}
  #cogllm-same-question-curves .sq-key.sq-b {border-top-style:dashed;}
  #cogllm-same-question-curves .sq-panels {display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:var(--font-size-base);}
  #cogllm-same-question-curves .sq-panels section {display:grid; align-content:start; gap:calc(var(--font-size-base) * .25);}
  #cogllm-same-question-curves .sq-plot svg {display:block; inline-size:100%; block-size:auto;}
  #cogllm-same-question-curves .sq-plot text {fill:var(--muted-foreground);}
  #cogllm-same-question-curves .sq-plot .sq-grid {stroke:var(--border);}
  #cogllm-same-question-curves .sq-plot .sq-band {fill:var(--accent); opacity:.5; pointer-events:none;}
  #cogllm-same-question-curves .sq-plot .sq-edge {stroke:var(--muted-foreground); stroke-dasharray:3 3;}
  #cogllm-same-question-curves .sq-plot .sq-decisive {stroke:var(--muted-foreground); stroke-width:1.5;}
  #cogllm-same-question-curves .sq-plot .sq-series {fill:none; stroke:currentColor; stroke-width:2; pointer-events:none;}
  #cogllm-same-question-curves .sq-plot .sq-series.sq-b {stroke-dasharray:5 4;}
  @media(max-width:620px) {#cogllm-same-question-curves .sq-panels {grid-template-columns:minmax(0,1fr);}}
</style>
<script type="module">
  const root = document.getElementById('cogllm-same-question-curves');
  const data = JSON.parse(root.querySelector('[data-curves]').textContent);
  const family = root.querySelector('[data-family]');
  const method = root.querySelector('[data-method]');
  const identities = ['A', 'B'];
  const families = Object.keys(data.families);
  const panels = identities.map(identity => ({
    identity, plot: root.querySelector(`[data-plot="${identity}"]`),
    question: root.querySelector(`[data-question="${identity}"]`),
    cue: root.querySelector(`[data-cue="${identity}"]`), end: root.querySelector(`[data-end="${identity}"]`)
  }));
  const display = new Intl.NumberFormat('zh-TW', {maximumFractionDigits: 5});

  function fill(select, entries) {
    for (const [value, label] of entries) {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = label;
      select.append(option);
    }
  }

  function selectState(saved) {
    family.value = families.includes(saved?.family) ? saved.family : families[0];
    method.value = saved?.defaultMethod === data.defaultMethod && Object.hasOwn(data.methods, saved.method)
      ? saved.method : data.defaultMethod;
  }

  function graph(panel, group, branch) {
    const {identity, plot} = panel;
    const width = Math.max(260, Math.floor(plot.clientWidth));
    const height = 192, left = 37, right = width - 15, top = 16, bottom = 143;
    const {initial, left: cueStart, right: cueEnd} = group;
    const series = branch.values[method.value];
    const x = tick => left + (tick - initial) * (right - left) / (cueEnd - initial);
    const y = value => bottom - value * (bottom - top);
    const makePath = key => series[key].map((value, index) =>
      `${index === 0 ? 'M' : 'L'}${x(initial + index).toFixed(2)},${y(value).toFixed(2)}`).join(' ');
    const yGrid = [0, .25, .5, .75, 1].map(value =>
      `<line class="sq-grid" x1="${left}" x2="${right}" y1="${y(value)}" y2="${y(value)}"/>` +
      `<text class="text-small" x="${left - 7}" y="${y(value) + 4}" text-anchor="end">${value}</text>`);
    const xTicks = [...new Set([initial, Math.floor((initial + cueEnd) / 2), cueEnd])].map(tick =>
      `<text class="text-small" x="${x(tick)}" y="${bottom + 19}" text-anchor="middle">${tick}</text>`);
    const tips = series.A.map((value, index) => {
      const token = initial + index;
      const lo = x(Math.max(initial, token - .5));
      const hi = x(Math.min(cueEnd, token + .5));
      const label = `已讀 ${token} 個 token · A ${display.format(value)} · B ${display.format(series.B[index])}`;
      return `<rect x="${lo}" y="${top}" width="${hi - lo}" height="${bottom - top}" fill="transparent" data-tooltip="${label}"/>`;
    });
    const decisive = branch.decisive;
    const marker = decisive === null ? '' :
      `<line class="sq-decisive" x1="${x(decisive)}" x2="${x(decisive)}" y1="${top}" y2="${bottom}"/>`;
    const decisiveText = decisive === null ? '' : `作者指定答案停點為 token ${decisive}。`;
    const label = `回答 ${identity} 的續文。縱軸是未解機率，介於零與一。線索區間是 token ${cueStart + 1} 至 ${cueEnd}。${decisiveText}`;
    plot.innerHTML = `<svg role="img" aria-label="${label}" viewBox="0 0 ${width} ${height}">
      <text class="text-small" x="${left}" y="12">未解機率</text>
      <rect class="sq-band" x="${x(cueStart)}" y="${top}" width="${x(cueEnd)-x(cueStart)}" height="${bottom-top}"/>
      ${yGrid.join('')}${xTicks.join('')}
      <line class="sq-edge" x1="${x(cueStart)}" x2="${x(cueStart)}" y1="${top}" y2="${bottom}"/>
      ${marker}
      <path class="sq-series sq-a" d="${makePath('A')}"/>
      <path class="sq-series sq-b" d="${makePath('B')}"/>
      ${tips.join('')}
      <text class="text-small" x="${(left + right) / 2}" y="${height - 7}" text-anchor="middle">已讀 token 數</text>
    </svg>`;
  }

  function render() {
    const group = data.families[family.value];
    for (const panel of panels) {
      const key = panel.identity;
      const branch = group.branches[key];
      const series = branch.values[method.value];
      panel.question.textContent = `${key} ${group.questions[key]}`;
      const decisive = branch.decisive === null ? '' : ` · 作者指定答案停點 ${branch.decisive}`;
      panel.cue.textContent = `線索段（${group.left + 1}–${group.right}${decisive}） ${branch.cue}`;
      panel.end.textContent = `線索末未解機率  A ${display.format(series.A.at(-1))} · B ${display.format(series.B.at(-1))}`;
      graph(panel, group, branch);
    }
  }

  fill(family, families.map(key => [key, key.replaceAll('-', ' ')]));
  fill(method, Object.entries(data.methods));
  selectState(window.openai?.widgetState?.modelContent);
  render();
  const size = new ResizeObserver(render);
  for (const panel of panels) size.observe(panel.plot);
  for (const control of [family, method]) {
    control.addEventListener('change', () => {
      render();
      window.openai?.setWidgetState?.({modelContent:{family:family.value, method:method.value,
        defaultMethod:data.defaultMethod}}).catch(() => {});
    });
  }
  window.addEventListener('openai:set_globals', event => {
    const saved = event.detail?.globals?.widgetState;
    if (saved) { selectState(saved.modelContent); render(); }
  });
</script>
'''


def render(payload):
    values = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(',', ':')).replace('<', '\\u003c')
    fragment = HEAD + values + TAIL
    if len(fragment.encode('utf-8')) >= MAX_INLINE_BYTES:
        raise ValueError('The inline figure exceeds the size limit')
    return fragment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('paths', 'pairs', 'inputs', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--timing-paths', type=Path)
    parser.add_argument('--decisive-timing', type=Path)
    args = parser.parse_args()
    timing_paths = stream_rows(args.timing_paths) if args.timing_paths is not None else None
    decisive = (read_decisive(args.decisive_timing, (args.paths, args.timing_paths))
                if args.decisive_timing is not None else None)
    fragment = render(data(stream_rows(args.paths), stream_rows(args.pairs), indexed(args.inputs),
                           timing_paths=timing_paths, decisive=decisive))
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(fragment)


if __name__ == '__main__':
    main()
