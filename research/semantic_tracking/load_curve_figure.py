"""Render a selectable inline comparison of real, frozen load-curve trajectories."""

import argparse
from bisect import bisect_left
from collections import defaultdict
import json
from math import isfinite
from pathlib import Path


CONDITIONS = {'early': '段內解答', 'delayed': '下一段解答', 'abrupt': '尚未解答便轉場'}
METHODS = {'independent': '獨立讀出', 'joint_semantic': '共享讀出（疑難＋證據）',
           'joint': '共享讀出（三任務）', 'constant_train_rate': '常數基線'}
CHANNELS = {'cohort': '共同前文疑問的未解量', 'load': '全部疑問的未解量',
            'opening': '當下開啟量', 'closing': '當下解除量'}
FAMILY_SIZE = 12
MAX_INLINE_BYTES = 1_000_000


def indexed(path):
    records = {}
    with path.open(encoding='utf-8') as stream:
        for row in map(json.loads, stream):
            key = row['id']
            if key in records:
                raise ValueError(f'Duplicate document in {path.name}')
            records[key] = row
    return records


def token_geometry(trace, expected_end):
    ids, offsets = trace['token_ids'], trace['offsets']
    if (not ids or len(ids) > 2000 or any(type(token) is not int or token < 0 for token in ids)
            or len(ids) != len(offsets)
            or any(type(left) is not int or type(right) is not int or not 0 <= left < right <= expected_end
                   for left, right in offsets)):
        raise ValueError('Expected finite native token indices and nonempty Unicode offsets')
    ends = [right for _, right in offsets]
    if ends != sorted(ends) or ends[-1] != expected_end:
        raise ValueError('Token coordinates do not span the supplied article')
    return ids, offsets, ends


def reference_position(ends, char_end):
    index = bisect_left(ends, char_end)
    if index == len(ends) or ends[index] != char_end:
        raise ValueError('A phase end does not have an exact native token coordinate')
    while index + 1 < len(ends) and ends[index + 1] == char_end:
        index += 1
    return index + 1


def data(labels, traces, inputs):
    if set(traces) != set(METHODS) or any(set(rows) != set(labels) for rows in traces.values()):
        raise ValueError('Expected every authored document in each frozen readout')
    if inputs is not None and set(inputs) != set(labels):
        raise ValueError('Authored text must cover the same document IDs')
    families = defaultdict(dict)
    largest = dict.fromkeys(CHANNELS, 0.0)
    geometry = {}
    for key, label in labels.items():
        family, condition = label['family_id'], label['condition']
        if (label['source'] != 'constructed_control' or condition not in CONDITIONS
                or condition in families[family]):
            raise ValueError('Expected unique authored continuation conditions')
        meta = label['offsets']
        sentence_ends = meta['sentence_ends']
        if (len(sentence_ends) != 5 or sentence_ends != sorted(set(sentence_ends))
                or (meta['common_prefix_end'], meta['paragraph_end'], meta['late_comparison_end'])
                != (sentence_ends[1], sentence_ends[2], sentence_ends[4])):
            raise ValueError('Inconsistent authored segment boundaries')
        marker = meta['shift'] if condition == 'abrupt' else meta['cue']
        if marker is None or not 0 < marker['start'] < marker['end'] <= sentence_ends[-1]:
            raise ValueError('An authored continuation has no cue or switch span')
        base_ids, base_offsets, ends = token_geometry(traces['independent'][key], sentence_ends[-1])
        measures = {}
        for method, rows in traces.items():
            record = rows[key]
            if record['token_ids'] != base_ids or record['offsets'] != base_offsets:
                raise ValueError('Readouts disagree on token identity or position')
            measures[method] = {}
            for channel in CHANNELS:
                values = record[channel]
                if (len(values) != len(base_ids) or any(type(x) not in (float, int) or not isfinite(x) or x < 0
                                                      for x in values)):
                    raise ValueError(f'Invalid {channel} values for {key}')
                largest[channel] = max(largest[channel], max(values))
                measures[method][channel] = values
        phase = {'common': reference_position(ends, meta['common_prefix_end']),
                 'paragraph': reference_position(ends, meta['paragraph_end']),
                 'event': reference_position(ends, marker['end'])}
        text = inputs[key]['text'] if inputs is not None else None
        if text is not None and (len(text) != sentence_ends[-1]
                                 or text[meta['paragraph_end']:meta['continuation_start']] != '\n\n'):
            raise ValueError('Authored text does not reproduce the expected document boundaries')
        families[family][condition] = {'question': label['question'], 'domain': label['domain'], 'values': measures,
                                       'phases': phase, 'length': len(base_ids), 'text': text}
        prefix = phase['common']
        geometry[family, condition] = tuple(base_ids[:prefix]), tuple(map(tuple, base_offsets[:prefix]))
    if len(families) != FAMILY_SIZE or any(set(rows) != set(CONDITIONS) for rows in families.values()):
        raise ValueError(f'Expected {FAMILY_SIZE} complete authored triplets')
    for family, rows in families.items():
        if len({(row['question'], row['domain']) for row in rows.values()}) != 1:
            raise ValueError('An authored family mixes prompts or domains')
        common_tokens = {row['phases']['common'] for row in rows.values()}
        if len(common_tokens) != 1:
            raise ValueError('Common-prefix token indices differ within a triplet')
        if len({geometry[family, condition] for condition in CONDITIONS}) != 1:
            raise ValueError('Common-prefix token identity differs within a triplet')
    return {'families': dict(sorted(families.items())), 'conditions': CONDITIONS, 'methods': METHODS,
            'channels': CHANNELS, 'upper': {name: top * 1.05 if top > 0 else 1.0 for name, top in largest.items()}}


FRAGMENT_HEAD = '''<section id="cogllm-actual-load-curves">
  <h2>三種續寫的逐 token 曲線</h2>
  <p class="text-muted text-small">自編文本 · 模型估計</p>
  <div class="viz-controls">
    <label class="form-label">題材 <select class="form-select" data-family></select></label>
    <label class="form-label">讀出方式 <select class="form-select" data-method></select></label>
    <label class="form-label">顯示量 <select class="form-select" data-channel></select></label>
  </div>
  <p data-question aria-live="polite"></p>
  <section data-viz-composition>
    <div data-viz-container></div>
  </section>
  <details data-text hidden><summary>三種續寫原文</summary><div data-paragraphs></div></details>
  <script type="application/json" data-curve-data>
'''

FRAGMENT_TAIL = '''
  </script>
</section>
<style>
  #cogllm-actual-load-curves {display: grid; gap: calc(var(--font-size-base) * .60);}
  #cogllm-actual-load-curves [data-paragraphs] {display: grid; gap: var(--font-size-base); margin-top: var(--font-size-base);}
  #cogllm-actual-load-curves [data-paragraphs] p {white-space: pre-wrap; overflow-wrap: anywhere;}
</style>
<script type="module">
  import { mountChart } from 'https://persistent.oaistatic.com/plugins/visualize/charts/1.8.1/element.js';
  const root = document.getElementById('cogllm-actual-load-curves');
  const data = JSON.parse(root.querySelector('[data-curve-data]').textContent);
  const family = root.querySelector('[data-family]');
  const method = root.querySelector('[data-method]');
  const channel = root.querySelector('[data-channel]');
  const question = root.querySelector('[data-question]');
  const text = root.querySelector('[data-text]');
  const paragraphs = root.querySelector('[data-paragraphs]');
  const groups = Object.keys(data.families);
  const conditions = Object.entries(data.conditions);
  const defaults = { family: groups[0], method: 'independent', channel: 'cohort' };

  function options(select, items) {
    for (const [value, label] of items) {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = label;
      select.append(option);
    }
  }

  options(family, groups.map((key, index) => [key, `題材 ${index + 1} · ${key.replaceAll('_', ' ')}`]));
  options(method, Object.entries(data.methods));
  options(channel, Object.entries(data.channels));

  function selectState(saved) {
    family.value = groups.includes(saved?.family) ? saved.family : defaults.family;
    method.value = Object.hasOwn(data.methods, saved?.method) ? saved.method : defaults.method;
    channel.value = Object.hasOwn(data.channels, saved?.channel) ? saved.channel : defaults.channel;
  }

  function configuration() {
    const choice = data.families[family.value];
    const xmax = Math.max(...Object.values(choice).map(row => row.length));
    question.textContent = choice.early.question;
    text.hidden = choice.early.text === null;
    paragraphs.replaceChildren();
    if (!text.hidden) {
      for (const [id, title] of conditions) {
        const section = document.createElement('section');
        const heading = document.createElement('h3');
        const paragraph = document.createElement('p');
        heading.textContent = title;
        paragraph.textContent = choice[id].text;
        section.append(heading, paragraph);
        paragraphs.append(section);
      }
    }
    return { charts: conditions.map(([id, title]) => {
      const row = choice[id];
      const marker = id === 'abrupt' ? '轉場片語末' : '解答線索末';
      return {
        id: `load-curve-${id}`, title, height: 208,
        summary: `${title}。共同前文末位於 token ${row.phases.common}，第三句末位於 token ${row.phases.paragraph}，${marker}位於 token ${row.phases.event}。`,
        spec: {
          type: 'line', x: 'token', y: 'value', xLabel: 'Token 序號', yLabel: data.channels[channel.value],
          xDomain: [1, xmax], yDomain: [0, data.upper[channel.value]], showLegend: false,
          referenceLines: [
            { axis: 'x', value: row.phases.common, label: '共同前文末' },
            { axis: 'x', value: row.phases.paragraph, label: '第三句末' }
          ],
          annotations: [{ kind: 'point', id: `event-${id}`, field: 'value', at: row.phases.event, label: marker }],
          colors: { value: 'var(--chart-1)' }
        },
        columns: ['token', 'value'], rows: row.values[method.value][channel.value].map((y, i) => [i + 1, y])
      };
    }) };
  }

  selectState(window.openai?.widgetState?.modelContent);
  const chart = mountChart(root.querySelector('[data-viz-container]'), configuration());
  for (const control of [family, method, channel]) {
    control.addEventListener('change', () => {
      chart.config = configuration();
      window.openai?.setWidgetState?.({ modelContent: {
        family: family.value, method: method.value, channel: channel.value
      }}).catch(() => {});
    });
  }
  window.addEventListener('openai:set_globals', event => {
    const state = event.detail?.globals?.widgetState;
    if (state) {
      selectState(state.modelContent);
      chart.config = configuration();
    }
  });
</script>
'''


def render(payload):
    embedded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(',', ':')).replace('<', '\\u003c')
    fragment = FRAGMENT_HEAD + embedded + FRAGMENT_TAIL
    if len(fragment.encode('utf-8')) >= MAX_INLINE_BYTES:
        raise ValueError('The inline figure must remain below one megabyte')
    return fragment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--traces', type=Path, required=True)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    payload = data(indexed(args.labels), json.loads(args.traces.read_bytes()),
                   None if args.inputs is None else indexed(args.inputs))
    fragment = render(payload)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(fragment)


if __name__ == '__main__':
    main()
