"""Freeze equal prompt history for the auxiliary expert edit comparison."""
import hashlib
import json
from pathlib import Path

from research.reading_validation.transition_external_prepare import OUT, ROOT


def prepare():
    docs = [json.loads(s) for s in (ROOT / 'cohesion-stimuli.jsonl').read_text().splitlines()]
    anchors = {s['id']: (doc, pos) for doc in docs for pos, s in enumerate(doc['stops'])}
    observations = json.loads((OUT / 'stimuli.json').read_text())['material']
    packet = []
    for row in observations:
        if row['source'] != 'decor_rewrites':
            continue
        original, i = anchors[row['counterpart']]
        edit, j = anchors[row['id']]
        a = original['stops'][i]
        b = edit['stops'][j]
        if original['text'][:a['start']] != edit['text'][:b['start']]:
            raise ValueError('Edit and source must expose exactly the same context')
        history = original['stops'][:i]
        for identifier, doc, stop in [(row['counterpart'], original, a), (row['id'], edit, b)]:
            packet.append({'id':identifier, 'text':doc['text'][:stop['end']], 'stops':[*history,stop]})
    distinct = {}
    for passage in packet:
        if distinct.setdefault(passage['id'], passage) != passage:
            raise ValueError('The same hashed passage has two different exposures')
    if len(packet) != 426 or len(distinct) != 424:
        raise ValueError('Expected 213 edit pairs including two unchanged release entries')
    path = ROOT / 'cohesion-matched-stimuli.jsonl'
    path.write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in distinct.values()))
    print(path, hashlib.sha256(path.read_bytes()).hexdigest(), len(distinct))


if __name__ == '__main__':
    prepare()
