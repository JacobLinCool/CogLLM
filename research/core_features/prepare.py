"""Seal labeled training rows and outcome-free evaluation anchors for Colab."""

import argparse
import hashlib
import json
from pathlib import Path

from research.stage3.probe import _jsonl


def write_rows(path, rows):
    with path.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def run(cwi, readback, output):
    output.mkdir(parents=True, exist_ok=False)
    stimuli = []
    keys = set()
    inventory = {}
    for name, folder in (('cwi', cwi), ('readback', readback)):
        material = list(_jsonl(folder / 'stimuli.jsonl'))
        docs = {row['id']: set(row['stops']) for row in material}
        if len(docs) != len(material) or keys & docs.keys():
            raise ValueError(f'Duplicate text IDs in {name}')
        keys.update(docs)
        stimuli.extend(material)
        training, evaluation, roles = [], {}, {}
        anchor_fields = ['id', 'group_id', 'stop', 'split']
        if name == 'cwi':
            anchor_fields.append('start')
            label_fields = ['native_total', 'nonnative_total', 'native_marked', 'nonnative_marked']
        else:
            label_fields = ['readback', 'source_stop']
        for row in _jsonl(folder / 'outcomes.jsonl'):
            split = row['split']
            roles.setdefault(row['group_id'], split)
            if roles[row['group_id']] != split or row['stop'] not in docs[row['id']]:
                raise ValueError(f'Split leakage or missing anchor in {name}')
            anchor = {key: row[key] for key in anchor_fields}
            if split == 'train':
                training.append({**anchor, **{key: row[key] for key in label_fields}})
            elif split in {'dev', 'test'}:
                evaluation[tuple(anchor.values())] = anchor
            else:
                raise ValueError(f'Unexpected split in {name}: {split}')
        if not training or not evaluation:
            raise ValueError(f'Train and evaluation both need observations in {name}')
        write_rows(output / f'{name}-train.jsonl', training)
        write_rows(output / f'{name}-eval.jsonl', evaluation.values())
        inventory[name] = {'texts': len(material), 'training_rows': len(training),
                           'unique_evaluation_anchors': len(evaluation)}
    write_rows(output / 'stimuli.jsonl', stimuli)
    audit = {'inventory': inventory, 'file_sha256': {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output.glob('*.jsonl'))
    }}
    (output / 'payload-audit.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('cwi', 'readback', 'output'):
        parser.add_argument(f'--{key}', type=Path, required=True)
    args = parser.parse_args()
    run(args.cwi, args.readback, args.output)
