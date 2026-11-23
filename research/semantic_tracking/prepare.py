"""Separate semantic tracking training labels from the test prediction payload."""

import argparse
import hashlib
import json
from pathlib import Path

from research.core_features.prepare import write_rows
from research.stage3.probe import _jsonl


STIMULUS_FIELDS = {'id', 'text', 'stops'}


def run(source, onset, output, local_outcomes):
    output.mkdir(parents=True, exist_ok=False)
    local_outcomes.mkdir(parents=True, exist_ok=False)
    inventory = {}
    combined = []
    identifiers = set()
    for task, directory in (('source', source), ('onset', onset)):
        stimuli = list(_jsonl(directory / 'stimuli.jsonl'))
        by_id = {row['id']: set(row['stops']) for row in stimuli}
        if len(stimuli) != len(by_id) or identifiers & by_id.keys() or any(set(row) != STIMULUS_FIELDS for row in stimuli):
            raise ValueError(f'Expected distinct and outcome-free texts for {task}')
        identifiers.update(by_id)
        combined.extend(stimuli)
        roles = {}
        train = []
        evaluation = {}
        truth = []
        observations = list(_jsonl(directory / 'outcomes.jsonl'))
        for index, row in enumerate(observations):
            if task == 'onset' and not row['initial_risk']:
                continue
            key, group, stop, split = (row.get(field) for field in ('id', 'group_id', 'stop', 'split'))
            if stop not in by_id[key] or split not in {'train', 'test'} or roles.setdefault(group, split) != split:
                raise ValueError(f'Invalid split or uncertified stop inventory for {task}: {key}')
            anchor = {'id': key, 'group_id': group, 'stop': stop}
            if task == 'source':
                identifier = f'source-annotation-{index:05d}'
                candidate = row['source_candidates']
                start, end = row['source_start'], row['source_stop']
                if not 0 <= start < end < stop or not set(candidate) <= by_id[key]:
                    raise ValueError('Source must precede the marked inconsistency')
                anchor.update(row_id=identifier, source_candidates=candidate)
                labels = {'source_start': start, 'source_stop': end}
                test_key = identifier
            else:
                labels = {name: row[name] for name in ('readback', 'source_stop')}
                test_key = (key, stop)
            if split == 'train':
                train.append({**anchor, **labels, 'split': split})
            else:
                evaluation[test_key] = anchor
                truth.append({**row, **anchor})
        if not train or not evaluation:
            raise ValueError(f'No training or no held-out rows for {task}')
        write_rows(output / f'{task}-train.jsonl', train)
        write_rows(output / f'{task}-eval.jsonl', evaluation.values())
        write_rows(local_outcomes / f'{task}-test.jsonl', truth)
        inventory[task] = {'texts': len(stimuli), 'train': len(train), 'test_rows': len(truth),
                           'test_unique_inputs': len(evaluation)}
    write_rows(output / 'stimuli.jsonl', combined)
    audit = {'inventory': inventory, 'file_sha256': {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output.glob('*.jsonl'))
    }}
    (output / 'payload-audit.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'onset', 'output', 'local-outcomes'):
        parser.add_argument(f'--{key}', type=Path, required=True)
    run(**vars(parser.parse_args()))
