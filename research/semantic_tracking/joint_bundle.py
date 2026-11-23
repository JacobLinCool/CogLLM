"""Seal the existing lexical assays and coordinate-only closure data for Colab."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil


TEXT_FILES = (
    'stimuli.jsonl', 'source-train.jsonl', 'source-eval.jsonl',
    'onset-train.jsonl', 'onset-eval.jsonl',
)
CLOSURE_FILE = 'closure-intervals.jsonl'
# The three workbooks measure original text lines. The fourth story's published
# derived word coordinates repeat line labels and are not comparable exposures.
STORIES = {'Katze': {'AA'}, 'Auftrag': {'AA', 'JE'}, 'Thomas': {'AA', 'JE'}}
CLOSURE_FIELDS = {'id', 'story', 'team', 'open_line', 'lower_line', 'upper_line', 'last_line', 'line_opportunities'}
ISSUE_EVAL_FIELDS = {'row_id', 'id', 'group_id', 'stop', 'source_candidates'}
ONSET_EVAL_FIELDS = {'id', 'group_id', 'stop'}


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    with path.open(encoding='utf-8') as source:
        return [json.loads(line) for line in source]


def load_closures(path):
    records = rows(path)
    names = set()
    totals = Counter()
    censored = Counter()
    for row in records:
        if row.keys() != CLOSURE_FIELDS:
            raise ValueError('Closure payload may contain only question identifiers and line coordinates')
        identity, story, team = (row[field] for field in ('id', 'story', 'team'))
        opening, lower, upper, last = (row[field] for field in ('open_line', 'lower_line', 'upper_line', 'last_line'))
        exposure = row['line_opportunities']
        if (
            not isinstance(identity, str) or identity in names or story not in STORIES or team not in STORIES[story]
            or not re.fullmatch(rf'onea-{story}-{team}-[0-9]{{3}}', identity)
            or any(type(value) is not int for value in (opening, lower, last))
            or not 1 <= opening <= lower <= last <= 10_000
            or upper is not None and (type(upper) is not int or not lower < upper <= last)
            or type(exposure) is not int or exposure < 1 or exposure != (upper if upper is not None else lower) - opening
        ):
            raise ValueError('Invalid metadata-only opening, closure or censoring coordinates')
        names.add(identity)
        totals[story] += 1
        censored[story] += upper is None
    if totals.keys() != STORIES.keys() or any(totals[story] <= censored[story] for story in totals):
        raise ValueError('Expected all three published line-annotation stories and observed closures')
    expected_teams = {(story, team) for story, teams in STORIES.items() for team in teams}
    if {(row['story'], row['team']) for row in records} != expected_teams:
        raise ValueError('Expected the published consolidated and team annotation lists')
    inventory = {story: {'questions': totals[story], 'right_censored': censored[story]} for story in sorted(totals)}
    return records, inventory


def validate_text_bundle(prepared):
    historical = read_json(prepared / 'payload-audit.json')
    digests = {name: sha256(prepared / name) for name in TEXT_FILES}
    if historical['file_sha256'] != digests:
        raise ValueError('Historical text, training rows or old evaluation inventories changed')
    tokenizer = read_json(prepared / 'token-inventory.json')
    if tokenizer['stimuli_sha256'] != digests['stimuli.jsonl'] or tokenizer['excluded_stops']:
        raise ValueError('Expect the fully certified historical text inventory')
    for task, fields in (('source', ISSUE_EVAL_FIELDS), ('onset', ONSET_EVAL_FIELDS)):
        train = rows(prepared / f'{task}-train.jsonl')
        evaluation = rows(prepared / f'{task}-eval.jsonl')
        coverage = historical['inventory'][task]
        if (
            len(train) != coverage['train'] or len(evaluation) != coverage['test_unique_inputs']
            or any(row.get('split') != 'train' for row in train)
            or any(row.keys() != fields for row in evaluation)
            or {row['group_id'] for row in train} & {row['group_id'] for row in evaluation}
        ):
            raise ValueError(f'Invalid labels or material groups in the frozen {task} split')
    return historical, tokenizer, digests


def run(prepared, closure, output):
    previous, tokenizer, digests = validate_text_bundle(prepared)
    closure_hash = sha256(closure)
    _, inventory = load_closures(closure)
    output.mkdir(parents=True, exist_ok=False)
    for name, digest in digests.items():
        shutil.copyfile(prepared / name, output / name)
        if sha256(output / name) != digest:
            raise RuntimeError(f'Historical {name} changed while building the payload')
    shutil.copyfile(closure, output / CLOSURE_FILE)
    if sha256(output / CLOSURE_FILE) != closure_hash:
        raise RuntimeError('The metadata-only closure file changed while building the payload')
    (output / 'token-inventory.json').write_text(json.dumps(tokenizer, indent=2) + '\n', encoding='utf-8')
    audit = {
        'inventory': {**previous['inventory'], 'closure': inventory},
        'file_sha256': {**digests, CLOSURE_FILE: closure_hash},
        'evaluation_separation': {
            'onset_and_source': 'Material-group outcomes for both fixed test folds stay local.',
            'closure': 'All numeric rows are present for story rotation. Fit code must exclude the current test story.',
            'closure_story_text_in_payload': False,
        },
    }
    (output / 'payload-audit.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(audit['inventory'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared', 'closure', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    run(**vars(parser.parse_args()))
