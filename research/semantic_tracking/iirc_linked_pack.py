"""Seal the linked-evidence inputs available before held-out scoring."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

from research.semantic_tracking.iirc_prepare import digest
from research.stage2.records import write_jsonl
from research.stage3.probe import _jsonl
from research.stage3.run import save_json


PUBLIC_MEMBERS = ('train/stimuli.jsonl', 'train/queries.jsonl', 'dev/stimuli.jsonl', 'dev/queries.jsonl')
TRAIN_TARGETS = 'train/locked-labels.jsonl'
HELD_TARGETS = ('dev/locked-labels.jsonl', 'dev/locked-matched-swaps.jsonl')
METHODS = ('geometry', 'path_text', 'anchor_conditioned')
NLL_MINIMUM_GAIN = 0.02
BRIER_TOLERANCE = 0.01


def validate_split(root, split, expected):
    stimuli = _jsonl(root / split / 'stimuli.jsonl')
    queries = _jsonl(root / split / 'queries.jsonl')
    cases = {}
    for stimulus, query in zip(stimuli, queries, strict=True):
        if set(stimulus) != {'id', 'text', 'stops'} or set(query) != {
            'case_id', 'anchor_start', 'anchor_end', 'linked_start',
        }:
            raise ValueError(f'Unexpected input fields in {split}')
        case, stops = stimulus['id'], stimulus['stops']
        text = stimulus['text']
        start, end, linked = (query[name] for name in ('anchor_start', 'anchor_end', 'linked_start'))
        if (case != query['case_id'] or case in cases or len(stops) < 2 or stops[0] != end
                or stops[-1] != len(text) or any(a >= b for a, b in zip(stops, stops[1:]))
                or not 0 <= start < end < linked < stops[1] or text[linked - 2:linked] != '\n\n'):
            raise ValueError(f'Invalid linked path or boundary for {case}')
        cases[case] = len(stops) - 1
    if len(cases) != expected:
        raise ValueError(f'Unexpected number of linked input paths in {split}')
    return cases


def fit_intervals(path, cases):
    retained = {}
    for row in _jsonl(path):
        case, lower, upper = (row[name] for name in ('case_id', 'lower', 'upper'))
        maximum = cases[case]
        if case in retained or type(lower) is not int or not 0 <= lower <= maximum:
            raise ValueError(f'Invalid fit interval for {case}')
        if (upper is None and lower != maximum
                or upper is not None and (type(upper) is not int or upper != lower + 1 or upper > maximum)):
            raise ValueError(f'Invalid fit interval for {case}')
        retained[case] = {'case_id': case, 'lower': lower, 'upper': upper}
    if retained.keys() != cases.keys():
        raise ValueError('Fit interval IDs do not match the training paths')
    return retained.values()


def write_archive(path, files):
    with path.open('xb') as output, gzip.GzipFile(fileobj=output, mode='wb', mtime=0, filename='') as compressed:
        with tarfile.open(fileobj=compressed, mode='w|') as archive:
            for name, source in sorted(files.items()):
                member = tarfile.TarInfo(name)
                member.size = source.stat().st_size
                with source.open('rb') as handle:
                    archive.addfile(member, handle)


def run(prepared, audit, output):
    provenance = json.loads(audit.read_text(encoding='utf-8'))
    hashes = provenance['files_sha256']
    files = {name: prepared / name for name in PUBLIC_MEMBERS}
    for name in (*PUBLIC_MEMBERS, TRAIN_TARGETS):
        if digest(prepared / name) != hashes[name]:
            raise ValueError(f'Linked preparation changed before packaging: {name}')
    train = validate_split(prepared, 'train', provenance['splits']['train']['retained'])
    dev = validate_split(prepared, 'dev', provenance['splits']['dev']['retained'])
    if train.keys() & dev.keys() or any(provenance['overlap_after_filter'].values()):
        raise ValueError('Overlapping fit and held-out cases or content families')
    intervals = fit_intervals(prepared / TRAIN_TARGETS, train)
    output.mkdir(parents=True, exist_ok=False)
    targets = output / 'fit-intervals.jsonl'
    write_jsonl(targets, intervals)
    files['train/intervals.jsonl'] = targets

    manifest = output / 'assay.json'
    save_json(manifest, {
        'schema_version': 1,
        'scope': 'First linked stop that covers a complete retrospective evidence excerpt for a given IIRC anchor.',
        'preparation_audit_sha256': digest(audit),
        'model': provenance['model'], 'revision': provenance['revision'],
        'method_fields': list(METHODS),
        'forecast_row': {'case_id': 'prepared case identifier', 'methods': 'per-method conditional event probabilities '
                         'aligned to linked stops only, in ascending character order'},
        'cases': {'train': len(train), 'dev': len(dev)},
        'tokens': {split: provenance['splits'][split]['tokens'] for split in ('train', 'dev')},
        'held_out_not_in_archive': {name: hashes[name] for name in HELD_TARGETS},
        'inputs_sha256': {name: digest(source) for name, source in files.items()},
        'acceptance': {
            'bootstrap': '10,000 paired resamples grouped by connected article family',
            'interval_nll_gain_required_nats_vs_geometry': NLL_MINIMUM_GAIN,
            'max_brier_increase_vs_geometry': BRIER_TOLERANCE,
            'explicit_anchor_requirement': 'Upper paired NLL difference bound below zero vs same joined-path '
                                           'model using linked text states without explicit anchor vectors',
            'same_document_swaps': 'Descriptive only. A transfer gain does not establish identity-specific prediction.',
        },
    })
    files['assay.json'] = manifest
    archive = output / 'linked-evidence-model-inputs.tar.gz'
    write_archive(archive, files)
    with tarfile.open(archive, mode='r|gz') as reader:
        verified = {}
        for member in reader:
            if not member.isfile() or member.name not in files or member.name in verified:
                raise ValueError(f'Unexpected archive member: {member.name}')
            with reader.extractfile(member) as source:
                verified[member.name] = hashlib.file_digest(source, 'sha256').hexdigest()
    if verified != {name: digest(source) for name, source in files.items()}:
        raise ValueError('Archive member hashes disagree with packaged files')
    save_json(output / 'pack-verification.json', {
        'archive_sha256': digest(archive), 'members_sha256': verified,
        'held_out_target_hashes_recorded_without_reading_labels': True,
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared', 'audit', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    run(**vars(parser.parse_args()))
