"""Independently recompute Oliver Experiment 1 endpoints from typed local rows."""

from collections import Counter
from dataclasses import dataclass
import argparse
from hashlib import file_digest, sha256
import json
from pathlib import Path

import numpy as np


SPLIT_PREFIX = 'cogllm-oliver-e1-v1:'
SEED = 608315
DRAWS = 10_000
SOURCE_RECEIPT = Path(__file__).resolve().parents[1] / 'results' / 'oliver-retrieval-source-audit.json'


@dataclass(frozen=True)
class Observation:
    reader: str
    family: str
    first_variant: int | None
    second_variant: int | None
    same_written_definition: bool
    first_document: str | None
    first: str
    second: str
    pressed_know: bool | None
    confirmed: bool | None
    valid_first: bool | None
    valid_second: bool | None


def family_split(targets):
    salted = {family: sha256((SPLIT_PREFIX + target.strip().casefold()).encode('utf-8')).hexdigest()
              for family, target in targets.items()}
    if len(salted) != 44 or len(set(salted.values())) != 44:
        raise ValueError('Expected the 44 distinct Experiment 1 target families before alignment exclusions')
    ranked = sorted(salted, key=salted.get)
    return set(ranked[:30]), set(ranked[30:])


def count_rows(rows):
    def phase_counts(items):
        return {
            'pairs': len(items),
            'first_report': dict(sorted(Counter(row.first for row in items).items())),
            'retest_report': dict(sorted(Counter(row.second for row in items).items())),
            'first_to_retest': {'|'.join(pair): n for pair, n in
                                sorted(Counter((row.first, row.second) for row in items).items())},
        }

    wording = {(row.first_variant, row.second_variant) for row in rows if row.first_variant is not None}
    return {
        **phase_counts(rows),
        'readers': len({row.reader for row in rows}),
        'families': len({row.family for row in rows}),
        'stage1_by_initial_report': {
            first: sorted(({'delayed_know_press': press, 'post_reveal_same_target': yes, 'count': count}
                           for (press, yes), count in Counter((row.pressed_know, row.confirmed)
                                                              for row in rows if row.first == first).items()),
                          key=lambda record: (str(record['delayed_know_press']), str(record['post_reveal_same_target'])))
            for first in sorted({row.first for row in rows})
        },
        'retest_by_wording_direction': {
            f'{a}->{b}': phase_counts([row for row in rows if (row.first_variant, row.second_variant) == (a, b)])
            for a, b in sorted(wording)
        },
        'retest_by_wording_match': {
            name: phase_counts([row for row in rows if row.same_written_definition is same])
            for name, same in (('identical', True), ('alternate', False))
        },
    }


def effect_intervals(rows, *, draws=DRAWS, seed=SEED):
    readers = sorted({row.reader for row in rows})
    families = sorted({row.family for row in rows})
    reader_index = {reader: i for i, reader in enumerate(readers)}
    family_index = {family: i for i, family in enumerate(families)}
    i_reader = np.array([reader_index[row.reader] for row in rows])
    i_family = np.array([family_index[row.family] for row in rows])
    groups = (np.array([row.first == 'tot' for row in rows]), np.array([row.first == 'dont_know' for row in rows]))
    outcomes = ([row.pressed_know for row in rows], [row.confirmed for row in rows])
    numerator, denominator, counts = [], [], []
    for endpoint in outcomes:
        yes = np.array([value is True for value in endpoint])
        observed = np.array([value is not None for value in endpoint])
        numerator.append(np.column_stack([selected & yes for selected in groups]))
        denominator.append(np.column_stack([selected & observed for selected in groups]))
        counts.append({'tot': [int(np.sum(yes & groups[0])), int(np.sum(observed & groups[0]))],
                       'dont_know': [int(np.sum(yes & groups[1])), int(np.sum(observed & groups[1]))]})
    numerator = np.concatenate(numerator, axis=1).astype(float)
    denominator = np.concatenate(denominator, axis=1).astype(float)

    def contrast(weights):
        totals = weights @ denominator
        if np.any(totals == 0):
            raise ValueError('A repeated-sampling draw has an unobserved endpoint denominator')
        rate = (weights @ numerator) / totals
        return rate[:, ::2] - rate[:, 1::2]

    point = contrast(np.ones((1, len(rows))))[0]
    rng = np.random.default_rng(seed)
    # Keep draw batches deterministic while independent of the main score reducer.
    draws_collected = []
    for start in range(0, draws, 128):
        size = min(128, draws - start)
        reader_weights = rng.multinomial(len(readers), np.full(len(readers), 1 / len(readers)), size=size)
        family_weights = rng.multinomial(len(families), np.full(len(families), 1 / len(families)), size=size)
        draws_collected.append(contrast(reader_weights[:, i_reader] * family_weights[:, i_family]))
    effects = np.concatenate(draws_collected)
    return {name: {'estimate': point[j].item(), 'observed': counts[j],
                   'ci95': np.quantile(effects[:, j], [0.025, 0.975]).tolist()}
            for j, name in enumerate(('know_press_tot_minus_dont_know', 'confirmation_tot_minus_dont_know'))}


TABLE1 = {
    'identical': {'know': (380, 1, 4), 'dont_know_no': (145, 110, 37), 'dont_know_yes': (26, 6, 4),
                  'tot_no': (59, 6, 18), 'tot_yes': (50, 1, 3)},
    'alternate': {'know': (357, 15, 10), 'dont_know_no': (144, 103, 38), 'dont_know_yes': (27, 5, 2),
                  'tot_no': (49, 10, 11), 'tot_yes': (54, 7, 4)},
}


def published_table_diagnostic(rows):
    valid_pairs = [row for row in rows if row.valid_first is True and row.valid_second is True]
    # Printed frequencies and source flags disagree for this exact recorded stratum.
    exception_rows = [row for row in rows if row.first == 'dont_know' and row.pressed_know
                      and row.confirmed is False and row.valid_first is False and row.valid_second is True]
    result = {}
    exception_counts = {}
    adjusted = {}
    for wording, same in (('identical', True), ('alternate', False)):
        observed = Counter()
        for row in valid_pairs:
            if row.same_written_definition is same:
                first = row.first
                if first != 'know':
                    first += '_yes' if row.pressed_know and row.confirmed else '_no'
                observed[first, row.second] += 1
        result[wording] = {kind: tuple(observed[kind, second] for second in ('know', 'dont_know', 'tot'))
                           for kind in TABLE1[wording]}
        added = Counter(row.second for row in exception_rows if row.same_written_definition is same)
        exception_counts[wording] = tuple(added[state] for state in ('know', 'dont_know', 'tot'))
        adjusted[wording] = dict(result[wording])
        adjusted[wording]['dont_know_yes'] = tuple(a + b for a, b in zip(result[wording]['dont_know_yes'],
                                                                     exception_counts[wording], strict=True))
    return {'retained_pairs': len(valid_pairs), 'excluded_pairs': len(rows) - len(valid_pairs),
            'cross_tab': result, 'matches_published_table_exactly': result == TABLE1,
            'additional_dont_know_with_press_no_confirmation_first_invalid_second_valid': exception_counts,
            'matches_published_table_after_counting_that_stratum_as_resolved': adjusted == TABLE1}


def load_rows(path):
    with path.open(encoding='utf-8') as source:
        return [json.loads(line) for line in source]


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def audit(prepared, receipt=SOURCE_RECEIPT):
    hashes = {name: digest(prepared / f'{name}.jsonl') for name in ('stimuli', 'layout', 'targets', 'outcomes')}
    if json.loads(receipt.read_text(encoding='utf-8'))['prepared_sha256'] != hashes:
        raise ValueError('The prepared source differs from its released receipt')
    stimuli = load_rows(prepared / 'stimuli.jsonl')
    layout = {row['id']: (row['family'], row['version']) for row in load_rows(prepared / 'layout.jsonl')}
    targets = {row['family']: row['target'] for row in load_rows(prepared / 'targets.jsonl')}
    inputs = load_rows(prepared / 'outcomes.jsonl')
    if (len(stimuli) != 86 or len(layout) != 86 or {row['id'] for row in stimuli} != layout.keys()
            or any(row.keys() != {'id', 'text', 'stops'} or row['stops'] != [len(row['text'])] for row in stimuli)):
        raise ValueError('Expected two exact definition variants for each of the 43 retained target families')
    train, held = family_split(targets)
    observations = []
    keys = set()
    for row in inputs:
        family, reader = row['family'], row['reader_id']
        first, second = row['first'], row['second']
        key = reader, family
        written_match = row['same_written_definition']
        recall, confirmed = first['recall_signaled_before_reveal'], first['same_target_confirmed']
        ver1 = ver2 = None
        if first['id'] is not None and second['id'] is not None:
            fam1, ver1 = layout[first['id']]
            fam2, ver2 = layout[second['id']]
            if not (fam1 == fam2 == family and (ver1 == ver2) is written_match):
                raise ValueError('A wording pair does not map to the same source family and assigned contrast')
        elif first['id'] is not None or second['id'] is not None:
            raise ValueError('Quarantined families must have both source definitions withheld')
        if (key in keys or family not in targets or first['state'] not in ('know', 'dont_know', 'tot')
                or second['state'] not in ('know', 'dont_know', 'tot') or type(written_match) is not bool
                or (first['state'] == 'know' and recall is not None)
                or (first['state'] != 'know' and type(recall) is not bool) or type(confirmed) is not bool
                or type(first['source_valid']) is not bool or type(second['source_valid']) is not bool
                or first['spoken_equals_target_before_reveal'] is not None):
            raise ValueError('Invalid paired initial phase, route, or post-reveal observation')
        keys.add(key)
        observations.append(Observation(reader, family, ver1, ver2, written_match, first['id'], first['state'],
                                        second['state'], recall, confirmed, first['source_valid'], second['source_valid']))
    readers = {row.reader for row in observations}
    if len(readers) != 52 or keys != {(reader, family) for reader in readers for family in targets}:
        raise ValueError('The raw joined study is not the full crossed 52-reader by 44-family cohort')
    aligned = [row for row in observations if row.first_document is not None]
    train_rows = [row for row in aligned if row.family in train]
    test_rows = [row for row in aligned if row.family in held]
    if len(aligned) != 2_236 or len(train_rows) + len(test_rows) != len(aligned):
        raise ValueError('The model-aligned corpus does not reconcile to the locked split')
    return {
        'source_receipt_sha256': digest(receipt), 'prepared_sha256': hashes,
        'all': count_rows(observations), 'aligned': count_rows(aligned),
        'aligned_train': count_rows(train_rows), 'aligned_held_out': count_rows(test_rows),
        'aligned_held_out_effects': effect_intervals(test_rows),
        'published_validity_diagnostic_excluded_from_model_scoring': published_table_diagnostic(observations),
        'bootstrap': {'seed': SEED, 'draws': DRAWS, 'sampling': 'reader by target-family independent multiplicities'},
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prepared', type=Path)
    parser.add_argument('--receipt', type=Path, default=SOURCE_RECEIPT)
    args = parser.parse_args()
    print(json.dumps(audit(args.prepared, args.receipt), ensure_ascii=False, indent=2, allow_nan=False))
