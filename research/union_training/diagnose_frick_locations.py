"""Audit already-fitted Frick first-report forecasts on the model-selection split."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from research.backbone_transfer.extract import read_json, stream_rows, write_json
from research.union_training.verify_observations import digest, require, text_digest, verified_member


TASK = 'frick_first_report'
AFTER_TASK = 'frick_after_first_report'
CONDITIONS = ('01', '02')
METHODS = ('epoch31', 'epoch12', 'position_hazard')
PHASES = ('before_target', 'target', 'after_target', 'no_report')
ARCHIVED_EFFECTS = {'frick_target_first_report': 'first_report',
                    'frick_sentence_inconsistent': 'sentence_inconsistent'}


@dataclass(frozen=True)
class Story:
    family: str
    condition: str
    identifier: str
    coordinates: tuple
    sentences: tuple
    target: int
    changed: int


def read_stories(source):
    audit = read_json(source / 'source-audit.json')
    files = ('stimuli.jsonl', 'metadata.jsonl', 'outcomes.jsonl')
    for name in files:
        verified_member(source, Path(name), audit['output_sha256'][name])
    displayed = {row['id']: row for row in stream_rows(source / 'stimuli.jsonl')}
    stories, families = {}, defaultdict(dict)
    for row in stream_rows(source / 'metadata.jsonl'):
        key = row['id']
        material = displayed[key]
        text = material['text']
        spans = row['sentences']
        positions = tuple(span['stop'] for span in spans)
        sentences = tuple(text[span['start']:span['stop']] for span in spans)
        for span, sentence in zip(spans, sentences, strict=True):
            single = displayed[span['single_id']]
            require(single['text'] == sentence and single['id'] == text_digest(sentence)
                    and single['stops'] == [len(sentence)], 'Original Frick display geometry changed')
        require(key == text_digest(text) and material['stops'] == list(positions)
                and '\n\n'.join(sentences) == text, 'Frick path is not the exact displayed text')
        target, changed = row['target_index'], row['changed_index']
        require(1 <= changed < target < len(positions) - 1 and 7 <= target - changed <= 10,
                'Unexpected position of the manipulation or common target')
        family, condition = f"frick:{row['story']}", row['condition']
        require(key not in stories and condition in CONDITIONS and condition not in families[family],
                'Duplicate or malformed Frick variant')
        story = Story(family, condition, key, tuple((key, stop) for stop in positions[1:]),
                      sentences, target, changed)
        stories[key] = story
        families[family][condition] = story
    for pair in families.values():
        plain, altered = (pair[condition] for condition in CONDITIONS)
        differences = [i for i, (a, b) in enumerate(zip(plain.sentences, altered.sentences, strict=True)) if a != b]
        require(plain.target == altered.target and differences == [plain.changed] == [altered.changed],
                'The sentence paired for comparison must be identical after changing the earlier context')

    counts = {key: Counter() for key in stories}
    all_reports = {key: Counter() for key in stories}
    seen, readers = set(), defaultdict(set)
    for row in stream_rows(source / 'outcomes.jsonl'):
        story = stories[row['id']]
        reader = row['reader_id']
        key = story.family, reader
        keys = row['keys']
        require(key not in seen and len(keys) == len(story.sentences) and set(keys) <= {'k', 'd'},
                'Original Frick report path or reader assignment changed')
        seen.add(key)
        readers[story.identifier].add(reader)
        first = next((i for i, keypress in enumerate(keys[1:]) if keypress == 'd'), len(story.coordinates))
        counts[story.identifier][first] += 1
        all_reports[story.identifier].update(i for i, keypress in enumerate(keys[1:]) if keypress == 'd')
    require(len(stories) == 40 and len(families) == 20 and len(seen) == 1800,
            'Original Frick cohort is incomplete')
    return stories, counts, all_reports, readers, audit['output_sha256']


def read_forecasts(directory, packet_hash, role_indices, rows, after_rows, epoch):
    seal = read_json(directory / 'seal.json')
    require(seal['packet_receipt_sha256'] == packet_hash and seal['backbone_updates'] == 0
            and seal['one_forward_per_stimulus'], 'Forecasts do not refer to the frozen observation run')
    for name in ('fitting.json', 'predictions.npz'):
        verified_member(directory, Path(name), seal['files_sha256'][name])
    require(read_json(directory / 'fitting.json')['selected_epoch'] == epoch, 'Checkpoint epoch differs')
    prefix = f'{TASK}.development.'
    after_prefix = f'{AFTER_TASK}.development.'
    with np.load(directory / 'predictions.npz', allow_pickle=False) as file:
        indices, matrix = file[prefix + 'rows'], file[prefix + 'probabilities']
        after_indices = file[after_prefix + 'rows']
        after_matrix = file[after_prefix + 'probabilities'].astype(np.float64)
    require(indices.dtype.kind in 'iu' and np.array_equal(indices, role_indices)
            and matrix.ndim == 2 and matrix.shape[0] == len(indices), 'Forecast row order differs from the packet')
    result = {}
    for i, value in zip(indices, matrix.astype(np.float64), strict=True):
        length = len(rows[i]['coordinates'])
        require(len(value) > length and np.all(value[length:-1] == 0), 'Frick path padding is inconsistent')
        probabilities = np.r_[value[:length], value[-1]]
        require(np.isfinite(probabilities).all() and np.all(probabilities > 0)
                and np.isclose(probabilities.sum(), 1, atol=1e-6, rtol=0), 'Invalid first-report probability mass')
        result[i] = probabilities
    expected_after = [i for i, row in enumerate(after_rows) if row['role'] == 1]
    require(after_indices.dtype.kind in 'iu' and np.array_equal(after_indices, expected_after)
            and after_matrix.shape == (len(expected_after), 2) and np.isfinite(after_matrix).all()
            and np.all(after_matrix >= 0) and np.allclose(after_matrix.sum(axis=1), 1, atol=1e-6, rtol=0),
            'Invalid conditional Frick report forecasts or row order')
    after = {tuple(after_rows[i]['coordinates'][0]): (after_rows[i]['outcome'], float(value[1]))
             for i, value in zip(after_indices, after_matrix, strict=True)}
    require(len(after) == len(expected_after), 'Duplicate after-first report coordinate')
    return result, after, seal['files_sha256']['predictions.npz']


def fit_position_hazards(training):
    risk_by_stop, event_by_stop = Counter(), Counter()
    for vector in training:
        for stop, events in enumerate(vector[:-1]):
            event_by_stop[stop] += events
            risk_by_stop[stop] += sum(vector[stop:])
    hazards = [(event_by_stop[i] + 0.5) / (risk + 1) for i, risk in sorted(risk_by_stop.items())]
    require(list(sorted(risk_by_stop)) == list(range(len(hazards))) and min(risk_by_stop.values()) > 0,
            'Training risks do not support contiguous absolute positions')
    return hazards, [{'sentence': i + 2, 'risk': risk_by_stop[i], 'events': event_by_stop[i], 'hazard': hazard}
                     for i, hazard in enumerate(hazards)]


def position_forecast(hazards, length):
    require(length <= len(hazards), 'Held-out path exceeds train-only baseline support')
    chance = np.asarray(hazards[:length])
    survival = np.r_[1, np.cumprod(1 - chance)]
    return np.r_[chance * survival[:-1], survival[-1]]


def phase_mass(values, target):
    return np.array([values[:target].sum(), values[target], values[target + 1:-1].sum(), values[-1]])


def nll_total(counts, probabilities):
    observed = counts > 0
    return float(-counts[observed] @ np.log(probabilities[observed]))


def variant_result(story, counts, forecasts):
    # `target` is zero-based in display order and the modeled path starts at sentence 2.
    target = story.target - 1
    memberships = []
    for index, (_, stop) in enumerate(story.coordinates):
        memberships.append({'sentence': index + 2, 'end_exclusive': stop, 'human_first_reports': int(counts[index]),
                            'human_at_risk': int(counts[index:].sum()),
                            'first_report_probability': {name: float(p[index]) for name, p in forecasts.items()}})
    groups = phase_mass(counts, target)
    totals = int(counts.sum())
    at_risk = int(counts[target:].sum())
    conditional = {}
    for name, probabilities in forecasts.items():
        survival = float(probabilities[target:].sum())
        conditional[name] = {'survival_entering_target': survival,
                             'hazard': float(probabilities[target]) / survival}
    return {
        'condition': story.condition, 'stimulus_sha256': story.identifier,
        'trials': totals, 'target_sentence': story.target + 1,
        'changed_sentence': story.changed + 1, 'changed_sentence_text': story.sentences[story.changed],
        'sentences': memberships, 'no_report': {'human_count': int(counts[-1]),
                                                'probability': {name: float(p[-1]) for name, p in forecasts.items()}},
        'human_phase_counts': dict(zip(PHASES, groups.tolist(), strict=True)),
        'target_conditional_first_report': {'human_at_risk': at_risk, 'human_first_reports': int(counts[target]),
                                            'human_hazard': int(counts[target]) / at_risk, 'forecast': conditional},
        'forecast_phase_probabilities': {name: dict(zip(PHASES, phase_mass(p, target).tolist(), strict=True))
                                         for name, p in forecasts.items()},
        'full_path_nll_nats_per_trial': {name: nll_total(counts, p) / totals for name, p in forecasts.items()},
        'four_phase_nll_nats_per_trial': {name: nll_total(groups, phase_mass(p, target)) / totals
                                          for name, p in forecasts.items()},
    }


def summarize(variants):
    count = sum(row['trials'] for row in variants)
    result = {'trials': count, 'variants': len(variants),
              'human_phase_counts': {phase: sum(row['human_phase_counts'][phase] for row in variants) for phase in PHASES}}
    result['human_phase_rates'] = {phase: total / count for phase, total in result['human_phase_counts'].items()}
    result['forecast_phase_probabilities'] = {
        name: {phase: sum(row['trials'] * row['forecast_phase_probabilities'][name][phase] for row in variants) / count
               for phase in PHASES} for name in METHODS}
    for metric in ('full_path_nll_nats_per_trial', 'four_phase_nll_nats_per_trial'):
        result[metric] = {name: sum(row['trials'] * row[metric][name] for row in variants) / count for name in METHODS}
    return result


def target_gap(plain, altered):
    human = 100 * (altered['human_phase_rates']['target'] - plain['human_phase_rates']['target'])
    model = {name: 100 * (altered['forecast_phase_probabilities'][name]['target']
                          - plain['forecast_phase_probabilities'][name]['target']) for name in METHODS}
    return {'human': human, 'forecast': model, 'absolute_error': {name: abs(value - human) for name, value in model.items()}}


def sentence_reports(story, counts, all_reports, forecasts, after, position):
    key = story.coordinates[position]
    earlier = sum(counts[:position])
    trials = sum(counts)
    repeated = all_reports[position] - counts[position]
    require(0 <= repeated <= earlier, 'A report after the first has an invalid observed risk set')
    result = {}
    for name, probability in forecasts.items():
        outcome, conditional = after[name][key]
        require(outcome == [earlier - repeated, repeated], 'Conditional Frick counts differ from the original key paths')
        first = float(probability[position])
        prior = float(probability[:position].sum())
        result[name] = {'first_report': first, 'sentence_inconsistent': first + prior * conditional}
    return {'display_sentence': position + 2, 'end_exclusive': key[1], 'trials': trials,
            'human_counts': {'first_report': counts[position], 'sentence_inconsistent': all_reports[position]},
            'forecasts': result}


def report_effect(points, endpoint):
    rates = {}
    for condition in CONDITIONS:
        rows = points[condition]
        denominators = sum(row['trials'] for row in rows)
        rates[condition] = {
            'human': sum(row['human_counts'][endpoint] for row in rows) / denominators,
            **{name: sum(row['trials'] * row['forecasts'][name][endpoint] for row in rows) / denominators
               for name in METHODS[:2]},
        }
    return {name: 100 * (rates['02'][name] - rates['01'][name]) for name in rates['01']}


def pooled_target_hazard(variants):
    observed = [row['target_conditional_first_report'] for row in variants]
    at_risk = sum(row['human_at_risk'] for row in observed)
    first = sum(row['human_first_reports'] for row in observed)
    predictions = {}
    for name in METHODS:
        model_risk = [row['trials'] * point['forecast'][name]['survival_entering_target']
                      for row, point in zip(variants, observed, strict=True)]
        hazards = [row['forecast'][name]['hazard'] for row in observed]
        predictions[name] = {
            'model_risk_weighted_hazard': float(np.average(hazards, weights=model_risk)),
            'human_risk_weighted_hazard': float(np.average(hazards, weights=[row['human_at_risk'] for row in observed])),
        }
    return {'human_at_risk': at_risk, 'human_first_reports': first, 'human_hazard': first / at_risk,
            'forecast': predictions}


def archived_alignment(base, source_receipt, forecasts, actual, paths):
    base_hash = digest(base / 'receipt.json')
    require(base_hash == source_receipt['base_receipt_sha256'], 'The archive does not match the original packet receipt')
    checksums = read_json(base / 'receipt.json')['packet_sha256']
    tasks = read_json(verified_member(base, Path('index.json'), checksums['index.json']))['tasks']
    archive = {key: read_json(path / 'development-effects.json') for key, path in paths.items()}
    for key, row in archive.items():
        require(row['base_receipt_sha256'] == base_hash and row['predictions_sha256'] == forecasts[key],
                'The earlier effect report refers to a different fit')
    result = {}
    for task in tasks:
        name = task['name']
        if name not in ARCHIVED_EFFECTS:
            continue
        file = Path(task['file'])
        rows = list(stream_rows(verified_member(base, file, checksums[str(file)])))
        effect, = task['dev_effects']
        endpoint = ARCHIVED_EFFECTS[name]
        totals = Counter()
        mapped = set()
        for index, coefficient in zip(effect['indices'], effect['coefficients'], strict=True):
            row = rows[index]
            before, target = actual[row['doc']]
            require(row['stop'] == before['end_exclusive'] and row['stop'] != target['end_exclusive']
                    and before['display_sentence'] == target['display_sentence'] - 1 and row['split'] == 1,
                    'A historical Frick contrast does not match the checked preceding display position')
            totals['human'] += coefficient * before['human_counts'][endpoint] / before['trials']
            for model in paths:
                totals[model] += coefficient * before['forecasts'][model][endpoint]
            mapped.add(row['doc'])
        require(mapped == actual.keys(), 'Historical Frick alignment does not cover every scored variant')
        for model, report in archive.items():
            prior, = (row for row in report['effects'] if row['task'] == name)
            require(np.isclose(totals['human'], prior['human_difference'], atol=1e-10, rtol=0)
                    and np.isclose(totals[model], prior['model_difference'], atol=1e-10, rtol=0),
                    'Historical Frick metric does not reproduce at the preceding sentence')
        result[endpoint] = {key: 100 * value for key, value in totals.items()}
    require(result.keys() == set(ARCHIVED_EFFECTS.values()), 'Missing historical Frick effects')
    return {'status': 'Archived contrast replayed at the preceding sentence, one position before the true target',
            'base_receipt_sha256': base_hash,
            'report_sha256': {key: digest(path / 'development-effects.json') for key, path in paths.items()},
            'archived_effects_percentage_points': result}


def run(packet, source, base, settled, original, output):
    receipt = packet / 'receipt.json'
    packet_hash = digest(receipt)
    receipt_body = read_json(receipt)
    filename = Path(f'{TASK}.jsonl')
    after_file = Path(f'{AFTER_TASK}.jsonl')
    row_hash = receipt_body['packet_sha256'][str(filename)]
    after_hash = receipt_body['packet_sha256'][str(after_file)]
    rows = list(stream_rows(verified_member(packet, filename, row_hash)))
    after_rows = list(stream_rows(verified_member(packet, after_file, after_hash)))
    stories, raw_counts, all_reports, readers, source_hashes = read_stories(source)
    roles, variant_rows = defaultdict(list), {}
    for i, row in enumerate(rows):
        story = stories[row['coordinates'][0][0]]
        require(story.identifier not in variant_rows and row['family'] == story.family
                and tuple(map(tuple, row['coordinates'])) == story.coordinates and row['role'] in (0, 1),
                'Packet roles or coordinates do not match the original Frick design')
        expected = [raw_counts[story.identifier][j] for j in range(len(story.coordinates) + 1)]
        require(expected == row['outcome'], 'First reports or censoring do not reproduce from the original responses')
        variant_rows[story.identifier] = row
        roles[row['role']].append(i)
    require(variant_rows.keys() == stories.keys(), 'Some study versions are missing')
    role_families = {role: {rows[i]['family'] for i in indices} for role, indices in roles.items()}
    require(role_families.keys() == {0, 1} and not role_families[0] & role_families[1],
            'The baseline fitting split must exclude both versions of every scored story')
    hazards, position_audit = fit_position_hazards([rows[i]['outcome'] for i in roles[0]])
    current, current_after, current_hash = read_forecasts(settled, packet_hash, roles[1], rows, after_rows, 31)
    earlier, earlier_after, earlier_hash = read_forecasts(original, packet_hash, roles[1], rows, after_rows, 12)
    variants, pairs = defaultdict(list), defaultdict(dict)
    actual = {}
    for i in roles[1]:
        row = rows[i]
        story = stories[row['coordinates'][0][0]]
        model = dict(zip(METHODS, (current[i], earlier[i], position_forecast(hazards, len(story.coordinates))), strict=True))
        variant = variant_result(story, np.asarray(row['outcome']), model)
        head = {'epoch31': current[i], 'epoch12': earlier[i]}
        after = {'epoch31': current_after, 'epoch12': earlier_after}
        before, target = (sentence_reports(story, row['outcome'], all_reports[story.identifier], head, after, position)
                          for position in (story.target - 2, story.target - 1))
        actual[story.identifier] = before, target
        variant['actual_target_reports'] = target
        variants[story.condition].append(variant)
        pair = pairs[story.family]
        pair['target_sentence_text'] = story.sentences[story.target]
        pair[story.condition] = variant
    by_condition = {condition: summarize(variants[condition]) for condition in CONDITIONS}
    pair_rows = []
    for family, pair in sorted(pairs.items()):
        plain, altered = (summarize([pair[condition]]) for condition in CONDITIONS)
        pair_rows.append({'family': family, 'target_sentence_text': pair['target_sentence_text'],
                          'conditions': {condition: pair[condition] for condition in CONDITIONS},
                          'target_effect_percentage_points': target_gap(plain, altered)})
    with_readers = {role: set().union(*(readers[rows[i]['coordinates'][0][0]] for i in indices))
                    for role, indices in roles.items()}
    forecasts = {'epoch31': current_hash, 'epoch12': earlier_hash}
    archived = archived_alignment(base, receipt_body, forecasts, actual,
                                  {'epoch31': settled, 'epoch12': original})
    target_points = {condition: [v['actual_target_reports'] for v in variants[condition]] for condition in CONDITIONS}
    trial_pooled = {endpoint: report_effect(target_points, endpoint) for endpoint in ARCHIVED_EFFECTS.values()}
    family_effects = [
        {endpoint: report_effect({condition: [pair[condition]['actual_target_reports']] for condition in CONDITIONS}, endpoint)
         for endpoint in ARCHIVED_EFFECTS.values()}
        for pair in pairs.values()
    ]
    equal_family = {endpoint: {model: float(np.mean([family[endpoint][model] for family in family_effects]))
                               for model in gap} for endpoint, gap in trial_pooled.items()}
    write_json(output, {
        'analysis': 'Model-selection development diagnostic on originally assigned Frick story families',
        'task': 'First prompted report of inconsistency with prior text, from displayed sentence 2; no report is terminal censoring',
        'packet_sha256': packet_hash, 'task_rows_sha256': row_hash, 'conditional_rows_sha256': after_hash,
        'original_source_sha256': source_hashes, 'predictions_sha256': forecasts,
        'split': {name: {'families': len(role_families[role]), 'variants': len(roles[role]),
                         'trials': sum(sum(rows[i]['outcome']) for i in roles[role]), 'readers': len(with_readers[role])}
                  for role, name in enumerate(('training', 'development'))},
        'readers_in_both_roles': len(with_readers[0] & with_readers[1]),
        'baseline': {'definition': 'Train-only absolute sentence hazard pooled across both conditions with Beta(0.5, 0.5) smoothing',
                     'positions': position_audit},
        'condition_labels': {'01': 'consistent_context', '02': 'inconsistent_context'},
        'development': summarize(variants['01'] + variants['02']), 'development_by_condition': by_condition,
        'target_conditional_first_report_by_condition': {condition: pooled_target_hazard(variants[condition])
                                                        for condition in CONDITIONS},
        'target_effect_percentage_points': target_gap(by_condition['01'], by_condition['02']),
        'mean_absolute_within_family_target_effect_error_percentage_points': {
            name: float(np.mean([pair['target_effect_percentage_points']['absolute_error'][name] for pair in pair_rows]))
            for name in METHODS},
        'target_alignment_audit': {**archived, 'actual_target_effects_percentage_points': {
            'trial_pooled': trial_pooled, 'equal_family': equal_family}},
        'pairs': pair_rows,
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('packet', 'source', 'base', 'settled', 'original', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    run(**vars(parser.parse_args()))
