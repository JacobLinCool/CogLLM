"""Score sealed portable heads at common original-text observations."""

from argparse import ArgumentParser
from bisect import bisect_right
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from research.backbone_transfer.extract import checked_digest, digest, read_json, stream_rows, write_json
from research.token_readout.data import TOKENIZER
from research.token_readout.score import (
    CONTROL_DIFF, PREFIX_ATOL, conditional_window, evidence_summary, first_separation,
    frick_summary, log_interval, mean, paired_comparison, probabilities, read_frick, survival,
)
from research.token_readout.transfer_data import MODELS, TOKENIZER_ROOT, EncodedText
from research.union_training.verify_observations import verified_member


ARMS = ('position', 'point', 'span', 'span-calibrated')
ASSAYS = ('frick', 'evidence', 'controls', 'ab')


def load_bundle(root):
    receipt = read_json(root / 'receipt.json')
    if receipt['schema'] != 1:
        raise ValueError('Unknown portable readout data schema')
    files = {name: verified_member(root, Path(name), value) for name, value in receipt['file_sha256'].items()}
    texts = {row['id']: row['text'] for row in stream_rows(files['stimuli.jsonl'])}
    design = read_json(files['design.json'])
    if len(texts) != receipt['documents'] or design['schema_version'] != 2:
        raise ValueError('Portable label and stimulus count disagree')
    return receipt, files, texts, design


def load_forecasts(run, model, protocol, expected):
    seal = read_json(run / 'seal.json')
    if (seal['protocol_sha256'] != protocol or seal.get('candidate', model) != model
            or seal.get('smoke', False)):
        raise ValueError(f'{model} was fitted against another protocol or was only a smoke run')
    members = {name: verified_member(run, Path(name), sha) for name, sha in seal['files_sha256'].items()}
    files = {f'forecast-{arm}.json': arm for arm in ARMS}
    if not {f'forecast-{arm}.json' for arm in ('position', 'point', 'span')} <= members.keys():
        raise ValueError(f'{model} is missing one or more declared forecast arms')
    forecasts = {}
    for filename, arm in files.items():
        if filename not in members:
            continue
        values = read_json(members[filename])
        if (set(values) != set(expected) or any(set(values[kind]) != ids for kind, ids in expected.items())):
            raise ValueError(f'{model}/{arm} forecast is missing test rows or contains a non-test row')
        forecasts[arm] = values
    return forecasts, {'run_seal_sha256': digest(run / 'seal.json'),
                       'forecasts_sha256': {arm: seal['files_sha256'][filename]
                                            for filename, arm in files.items() if filename in members}}


def evidence_rows(annotations, encoded, layout, predictions):
    output = []
    for row in annotations:
        key = row['id']
        entry, geometry = encoded[key], layout[key]
        start = geometry['risk_start']
        hazards = predictions[key]['hazard']
        if set(hazards) != set(geometry['anchor_tokens']):
            raise ValueError(f'Incorrect evidence candidate IDs in {key}')
        logs = {name: survival(probabilities(values, len(entry.ids), f'{key}/{name}'), start)
                for name, values in hazards.items()}
        label_windows = {q['id']: entry.overlap_window(q['evidence_window']) if q['evidence_window'] else None
                         for q in row['queries']}
        by_query = {}
        for name, window in label_windows.items():
            mass = logs[name]
            if window is None:
                by_query[name] = {'nll': -float(mass[-1]), 'right_censored': True}
            else:
                left, right = window
                by_query[name] = {'nll': -log_interval(mass, left, right, start), 'right_censored': False,
                                  'premature_probability': float(-np.expm1(mass[left - start]))}
        sample = {'group_id': row['group_id'], 'questions': by_query,
                  'nll': float(np.mean([value['nll'] for value in by_query.values()]))}
        early, late = row['earlier_query'], row['later_query']
        if label_windows[early] is not None:
            left, right = label_windows[early]
            first_shown = min(row['queries'], key=lambda query: query['span'][0])['id']
            early_mass = conditional_window(logs[early], left, right, start)
            late_mass = conditional_window(logs[late], left, right, start)
            gap = early_mass - late_mass
            first, timing = first_separation(np.exp(logs[late][1:]) - np.exp(logs[early][1:]),
                                             left - start, right - start - 1)
            absolute = first + start if first is not None else None
            sample.update({'earlier_question_shown_first': early == first_shown,
                           'earlier_window_conditional_advantage': gap, 'pair_ranking': (
                0.5 if abs(gap) < 1e-9 else float(gap > 0)),
                'earliest_nontrivial_advantage_token': absolute,
                'earliest_nontrivial_advantage_char_end': entry.ends[absolute] if absolute is not None else None,
                'earliest_nontrivial_advantage_window': timing})
            if row['earlier_sham']:
                a, b = entry.overlap_window(row['earlier_sham'])
                sample['earlier_window_advantage_over_sham'] = early_mass - conditional_window(logs[early], a, b, start)
        output.append(sample)
    return output


def ranking_by_display_order(rows):
    judged = [row for row in rows if 'pair_ranking' in row]
    strata = {name: [row['pair_ranking'] for row in judged if row['earlier_question_shown_first'] == first]
              for name, first in (('shown_first', True), ('shown_second', False))}
    accuracy = {name: float(np.mean(values)) for name, values in strata.items() if values}
    return {'pairs': {name: len(values) for name, values in strata.items()},
            'ranking_accuracy': accuracy,
            'equal_stratum_accuracy': float(np.mean(list(accuracy.values()))) if len(accuracy) == 2 else None,
            'always_first_accuracy': len(strata['shown_first']) / len(judged) if judged else None}


def resolution_curves(keys, encoded, layout, values):
    result = {}
    for key in keys:
        n = len(encoded[key].ids)
        anchors = layout[key]['anchor_tokens']
        hazards = values[key]['hazard']
        if set(hazards) != set(anchors):
            raise ValueError(f'Incorrect prose candidate IDs in {key}')
        state = {}
        for name, (_, risk_start) in anchors.items():
            hazard = probabilities(hazards[name], n, f'{key}/{name}')
            state[name] = np.r_[np.zeros(risk_start), -np.expm1(survival(hazard, risk_start))]
        result[key] = state
    return result


def curve_at(values, ends, stop):
    """Use only complete tokens by the original-character cutoff."""
    return float(values[bisect_right(ends, stop)])


def prefix_error(keys, curves, encoded, shared_stop):
    reference, *others = keys
    count = bisect_right(encoded[reference].ends, shared_stop)
    for other in others:
        if (encoded[other].ends[:count] != encoded[reference].ends[:count]
                or encoded[other].ids[:count] != encoded[reference].ids[:count]
                or bisect_right(encoded[other].ends, shared_stop) != count):
            raise ValueError('Verified common prefix offsets/token IDs do not agree')
    return max(float(np.max(np.abs(curves[reference][anchor][:count + 1] - curves[other][anchor][:count + 1])))
               for other in others for anchor in curves[reference])


def stage_ends(entry):
    parts = entry.text.split('\n\n')
    if len(parts) != 3:
        raise ValueError('Expected an authored three-segment prose stimulus')
    return [len(parts[0]), len(parts[0]) + len(parts[1]) + 2, len(entry.text)]


def matched_character_stop(stop, target, rival):
    """Align progress through matched prose stages by a conservative char cutoff."""
    for (t0, t1), (r0, r1) in zip(zip(target, target[1:]), zip(rival, rival[1:]), strict=True):
        if stop <= t1:
            return r0 + (stop - t0) * (r1 - r0) // (t1 - t0)
    raise ValueError('Prose observation lies outside its matching stage')


def pair_curve(key, rival, anchor, curves, encoded, target_stages, rival_stages):
    left, right = encoded[key].ends, encoded[rival].ends
    alignment = {stop: matched_character_stop(stop, target_stages, rival_stages)
                 for stop in set(left) if stop >= target_stages[0]}
    observations = sorted(alignment)
    values = {stop: curve_at(curves[key][anchor], left, stop)
              - curve_at(curves[rival][anchor], right, alignment[stop]) for stop in observations}
    return observations, values


def timed_pair(key, rival, anchor, curves, encoded, target_stages, rival_stages, window):
    observations, advantage = pair_curve(key, rival, anchor, curves, encoded, target_stages, rival_stages)
    start, end = window
    initial = next((stop for stop in observations if advantage[stop] >= CONTROL_DIFF), None)
    timing = ('never' if initial is None else 'before' if initial <= start else 'inside' if initial <= end else 'late')
    target_end = max(stop for stop in observations if stop <= end)
    return {'timing': timing, 'first_char_end': initial, 'cue_char_window': window,
            'decisive_visible_char_end': target_end, 'advantage_at_decisive': advantage[target_end]}


def controls_rows(annotations, encoded, layout, predictions, shared_stops):
    output = []
    for row in annotations:
        variants = row['variants']
        ids = {name: value['id'] for name, value in variants.items()}
        curves = resolution_curves(ids.values(), encoded, layout, predictions)
        stops = shared_stops[row['group_id']]
        prefix = stops['common_prefix_end']
        middle = stops['late_none_prefix_end']
        no = ids['none']
        none_closure = lambda cutoff: curve_at(curves[no]['main'], encoded[no].ends, cutoff)
        not_closed_before = 1 - none_closure(prefix)
        output.append({'group_id': row['group_id'],
                       'prefix_max': max(prefix_error(list(ids.values()), curves, encoded, prefix),
                                         prefix_error([ids['late'], no], curves, encoded, middle)),
                       'no_answer_retention': float(1 - curves[no]['main'][-1]),
                       'no_answer_transition_retention': ((1 - none_closure(middle)) / not_closed_before
                                                          if not_closed_before > 0 else None)})
        for name, contrast, divergence in (('early', 'late', prefix), ('late', 'none', middle)):
            key, rival = ids[name], ids[contrast]
            stages = [[divergence, *[stop for stop in stage_ends(encoded[current]) if stop > divergence]]
                      for current in (key, rival)]
            window = [variants[name]['cue_span'][0], variants[name]['decisive_end']]
            output[-1][name] = timed_pair(key, rival, 'main', curves, encoded, *stages, window)
    return output


def ab_rows(annotations, encoded, layout, predictions, shared_stops):
    output = []
    for row in annotations:
        variants = row['variants']
        ids = {name: value['id'] for name, value in variants.items()}
        curves = resolution_curves(ids.values(), encoded, layout, predictions)
        prefix = shared_stops[row['group_id']]['common_prefix_end']
        no = ids['none']
        family = {'group_id': row['group_id'], 'prefix_max': prefix_error(list(ids.values()), curves, encoded, prefix)}
        for target, other in (('A', 'B'), ('B', 'A')):
            key = ids[target]
            window = variants[target]['cue_span']
            control_window = variants['none']['cue_span']
            result = timed_pair(key, no, target, curves, encoded, [prefix, window[1], len(encoded[key].text)],
                                [prefix, control_window[1], len(encoded[no].text)], window)

            def increase(document, anchor, stop):
                return (curve_at(curves[document][anchor], encoded[document].ends, stop)
                        - curve_at(curves[document][anchor], encoded[document].ends, prefix))

            target_gain = increase(key, target, window[1])
            other_gain = increase(key, other, window[1])
            sham_target = increase(no, target, control_window[1])
            sham_other = increase(no, other, control_window[1])
            family[target] = {**result, 'target_gain': target_gain, 'other_gain': other_gain,
                              'sham_target_gain': sham_target, 'sham_other_gain': sham_other,
                              'selectivity': (target_gain - sham_target) - (other_gain - sham_other),
                              'passes': bool(result['timing'] == 'inside' and target_gain - sham_target >= CONTROL_DIFF
                                             and target_gain - other_gain >= CONTROL_DIFF
                                             and other_gain - sham_other < CONTROL_DIFF)}
        output.append(family)
    return output


def control_summary(rows):
    retention = [row['no_answer_transition_retention'] for row in rows if row['no_answer_transition_retention'] is not None]
    by_cue = {label: {status: sum(row[label]['timing'] == status for row in rows)
                      for status in ('before', 'inside', 'late', 'never')} for label in ('early', 'late')}
    both = sum(all(row[label]['timing'] == 'inside' for label in ('early', 'late')) for row in rows)
    prefix_max = max(row['prefix_max'] for row in rows)
    min_families = (3 * len(rows) + 3) // 4
    return {'families': len(rows), 'prefix_max': prefix_max, 'both_located': both, 'both_gate_threshold': min_families,
            'cue_timing': by_cue, 'mean_transition_retention': float(np.mean(retention)) if retention else None,
            'transition_at_least_80_percent': sum(value >= 0.8 for value in retention),
            'point_gate': prefix_max <= PREFIX_ATOL and both >= min_families and not any(
                counts['before'] for counts in by_cue.values())}


def ab_summary(rows):
    return {'families': len(rows), 'prefix_max': max(row['prefix_max'] for row in rows),
            'selective_cues': sum(row[anchor]['passes'] for row in rows for anchor in ('A', 'B')),
            'both_selective': sum(row['A']['passes'] and row['B']['passes'] for row in rows),
            'mean_sham_adjusted_selectivity': float(np.mean([row[anchor]['selectivity']
                                                           for row in rows for anchor in ('A', 'B')]))}


def compare(candidate, position):
    return {'frick_path_nll': paired_comparison(candidate['frick'], position['frick'], 'path_nll'),
            'evidence_interval_nll': paired_comparison(candidate['evidence'], position['evidence'], 'nll'),
            'evidence_same_window_ranking': paired_comparison(candidate['evidence'], position['evidence'], 'pair_ranking'),
            'controls_both_located': paired_comparison(
                *[[{'group_id': row['group_id'], 'hit': int(all(row[k]['timing'] == 'inside' for k in ('early', 'late')))}
                  for row in arm['controls']] for arm in (candidate, position)], 'hit'),
            'ab_both_selective': paired_comparison(
                *[[{'group_id': row['group_id'], 'hit': int(row['A']['passes'] and row['B']['passes'])}
                  for row in arm['ab']] for arm in (candidate, position)], 'hit')}


def evaluate(bundle, paths, protocol_sha256, runs, output):
    protocol = checked_digest(protocol_sha256)
    receipt, files, texts, design = load_bundle(bundle)
    frick = [row for row in design['frick'] if row['split'] == 'test']
    evidence = [row for row in design['evidence'] if row['split'] == 'test']
    stops = receipt['structural_alignment']['control_shared_stops']
    results = {}
    raw = {}
    for model, run in runs.items():
        path = paths[model]
        if digest(path) != receipt['tokenizer_sha256'][model]:
            raise ValueError(f'{model} evaluation tokenizer differs from the certified one')
        tokenizer = Tokenizer.from_file(str(path))
        data = read_json(files[f'{model}.json'])
        if data['model'] != model or data['schema'] != 1:
            raise ValueError(f'{model} evaluation layout differs from the certified one')
        layout = {key: row for key, row in data['rows'].items() if row['split'] == 'test'}
        expected = {kind: {key for key, row in layout.items() if row['kind'] == kind} for kind in ASSAYS}
        encodings = {key: EncodedText.encode(texts[key], tokenizer) for key in layout}
        if any(entry.offsets != tuple(map(tuple, layout[key]['offsets'])) for key, entry in encodings.items()):
            raise ValueError(f'{model} does not reproduce its original Unicode layout')
        forecasts, verification = load_forecasts(run, model, protocol, expected)
        scores = {arm: {'frick': read_frick(frick, (), False, prediction['frick']),
                        'evidence': evidence_rows(evidence, encodings, layout, prediction['evidence']),
                        'controls': controls_rows(design['controls'], encodings, layout, prediction['controls'], stops),
                        'ab': ab_rows(design['ab'], encodings, layout, prediction['ab'], stops)}
                  for arm, prediction in forecasts.items()}
        raw[model] = scores
        base = scores['position']
        results[model] = {'verification': verification,
                          'arms': {arm: {'frick': frick_summary(value['frick'], base['frick']),
                                         'evidence': {**evidence_summary(value['evidence'], base['evidence']),
                                                      'by_display_order': ranking_by_display_order(value['evidence'])},
                                         'controls': control_summary(value['controls']), 'ab': ab_summary(value['ab']),
                                         'details': value}
                                   for arm, value in scores.items()},
                          'span_vs_point': compare(scores['span'], scores['point'])}
        if 'span-calibrated' in scores:
            results[model]['span_calibrated_vs_span'] = compare(scores['span-calibrated'], scores['span'])
    cross = {f'{model}_vs_qwen': {arm: compare(raw[model][arm], raw['qwen'][arm])
                                 for arm in raw['qwen'] if arm in raw[model]}
             for model in raw if model != 'qwen' and 'qwen' in raw}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, {'schema': 1, 'protocol_sha256': protocol, 'bundle_receipt_sha256': digest(bundle / 'receipt.json'),
                        'models': results, 'paired_cross_model': cross,
                        'alignment': 'same declared original Unicode windows; conservative complete native tokens; '
                                     'matched control sentences aligned by relative character progress',
                        'interpretation': {'frick': 'human first report at sentence exposures',
                                           'evidence': 'answer windows given a question, rated retrospectively',
                                           'controls': 'earlier authored diagnostics, no human observations',
                                           'ab': 'new final-heldout authored diagnostics, no human observations'}})


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--run', action='append', required=True, help='one model=sealed-run-dir pair, repeatable')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--qwen-tokenizer', type=Path, default=TOKENIZER)
    parser.add_argument('--tokenizer-root', type=Path, default=TOKENIZER_ROOT)
    args = parser.parse_args()
    runs = {name: Path(path) for name, path in (item.split('=', 1) for item in args.run)}
    if len(runs) != len(args.run) or not set(runs) <= set(MODELS):
        parser.error('Expected one distinct sealed run directory per supported model')
    paths = {'qwen': args.qwen_tokenizer, **{name: args.tokenizer_root / name / 'tokenizer.json' for name in MODELS[1:]}}
    evaluate(args.prepared, paths, args.protocol_sha256, runs, args.output)
