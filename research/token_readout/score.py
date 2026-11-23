"""Score sealed CogLLM forecasts against separately stored, local annotations."""

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from research.backbone_transfer.extract import checked_digest, digest, read_json, stream_rows, write_json
from research.token_readout.data import TOKENIZER, TOKENIZER_SHA256, certified_end, certified_window, encode
from research.union_training.verify_observations import require, verified_member


ARMS = ('position', 'frozen', 'lora')
ASSAYS = ('frick', 'evidence', 'controls')
PROBABILITY_FLOOR = 1e-12
CONTROL_DIFF = 0.05
TRANSITION_REFERENCE_RETENTION = 0.80
PREFIX_ATOL = 3e-4
FRICK_MARGIN = 0.10
BOOTSTRAP_SEED = 735972
BOOTSTRAP_REPETITIONS = 10_000


@dataclass(frozen=True)
class Documents:
    texts: dict
    encodings: dict
    tokenizer: Tokenizer

    def end(self, key, offset):
        return certified_end(self.tokenizer, self.texts[key], self.encodings[key], offset)

    def window(self, key, span):
        return certified_window(self.tokenizer, self.texts[key], self.encodings[key], span)

    def verify_offsets(self, key, offsets):
        require(offsets == [list(value) for value in self.encodings[key].offsets], f'Token offsets differ on {key}')


def load_prepared(prepared, tokenizer_path):
    receipt = read_json(prepared / 'receipt.json')
    require(receipt['schema_version'] == 1 and receipt['tokenizer_sha256'] == TOKENIZER_SHA256
            and digest(tokenizer_path) == TOKENIZER_SHA256, 'Unrecognized text and label packet')
    mapping = {name: verified_member(prepared, Path(name), receipt['file_sha256'][name])
               for name in ('stimuli.jsonl', 'design.json')}
    inputs = list(stream_rows(mapping['stimuli.jsonl']))
    require(all(set(row) == {'id', 'text'} for row in inputs), 'Annotations are present in the text-only packet')
    texts = {row['id']: row['text'] for row in inputs}
    require(len(inputs) == len(texts) == receipt['documents'], 'Prepared document count differs')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    docs = Documents(texts, {key: encode(tokenizer, text) for key, text in texts.items()}, tokenizer)
    return read_json(mapping['design.json']), docs, receipt


def load_forecasts(predictions, protocol_sha256, expected):
    seal = read_json(predictions / 'seal.json')
    require(seal['protocol_sha256'] == protocol_sha256 and seal['smoke'] is False,
            'Training artifact lacks the exact preregistered protocol or contains a smoke result')
    files = seal['files_sha256']
    for name, sha in files.items():
        verified_member(predictions, Path(name), sha)
    possible = {f'forecast-{mode}.json' for mode in ARMS}
    found = {path.name for path in predictions.glob('forecast-*.json')}
    require(found <= possible and found <= set(files) and 'forecast-position.json' in found
            and any(name != 'forecast-position.json' for name in found), 'Missing, unsealed, or unknown forecasts')
    result = {}
    for mode in ARMS:
        name = f'forecast-{mode}.json'
        if name not in found:
            continue
        arm = read_json(predictions / name)
        require(set(arm) == set(ASSAYS) and all(set(arm[kind]) == expected[kind] for kind in ASSAYS),
                f'{mode} forecast does not cover exactly the sealed scoring rows')
        result[mode] = arm
    return result, seal


def probabilities(values, length, name):
    values = np.asarray(values, dtype=np.float64)
    require(values.shape == (length,) and np.isfinite(values).all() and ((0 <= values) & (values <= 1)).all(),
            f'Invalid forecast probabilities for {name}')
    return values


def read_frick(annotated, baseline_positions, position_checks, predictions):
    grouped = {}
    for row in annotated:
        key = row['id']
        p = predictions[key]
        counts = np.asarray([*row['event_counts'], row['censor_count']], dtype=np.float64)
        n = len(counts) - 1
        hazards = probabilities(p['hazard'], n, key)
        reported = probabilities(p['first_report'], n + 1, key)
        survival = np.r_[1.0, np.cumprod(1 - hazards)]
        direct = np.r_[survival[:-1] * hazards, survival[-1]]
        require(np.allclose(reported, direct, atol=2e-6, rtol=2e-5) and abs(reported.sum() - 1) < 2e-6,
                f'Inconsistent Frick hazard and mass for {key}')
        if position_checks:
            require(np.allclose(hazards, baseline_positions[:n], rtol=1e-5, atol=1e-7),
                    f'Position hazard for {key} differs from the train-only baseline')
        target = row['target_sentence_index'] - 1
        require(0 <= target < n and row['event_ends'][target] == row['target_span'][1], 'Wrong Frick target stop')
        total = counts.sum()
        require(total == row['observations'] and total > 0 and counts[target:].sum() > 0,
                'Frick outcome denominator differs')
        stats = {'trials': int(total), 'target_at_risk': int(counts[target:].sum()),
                 'target_first_reports': int(counts[target]),
                 'path_nll': float(-counts @ np.log(np.maximum(reported, PROBABILITY_FLOOR)) / total),
                 'human_target_probability': float(counts[target] / total),
                 'model_target_probability': float(reported[target]),
                 'human_premature_probability': float(counts[:target].sum() / total),
                 'model_premature_probability': float(reported[:target].sum()),
                 'human_target_hazard': float(counts[target] / counts[target:].sum()),
                 'model_target_hazard': float(hazards[target]),
                 'target_sentence_number': target + 2}
        group = grouped.setdefault(row['group_id'], {})
        require(row['condition'] not in group, 'Duplicate condition within a Frick family')
        group[row['condition']] = stats
    scored = []
    for group_id, pair in sorted(grouped.items()):
        require(set(pair) == {'01', '02'} and pair['01']['target_sentence_number'] == pair['02']['target_sentence_number'],
                'Frick family is missing a matched target')
        left, right = pair['01'], pair['02']
        summary = {'group_id': group_id, 'conditions': pair,
                   'path_nll': (left['path_nll'] + right['path_nll']) / 2}
        for who in ('human', 'model'):
            for field in ('target_probability', 'target_hazard'):
                measure = f'{who}_{field}'
                summary[f'{measure}_delta'] = right[measure] - left[measure]
        for field in ('target_probability_delta', 'target_hazard_delta'):
            summary[f'{field}_error'] = summary[f'model_{field}'] - summary[f'human_{field}']
        for condition, row in pair.items():
            summary[f'premature_error_{condition}'] = (row['model_premature_probability']
                                                       - row['human_premature_probability'])
        scored.append(summary)
    return scored


def survival(hazard, risk_start):
    calibrated = np.clip(hazard[risk_start:], PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR)
    return np.r_[0.0, np.log1p(-calibrated).cumsum()]


def log_interval(log_survival, start, end, risk_start):
    require(risk_start <= start < end <= risk_start + len(log_survival) - 1, 'Invalid evidence window')
    before, after = log_survival[np.asarray([start, end]) - risk_start]
    return float(before + np.log(-np.expm1(after - before)))


def conditional_window(log_survival, start, end, risk_start):
    before, after = log_survival[np.asarray([start, end]) - risk_start]
    return float(-np.expm1(after - before))


def first_separation(difference, start, decisive, threshold=CONTROL_DIFF):
    first = np.flatnonzero(difference >= threshold)
    if len(first) == 0:
        return None, 'never'
    index = int(first[0])
    if index < start:
        return index, 'before'
    return index, 'inside' if index <= decisive else 'late'


def read_evidence(annotated, docs, predictions):
    scored = []
    for row in annotated:
        key = row['id']
        p = predictions[key]
        docs.verify_offsets(key, p['offsets'])
        offsets = docs.encodings[key].offsets
        n = len(offsets)
        start = next(i for i, (_, end) in enumerate(offsets) if end > row['article_start'])
        require(p['risk_start'] == start, 'Evidence risk start differs from displayed text')
        require(set(p['hazard']) == {q['id'] for q in row['queries']}, 'Evidence forecast question IDs differ')
        log_survival = {name: survival(probabilities(value, n, name), start) for name, value in p['hazard'].items()}
        by_query = {}
        for query in row['queries']:
            name = query['id']
            require(docs.end(key, query['anchor_span'][1]) - 1 < start, 'Question is not fully visible before passage')
            span = query['evidence_window']
            if span is None:
                by_query[name] = {'nll': float(-log_survival[name][-1]), 'right_censored': True}
            else:
                begin, end = docs.window(key, span)
                by_query[name] = {'nll': -log_interval(log_survival[name], begin, end, start), 'right_censored': False,
                                  'premature_probability': float(-np.expm1(log_survival[name][begin - start]))}
        sample = {'group_id': row['group_id'], 'questions': by_query,
                  'nll': float(np.mean([item['nll'] for item in by_query.values()]))}
        early, late = row['earlier_query'], row['later_query']
        span = next(q['evidence_window'] for q in row['queries'] if q['id'] == early)
        if span is not None:
            begin, end = docs.window(key, span)
            early_mass = conditional_window(log_survival[early], begin, end, start)
            late_mass = conditional_window(log_survival[late], begin, end, start)
            ranking = 0.5 if abs(early_mass - late_mass) < 1e-9 else float(early_mass > late_mass)
            curve_gap = np.exp(log_survival[late][1:]) - np.exp(log_survival[early][1:])
            initial, status = first_separation(curve_gap, begin - start, end - start - 1)
            sample.update({'earlier_window_conditional_advantage': early_mass - late_mass, 'pair_ranking': ranking,
                           'earliest_nontrivial_advantage_token': initial + start if initial is not None else None,
                           'earliest_nontrivial_advantage_window': status})
            sham = row['earlier_sham']
            if sham is not None:
                a, b = docs.window(key, sham)
                sample['earlier_window_advantage_over_sham'] = (
                    early_mass - conditional_window(log_survival[early], a, b, start))
        scored.append(sample)
    return scored


def anchor_curve(key, source_span, docs, prediction):
    docs.verify_offsets(key, prediction['offsets'])
    n = len(prediction['offsets'])
    anchor = docs.end(key, source_span[1]) - 1
    require(prediction['anchor'] == anchor, 'Scored anchor differs from the same predeclared position')
    opening = probabilities(prediction['opening'], n, key)
    remaining = probabilities(prediction['anchor_unresolved'], n, key)
    hits = probabilities(prediction['anchor_resolved'], n, key)
    require(abs(opening[anchor] - remaining[anchor]) < 2e-6 and opening[anchor] > 0
            and np.all(remaining[:anchor] == 0.0) and np.all(hits[:anchor + 1] == 0.0),
            'Anchor has mass before becoming visible')
    require(np.allclose(remaining[anchor:] + np.cumsum(hits)[anchor:], opening[anchor], atol=3e-6, rtol=2e-4),
            'Anchor mass does not conserve its opening probability')
    closed = np.cumsum(hits) / opening[anchor]
    load = np.asarray(prediction['total_unresolved'], dtype=np.float64)
    require(load.shape == (n,) and np.isfinite(load).all() and np.all(load >= 0), 'Invalid unresolved candidate totals')
    return {'opening': opening, 'remaining': remaining, 'closed': closed, 'load': load}


def prefix_difference(a, b, length):
    return max(float(np.max(np.abs(a[name][:length] - b[name][:length]))) for name in a)


def read_controls(annotated, docs, predictions):
    scored = []
    for row in annotated:
        variants = row['variants']
        ids = {name: value['id'] for name, value in variants.items()}
        require(set(ids) == {'early', 'late', 'none'} and variants['none']['cue_span'] is None, 'Incomplete control')
        curves = {name: anchor_curve(key, row['anchor_span'], docs, predictions[key]) for name, key in ids.items()}
        shared = docs.end(ids['none'], row['common_prefix_end'])
        middle = docs.end(ids['none'], row['late_none_prefix_end'])
        require(all(len(curve['closed']) == variants['none']['tokens'] for curve in curves.values()),
                'Controls cannot compare across unequal token budgets')
        require(0 < shared < middle <= variants['none']['tokens'], 'Invalid no-answer transition limits')
        identical = max(prefix_difference(curves['early'], curves['none'], shared),
                        prefix_difference(curves['late'], curves['none'], middle))
        mass_before = float(curves['none']['remaining'][shared - 1])
        mass_after = float(curves['none']['remaining'][middle - 1])
        retention = mass_after / mass_before if mass_before > 0 else None
        sample = {'group_id': row['group_id'], 'maximum_equal_prefix_disagreement': identical,
                  'no_answer_retention': float(1 - curves['none']['closed'][-1]),
                  'no_answer_transition_retention': {
                      'prefix_end_token': shared - 1, 'transition_end_token': middle - 1,
                      'mass_before': mass_before, 'mass_after': mass_after, 'survival_ratio': retention,
                      'status': 'defined' if retention is not None else 'no_remaining_mass_at_transition_start',
                  },
                  'resolved_probability': {name: curve['closed'].tolist() for name, curve in curves.items()}}
        for name, rival in (('early', 'late'), ('late', 'none')):
            variant = variants[name]
            cue_start, cue_end = docs.window(ids[name], variant['cue_span'])
            decisive = docs.end(ids[name], variant['decisive_end']) - 1
            require(cue_start <= decisive < cue_end and variant['decisive_tokens'] == decisive + 1,
                    'Decisive control offset differs from the sealed position')
            gap = curves[name]['closed'] - curves[rival]['closed']
            first, status = first_separation(gap, cue_start, decisive)
            sample[name] = {'rival': rival, 'cue_token_start': cue_start, 'decisive_token': decisive,
                            'first_separation_token': first, 'first_separation_timing': status,
                            'advantage_at_decisive_token': float(gap[decisive]),
                            'advantage_at_end_against_no_answer': float(
                                curves[name]['closed'][-1] - curves['none']['closed'][-1])}
        scored.append(sample)
    return scored


def mean(rows, key):
    return float(np.mean([value[key] for value in rows]))


def interval(values):
    a = np.asarray(values, dtype=np.float64)
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(len(a), size=(BOOTSTRAP_REPETITIONS, len(a)))
    lower, upper = np.quantile(a[draws].mean(1), [0.025, 0.975])
    return [float(lower), float(upper)]


def paired_comparison(candidate, reference, name):
    lookup = {row['group_id']: row for row in reference}
    require(len(lookup) == len(reference) == len(candidate)
            and {row['group_id'] for row in candidate} == lookup.keys(), 'Paired score families differ')
    difference = [row[name] - lookup[row['group_id']][name] for row in candidate]
    return {'difference': float(np.mean(difference)), 'group_bootstrap_95_interval': interval(difference)}


def frick_summary(rows, reference):
    reliability = {}
    for key in ('target_probability_delta_error', 'premature_error_01', 'premature_error_02'):
        values = [row[key] for row in rows]
        error = float(np.mean(values))
        limits = interval(values)
        reliability[key] = {'error': error, 'group_bootstrap_95_interval': limits,
                            'within_point_margin': abs(error) <= FRICK_MARGIN,
                            'within_interval_margin': -FRICK_MARGIN <= limits[0] and limits[1] <= FRICK_MARGIN}
    nll = paired_comparison(rows, reference, 'path_nll')
    total = sum(row['trials'] for group in rows for row in group['conditions'].values())
    pooled = {}
    for condition in ('01', '02'):
        group = [row['conditions'][condition] for row in rows]
        weights = [row['trials'] for row in group]
        risk_weights = [row['target_at_risk'] for row in group]
        pooled[condition] = {'trials': sum(weights), 'target_at_risk': sum(risk_weights)}
        for field in ('path_nll', 'human_target_probability', 'model_target_probability',
                      'human_premature_probability', 'model_premature_probability'):
            pooled[condition][field] = float(np.average([row[field] for row in group], weights=weights))
        for field in ('human_target_hazard', 'model_target_hazard'):
            pooled[condition][field] = float(np.average([row[field] for row in group], weights=risk_weights))
    pooled_nll = sum(row['path_nll'] * row['trials'] for row in pooled.values()) / total
    return {'story_families': len(rows), 'trials': total, 'mean_equal_family_path_nll': mean(rows, 'path_nll'),
            'mean_trial_weighted_path_nll': pooled_nll, 'trial_weighted_conditions': pooled,
            'path_nll_difference_from_position': nll,
            'human_target_first_report_difference': mean(rows, 'human_target_probability_delta'),
            'model_target_first_report_difference': mean(rows, 'model_target_probability_delta'),
            'human_target_hazard_difference': mean(rows, 'human_target_hazard_delta'),
            'model_target_hazard_difference': mean(rows, 'model_target_hazard_delta'),
            'calibration': reliability,
            'point_gate': bool(nll['difference'] < 0 and mean(rows, 'model_target_probability_delta') > 0
                               and all(row['within_point_margin'] for row in reliability.values()))}


def evidence_summary(rows, reference):
    pair = [row for row in rows if 'pair_ranking' in row]
    sham = [row for row in pair if 'earlier_window_advantage_over_sham' in row]
    nll = paired_comparison(rows, reference, 'nll')
    positions = {kind: sum(row['earliest_nontrivial_advantage_window'] == kind for row in pair)
                 for kind in ('before', 'inside', 'late', 'never')}
    return {'articles': len(rows), 'right_censored_questions': sum(
        question['right_censored'] for row in rows for question in row['questions'].values()),
        'nll': mean(rows, 'nll'), 'nll_difference_from_position': nll, 'pairs': len(pair),
        'pair_ranking_accuracy': mean(pair, 'pair_ranking'), 'first_separation_window_counts': positions,
        'window_advantage': mean(pair, 'earlier_window_conditional_advantage'), 'sham_pairs': len(sham),
        'window_advantage_over_sham': mean(sham, 'earlier_window_advantage_over_sham') if sham else None,
        'point_gate': bool(nll['difference'] < 0 and mean(pair, 'pair_ranking') > 0.5)}


def control_summary(rows, reference):
    prefix_max = max(row['maximum_equal_prefix_disagreement'] for row in rows)
    valid = {side: sum(row[side]['first_separation_timing'] == 'inside' for row in rows)
             for side in ('early', 'late')}
    premature = sum(row[side]['first_separation_timing'] == 'before' for side in ('early', 'late') for row in rows)
    jointly_located = sum(row['early']['first_separation_timing'] == row['late']['first_separation_timing'] == 'inside'
                          for row in rows)
    retention = paired_comparison(rows, reference, 'no_answer_retention')
    transition_ratios = [entry['survival_ratio'] for entry in (row['no_answer_transition_retention'] for row in rows)
                         if entry['survival_ratio'] is not None]
    transition = {'reference_threshold': TRANSITION_REFERENCE_RETENTION, 'families_with_defined_ratio': len(transition_ratios),
                  'families_without_initial_mass': len(rows) - len(transition_ratios),
                  'families_at_or_above_reference': sum(value >= TRANSITION_REFERENCE_RETENTION
                                                       for value in transition_ratios),
                  'mean_ratio_when_defined': float(np.mean(transition_ratios)) if transition_ratios else None}
    timing = {}
    for side in ('early', 'late'):
        main = [{'group_id': row['group_id'], **row[side]} for row in rows]
        position = [{'group_id': row['group_id'], **row[side]} for row in reference]
        timing[side] = {'first_separation_window_counts': {
            kind: sum(row['first_separation_timing'] == kind for row in main)
            for kind in ('before', 'inside', 'late', 'never')},
            'mean_advantage_at_decisive_token': mean(main, 'advantage_at_decisive_token'),
            'advantage_at_decisive_token_difference_from_position': paired_comparison(
                main, position, 'advantage_at_decisive_token')}
    # At least 3/4 of the authored families must pass both windows; fail on premature positives.
    target = (3 * len(rows) + 3) // 4
    return {'engineering_families': len(rows), 'maximum_equal_prefix_disagreement': prefix_max,
            'prefix_within_numeric_tolerance': prefix_max <= PREFIX_ATOL, 'cue_windows_hit': valid,
            'both_cues_hit': jointly_located, 'premature_windows': premature,
            'joint_cue_threshold': target, 'no_answer_retention': mean(rows, 'no_answer_retention'),
            'no_answer_retention_difference_from_position': retention,
            'no_answer_transition_retention': transition,
            'cue_timing': timing,
            'point_gate': bool(prefix_max <= PREFIX_ATOL and jointly_located >= target and premature == 0)}


def lora_comparison(lora, frozen):
    """Compare the two ready arms on the identical annotated families."""
    lora_frick = lora['frick']
    frozen_frick = frozen['frick']
    absolute_errors = [
        [{'group_id': row['group_id'], 'absolute_target_error': abs(row['target_probability_delta_error'])}
         for row in arm]
        for arm in (lora_frick, frozen_frick)
    ]
    evidence_ranking = [
        [row for row in arm['evidence'] if 'pair_ranking' in row]
        for arm in (lora, frozen)
    ]
    controls = {'no_answer_retention': paired_comparison(lora['controls'], frozen['controls'], 'no_answer_retention')}
    for side in ('early', 'late'):
        variants = [[{'group_id': row['group_id'], **row[side]} for row in arm['controls']]
                    for arm in (lora, frozen)]
        controls[f'{side}_cue_advantage_at_decisive_token'] = paired_comparison(
            *variants, 'advantage_at_decisive_token')
    return {'frick': {'path_nll': paired_comparison(lora_frick, frozen_frick, 'path_nll'),
                      'absolute_target_calibration_error': paired_comparison(*absolute_errors, 'absolute_target_error')},
            'evidence': {'interval_nll': paired_comparison(lora['evidence'], frozen['evidence'], 'nll'),
                         'same_window_ranking_accuracy': paired_comparison(*evidence_ranking, 'pair_ranking')},
            'controls': controls}


def score(prepared, tokenizer, predictions, protocol_sha256, output):
    require(not output.exists(), f'Scores already exist at {output}')
    design, docs, receipt = load_prepared(prepared, tokenizer)
    expected = {kind: {row['id'] for row in design[kind] if row['split'] == 'test'}
                for kind in ('frick', 'evidence')}
    expected['controls'] = {variant['id'] for row in design['controls'] for variant in row['variants'].values()}
    modes, seal = load_forecasts(predictions, protocol_sha256, expected)
    require('fitting.json' in seal['files_sha256'], 'Missing sealed training result')
    fitting = read_json(predictions / 'fitting.json')
    fitted = {arm['mode']: arm for arm in fitting['fits']}
    report = {}
    for mode, forecast in modes.items():
        per_group = {
            'frick': read_frick([row for row in design['frick'] if row['split'] == 'test'],
                                np.asarray([value['hazard'] for value in design['frick_position_baseline']['positions']]),
                                mode == 'position', forecast['frick']),
            'evidence': read_evidence([row for row in design['evidence'] if row['split'] == 'test'], docs,
                                      forecast['evidence']),
            'controls': read_controls(design['controls'], docs, forecast['controls']),
        }
        reference = report['position']['per_group'] if mode != 'position' else per_group
        report[mode] = {'frick': frick_summary(per_group['frick'], reference['frick']),
                        'evidence': evidence_summary(per_group['evidence'], reference['evidence']),
                        'controls': control_summary(per_group['controls'], reference['controls']),
                        'per_group': per_group}
        if mode != 'position':
            require(mode in fitted and 'per_task_training_plateau' in fitted[mode], 'Missing trained arm record')
            report[mode]['training'] = fitted[mode]
            report[mode]['joint_point_gate'] = (fitted[mode]['per_task_training_plateau'] is True
                                                  and all(report[mode][kind]['point_gate'] for kind in ASSAYS))
    paired = lora_comparison(report['lora']['per_group'], report['frozen']['per_group']) if 'lora' in report else None
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, {'schema_version': 1, 'protocol_sha256': protocol_sha256,
                        'forecast_seal_sha256': digest(predictions / 'seal.json'), 'files_sha256': seal['files_sha256'],
                        'prepared_receipt_sha256': digest(prepared / 'receipt.json'),
                        'prepared_file_sha256': receipt['file_sha256'], 'arms': report, 'fitting': fitting,
                        'lora_minus_frozen': paired,
                        'definition': 'Text and forecasts sealed before joining local annotations; historical development families '
                                      'for Frick and ExpMRC, engineering-only for controls. NLL uses finite numeric floor.',
                        'predeclared': {'probability_floor': PROBABILITY_FLOOR, 'control_threshold': CONTROL_DIFF,
                                        'prefix_tolerance': PREFIX_ATOL, 'frick_margin': FRICK_MARGIN}})


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', default=Path('.cache/cogllm/token-readout/prepared'), type=Path)
    parser.add_argument('--tokenizer', default=TOKENIZER, type=Path)
    parser.add_argument('--predictions', required=True, type=Path)
    parser.add_argument('--protocol-sha256', required=True, type=checked_digest)
    parser.add_argument('--output', required=True, type=Path)
    score(**vars(parser.parse_args()))
