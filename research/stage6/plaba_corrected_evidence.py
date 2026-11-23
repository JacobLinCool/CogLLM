"""Score sealed PLABA out-of-fold readouts with the corrected evidence inventory."""

import argparse
import json
from pathlib import Path

import numpy as np

from research.semantic_tracking.iirc_prepare import digest
from research.stage3.causal_ordinal import reader_components
from research.stage3.probe import _jsonl, _load_plaba
from research.stage3.run import save_json


METHODS = ('candidate_evidence', 'length_evidence', 'candidate_article')
CANDIDATE = 'qwen_and_rewrite_effect'
DRAWS, BLOCK, SEED = 10_000, 64, 43
GRADES = np.arange(1, 6)
PROTOCOL = Path(__file__).with_name('plaba-corrected-evidence.md')


def fixed_protocol(data, ordinal):
    manifest = json.loads((ordinal / 'manifest.json').read_bytes())
    stages = manifest['data']['current_stage2_sha256']
    inputs = {name: digest(data / name) for name in ('splits.json', 'stimuli.jsonl', 'trials.jsonl')}
    forecast = digest(ordinal / 'oof-predictions.jsonl')
    if (stages != inputs or manifest['oof_predictions_sha256'] != forecast
            or manifest['ratings'] != GRADES.tolist() or manifest['target_order'][1] != 'perceived_understanding'
            or manifest['all_data_head_not_scored_out_of_fold'] is not True):
        raise ValueError('Expected the original five-fold ordinal forecasts and corrected evidence inputs')
    return {'protocol_sha256': digest(PROTOCOL), 'scorer_sha256': digest(Path(__file__)),
            'stage2_sha256': inputs, 'oof_predictions_sha256': forecast,
            'ordinal_manifest_sha256': digest(ordinal / 'manifest.json'), 'methods': list(METHODS),
            'ordinal_method': CANDIDATE, 'folds': 5, 'bootstrap_groups': 25, 'bootstrap_draws': DRAWS,
            'bootstrap_seed': SEED, 'primary': 'AUROC', 'secondary': 'average_precision'}


def scored_trials(data, ordinal):
    stimuli, inventory, _, fold_of, _ = _load_plaba(data)
    predictions = {}
    for row in _jsonl(ordinal / 'oof-predictions.jsonl'):
        key = row['sentence_id']
        document, _ = inventory[key]
        if key in predictions or row['fold'] != fold_of[stimuli[document]['family_id']]:
            raise ValueError('Invalid fold or duplicate ordinal prediction')
        levels = np.array([row['probability'][name][1] for name in (CANDIDATE, 'length')])
        if (levels.shape != (2, 5) or not np.isfinite(levels).all() or np.any(levels < 0)
                or not np.allclose(levels.sum(axis=1), 1, rtol=0, atol=1e-10)):
            raise ValueError('Expected valid distributions for both evidence scores')
        predictions[key] = levels @ GRADES
    if predictions.keys() != inventory.keys():
        raise ValueError('Out-of-fold sentence inventory is incomplete')
    articles = {key: np.mean([predictions[s['id']][0] for s in value['segments']])
                for key, value in stimuli.items()}
    seen = set()
    families, scores, errors = [], [], []
    observed = excluded = multi = old_mismatch = 0
    lengths = []
    for trial in _jsonl(data / 'trials.jsonl'):
        if trial['exposure']['task'] != 'open_book_mcq':
            continue
        document = trial['stimulus_id']
        records = trial['observations']
        evidence, correct = (records[name] for name in ('item_evidence_segment_ids', 'mcq_correct'))
        observed += 1
        if trial['id'] in seen or correct['status'] != 'observed' or type(correct['value']) is not bool:
            raise ValueError('Unexpected or duplicate open-book trial')
        seen.add(trial['id'])
        if evidence['status'] == 'missing':
            excluded += 1
            continue
        segments = evidence['value']
        if (evidence['status'] != 'observed' or not segments or len(segments) != len(set(segments))
                or any(inventory[key][0] != document for key in segments)):
            raise ValueError('Evidence does not refer to distinct sentences in the current document')
        candidate, length = np.mean([predictions[x] for x in segments], axis=0)
        scores.append((-candidate, -length, -articles[document]))
        errors.append(not correct['value'])
        families.append(stimuli[document]['family_id'])
        multi += len(segments) > 1
        prior = records['response_indexed_segment']
        old_mismatch += prior['status'] == 'observed' and prior['value'] not in segments
        lengths.append(len(segments))
    controls = {'open_book_trials': observed, 'scored_trials': len(scores), 'excluded_missing_evidence': excluded,
                'multi_sentence_evidence_trials': multi, 'old_index_outside_evidence_trials': old_mismatch,
                'incorrect_trials': int(np.count_nonzero(errors)), 'maximum_evidence_sentences': max(lengths)}
    return stimuli, families, np.asarray(scores), np.asarray(errors), controls


def ranking(weights, scores, errors):
    order = np.argsort(-scores, kind='stable')
    boundaries = np.r_[0, np.flatnonzero(np.diff(scores[order])) + 1]
    positive = np.add.reduceat(weights[:, order] * errors[order], boundaries, axis=1)
    negative = np.add.reduceat(weights[:, order] * ~errors[order], boundaries, axis=1)
    cumulative_p, cumulative_n = np.cumsum(positive, axis=1), np.cumsum(negative, axis=1)
    total_p, total_n = cumulative_p[:, -1], cumulative_n[:, -1]
    if np.any((total_p == 0) | (total_n == 0)):
        raise ValueError('AUROC requires examples of both response classes')
    auc = np.sum(positive * (total_n[:, None] - cumulative_n + 0.5 * negative), axis=1) / (total_p * total_n)
    denominator = cumulative_p + cumulative_n
    precision = np.divide(cumulative_p, denominator, out=np.zeros_like(denominator, dtype=float), where=denominator > 0)
    ap = np.sum(precision * positive, axis=1) / total_p
    return np.column_stack((auc, ap))


def evaluate(scores, errors, group_count, groups):
    def paired(weights):
        return np.stack([ranking(weights, values, errors) for values in scores.T], axis=1)

    groups = np.array([row['group_id'] for row in groups])
    if set(groups) != set(range(group_count)):
        raise ValueError('The evaluated response set must cover all prespecified reader/article groups')
    point = paired(np.ones((1, len(scores)), dtype=int))[0]
    rng = np.random.default_rng(SEED)
    p = np.full(group_count, 1 / group_count)
    sampled = np.concatenate([paired(rng.multinomial(group_count, p, min(BLOCK, DRAWS - i))[:, groups])
                              for i in range(0, DRAWS, BLOCK)])

    def interval(index, other=None):
        center, draw = point[index], sampled[:, index]
        if other is not None:
            center, draw = center - point[other], draw - sampled[:, other]
        ci = np.quantile(draw, (0.025, 0.975), axis=0).T
        return {metric: {'estimate': float(value), 'ci95': bounds.tolist()}
                for metric, value, bounds in zip(('AUROC', 'average_precision'), center, ci, strict=True)}

    methods = {name: interval(i) for i, name in enumerate(METHODS)}
    contrasts = {f'{METHODS[0]}_minus_{name}': interval(0, i) for i, name in enumerate(METHODS[1:], 1)}
    return {'methods': methods, 'contrasts': contrasts,
            'passes_both_auroc_comparisons': all(s['AUROC']['ci95'][0] > 0 for s in contrasts.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    for name in ('lock', 'score'):
        action = actions.add_parser(name)
        for field in ('data', 'ordinal', 'output') + (('lock',) if name == 'score' else ()):
            action.add_argument(f'--{field}', type=Path, required=True)
    args = parser.parse_args()
    spec = fixed_protocol(args.data, args.ordinal)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.action == 'lock':
        save_json(args.output, spec)
        return
    if json.loads(args.lock.read_bytes()) != spec:
        raise ValueError('Scoring code or source bytes differ from the registered comparison')
    stimuli, families, scores, errors, controls = scored_trials(args.data, args.ordinal)
    count, groups = reader_components(stimuli, args.data / 'trials.jsonl', families)
    if count != spec['bootstrap_groups'] or controls['scored_trials'] != 932:
        raise ValueError('Corrected PLABA scored inventory differs from Stage 2')
    save_json(args.output, {'protocol_sha256': digest(args.lock), 'coverage': controls,
                            **evaluate(scores, errors, count, groups)})


if __name__ == '__main__':
    main()
