"""Hold out whole article groups containing strict same-document evidence swaps."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from research.core_features.readback_score import paired_material_interval
from research.semantic_tracking import iirc_linked_colab_worker as worker
from research.semantic_tracking.iirc_linked_extract import shard_key
from research.semantic_tracking.iirc_linked_fit import expected_projection
from research.semantic_tracking.iirc_linked_pack import METHODS, validate_split, write_archive
from research.semantic_tracking.iirc_prepare import digest
from research.semantic_tracking.joint_score import interval_nll, probability_logits, rows_by_key
from research.stage2.records import write_jsonl
from research.stage3.probe import _jsonl
from research.stage3.run import save_json


PROTOCOL = Path(__file__).resolve().parents[1] / 'iirc-identity-holdout.md'
EXPECTED = {'fit_paths': 3240, 'holdout_paths': 114, 'pairs': 37, 'paired_paths': 72, 'paired_groups': 33}


def read_json(path):
    return json.loads(path.read_bytes())


def verify_pair(case_a, case_b, labels, stims, queries):
    left, right = (labels[key] for key in (case_a, case_b))
    if (left['article_cluster'] != right['article_cluster'] or left['seed_family'] == right['seed_family']
            or left['linked_text_sha256'] != right['linked_text_sha256']
            or left['upper'] is None or right['upper'] is None or left['upper'] == right['upper']
            or max(left['evidence_start'], right['evidence_start']) < min(left['evidence_end'], right['evidence_end'])):
        raise ValueError('Expected non-overlapping located evidence in the same document from distinct seed families')
    first, second = (stims[key] for key in (case_a, case_b))
    a, b = (queries[key]['linked_start'] for key in (case_a, case_b))
    if (first['text'][a:] != second['text'][b:]
            or [s - a for s in first['stops'][1:]] != [s - b for s in second['stops'][1:]]):
        raise ValueError('Paired model paths must show the same linked text at each corresponding stop')


def prepare(original, audit_path, output):
    audit = read_json(audit_path)
    train = original / 'train'
    files = ('stimuli.jsonl', 'queries.jsonl', 'locked-labels.jsonl', 'locked-matched-swaps.jsonl')
    if any(digest(train / name) != audit['files_sha256'][f'train/{name}'] for name in files):
        raise ValueError('Use the immutable original training materials and strict-swap pairs')
    stims = rows_by_key(train / 'stimuli.jsonl', lambda row: row['id'])
    queries = rows_by_key(train / 'queries.jsonl', lambda row: row['case_id'])
    labels = rows_by_key(train / 'locked-labels.jsonl', lambda row: row['case_id'])
    if stims.keys() != queries.keys() or stims.keys() != labels.keys():
        raise ValueError('Original training case IDs do not match')
    pairs = list(_jsonl(train / 'locked-matched-swaps.jsonl'))
    seen = set()
    for pair in pairs:
        a, b = pair['case_ids']
        key = frozenset((a, b))
        if a == b or key in seen or pair['article_cluster'] != labels[a]['article_cluster']:
            raise ValueError('Unexpected or duplicate paired case IDs or cluster')
        verify_pair(a, b, labels, stims, queries)
        seen.add(key)
    held_clusters = {pair['article_cluster'] for pair in pairs}
    ordered = {
        'train': [k for k in stims if labels[k]['article_cluster'] not in held_clusters],
        'dev': [k for k in stims if labels[k]['article_cluster'] in held_clusters],
    }
    keys = ('seed_family', 'linked_family', 'seed_text_sha256', 'linked_text_sha256')
    titles_and_texts = [{labels[case][name] for case in ids for name in keys} for ids in ordered.values()]
    counts = {'fit_paths': len(ordered['train']), 'holdout_paths': len(ordered['dev']), 'pairs': len(pairs),
              'paired_paths': len(set().union(*seen)), 'paired_groups': len(held_clusters)}
    if counts != EXPECTED or titles_and_texts[0] & titles_and_texts[1]:
        raise ValueError('Matched split differs from the prespecified article-separated inventory')
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.iirc-identity-') as temporary:
        stage = Path(temporary) / 'assay'
        tables = {}
        for split, ids in ordered.items():
            (stage / split).mkdir(parents=True, mode=0o700)
            tables[f'{split}/stimuli.jsonl'] = [stims[k] for k in ids]
            tables[f'{split}/queries.jsonl'] = [queries[k] for k in ids]
        fit_keys = ('case_id', 'lower', 'upper')
        tables['train/intervals.jsonl'] = [{key: labels[k][key] for key in fit_keys} for k in ordered['train']]
        protected = {'dev/locked-labels.jsonl': [labels[k] for k in ordered['dev']],
                     'dev/locked-matched-swaps.jsonl': pairs}
        for name, rows in {**tables, **protected}.items():
            write_jsonl(stage / name, rows)
        for split, ids in ordered.items():
            validate_split(stage, split, len(ids))
        assay = {'schema_version': 1, 'model': audit['model'], 'revision': audit['revision'],
                 'method_fields': list(METHODS), 'cases': {k: len(v) for k, v in ordered.items()}, 'counts': counts,
                 'protocol_sha256': digest(PROTOCOL), 'code_sha256': digest(Path(__file__)),
                 'original_audit_sha256': digest(audit_path),
                 'original_stimuli_sha256': {split: audit['files_sha256'][f'{split}/stimuli.jsonl'] for split in ordered},
                 'held_out_not_in_archive': {k: digest(stage / k) for k in protected},
                 'inputs_sha256': {k: digest(stage / k) for k in tables}}
        save_json(stage / 'assay.json', assay)
        write_archive(stage / 'fit-inputs.tar.gz', {k: stage / k for k in (*tables, 'assay.json')})
        stage.rename(output)
    return assay


def replay(prepared, job, output):
    assay_path = prepared / 'assay.json'
    assay = read_json(assay_path)
    manifest = read_json(job / 'manifest.json')
    checkpoints = job / 'checkpoints'
    receipt_path = checkpoints / 'receipt.json'
    receipt = read_json(receipt_path)
    worker.check_proofs(checkpoints, receipt, manifest['data_members_sha256']['assay.json'],
                        manifest['payload_members_sha256'], manifest['data_members_sha256'])
    if (len(receipt['shards']) != receipt['shards_total'] or receipt['run_id'] != job.name
            or {split: manifest['data_members_sha256'][f'{split}/stimuli.jsonl'] for split in ('train', 'dev')}
            != assay['original_stimuli_sha256']):
        raise ValueError('Require the complete sealed extraction of the original paths')
    keys, positions = [], []
    for split in ('train', 'dev'):
        path = prepared / split / 'stimuli.jsonl'
        if digest(path) != assay['inputs_sha256'][f'{split}/stimuli.jsonl']:
            raise ValueError('Identity inputs changed after preparation')
        for row in _jsonl(path):
            keys.extend([row['id']] * len(row['stops']))
            positions.extend(row['stops'])
    index = {key: i for i, key in enumerate(zip(keys, positions, strict=True))}
    if len(index) != len(keys):
        raise ValueError('Matched split contains repeated source stops')
    vectors = np.empty((len(keys), 128), dtype=np.float32)
    remaining = set(index)
    for item in receipt['shards']:
        path = checkpoints / worker.shard_name(item['number'])
        worker.check_shard(path, item)
        with np.load(path, allow_pickle=False) as archive:
            ids, stops, states = (archive[name] for name in ('ids', 'stops', 'states'))
            if (archive['signature'].item() != shard_key(receipt['plan_sha256'], item['number'], ids, stops)
                    or states.shape != (len(ids), 128) or states.dtype != np.float32
                    or not np.allclose(np.linalg.norm(states, axis=1), 1, rtol=0, atol=0.00005)):
                raise ValueError('Checkpoint signature, projection dimensions or state norms changed')
            for i, key in enumerate(zip(ids.tolist(), stops.tolist(), strict=True)):
                if key in index:
                    remaining.remove(key)
                    vectors[index[key]] = states[i]
    if remaining:
        raise ValueError('Checkpoint set omitted planned matched-assay coordinates')
    output.mkdir(parents=True, exist_ok=False)
    features = output / 'features.npz'
    np.savez(features, ids=np.asarray(keys), stops=np.asarray(positions), states=vectors, projection=expected_projection())
    save_json(output / 'extraction.json', {'assay_sha256': digest(assay_path), 'features_sha256': digest(features),
              'model': assay['model'], 'revision': assay['revision'], 'source_checkpoint_sha256': digest(receipt_path),
              'inputs_sha256': {'assay': digest(assay_path), **{f'{s}/stimuli.jsonl': assay['inputs_sha256'][f'{s}/stimuli.jsonl']
                                                             for s in ('train', 'dev')}}})


def score(prepared, fit, output):
    assay_path = prepared / 'assay.json'
    assay = read_json(assay_path)
    seal_path = fit / 'fit-manifest.json'
    seal = read_json(seal_path)
    predictions = fit / 'predictions.jsonl'
    if (assay['protocol_sha256'] != digest(PROTOCOL) or assay['code_sha256'] != digest(Path(__file__))
            or seal['assay_sha256'] != digest(assay_path) or seal['predictions_sha256'] != digest(predictions)
            or seal['heads_sha256'] != digest(fit / 'heads.npz') or seal['inputs_sha256'] != assay['inputs_sha256']
            or (seal['model'], seal['revision']) != (assay['model'], assay['revision'])
            or seal['fit_paths'] != EXPECTED['fit_paths'] or seal['heldout_forecast_paths'] != EXPECTED['holdout_paths']):
        raise ValueError('Expected the predictions and refitted heads from the matched holdout seal')
    held_inputs = {name: expected for name, expected in assay['inputs_sha256'].items() if name.startswith('dev/')}
    if any(digest(prepared / name) != expected for name, expected in
           {**assay['held_out_not_in_archive'], **held_inputs}.items()):
        raise ValueError('The original matched holdout inputs or labels changed')
    pairs = list(_jsonl(prepared / 'dev/locked-matched-swaps.jsonl'))
    labels = rows_by_key(prepared / 'dev/locked-labels.jsonl', lambda row: row['case_id'])
    forecasts = rows_by_key(predictions, lambda row: row['case_id'])
    if forecasts.keys() != labels.keys() or len(pairs) != EXPECTED['pairs']:
        raise ValueError('Forecasts or pair counts do not match the planned holdout')
    stops = validate_split(prepared, 'dev', EXPECTED['holdout_paths'])
    logits = {}
    for key, forecast in forecasts.items():
        if set(forecast) != {'case_id', 'methods'} or set(forecast['methods']) != set(METHODS):
            raise ValueError('Unexpected matched forecast fields')
        logits[key] = {method: probability_logits(p, (stops[key],)) for method, p in forecast['methods'].items()}
    margins = defaultdict(list)
    for pair in pairs:
        case_a, case_b = pair['case_ids']
        for method in METHODS:
            by_case = []
            for key, other in ((case_a, case_b), (case_b, case_a)):
                p = logits[key][method]
                original, exchange = labels[key], labels[other]
                by_case.append(interval_nll(p, exchange['lower'], exchange['upper']) -
                               interval_nll(p, original['lower'], original['upper']))
            margins[method].append(by_case)
    values = {name: np.asarray(rows).mean(axis=1) for name, rows in margins.items()}
    groups = [{'group_id': p['article_cluster']} for p in pairs]
    summary = {name: paired_material_interval(rows, groups)
               for name, rows in {**values, 'anchor_minus_path_text': values['anchor_conditioned'] - values['path_text']}.items()}
    accepted = all(summary[k]['descriptive_paired_material_95_ci'][0] > 0
                   for k in ('anchor_conditioned', 'anchor_minus_path_text'))
    save_json(output, {'counts': EXPECTED, 'fit_seal_sha256': digest(seal_path), 'assay_sha256': digest(assay_path),
                       'swap_minus_own_interval_nll_nats': summary, 'acceptance': accepted,
                       'both_paths_favor_own_interval': {k: int(np.count_nonzero(np.all(np.asarray(v) > 0, axis=1)))
                                                         for k, v in margins.items()}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    for name, function, fields in (('prepare', prepare, ('original', 'audit_path', 'output')),
                                   ('replay', replay, ('prepared', 'job', 'output')),
                                   ('score', score, ('prepared', 'fit', 'output'))):
        action = actions.add_parser(name)
        action.set_defaults(function=function)
        for field in fields:
            action.add_argument(f'--{field.replace("_", "-")}', type=Path, required=True)
    args = vars(parser.parse_args())
    function = args.pop('function')
    args.pop('action')
    function(**args)


if __name__ == '__main__':
    main()
