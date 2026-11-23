"""Repeat the same assay with equal scaling and penalties on shared feature blocks."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from research.backbone_transfer.extract import digest, stream_rows, write_json, write_rows
from research.semantic_tracking.load_curve_fit import cluster_interval, projection
from research.semantic_tracking.same_question_fit import (
    PROTOCOL, Features, authored_windows, end_token, evaluate, evidence_windows, family_scores, feature_sets, train,
)


def nested_features(features):
    local = features.x['local']
    overlap = features.x['lexical'][:, 2:]
    return Features(features.paths, {**features.x, 'local_lexical': np.column_stack((local, overlap))})


def score(training, testing, width):
    blocks = {'steps': (2,), 'local_lexical': (2, width, 2),
              'question': (2, width, 2, 2, width, width)}
    summaries, paths = {}, {}
    for method, sizes in blocks.items():
        fitted = train(training, method, block_widths=sizes)
        paths[method] = evaluate(testing, method, fitted)
        summaries[method] = family_scores(paths[method])
    groups = sorted(summaries['steps'])
    output = {}
    for method in blocks:
        value = {}
        for metric in ('nll', 'matched_minus_unmatched_drop', 'matched_order_score'):
            raw = np.array([summaries[method][g][metric] for g in groups])
            weight = np.ones(len(groups))
            value[metric] = cluster_interval(raw, groups, weight)
            for baseline in ('steps', 'local_lexical'):
                ref = np.array([summaries[baseline][g][metric] for g in groups])
                value[f'{metric}_minus_{baseline}'] = cluster_interval(raw - ref, groups, weight)
        sep = [row for group in summaries[method].values() for row in group['first_5pp_separation']]
        value['five_point_separations'] = {'in_cue': sum(row['in_cue'] for row in sep),
                                            'early': sum(row['token'] is not None and not row['in_cue'] for row in sep),
                                            'total': len(sep)}
        output[method] = value
    return output, paths


def exact_training_windows(paths, markup, docs):
    boundaries = {row['id']: row['first_sufficient_end'] for row in stream_rows(markup)}
    if boundaries.keys() != {row.doc for row in paths if row.split == 'train'}:
        raise ValueError('Token-timing supervision must cover only the previously fixed training set')
    adjusted = []
    for p in paths:
        if p.split != 'train':
            adjusted.append(p)
            continue
        boundary = boundaries[p.doc]
        if boundary is None:
            continue
        target = end_token(docs[p.doc], boundary) - 1
        if not p.left <= target < p.end:
            raise ValueError('An authored exact target must be inside its earlier registered cue window')
        adjusted.append(replace(p, left=target, end=target + 1) if p.hit else p)
    return adjusted


def run(job, authored, evidence, output, train_timing=None):
    docs, seal = projection(job, protocol=PROTOCOL)
    a, _ = authored_windows(list(stream_rows(authored)), docs)
    e = evidence_windows(list(stream_rows(evidence)), docs)
    output.mkdir()
    results = {}
    studies = (('authored', a), ('evidence', e))
    if train_timing is not None:
        studies = (('authored', exact_training_windows(a, train_timing, docs)),)
    for name, records in studies:
        train_set, test_set = (nested_features(feature_sets([row for row in records if row.split == split], docs))
                               for split in ('train', 'test'))
        width = int(seal['shape'][1])
        results[name], scores = score(train_set, test_set, width)
        write_rows(output / f'{name}-paths.jsonl', (
            {**row, 'method': method} for method, paths in scores.items() for row in paths))
    write_json(output / 'metrics.json', {'scope': 'Post hoc block-scaling diagnostic; same frozen cohorts and objective.',
                                        'trained_with_authored_exact_tokens': train_timing is not None,
                                        'train_timing_sha256': digest(train_timing) if train_timing is not None else None,
                                        'authored_pairs_sha256': digest(authored), 'evidence_pairs_sha256': digest(evidence),
                                        'results': results})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('job', 'authored', 'evidence', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--train-timing', type=Path)
    run(**vars(parser.parse_args()))
