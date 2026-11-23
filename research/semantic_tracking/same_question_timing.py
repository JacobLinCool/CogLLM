"""Score prespecified decisive-token ranges on authored held-out traces."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from research.backbone_transfer.extract import digest, stream_rows, write_json
from research.semantic_tracking.load_curve_fit import cluster_interval
from research.semantic_tracking.same_question_fit import SEPARATION


def decisive_score(paths, markup, coordinates):
    offsets = {row['id']: row['offsets'] for row in stream_rows(coordinates)}
    anchors = {row['id']: row for row in stream_rows(markup)}
    traces = defaultdict(dict)
    for row in stream_rows(paths):
        if not row['focus']:
            continue
        pair = traces[row['method'], row['doc']]
        if row['hit'] in pair:
            raise ValueError('Duplicate question identity outcome')
        pair[row['hit']] = row
    if {key for _, key in traces} != anchors.keys() or not all(p.keys() == {True, False} for p in traces.values()):
        raise ValueError('Each held continuation must have both paths and separately authored markup')
    summaries, by_family, details = defaultdict(Counter), defaultdict(lambda: defaultdict(list)), []
    for (method, key), data in sorted(traces.items()):
        event = anchors[key]
        hit, other = data[True], data[False]
        boundary = event['first_sufficient_end']
        if boundary is None:
            summaries[method]['authored_boundary_ambiguous'] += 1
            continue
        tokens = [i for i, (_, end) in enumerate(offsets[key]) if end == boundary]
        if len(tokens) != 1 or not hit['cue_left_token'] <= tokens[0] < hit['cue_end_exclusive']:
            raise ValueError('Decisive suffix boundary differs from the frozen full-cue coordinates')
        match = np.asarray(hit['survival'])
        mismatch = np.asarray(other['survival'])
        gap = mismatch - match
        detections = np.flatnonzero(gap >= SEPARATION)
        first = int(detections[0]) + hit['initial_token'] if len(detections) else None
        difference = first - tokens[0] if first is not None else None
        category = 'no_5pp_gap' if difference is None else ('early' if difference < 0 else 'at' if difference == 0 else 'late')
        gain = np.diff(np.r_[0.0, gap])
        maximum = int(np.argmax(gain)) + hit['initial_token'] if np.max(gain) > 1e-12 else None
        relative = tokens[0] - hit['initial_token']
        before = match[relative - 1] if relative else 1.0
        exact_mass = max(before - match[relative], np.finfo(float).tiny)
        exact_nll = (-np.log(exact_mass) - np.log(mismatch[-1])) / 2
        by_family[method][hit['group']].append(float(exact_nll))
        summaries[method][category] += 1
        if maximum is not None:
            summaries[method]['max_increment_at_exact_token'] += int(maximum == tokens[0])
        details.append({'id': key, 'group': hit['group'], 'method': method,
                        'authored_decisive_token': tokens[0], 'first_5pp_separation_token': first,
                        'first_5pp_offset': difference, 'maximum_positive_gap_increment_token': maximum,
                        'matched_event_mass_at_exact_token': float(exact_mass),
                        'unmatched_survival_at_cue_end': float(mismatch[-1])})
    groups = sorted(by_family['steps'])
    weight = np.ones(len(groups))
    baseline = np.array([np.mean(by_family['steps'][group]) for group in groups])
    metrics = {}
    for method, series in by_family.items():
        if series.keys() != set(groups):
            raise ValueError('Decisive-token likelihood must compare the same independent materials')
        values = np.array([np.mean(series[group]) for group in groups])
        metrics[method] = {'nll': cluster_interval(values, groups, weight),
                           'nll_minus_steps': cluster_interval(values - baseline, groups, weight)}
    return {'question_survival_gap_threshold': SEPARATION,
            'scope': 'Author-designated sufficiency in constructed sentences, not an observed reader threshold.',
            'paths_sha256': digest(paths), 'markup_sha256': digest(markup),
            'summary': dict(summaries), 'exact_token_or_censor_nll': metrics, 'details': details}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('paths', 'markup', 'coordinates', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    write_json(args.output, decisive_score(args.paths, args.markup, args.coordinates))


if __name__ == '__main__':
    main()
