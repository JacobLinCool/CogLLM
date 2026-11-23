"""Audit later IIRC support and define a causal, normalized availability forecast."""

import argparse
from bisect import bisect_left
from collections import Counter
import json
from pathlib import Path
from statistics import median
import tarfile
import unicodedata

import numpy as np

from .iirc_prepare import digest


TOKEN_STRIDE = 16
BASELINE_FEATURES = ('intercept', 'log_elapsed_tokens', 'log_interval_tokens')


def certified_stops(encoding, anchor_end, tokenizer, stride=TOKEN_STRIDE):
    """Check exact prefix identities. Return (exclusive token count, source end)."""
    if not 0 < anchor_end < len(encoding.text) or stride < 1:
        raise ValueError('Expected an interior anchor end and positive stride')

    def certify(end):
        prefix = tokenizer.encode(encoding.text[:end], add_special_tokens=False)
        count = len(prefix.ids)
        if (not count or tuple(prefix.ids) != encoding.tokens[:count]
                or tuple(prefix.offsets) != encoding.offsets[:count]
                or max(finish for _, finish in prefix.offsets) != end):
            return None
        return count, end

    anchor = certify(anchor_end)
    if anchor is None:
        raise ValueError('Cannot certify the anchor stop')
    assigned = []
    ends = np.maximum.accumulate([end for _, end in encoding.offsets])
    full = len(encoding.tokens)
    for nominal in (*range(anchor[0] + stride, full, stride), full):
        end = int(ends[nominal - 1])
        if end <= (assigned[-1][1] if assigned else anchor_end):
            continue
        stop = certify(end)
        if stop is not None and stop[0] <= nominal:
            assigned.append(stop)
    if not assigned:
        raise ValueError('Cannot certify any stop after the anchor')
    return anchor, tuple(assigned)


def availability_path(anchor, stops, projected, weights):
    """Compute probabilities for an annotated interval or no event by the last stop."""
    anchor_token, anchor_end = anchor
    counts, ends = np.asarray(stops, dtype=np.int64).T
    widths = np.diff(np.r_[anchor_token, counts])
    if np.any(widths <= 0) or counts[-1] > len(projected) or np.any(np.diff(np.r_[anchor_end, ends]) <= 0):
        raise ValueError('Expected increasing certified stops within the supplied state array')
    form = np.column_stack((np.ones(len(counts)), np.log1p(counts - anchor_token), np.log1p(widths)))
    weights = np.asarray(weights, dtype=np.float64)
    if weights.size > len(BASELINE_FEATURES):
        current = projected[counts - 1]
        anchor_state = projected[anchor_token - 1]
        form = np.column_stack((form, current, current * anchor_state * np.sqrt(projected.shape[1])))
    if weights.shape != (form.shape[1],) or not np.isfinite(form).all() or not np.isfinite(weights).all():
        raise ValueError('Invalid geometry or anchor-conditioned model coefficients')
    logits = form @ weights
    log_survival = np.cumsum(-np.logaddexp(0.0, logits))
    log_events = np.r_[0.0, log_survival[:-1]] - np.logaddexp(0.0, -logits)
    return np.r_[log_events, log_survival[-1]]


def interval_score(anchor_end, stops, log_probabilities, support_end):
    """One proper negative log score for first full visibility, with right censoring."""
    boundaries = [end for _, end in stops]
    if support_end <= anchor_end or len(log_probabilities) != len(boundaries) + 1:
        raise ValueError('Expected a later support and a mass for each stop plus censoring')
    correct = bisect_left(boundaries, support_end)
    return float(-log_probabilities[correct])


def audit(archive, prepared, seed_audit, output):
    expected = json.loads(seed_audit.read_text(encoding='utf-8'))
    if digest(archive) != expected['author_archive']['sha256']:
        raise ValueError('Unrecognized IIRC author archive')
    results, families = {}, {}
    with tarfile.open(archive, 'r:gz') as release:
        for split in ('train', 'dev'):
            relative = f'{split}/locked-evidence.jsonl'
            labels = prepared / relative
            if digest(labels) != expected['files_sha256'][relative]:
                raise ValueError('Prepared label hash mismatch')
            locked = {row['question_id']: row for line in labels.read_text(encoding='utf-8').splitlines()
                      if (row := json.loads(line))['direction'] == 'later'}
            actual = {}
            for paragraph in json.load(release.extractfile(f'iirc/{split}.json')):
                links = {}
                for node in paragraph['links']:
                    links.setdefault(node['target'], set()).add(tuple(node['indices']))
                family = ' '.join(unicodedata.normalize('NFKC', paragraph['title'].replace('_', ' ')).casefold().split())
                for query in paragraph['questions']:
                    targets = query['question_links']
                    if (len(targets) != 1 or len(links.get(targets[0], ())) != 1 or not query['question'].strip()
                            or query['answer']['type'] == 'none'):
                        continue
                    ((a, b),) = links[targets[0]]
                    disjoint = set()
                    for source in query['context']:
                        if source['passage'] == 'main':
                            x, y = source['indices']
                            if paragraph['text'][x:y] != source['text']:
                                raise ValueError('Author seed evidence does not match its original passage')
                            if y <= a or x >= b:
                                disjoint.add((x, y))
                    if len(disjoint) == 1 and (support := next(iter(disjoint)))[0] >= b:
                        actual[query['qid']] = (paragraph['pid'], family, a, b, *support, len(paragraph['text']))
            if actual.keys() != locked.keys() or len(actual) != expected['splits'][split]['eligible_later']:
                raise ValueError('Official author release disagrees with later cohort inventory')
            for qid, (pid, family, a, b, x, y, length) in actual.items():
                row = locked[qid]
                if ((pid, family, a, b, x, y) != (row['paragraph_id'], row['article_family'], row['anchor']['start'],
                                                  row['anchor']['end'], row['support_start'], row['support_end'])
                        or not 0 <= a < b <= x < y <= length):
                    raise ValueError('Locked later evidence does not match author boundaries')
            families[split] = {row[1] for row in actual.values()}
            lags = [y - b for _, _, _, b, _, y, _ in actual.values()]
            lengths = [y - x for _, _, _, _, x, y, _ in actual.values()]
            duplicates = Counter((pid, a, b) for pid, _, a, b, *_ in actual.values())
            results[split] = {
                'questions': len(actual), 'article_families': len(families[split]), 'distinct_anchors': len(duplicates),
                'questions_at_shared_anchors': sum(count for count in duplicates.values() if count > 1),
                'same_paragraph_identity_swap_pairs': expected['splits'][split]['later_contrast_pairs'],
                'characters_until_support_end': {'minimum': min(lags), 'median': median(lags), 'maximum': max(lags)},
                'support_characters': {'minimum': min(lengths), 'median': median(lengths), 'maximum': max(lengths)},
                'fits_8000_source_characters': sum(y <= 8000 for *_, y, _ in actual.values()),
            }
    if families['train'] & families['dev']:
        raise ValueError('Overlapping train and held-out source families')
    output.write_text(json.dumps({
        'author_archive_sha256': digest(archive), 'independent_seed_evidence_audit_sha256': digest(seed_audit),
        'independent_author_release_reconstruction': True, 'cohorts': results,
        'interval_event': 'First certified online stop where the whole answer-worker support span is visible. '
                          'The selected cohort has a later support by construction. No reader response time is recorded.',
        'hypothesis_tested': False,
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'prepared', 'seed-audit', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    audit(**vars(parser.parse_args()))
