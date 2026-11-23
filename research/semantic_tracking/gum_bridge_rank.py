"""Fit and evaluate causal associations to prior words at GUMBridge annotation stops."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import file_digest
import json
from pathlib import Path
import re
from unicodedata import normalize

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from research.semantic_tracking.gum_bridge_extract import CANDIDATE


WORD = re.compile(r"[^\W_]+(?:[’'\-][^\W_]+)*")
FORM = ('word_distance', 'paragraph_distance', 'word_length', 'capitalized', 'prefix_count', 'phrase_overlap')
FEATURES = FORM + ('qwen_cosine',)
METHODS = ('distance', 'form', 'qwen_form')
PENALTY = 0.05
PREMISE = Path(__file__).with_name('gum-bridge-protocol.md')


@dataclass(frozen=True)
class Example:
    document: str
    design: np.ndarray
    gold: np.ndarray


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def read_rows(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def read_projection(path, coordinates_path):
    coordinates = {}
    offset = 0
    for row in read_rows(coordinates_path):
        key, stops = row['id'], row['stops']
        if key in coordinates or not stops or sorted(set(stops)) != stops:
            raise ValueError(f'Invalid extracted coordinates for {key}')
        coordinates[key] = {value: offset + index for index, value in enumerate(stops)}
        offset += len(stops)
    vectors = np.load(path, allow_pickle=False, mmap_mode='r')
    if vectors.shape != (offset, 128) or vectors.dtype not in (np.dtype('float16'), np.dtype('float32')):
        raise ValueError('Vector dimensions do not match the ordered extraction coordinates')
    return coordinates, vectors


def paragraph_for(paragraphs, starts, span):
    start, end = span
    index = bisect_right(starts, start) - 1
    if index < 0 or not paragraphs[index][0] <= start < end <= paragraphs[index][1]:
        return None
    return index


def surface(value):
    return normalize('NFKC', value).casefold().replace('’', "'")


def example(article, truth, coordinates, vectors):
    body = article['text']
    paragraphs = article['paragraphs']
    starts = [start for start, _ in paragraphs]
    trigger = truth['trigger_span']
    target_spans = truth['antecedent_spans']
    onset, stop = trigger
    if (truth['event_stop'] != stop or not target_spans or not 0 <= onset < stop <= len(body)
            or any(not 0 <= start < end <= onset for start, end in target_spans)):
        raise ValueError(f'Invalid GUMBridge scoring spans for {article["id"]}')
    origin = paragraph_for(paragraphs, starts, trigger)
    targets = [paragraph_for(paragraphs, starts, span) for span in target_spans]
    if origin is None or None in targets:
        return 'unknown_paragraph', None
    if any(target >= origin for target in targets):
        return 'same_paragraph_target', None
    if stop not in coordinates:
        return 'trigger_tokenization', None

    words = list(WORD.finditer(body, 0, onset))
    candidates = []
    for index, item in enumerate(words):
        paragraph = paragraph_for(paragraphs, starts, item.span())
        if paragraph is not None and item.end() in coordinates:
            candidates.append((index, item, paragraph))
    target_ends = {end for _, end in target_spans}
    if not target_ends <= {item.end() for _, item, _ in candidates}:
        return 'target_tokenization_or_nonword', None
    if len(candidates) <= len(target_ends):
        return 'no_negative_candidate', None

    frequency = Counter(surface(item.group()) for item in WORD.finditer(body, 0, stop))
    trigger_terms = {surface(item.group()) for item in WORD.finditer(body, onset, stop)}
    values, positions, gold = [], [], []
    for index, item, paragraph in candidates:
        word = item.group()
        term = surface(word)
        values.append((np.log1p(len(words) - index), np.log1p(origin - paragraph), np.log1p(len(term)),
                       word[0].isupper(), np.log1p(frequency[term]), len(term) >= 3 and term in trigger_terms))
        positions.append(coordinates[item.end()])
        gold.append(item.end() in target_ends)
    anchor = vectors[coordinates[stop]].astype(np.float64)
    earlier = vectors[positions].astype(np.float64)
    lengths = np.linalg.norm(earlier, axis=1) * np.linalg.norm(anchor)
    if not np.all(np.isfinite(lengths) & (lengths > 0)):
        raise ValueError(f'Invalid values at extracted positions for {article["id"]}')
    cosine = earlier @ anchor / lengths
    return 'scored', Example(article['id'], np.column_stack((values, cosine)), np.array(gold, dtype=bool))


def build(root, split, coordinates, vectors):
    articles = read_rows(root / 'model_input' / f'{split}.jsonl')
    by_id = {row['id']: row for row in articles}
    if len(articles) != len(by_id) or not by_id.keys() <= coordinates.keys():
        raise ValueError(f'Missing or duplicate extracted articles in {split}')
    for key, article in by_id.items():
        if not coordinates[key].keys() <= set(article['stops']):
            raise ValueError(f'Extraction includes a stop outside the source inventory for {key}')
    audit = Counter()
    grouped = defaultdict(list)
    for target in read_rows(root / 'scoring' / f'{split}.jsonl'):
        key = target['id']
        reason, item = example(by_id[key], target, coordinates[key], vectors)
        audit[reason] += 1
        if item is not None:
            grouped[key].append(item)
    if not grouped:
        raise ValueError(f'{split} contains no scorable cross-paragraph annotations')
    examples = [item for key in sorted(grouped) for item in grouped[key]]
    weight = np.array([1 / (len(grouped) * len(grouped[item.document])) for item in examples])
    return examples, weight, {'articles_available': len(articles), 'articles_scored': len(grouped),
                              'annotations': dict(audit), 'median_candidates': float(np.median([len(row.gold) for row in examples]))}


def loss_and_gradient(beta, examples, weights, matrix):
    loss = PENALTY * 0.5 * (beta @ beta)
    gradient = PENALTY * beta
    for weight, design, row in zip(weights, matrix, examples, strict=True):
        scores = design @ beta
        normalization = logsumexp(scores)
        gold_scores = scores[row.gold]
        gold_normalization = logsumexp(gold_scores)
        posterior = np.exp(scores - normalization)
        posterior[row.gold] -= np.exp(gold_scores - gold_normalization)
        loss += weight * (normalization - gold_normalization)
        gradient += weight * (posterior @ design)
    return loss, gradient


def fit(examples, weights):
    means = np.stack([row.design.mean(axis=0) for row in examples])
    squared_means = np.stack([np.square(row.design).mean(axis=0) for row in examples])
    mean = weights @ means
    second = weights @ squared_means
    scale = np.sqrt(np.maximum(second - mean ** 2, 1e-12))
    scaled = [(row.design - mean) / scale for row in examples]
    fit_weights = {}
    for key, dimensions in (('form', len(FORM)), ('qwen_form', len(FEATURES))):
        subset = [arr[:, :dimensions] for arr in scaled]
        solution = minimize(loss_and_gradient, np.zeros(dimensions), args=(examples, weights, subset),
                            method='L-BFGS-B', jac=True, options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
        if not solution.success or not np.all(np.isfinite(solution.x)):
            raise ValueError(f'GUMBridge {key} fit failed: {solution.message}')
        fit_weights[key] = solution.x.tolist()
    return {'center': mean.tolist(), 'scale': scale.tolist(), 'weights': fit_weights}


def evaluate(examples, weights, fitted):
    center, scale = np.array(fitted['center']), np.array(fitted['scale'])
    models = [np.array(fitted['weights'][name]) for name in METHODS[1:]]
    by_article = defaultdict(list)
    for row in examples:
        scaled = (row.design - center) / scale
        scores = (-row.design[:, 0], *(scaled[:, :len(beta)] @ beta for beta in models))
        metrics = []
        for values in scores:
            best = max(values[row.gold])
            # In ties assign the average rank over every possible ordering of tied candidates.
            ahead = np.count_nonzero(values > best)
            tied = values == best
            positives = np.count_nonzero(tied & row.gold)
            total_tied = np.count_nonzero(tied)
            survival, reciprocal = 1.0, 0.0
            for skips in range(total_tied - positives + 1):
                hit = positives / (total_tied - skips)
                reciprocal += survival * hit / (ahead + skips + 1)
                survival *= 1.0 - hit
            metrics.append((logsumexp(values) - logsumexp(values[row.gold]),
                            positives / total_tied if ahead == 0 else 0.0, reciprocal))
        by_article[row.document].append(metrics)
    matrix = np.stack([np.mean(by_article[key], axis=0) for key in sorted(by_article)])
    if not np.isclose(weights.sum(), 1):
        raise ValueError('Expected equal document weighting')
    delta = matrix[:, 2, 0] - matrix[:, 1, 0]
    rng = np.random.default_rng(735972)
    index = rng.integers(0, len(matrix), size=(10_000, len(matrix)))
    ci = np.quantile(delta[index].mean(axis=1), [0.025, 0.975])
    return {'metrics': {name: dict(zip(('nll_nats', 'top1', 'mrr'), values.tolist(), strict=True))
                        for name, values in zip(METHODS, matrix.mean(axis=0), strict=True)},
            'qwen_minus_form_nll': float(delta.mean()), 'qwen_minus_form_ci95': ci.tolist(),
            'passes': bool(len(matrix) >= 20 and ci[-1] < 0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('fit', 'score'))
    for flag in ('root', 'extraction', 'fit', 'output'):
        parser.add_argument(f'--{flag}', required=flag != 'output', type=Path)
    args = parser.parse_args()
    coordinates_path = args.extraction / 'coordinates.jsonl'
    vectors_path = args.extraction / 'vectors.npy'
    seal_path = args.extraction / 'seal.json'
    seal = json.loads(seal_path.read_text(encoding='utf-8'))
    extraction_hashes = {path.name: digest(path) for path in (coordinates_path, vectors_path)}
    if (seal['candidate_id'] != CANDIDATE or any(seal['file_sha256'][name] != sha for name, sha in extraction_hashes.items())
            or seal['input_sha256'] != {f'{split}.jsonl': digest(args.root / 'model_input' / f'{split}.jsonl')
                                       for split in ('train', 'dev', 'test')}):
        raise ValueError('Expected the verified candidate projection for these three inputs')
    coordinates, vectors = read_projection(vectors_path, coordinates_path)
    provenance = {'protocol': digest(PREMISE), 'scorer': digest(Path(__file__)),
                  'extraction': digest(seal_path), **extraction_hashes,
                  'train_inputs': digest(args.root / 'model_input' / 'train.jsonl'),
                  'train_labels': digest(args.root / 'scoring' / 'train.jsonl')}
    if args.command == 'fit':
        examples, weights, audit = build(args.root, 'train', coordinates, vectors)
        result = {'provenance': provenance, 'train': audit, 'params': fit(examples, weights)}
        destination = args.fit
    else:
        if args.output is None:
            parser.error('score requires --output')
        receipt = json.loads(args.fit.read_text(encoding='utf-8'))
        if receipt['provenance'] != provenance:
            raise ValueError('Fit receipt differs from the locked data or analysis')
        result = {'fit_sha256': digest(args.fit), 'splits': {}}
        destination = args.output
        for split in ('dev', 'test'):
            examples, weights, audit = build(args.root, split, coordinates, vectors)
            result['splits'][split] = {'inventory': audit, **evaluate(examples, weights, receipt['params']),
                                       'inputs_sha256': digest(args.root / 'model_input' / f'{split}.jsonl'),
                                       'labels_sha256': digest(args.root / 'scoring' / f'{split}.jsonl')}
    with destination.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
