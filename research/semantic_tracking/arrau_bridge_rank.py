"""Fit a causal bridge reranker on GUM train and score sealed ARRAU transfer."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from math import log1p
from pathlib import Path

import numpy as np
import scipy

from research.semantic_tracking import bridge_interaction_fit as ridge
from research.semantic_tracking import bridge_rerank as previous
from research.semantic_tracking import gum_bridge_rank as gum


FEATURES = ('log_word_distance', 'log_word_length', 'capitalized', 'log_visible_frequency',
            'trigger_overlap', 'function_word', 'has_digit', 'following_punctuation',
            'preceding_preposition', 'qwen_cosine')
METHODS = ('distance', 'surface', 'cosine')
COLLECTIONS = ('pear', 'gnome')
STRATA = ('all', 'within_sentence', 'cross_sentence', 'unclassified')
PROJECTION_ARTIFACTS = ('coordinates.jsonl', 'vectors.npy')
PROJECTION_ID = ('candidate_id', 'model', 'revision', 'tokenizer_sha256', 'projection_sha256',
                 'projection_width', 'anchor_heads_sha256')
PUNCTUATION = ',.;:!?)]}'
BOOTSTRAP_SEED = 735972
BOOTSTRAP_REPLICATES = 10_000
GUM_ARCHIVE = Path(__file__).resolve().parents[1] / 'results' / 'gum-bridge-l4'


@dataclass(frozen=True)
class Article:
    identifier: str
    body: str
    coordinates: dict


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    with path.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def input_articles(path):
    rows = gum.read_rows(path)
    articles = {row['id']: row for row in rows}
    if len(articles) != len(rows):
        raise ValueError(f'Duplicate article identifiers in {path}')
    return articles


def code_identity():
    return {Path(module.__file__).name: gum.digest(Path(module.__file__)) for module in (ridge, previous, gum)} | {
        Path(__file__).name: gum.digest(Path(__file__)),
    }


def projection(extraction, expected_identity=None):
    seal = read_json(extraction / 'seal.json')
    identity = {key: seal[key] for key in PROJECTION_ID}
    if (identity['candidate_id'] != gum.CANDIDATE or identity['projection_width'] != 128
            or seal['content_forward_per_document'] != 1
            or type(seal['independent_causal_checks']) is not int or seal['independent_causal_checks'] < 1
            or (expected_identity is not None and identity != expected_identity)):
        raise ValueError(f'Unexpected frozen causal projection in {extraction}')
    for name in (*PROJECTION_ARTIFACTS, 'causality.json'):
        if gum.digest(extraction / name) != seal['file_sha256'][name]:
            raise ValueError(f'Projection artifact checksum differs for {name}')
    coordinates, vectors = gum.read_projection(extraction / 'vectors.npy', extraction / 'coordinates.jsonl')
    if list(vectors.shape) != seal['vectors_shape'] or len(coordinates) != seal['encoded_documents']:
        raise ValueError(f'Projection coordinates differ from the sealed inventory in {extraction}')
    return seal, identity, coordinates, vectors


def build_case(article, target, vectors, coverage):
    onset, stop = target['trigger_span']
    spans = target['antecedent_spans']
    if (target['event_stop'] != stop or not 0 <= onset < stop <= len(article.body) or not spans
            or any(not 0 <= start < end <= onset for start, end in spans)):
        raise ValueError(f'Invalid strictly prior scoring coordinates for {article.identifier}')

    words = tuple(gum.WORD.finditer(article.body, 0, stop))
    ends = tuple(word.end() for word in words)
    count = bisect_right(ends, onset)
    candidates = [index for index in range(count) if ends[index] in article.coordinates]
    word_at = {ends[index]: words[index] for index in candidates}
    exact = [(start, end) for start, end in spans if end in word_at and start <= word_at[end].start()]
    gold_ends = {end for _, end in exact}
    coverage.update(gold_spans=len(spans), exact_gold_spans=len(exact),
                    inaccessible_gold_spans=len(spans) - len(exact),
                    annotations_with_inaccessible_gold=len(exact) != len(spans),
                    candidate_words=len(candidates), uncertified_prior_words=count - len(candidates))
    if not article.coordinates:
        return 'missing_extraction_document', None
    if stop not in article.coordinates:
        return 'uncertified_trigger', None
    if not gold_ends:
        return 'no_exact_gold', None
    if len(candidates) == len(gold_ends):
        return 'no_negative_candidate', None

    terms = tuple(gum.surface(word.group()) for word in words)
    frequency = Counter(terms)
    trigger = {gum.surface(word.group()) for word in gum.WORD.finditer(article.body, onset, stop)}
    values, locations, gold = [], [], []
    for index in candidates:
        word = words[index]
        end = word.end()
        term = terms[index]
        prior = terms[index - 1] if index else ''
        values.append((
            log1p(count - index), log1p(len(term)), word.group()[0].isupper(),
            log1p(frequency[term]), len(term) >= 3 and term in trigger,
            term in previous.FUNCTION_WORDS, any(character.isdigit() for character in term),
            end < onset and article.body[end] in PUNCTUATION, prior in previous.PREPOSITIONS,
        ))
        locations.append(article.coordinates[end])
        gold.append(end in gold_ends)
    anchor = vectors[article.coordinates[stop]].astype(np.float32)
    earlier = vectors[locations].astype(np.float32)
    norms = np.linalg.norm(earlier, axis=1) * np.linalg.norm(anchor)
    if not np.all(np.isfinite(norms) & (norms > 0)):
        raise ValueError(f'Invalid frozen endpoint states in {article.identifier}')
    cosine = np.clip((earlier @ anchor) / norms, -1.0, 1.0)
    design = np.column_stack((np.asarray(values, dtype=np.float32), cosine))
    return 'scored', gum.Example(article.identifier, design, np.asarray(gold, dtype=bool))


def sentence_stratum(target):
    trigger = target['trigger_sentence']
    sentences = target['antecedent_sentences']
    if len(sentences) != len(target['antecedent_spans']):
        raise ValueError('Source sentence metadata does not align with antecedents')
    if trigger is None or None in sentences:
        return 'unclassified'
    if all(sentence == trigger for sentence in sentences):
        return 'within_sentence'
    if all(sentence != trigger for sentence in sentences):
        return 'cross_sentence'
    return 'unclassified'


def cases(articles, labels, coordinates, vectors, inventory, *, strata):
    article_cache = {}
    outcomes = Counter()
    coverage = Counter()
    stratum_counts = {name: Counter() for name in STRATA[1:] if strata}
    counts, available_case_documents = [], set()
    with labels.open(encoding='utf-8') as handle:
        for line in handle:
            target = json.loads(line)
            key = target['id']
            available_case_documents.add(key)
            if key not in article_cache:
                article_cache[key] = Article(key, articles[key]['text'], coordinates.get(key, {}))
            reason, item = build_case(article_cache[key], target, vectors, coverage)
            outcomes[reason] += 1
            group = sentence_stratum(target) if strata else None
            if strata:
                stratum_counts[group][reason] += 1
            if item is not None:
                counts.append(len(item.gold))
                yield group, item
    inventory.update(articles=len(articles), articles_with_annotations=len(available_case_documents),
                     articles_extracted=sum(key in coordinates for key in articles),
                     annotations=sum(outcomes.values()), outcomes=dict(outcomes), coverage=dict(coverage),
                     sentence_outcomes={key: dict(count) for key, count in stratum_counts.items()},
                     median_candidates=float(np.median(counts)) if counts else None)


def fitted_methods(examples):
    return {'surface': ridge.fit(examples, slice(0, len(FEATURES) - 1)), 'cosine': ridge.fit(examples)}


def fit(gum_assay, gum_extraction, protocol, output):
    archived_fit = read_json(GUM_ARCHIVE / 'fit.json')['provenance']
    inputs = gum_assay / 'model_input' / 'train.jsonl'
    labels = gum_assay / 'scoring' / 'train.jsonl'
    protocol_sha = gum.digest(protocol)
    sources = {'train_inputs': gum.digest(inputs), 'train_labels': gum.digest(labels),
               'extraction': gum.digest(gum_extraction / 'seal.json')}
    sources.update({name: gum.digest(gum_extraction / name) for name in PROJECTION_ARTIFACTS})
    if (any(archived_fit[key] != value for key, value in sources.items())
            or sources['extraction'] != gum.digest(GUM_ARCHIVE / 'extraction.json')):
        raise ValueError('Fitting inputs are not the archived GUM official training and extraction bytes')
    _, identity, coordinates, vectors = projection(gum_extraction)
    articles = input_articles(inputs)
    if any(key not in coordinates or not coordinates[key].keys() <= set(row['stops'])
           for key, row in articles.items()):
        raise ValueError('GUM train source and projection stops differ')
    inventory = {}
    examples = [item for _, item in cases(articles, labels, coordinates, vectors, inventory, strata=False)]
    fitted = fitted_methods(examples)
    receipt = {'schema': 1, 'fit_split': 'gum_train', 'protocol_sha256': protocol_sha, 'sources': sources,
               'code_sha256': code_identity(), 'projection_identity': identity, 'features': list(FEATURES),
               'ridge_penalty': ridge.PENALTY, 'inventory': inventory, 'fitted': fitted,
               'versions': {'numpy': np.__version__, 'scipy': scipy.__version__}}
    write_json(output, receipt)
    return receipt


def measures(item, fitted):
    logits = (-item.design[:, 0], ridge.predict(item.design, fitted['surface']),
              ridge.predict(item.design, fitted['cosine']))
    return np.asarray([previous.measure(row, item.gold) for row in logits])


def measured_summary(grouped, *, per_document=False):
    if not grouped:
        return {'articles': 0, 'cases': 0, 'methods': {}, 'cosine_minus_surface': {}}
    ids = sorted(grouped)
    values = np.asarray([np.mean(grouped[key], axis=0) for key in ids])
    differences = values[:, METHODS.index('cosine')] - values[:, METHODS.index('surface')]
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(len(ids), size=(BOOTSTRAP_REPLICATES, len(ids)))
    intervals = np.quantile(differences[indices].mean(axis=1), [0.025, 0.975], axis=0).T
    means = differences.mean(axis=0)
    metrics = previous.METRICS
    summary = {
        'articles': len(ids), 'cases': sum(len(grouped[key]) for key in ids),
        'methods': {method: dict(zip(metrics, map(float, values[:, index].mean(axis=0)), strict=True))
                    for index, method in enumerate(METHODS)},
        'cosine_minus_surface': {name: {'delta': float(mean), 'ci95': interval.tolist()}
                                 for name, mean, interval in zip(metrics, means, intervals, strict=True)},
    }
    if per_document:
        summary['paired_articles'] = {key: {'cases': len(grouped[key]), 'cosine_minus_surface':
                                            dict(zip(metrics, map(float, difference), strict=True))}
                                      for key, difference in zip(ids, differences, strict=True)}
    return summary


def arrau_inputs(arrau, packet, extraction, protocol_sha, identity):
    audit_path = arrau / 'audit.json'
    audit = read_json(audit_path)
    receipt = read_json(packet / 'preflight.json')
    stimuli_path = packet / 'stimuli.jsonl'
    stimuli_sha = gum.digest(stimuli_path)
    extract_preflight = packet / 'extractor-preflight.json'
    seal, _, coordinates, vectors = projection(extraction, identity)
    if (receipt['source_audit_sha256'] != gum.digest(audit_path) or receipt['stimuli_sha256'] != stimuli_sha
            or receipt['candidate_id'] != identity['candidate_id']
            or receipt['tokenizer_sha256'] != identity['tokenizer_sha256']
            or seal['assay_sha256'] != gum.digest(packet / 'preflight.json')
            or seal['preflight_sha256'] != gum.digest(extract_preflight)
            or seal['protocol_sha256'] != protocol_sha or seal['stimuli_sha256'] != stimuli_sha):
        raise ValueError('ARRAU extraction and preflight do not bind the exact source and protocol')

    stimuli = input_articles(stimuli_path)
    if (set(coordinates) != set(stimuli)
            or any(list(coordinates[key]) != row['stops'] for key, row in stimuli.items())):
        raise ValueError('Extraction does not exactly cover all certified ARRAU endpoints')
    articles = {}
    for collection in COLLECTIONS:
        name = f'model_input/{collection}.jsonl'
        path = arrau / name
        if gum.digest(path) != audit['output_sha256'][name] or audit['output_sha256'][name] != receipt['inputs_sha256'][name]:
            raise ValueError(f'Inputs differ from the sealed ARRAU {collection} inventory')
        rows = input_articles(path)
        documents = {key: row for key, row in receipt['documents'].items() if row['collection'] == collection}
        if (set(rows) != set(documents) or len(rows) != audit['collections'][collection]['documents']
                or {key for key, row in documents.items() if row['certified_stops']} != rows.keys() & stimuli.keys()):
            raise ValueError(f'Unexpected source or extracted documents in ARRAU {collection}')
        if any(stimuli[key]['text'] != rows[key]['text'] or not set(stimuli[key]['stops']) <= set(rows[key]['stops'])
               or len(stimuli[key]['stops']) != documents[key]['certified_stops'] for key in rows.keys() & stimuli.keys()):
            raise ValueError(f'Projection changed the original {collection} source words or stops')
        articles[collection] = rows
    pear_ids, gnome_ids = set(articles['pear']), set(articles['gnome'])
    if pear_ids & gnome_ids or set(receipt['documents']) != pear_ids | gnome_ids:
        raise ValueError('Unexpected overlap or omission in ARRAU collection assignment')
    return audit, receipt, articles, coordinates, vectors


def score(arrau, packet, extraction, protocol, fit_receipt, fit_sha256, output):
    protocol_sha = gum.digest(protocol)
    if gum.digest(fit_receipt) != fit_sha256:
        raise ValueError('Model coefficients do not match the separately sealed GUM fit digest')
    receipt = read_json(fit_receipt)
    provenance = read_json(GUM_ARCHIVE / 'fit.json')['provenance']
    frozen_sources = {key: provenance[key] for key in ('train_inputs', 'train_labels', 'extraction', *PROJECTION_ARTIFACTS)}
    widths = {'surface': len(FEATURES) - 1, 'cosine': len(FEATURES)}
    if (receipt['schema'] != 1 or receipt['fit_split'] != 'gum_train' or receipt['protocol_sha256'] != protocol_sha
            or receipt['features'] != list(FEATURES) or receipt['code_sha256'] != code_identity()
            or receipt['ridge_penalty'] != ridge.PENALTY or receipt['fitted'].keys() != widths.keys()
            or receipt['sources'] != frozen_sources
            or any(receipt['fitted'][method]['columns'] != list(range(width))
                   for method, width in widths.items())):
        raise ValueError('Incompatible frozen GUM fit for the locked ARRAU protocol')
    audit, preflight, articles, coordinates, vectors = arrau_inputs(
        arrau, packet, extraction, protocol_sha, receipt['projection_identity'])
    collections = {}
    for collection in COLLECTIONS:
        name = f'scoring/{collection}.jsonl'
        truth = arrau / name
        if gum.digest(truth) != audit['output_sha256'][name]:
            raise ValueError(f'ARRAU outcome bytes differ from the frozen source audit for {collection}')
        inventory = {}
        grouped = {name: defaultdict(list) for name in STRATA}
        for stratum, item in cases(articles[collection], truth, coordinates, vectors, inventory, strata=True):
            result = measures(item, receipt['fitted'])
            grouped['all'][item.document].append(result)
            grouped[stratum][item.document].append(result)
        if inventory['annotations'] != audit['collections'][collection]['scoring_rows']:
            raise ValueError(f'ARRAU source annotation count differs for {collection}')
        collections[collection] = {
            'intake': audit['collections'][collection], 'prefix_alignment': preflight['collections'][collection],
            'scoring_inventory': inventory,
            'strata': {name: measured_summary(rows, per_document=collection == 'gnome') for name, rows in grouped.items()},
        }
    pear = collections['pear']['strata']['all']
    contrast = pear['cosine_minus_surface']
    gates = {'articles_at_least_15': pear['articles'] >= 15,
             'nll_ci95_upper_below_zero': bool(contrast) and contrast['nll_nats']['ci95'][1] < 0,
             'top5_ci95_lower_above_zero': bool(contrast) and contrast['top5']['ci95'][0] > 0}
    result = {
        'schema': 1, 'fit_sha256': fit_sha256, 'protocol_sha256': protocol_sha, 'code_sha256': code_identity(),
        'source_audit_sha256': gum.digest(arrau / 'audit.json'), 'extraction_sha256': gum.digest(extraction / 'seal.json'),
        'bootstrap': {'replicates': BOOTSTRAP_REPLICATES, 'seed': BOOTSTRAP_SEED, 'paired_by': 'article'},
        'fit_inventory': receipt['inventory'], 'collections': collections,
        'pear_primary': {'pass': all(gates.values()), 'gates': gates},
    }
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    tasks = parser.add_subparsers(dest='task', required=True)
    train = tasks.add_parser('fit')
    holdout = tasks.add_parser('score')
    for name in ('gum-assay', 'gum-extraction'):
        train.add_argument(f'--{name}', type=Path, required=True)
    for name in ('arrau', 'packet', 'extraction'):
        holdout.add_argument(f'--{name}', type=Path, required=True)
    holdout.add_argument('--fit', dest='fit_receipt', type=Path, required=True)
    holdout.add_argument('--fit-sha256', required=True)
    for task in (train, holdout):
        task.add_argument('--protocol', type=Path, required=True)
        task.add_argument('--output', type=Path, required=True)
    args = vars(parser.parse_args())
    action = args.pop('task')
    if action == 'fit':
        fit(**args)
    else:
        score(**args)


if __name__ == '__main__':
    main()
