"""Score frozen plain-text IIRC anchor proposals on unseen article families."""

import argparse
from collections import defaultdict
import json
from math import isfinite
from pathlib import Path

import numpy as np

from research.core_features.readback_score import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, paired_material_interval
from research.stage3.run import save_json

from .iirc_prepare import digest
from .joint_score import rows_by_key


METHODS = ('form', 'qwen')
TOP_K = 5
MIN_IOU = 0.5


def span(row):
    return row['start'], row['end']


def intersection(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def iou(a, b):
    shared = intersection(a, b)
    return shared / (a[1] - a[0] + b[1] - b[0] - shared)


def proposals_for(row, text):
    if row.keys() != {'paragraph_id', 'methods'} or row['methods'].keys() != set(METHODS):
        raise ValueError('Unexpected IIRC prediction fields or methods')
    result = {}
    for method, proposals in row['methods'].items():
        if (not isinstance(proposals, list) or not 0 < len(proposals) <= TOP_K):
            raise ValueError(f'Expected 1–{TOP_K} proposals per paragraph and method')
        previous = 1.0
        offsets = []
        for proposed in proposals:
            if proposed.keys() != {'start', 'end', 'score'}:
                raise ValueError('Unexpected source span fields')
            start, end = span(proposed)
            score = proposed['score']
            if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text)
                    or not text[start:end].strip() or type(score) not in (int, float)
                    or not isfinite(score) or not 0 <= score <= previous
                    or any(intersection((start, end), seen) > 0 for seen in offsets)):
                raise ValueError('Invalid, unsorted or overlapping source span proposals')
            previous = score
            offsets.append((start, end))
        if sum(row['score'] for row in proposals) > 1.0 + 1e-7:
            raise ValueError('The top-k mass of one conditional distribution cannot exceed one')
        result[method] = offsets
    return result


def coverage(proposals, gold, threshold):
    return {index for index, candidate in enumerate(proposals) if iou(candidate, gold) >= threshold}


def metric(values, families):
    return {
        'means': {method: float(np.mean(values[method])) for method in METHODS},
        'qwen_minus_form': paired_material_interval(
            np.asarray(values['qwen'], dtype=np.float64) - np.asarray(values['form'], dtype=np.float64),
            [{'group_id': family} for family in families],
        ),
        'observations': len(families),
        'article_families': len(set(families)),
    }


def verify_seals(predictions, heads, seal, stimuli):
    metadata = json.loads(seal.read_text(encoding='utf-8'))
    for path, key in ((predictions, 'predictions_sha256'), (heads, 'heads_sha256'),
                      (stimuli, 'evaluation_sha256')):
        if metadata[key] != digest(path):
            raise ValueError(f'{path.name} differs from the frozen prediction manifest')


def run(stimuli, anchors, questions, pairs, predictions, heads, seal, output):
    verify_seals(predictions, heads, seal, stimuli)
    materials = rows_by_key(stimuli, lambda row: row['id'])
    inventory = rows_by_key(anchors, lambda row: row['paragraph_id'])
    forecasts = rows_by_key(predictions, lambda row: row['paragraph_id'])
    if not materials.keys() == inventory.keys() == forecasts.keys():
        raise ValueError('Missing or extra evaluation paragraphs')

    rendered = {}
    for identifier, material in materials.items():
        if material['stops'] != [len(material['text'])] or inventory[identifier]['stop'] != material['stops'][0]:
            raise ValueError('Evaluation stop does not match the original paragraph end')
        rendered[identifier] = proposals_for(forecasts[identifier], material['text'])

    links, families = {}, {}
    for id_, metadata in inventory.items():
        family = metadata['article_family']
        if not isinstance(family, str) or not family.strip():
            raise ValueError('Expected an article family for each paragraph')
        families[id_] = family
        seen = {}
        for item in metadata['anchors']:
            start, end = span(item)
            if (not 0 <= start < end <= metadata['stop'] or item['anchor_id'] in seen
                    or not materials[id_]['text'][start:end].strip()):
                raise ValueError('Invalid or repeated source anchor')
            seen[item['anchor_id']] = (start, end)
        links[id_] = seen

    labels = rows_by_key(questions, lambda row: row['question_id'])
    by_question = defaultdict(dict)
    observed = {method: [] for method in METHODS}
    exact = {method: [] for method in METHODS}
    brier = {method: [] for method in METHODS}
    bias = {method: [] for method in METHODS}
    question_families = []
    for question_id, label in labels.items():
        paragraph = label['paragraph_id']
        gold = links[paragraph][label['anchor_id']]
        question_families.append(families[paragraph])
        for method in METHODS:
            candidates = rendered[paragraph][method]
            matching = coverage(candidates, gold, MIN_IOU)
            by_question[question_id][method] = matching
            observed[method].append(bool(matching))
            is_exact = gold in candidates
            exact[method].append(is_exact)
            error = sum(row['score'] for row in forecasts[paragraph]['methods'][method]) - is_exact
            brier[method].append(error ** 2)
            bias[method].append(error)

    contrasts = rows_by_key(pairs, lambda row: (row['paragraph_id'], tuple(sorted(row['question_ids']))))
    paired, pair_families = {method: [] for method in METHODS}, []
    for (paragraph, (left, right)) in contrasts:
        a, b = labels[left], labels[right]
        if (left == right or a['paragraph_id'] != paragraph or b['paragraph_id'] != paragraph
                or intersection(links[paragraph][a['anchor_id']], links[paragraph][b['anchor_id']]) > 0):
            raise ValueError('Expected two different questions with disjoint anchors at the same stop')
        pair_families.append(families[paragraph])
        for method in METHODS:
            first, second = by_question[left][method], by_question[right][method]
            paired[method].append(any(x != y for x in first for y in second))

    per_question = metric(observed, question_families)
    both = metric(paired, pair_families)
    acceptance = {
        'coverage_superior_to_form': per_question['qwen_minus_form']['descriptive_paired_material_95_ci'][0] > 0,
        'distinct_pair_coverage_superior_to_form': both['qwen_minus_form']['descriptive_paired_material_95_ci'][0] > 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, {
        'scope': 'Predict a source span selected for an elicited question after reading a Wikipedia paragraph. '
                 'IIRC annotators saw hyperlinks. Plain-text predictions lack that cue.',
        'top_k': TOP_K, 'character_iou_threshold': MIN_IOU, 'paragraphs': len(materials),
        'per_question_coverage': per_question, 'distinct_same_stop_pair_coverage': both,
        'per_question_exact_coverage': metric(exact, question_families),
        'diagnostic_exact_hit_top_k_mass_brier': metric(brier, question_families),
        'diagnostic_exact_hit_top_k_mass_bias': metric(bias, question_families),
        'acceptance': {'conditions': acceptance, 'all_conditions_passed': all(acceptance.values())},
        'bootstrap': {'unit': 'source article family', 'draws': BOOTSTRAP_DRAWS, 'seed': BOOTSTRAP_SEED},
        'input_sha256': {key: digest(path) for key, path in (
            ('stimuli', stimuli), ('anchors', anchors), ('questions', questions), ('pairs', pairs),
            ('predictions', predictions), ('heads', heads), ('seal', seal), ('scorer', Path(__file__)),
        )},
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('stimuli', 'anchors', 'questions', 'pairs', 'predictions', 'heads', 'seal', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    run(**vars(parser.parse_args()))
