"""Score the fixed GUMBridge readouts on their independent GENTLE materials."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from research.semantic_tracking import gum_bridge_rank as gum
from research.stage6.external_projection_prepare import GENTLE_AUDIT, GUM_FIT, PROTOCOL, read


def evaluate(gentle, packet, extraction):
    source = read(GENTLE_AUDIT)
    assay_path = packet / 'assay.json'
    assay = read(assay_path)
    projected = read(extraction / 'seal.json')
    fit = read(GUM_FIT)
    if (gum.digest(GENTLE_AUDIT) != assay['gentle_audit_sha256'] or gum.digest(GUM_FIT) != assay['gum_fit_sha256']
            or gum.digest(Path(gum.__file__)) != fit['provenance']['scorer']
            or gum.digest(gum.PREMISE) != fit['provenance']['protocol']
            or any(gum.digest(gentle / name) != sha for name, sha in source['derived_sha256'].items())
            or source['derived_sha256'] != assay['gentle_material_sha256']
            or not gum.digest(PROTOCOL) == assay['protocol_sha256'] == projected['protocol_sha256']
            or projected['assay_sha256'] != gum.digest(assay_path) or projected['candidate_id'] != gum.CANDIDATE
            or any(gum.digest(extraction / name) != projected['file_sha256'][name]
                   for name in ('coordinates.jsonl', 'vectors.npy'))):
        raise ValueError('The independent source, original weights or candidate projection changed')
    coordinates, vectors = gum.read_projection(extraction / 'vectors.npy', extraction / 'coordinates.jsonl')
    articles = {row['id']: row for row in gum.read_rows(gentle / 'articles.jsonl')}
    if (list(articles) != assay['partitions']['gentle'] or not articles.keys() <= coordinates.keys()
            or any(not coordinates[article].keys() <= set(row['stops']) for article, row in articles.items())):
        raise ValueError('The extracted GENTLE article coordinates differ from the audited reading stops')
    prior_projection = read(GUM_FIT.with_name('extraction.json'))
    if projected['projection_sha256'] != prior_projection['projection_sha256']:
        raise ValueError('Expected the same IIRC projection as the original GUMBridge fit')
    included, outcomes = defaultdict(list), Counter()
    for target in gum.read_rows(gentle / 'scoring.jsonl'):
        key = target['id']
        reason, example = gum.example(articles[key], target, coordinates[key], vectors)
        outcomes[reason] += 1
        if example is not None:
            included[key].append(example)
    if not included:
        raise ValueError('No independent exact GENTLE examples')
    examples = [row for key in sorted(included) for row in included[key]]
    weights = np.array([1 / (len(included) * len(included[row.document])) for row in examples])
    report = gum.evaluate(examples, weights, fit['params'])
    return {'assay_sha256': projected['assay_sha256'], 'projection_seal_sha256': gum.digest(extraction / 'seal.json'),
            'original_fit_sha256': assay['gum_fit_sha256'], 'protocol_sha256': assay['protocol_sha256'],
            'articles_available': len(articles), 'articles_scored': len(included), 'annotation_inventory': dict(outcomes),
            'median_candidates': float(np.median([len(row.gold) for row in examples])),
            **report, 'passes_gentle_replication': bool(len(included) >= 12 and report['qwen_minus_form_ci95'][1] < 0)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('gentle', 'packet', 'extraction', 'output'):
        parser.add_argument(f'--{option}', type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.gentle, args.packet, args.extraction)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n')
