"""Replay the retrospective paired-state diagnostic without fitting to reader keys."""

import argparse
import json
from pathlib import Path

import numpy as np

from research.semantic_tracking.frick_consistency_score import CONDITIONS, PROTOCOL, load_layout, load_states


def diagnose(assay, extraction, packet, protocol):
    _, stimuli, metadata, families, _ = load_layout(assay)
    states, provenance = load_states(assay, extraction, packet, protocol, stimuli, metadata, families)
    differences, roles = [], []
    for story, pair in sorted(families.items()):
        a, b = (pair[condition] for condition in CONDITIONS)
        left, right = (states[key].astype(np.float64) for key in (a, b))
        unit_left, unit_right = (values / np.linalg.norm(values, axis=1, keepdims=True) for values in (left, right))
        differences.append(unit_right - unit_left)
        roles.append((story, metadata[b]['changed_index'], metadata[b]['target_index']))
    targets = np.stack([difference[target] for difference, (_, _, target) in zip(differences, roles, strict=True)])
    norms = np.linalg.norm(targets, axis=1, keepdims=True)
    if not np.all(norms > 0):
        raise ValueError('A target contrast has no finite nonzero direction')
    unit = targets / norms
    details = []
    for i, (difference, (story, changed, target)) in enumerate(zip(differences, roles, strict=True)):
        center = unit[np.arange(len(unit)) != i].mean(axis=0)
        direction = center / np.linalg.norm(center)
        contrast = difference @ direction
        details.append({
            'story': story, 'target_index': target, 'changed_index': changed,
            'target_contrast': float(contrast[target]), 'changed_contrast': float(contrast[changed]),
            'target_rank': 1 + int(np.count_nonzero(contrast[1:] > contrast[target])),
            'eligible_sentences': len(contrast) - 1,
        })
    return {
        'retrospective_not_a_gate': True, 'story_pairs': len(roles),
        'held_out_target_change_direction_agreement': sum(row['target_contrast'] > 0 for row in details),
        'target_peak_top1': sum(row['target_rank'] == 1 for row in details),
        'target_peak_top3': sum(row['target_rank'] <= 3 for row in details),
        'target_mean_rank': float(np.mean([row['target_rank'] for row in details])),
        'by_story': details, 'feature_seal_sha256': provenance['state_extraction_sha256'],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('assay', 'extraction', 'packet'):
        parser.add_argument(f'--{option}', required=True, type=Path)
    parser.add_argument('--protocol', default=PROTOCOL, type=Path)
    args = parser.parse_args()
    print(json.dumps(diagnose(**vars(args)), indent=2, sort_keys=True, allow_nan=False))
