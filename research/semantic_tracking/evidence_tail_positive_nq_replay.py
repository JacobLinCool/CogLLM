"""Replay both frozen QASPER clocks on the previously scored Natural Questions sample."""

import argparse
import json
from pathlib import Path

from research.semantic_tracking import evidence_tail_calibrate as primary
from research.semantic_tracking.evidence_tail_positive_control import compare

from nq_evidence_score import read_forecasts, read_sealed_outcomes, references_by_article


def run(assay_root, predictions_dir, outcomes, control, original_scores, output):
    experiment = json.loads(control.read_text(encoding='utf-8'))
    original = json.loads(original_scores.read_text(encoding='utf-8'))
    plan, hashes, articles, queries, schedules, _ = read_forecasts(assay_root, predictions_dir)
    if (experiment['input_sha256']['heads_sha256'] != hashes['heads_sha256']
            or any(hashes.get(key) != value for key, value in original['input_sha256'].items() if key in hashes)):
        raise ValueError('The clocks and original Natural Questions scores must share the same frozen holdout and head')
    selected, observations = read_sealed_outcomes(outcomes, plan, queries, hashes)
    references, inventory = references_by_article(assay_root, selected, observations, articles, queries, schedules)
    forecasts = primary.indexed_rows(predictions_dir / 'predictions.jsonl', {'id', 'article_id', 'stops', 'methods'})
    estimates = {'mixed_qasper_clock': experiment['mixed_estimate'],
                 'positive_only_qasper_clock': experiment['positive_only_train']['estimate']}
    cases, paths = [], {}
    for rows in references.values():
        for row in rows:
            cases.append(primary.ReferenceCase(row.identifier, row.family,
                                               tuple(None if ref is None else ref.stop for ref in row.references),
                                               tuple(None if ref is None else ref.source_end for ref in row.references)))
            layout = primary.PathLayout.from_hazards(forecasts[row.identifier]['methods']['token_rank'])
            paths[row.identifier] = {name: primary.probability_path(estimate, layout) for name, estimate in estimates.items()}
    primary.write_new(output, {
        'scope': 'Post-hoc external diagnostic on already scored Natural Questions; both QASPER clocks applied unchanged',
        'inventory': inventory, 'holdout': {scope: compare(cases, paths, scope) for scope in ('all', 'event', 'null')},
        'source_sha256': {**hashes, 'qasper_control_sha256': primary.digest(control),
                          'original_nq_scores_sha256': primary.digest(original_scores), 'replay_sha256': primary.digest(Path(__file__))},
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('assay-root', 'predictions-dir', 'outcomes', 'control', 'original-scores', 'output'):
        parser.add_argument(f'--{flag}', required=True, type=Path)
    run(**vars(parser.parse_args()))
