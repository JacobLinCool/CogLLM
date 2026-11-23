"""Estimate two calibration baselines for held-out OneStop rereading trials."""

import argparse
from collections import defaultdict
from hashlib import file_digest
import json
from pathlib import Path
from random import Random
from statistics import mean, quantiles

from .onestop_rereading_audit import SOURCES, load_releases, pairs


PROTOCOL = Path(__file__).with_name('onestop-rereading-baseline.md')
DRAWS = 10_000
SEED = 20261121


def prior(correct, total):
    return (correct + 0.5) / (total + 1)


def brier(probability, observed):
    return observed * (1 - probability) ** 2 + (1 - observed) * probability ** 2


def article_scores(heldout, pooled, by_exposure):
    cells = defaultdict(lambda: ([], []))
    for first, repeated in heldout:
        cells[(*first.material, first.question)][0].append(first.correct)
        cells[(*repeated.material, repeated.question)][1].append(repeated.correct)
    totals = defaultdict(list)
    for key, outcomes in sorted(cells.items()):
        if not all(outcomes):
            continue
        rates = tuple(mean(phase) for phase in outcomes)
        pooled_loss = mean(brier(pooled, rate) for rate in rates)
        exposure_loss = mean(brier(probability, rate) for probability, rate in zip(by_exposure, rates, strict=True))
        totals[key[:2]].append((pooled_loss, exposure_loss, rates[1] - rates[0]))
    if len(totals) != 10:
        raise ValueError('Expected one held-out article score for every article in batch 3')
    return [tuple(map(mean, zip(*rows, strict=True))) for rows in totals.values()]


def run(root):
    matched, _ = pairs(*load_releases(root))
    training = [pair for pair in matched if pair[0].batch in (1, 2)]
    heldout = [pair for pair in matched if pair[0].batch == 3]
    if {before.reader for before, _ in training} & {before.reader for before, _ in heldout}:
        raise ValueError('A reader appears in both development and holdout')
    counts = tuple(sum(trial.correct for trial in group) for group in zip(*training, strict=True))
    pooled = prior(sum(counts), 2 * len(training))
    conditional = tuple(prior(correct, len(training)) for correct in counts)
    scores = article_scores(heldout, pooled, conditional)
    pooled_brier, conditional_brier, observed_shift = map(mean, zip(*scores, strict=True))
    differences = [conditioned - baseline for baseline, conditioned, _ in scores]
    rng = Random(SEED)
    lower, *_, upper = quantiles(
        (mean(rng.choices(differences, k=len(scores))) for _ in range(DRAWS)), n=40, method='inclusive')
    predicted_shift = conditional[1] - conditional[0]
    with PROTOCOL.open('rb') as stream:
        protocol_sha256 = file_digest(stream, 'sha256').hexdigest()
    return {
        'protocol_sha256': protocol_sha256,
        'source_sha256': SOURCES,
        'development_pairs': len(training),
        'holdout_pairs_before_scoring_cell_filter': len(heldout),
        'holdout_articles': len(scores),
        'holdout_readers': len({first.reader for first, _ in heldout}),
        'probability_pooled': pooled,
        'probability_first_and_repeated': list(conditional),
        'holdout_pooled_brier': pooled_brier,
        'holdout_exposure_brier': conditional_brier,
        'holdout_exposure_minus_pooled_brier': conditional_brier - pooled_brier,
        'holdout_exposure_minus_pooled_brier_ci95': [lower, upper],
        'holdout_article_equal_repeated_minus_first_accuracy': observed_shift,
        'predicted_repeated_minus_first_accuracy': predicted_shift,
        'predicted_minus_observed_repeated_effect': predicted_shift - observed_shift,
        'brier_improvement_criterion_met': upper < 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(args.raw), indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
