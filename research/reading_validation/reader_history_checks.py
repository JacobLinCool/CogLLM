"""Check conjugate first updates and recover a known synthetic within-reader covariance."""
import numpy as np
from scipy.special import logsumexp
from research.backbone_transfer.extract import read_json, write_json
from research.reading_validation.reader_history_mode import (
    online_priors,
    pair_concentration,
)
from research.reading_validation.reader_history_peer import population_weights
from research.reading_validation.reader_history_materials import OUT
from research.reading_validation.reader_frequency_rank import LanguageModes, GAMMA, BETA


def check_exact_ranks():
    """Compare the serving shortcut with exhaustive probability sums on artificial words."""
    rng = np.random.default_rng(1911)
    n_words = 157
    log_frequency = rng.normal(size=n_words)
    dictionary = [{"freq": float(v)} for v in np.exp(log_frequency)]
    standardized = (log_frequency - log_frequency.mean()) / log_frequency.std()
    log_mass = rng.normal(size=(12, n_words + 1))
    law = LanguageModes(log_mass, dictionary, "de")
    sharpness = rng.dirichlet(np.ones(len(BETA)))
    preferences = np.vstack(
        [np.eye(len(GAMMA)), rng.dirichlet(np.ones(len(GAMMA)), 200)]
    )
    largest_error = 0.0
    different_winners = []
    for context, native in enumerate(log_mass[:, :-1]):
        logits = native[:, None, None] * BETA[None, :, None]
        logits = logits + standardized[:, None, None] * GAMMA
        conditional = np.exp(logits - logsumexp(logits, axis=0))
        direct = np.einsum("wkg,k->gw", conditional, sharpness)
        eligible, scores = law.ranked(context, sharpness)
        largest_error = max(largest_error, np.max(np.abs(scores - direct[:, eligible])))
        expected = (preferences @ direct).argmax(axis=1)
        actual = eligible[(preferences @ scores).argmax(axis=1)]
        if np.any(expected != actual):
            raise AssertionError(
                "Pruned exact-word ranking disagrees with full enumeration"
            )
        different_winners.append(len(np.unique(actual)))
    if largest_error > 1e-12 or max(different_winners) < 2:
        raise AssertionError(
            "Synthetic frequency preference failed to change an exact word"
        )
    return {
        "seed": 1911,
        "contexts": len(log_mass),
        "profiles_per_context": len(preferences),
        "words_per_context": n_words,
        "maximum_probability_discrepancy": float(largest_error),
        "maximum_distinct_winning_words_in_one_context": max(different_winners),
        "top_mismatches": 0,
    }


def main():
    rng = np.random.default_rng(1284)
    n = 700
    t = 70
    p = np.array([0.38, 0.62])
    kappa = 3.0
    lam = 1 / (1 + kappa)
    modes = np.array([0.12, 0.85])
    person = rng.dirichlet(kappa * p, n)
    y = rng.random((n, t)) < (person @ modes)[:, None]
    Q = np.where(y[:, :, None], modes, 1 - modes)
    # Use a disjoint cohort for fitting and a disjoint set of trials for each new person.
    train = 400
    probs = np.log(Q[:train, :35].reshape(-1, 2))
    people = np.repeat(np.arange(train), 35)
    prior, info = population_weights(probs, np.arange(len(probs)))
    dep, _ = pair_concentration(probs, people, prior)
    v = online_priors(
        Q[train:, :35],
        np.ones((n - train, 35), bool),
        np.ones((n - train, 35), bool),
        prior,
        dep,
    )
    resp = Q[train:, 0] * prior
    resp /= resp.sum(1, keepdims=True)
    baseline = prior + dep * (resp - prior)
    first_error = float(np.max(abs(baseline - v[:, 1])))
    held = Q[train:, 35:]
    own = np.einsum("ptk,pk->pt", held, v[:, -1])
    other = np.einsum("ptk,pk->pt", held, np.roll(v[:, -1], 1, axis=0))
    marg = held @ prior
    q = online_priors(
        Q[train:, :35],
        np.ones((n - train, 35), bool),
        np.ones((n - train, 35), bool),
        prior,
        0,
    )
    constant_error = float(np.max(abs(q - prior)))
    ymax = np.max(np.abs(v.sum(2) - 1))
    gain = float(np.mean(np.log(own / marg)))
    swap = float(np.mean(np.log(own / other)))
    if (
        first_error > 2e-7
        or constant_error > 2e-7
        or ymax > 2e-7
        or abs(dep - lam) > 0.08
        or abs(prior[0] - p[0]) > 0.08
        or min(gain, swap) <= 0
    ):
        raise ValueError("Known generative process check failed")
    evidence = dict(
        seed=1284,
        train_people=400,
        held_people=300,
        true_mixture=p.tolist(),
        fit_mixture=prior.tolist(),
        true_dependence=lam,
        fitted_dependence=dep,
        source_max_kkt=info["stationarity"],
        max_first_update_exact_dirichlet_mean_error=first_error,
        at_zero_dependence_max_history_effect=constant_error,
        maximum_probability_mass_error=float(ymax),
        history_trials=35,
        held_trials_per_person=35,
        rank_check=check_exact_ranks(),
        interpretation="Synthetic controls only",
        held_personal_gain_over_unknown=gain,
        held_personal_gain_over_other=swap,
    )
    path = OUT / "mechanism-verification.json"
    if path.exists():
        if read_json(path) != evidence:
            raise ValueError("The rerun has different simulation results")
    else:
        write_json(path, evidence)


if __name__ == "__main__":
    main()
