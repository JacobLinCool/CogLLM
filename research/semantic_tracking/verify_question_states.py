"""Verify the structured reading-state layer against independent tiny path enumeration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
import math


STATES = ("not_open", "unresolved", "set_aside", "resolved")
N, U, P, R = range(len(STATES))
ALLOWED = ((N, U), (U, P, R), (P, U, R), (R, U))


@dataclass(frozen=True)
class Enumerated:
    marginals: list
    events: list
    first_hit: list
    likelihood: float


def reference_transitions(logits, visible):
    probabilities = []
    for matrix, enabled in zip(logits, visible, strict=True):
        step = []
        for source, row in enumerate(matrix):
            destinations = ALLOWED[source] if enabled else (source,)
            values = [row[j] if enabled else 0.0 for j in destinations]
            maximum = max(values)
            normalizer = math.fsum(math.exp(x - maximum) for x in values)
            normalized = dict(zip(destinations, (math.exp(x - maximum) / normalizer for x in values), strict=True))
            step.append([normalized.get(j, 0.0) for j in range(len(STATES))])
        probabilities.append(step)
    return probabilities


def enumerate_tracks(transition, initial, emissions, observed, *, target=R):
    """Enumerate complete routes using ordinary probabilities on a short track."""
    if initial[target] != 0:
        raise ValueError("Reference first-hit requires an inactive target initially")
    size = len(transition)
    marginals = [[0.0] * len(STATES) for _ in range(size)]
    events = [[[0.0] * len(STATES) for _ in STATES] for _ in range(size)]
    first = [0.0] * (size + 1)
    likelihood = 0.0
    for start, start_mass in enumerate(initial):
        if not start_mass:
            continue
        for route in product(range(len(STATES)), repeat=size):
            edges = tuple(zip((start, *route)[:-1], route, strict=True))
            probability = start_mass * math.prod(transition[t][i][j] for t, (i, j) in enumerate(edges))
            if probability == 0.0:
                continue
            for t, (i, j) in enumerate(edges):
                marginals[t][j] += probability
                events[t][i][j] += probability
            first[next((t for t, j in enumerate(route) if j == target), size)] += probability
            weight = math.prod(emissions[t][j] for t, j in enumerate(route) if observed[t])
            likelihood += probability * weight
    return Enumerated(marginals, events, first, likelihood)


def fixture():
    visibility = [False, True, True, False, True]
    logits = [
        [[math.sin(13 * t + 5 * i + 3 * j) + (60 if j not in ALLOWED[i] else 0)
          for j in range(len(STATES))] for i in range(len(STATES))]
        for t in range(len(visibility))
    ]
    initial = [0.67, 0.33, 0.0, 0.0]
    emissions = [
        [0.4, 0.9, 1.0, 0.2], [0.2, 1.0, 0.7, 0.5], [0.0, 0.3, 0.9, 0.2],
        [1.0, 0.5, 0.4, 0.3], [0.1, 0.8, 0.4, 1.0],
    ]
    observed = [False, True, True, False, True]
    return logits, visibility, initial, emissions, observed


def censor_fixture():
    logits = [[[0.0 for _ in STATES] for _ in STATES] for _ in range(2)]
    initial = [0.0, 1.0, 0.0, 0.0]
    at_risk = [0.0, 1.0, 1.0, 0.0]
    resolved = [0.0, 0.0, 0.0, 1.0]
    unrestricted = [1.0] * len(STATES)
    cases = {
        "exact_first_resolution": ([at_risk, resolved], 2 / 9),
        "right_censored_first_resolution": ([at_risk, at_risk], 4 / 9),
        "unresolved_at_terminal_even_after_reopen": ([unrestricted, at_risk], 11 / 18),
    }
    return logits, initial, cases


def verify_reference():
    logits, visible, initial, emissions, observed = fixture()
    transition = reference_transitions(logits, visible)
    result = enumerate_tracks(transition, initial, emissions, observed)
    for t, rows in enumerate(transition):
        assert all(math.isclose(math.fsum(row), 1.0, abs_tol=1e-13) for row in rows)
        assert math.isclose(math.fsum(result.marginals[t]), 1.0, abs_tol=1e-13)
    assert math.isclose(math.fsum(result.first_hit), 1.0, abs_tol=1e-13)
    assert 0.0 < result.likelihood < 1.0
    logits, initial, cases = censor_fixture()
    transition = reference_transitions(logits, [True, True])
    for emissions, probability in cases.values():
        result = enumerate_tracks(transition, initial, emissions, [True, True])
        assert math.isclose(result.likelihood, probability, abs_tol=1e-13)
        assert all(math.isclose(got, want, abs_tol=1e-13)
                   for got, want in zip(result.first_hit, (1 / 3, 2 / 9, 4 / 9), strict=True))
    assert enumerate_tracks([], initial, [], []) == Enumerated([], [], [1.0], 1.0)


def verify_torch():
    import torch

    from research.semantic_tracking.question_states import (
        TRACE_STATES,
        first_hit_log_masses,
        trace_log_events,
        trace_log_marginals,
        trace_nll,
        transition_log_probs,
    )

    torch.set_num_threads(1)
    assert tuple(TRACE_STATES) == STATES

    def real(values, *, requires_grad=False):
        return torch.tensor(values, dtype=torch.float64, device="cpu", requires_grad=requires_grad)

    def mask(values):
        return torch.tensor(values, dtype=torch.bool, device="cpu")

    def close(actual, expected, *, exact=False):
        epsilon = 0 if exact else 3e-11
        torch.testing.assert_close(actual, expected, rtol=epsilon, atol=epsilon)

    logits, visible, initial, emissions, observed = fixture()
    reference = reference_transitions(logits, visible)
    truth = enumerate_tracks(reference, initial, emissions, observed)
    initial_lp = real(initial).log()
    raw = real(logits)
    active = mask(visible)
    observation = mask(observed)
    emission = real(emissions).log()
    transition = transition_log_probs(raw, active)
    close(transition.exp(), real(reference))
    assert torch.equal(torch.isneginf(transition), real(reference) == 0)
    close(transition[~active].exp(), torch.eye(len(STATES), dtype=torch.float64).expand(2, -1, -1), exact=True)
    close(trace_log_marginals(transition, initial_lp).exp(), real(truth.marginals))
    close(trace_log_events(transition, initial_lp).exp(), real(truth.events))
    close(first_hit_log_masses(transition, initial_lp).exp(), real(truth.first_hit))
    first_aside = enumerate_tracks(reference, initial, emissions, observed, target=P).first_hit
    close(first_hit_log_masses(transition, initial_lp, target=P).exp(), real(first_aside))
    loss = trace_nll(transition, initial_lp, emission, observation)
    assert loss.shape == ()
    close(loss, real(-math.log(truth.likelihood)))

    compact = transition[active]
    close(trace_log_marginals(compact, initial_lp), trace_log_marginals(transition, initial_lp)[active])
    close(trace_nll(compact, initial_lp, emission[active], observation[active]), loss)
    close(trace_log_marginals(transition, initial_lp)[0], initial_lp, exact=True)
    first_hit = first_hit_log_masses(transition, initial_lp)
    assert torch.isneginf(first_hit[:-1][~active]).all()
    close(first_hit_log_masses(compact, initial_lp), torch.cat((first_hit[:-1][active], first_hit[-1:])))

    changed_emissions = emission.clone()
    changed_emissions[~observation] = -torch.inf
    close(trace_nll(transition, initial_lp, changed_emissions, observation), loss, exact=True)
    changed_emissions[~observation] = torch.nan
    close(trace_nll(transition, initial_lp, changed_emissions, observation), loss, exact=True)
    no_observations = torch.zeros_like(observation)
    close(trace_nll(transition, initial_lp, changed_emissions, no_observations), real(0))
    cut = 3
    changed_raw = raw.clone()
    changed_raw[cut:] = raw[cut:].flip(-1) * 2
    changed_transition = transition_log_probs(changed_raw, active)
    close(trace_log_marginals(transition, initial_lp)[:cut], trace_log_marginals(changed_transition, initial_lp)[:cut], exact=True)
    close(trace_log_marginals(transition[:cut], initial_lp), trace_log_marginals(transition, initial_lp)[:cut], exact=True)
    close(trace_log_events(transition, initial_lp)[:cut], trace_log_events(changed_transition, initial_lp)[:cut], exact=True)
    close(first_hit_log_masses(transition, initial_lp)[:cut], first_hit_log_masses(changed_transition, initial_lp)[:cut], exact=True)
    prefix_only = observation & (torch.arange(len(observed)) < cut)
    close(
        trace_nll(changed_transition, initial_lp, emission, prefix_only),
        trace_nll(transition[:cut], initial_lp, emission[:cut], observation[:cut]),
    )

    # Reader questions overlap and resolve in either order without sharing a stack.
    expected_tracks = [[U, U, R, U, U], [N, U, U, P, R]]
    double_raw = torch.full((2, 5, 4, 4), -20.0, dtype=torch.float64)
    for track, expected in enumerate(expected_tracks):
        for t, (source, destination) in enumerate(zip((N, *expected[:-1]), expected, strict=True)):
            assert destination in ALLOWED[source]
            double_raw[track, t, source, destination] = 20.0
    double_transition = transition_log_probs(double_raw, mask([[True] * 5] * 2))
    double_initial = real([[1.0, 0.0, 0.0, 0.0]] * 2).log()
    marginal = trace_log_marginals(double_transition, double_initial)
    assert marginal.argmax(dim=-1).tolist() == expected_tracks
    assert trace_log_events(double_transition, double_initial)[0, 3, R, U].exp() > 0.999
    assert first_hit_log_masses(double_transition, double_initial).argmax(dim=-1).tolist() == [2, 4]
    batched = trace_nll(double_transition, double_initial, real([[[0.7] * 4] * 5] * 2).log(), mask([[True] * 5] * 2))
    assert batched.shape == (2,)
    close(batched, real([-5 * math.log(0.7)] * 2))
    two_axes = double_transition.expand(3, -1, -1, -1, -1)
    close(trace_log_marginals(two_axes, double_initial.expand(3, -1, -1)), marginal.expand(3, -1, -1, -1))

    logits, initial, cases = censor_fixture()
    transition = transition_log_probs(real(logits), mask([True, True]))
    initial_lp = real(initial).log()
    for emissions, probability in cases.values():
        close(trace_nll(transition, initial_lp, real(emissions).log(), mask([True, True])), real(-math.log(probability)))
    close(first_hit_log_masses(transition, initial_lp).exp(), real([1 / 3, 2 / 9, 4 / 9]))
    empty = transition_log_probs(real(logits)[:0], mask([]))
    assert trace_log_marginals(empty, initial_lp).shape == (0, len(STATES))
    assert trace_log_events(empty, initial_lp).shape == (0, len(STATES), len(STATES))
    close(trace_nll(empty, initial_lp, empty.new_empty((0, len(STATES))), mask([])), real(0), exact=True)
    close(first_hit_log_masses(empty, initial_lp).exp(), real([1.0]), exact=True)
    identity = torch.eye(len(STATES), dtype=torch.float64).log().expand_as(transition)
    close(transition_log_probs(identity, mask([True, True])), identity, exact=True)

    logits, visible, initial, emissions, observed = fixture()
    raw = real(logits, requires_grad=True)
    active = mask(visible)
    # Exact not-open initialization makes some destination columns entirely unreachable early.
    born_closed = real([1.0, 0.0, 0.0, 0.0]).log()
    emission = real(emissions).log()
    observation = mask(observed)

    def risk_loss(values):
        return trace_nll(transition_log_probs(values, active), born_closed, emission, observation)

    (gradient,) = torch.autograd.grad(risk_loss(raw), raw)
    assert torch.isfinite(gradient).all()
    assert torch.equal(gradient[~active], torch.zeros_like(gradient[~active]))
    allowed = mask([[j in ALLOWED[i] for j in range(4)] for i in range(4)])
    assert torch.equal(gradient[:, ~allowed], torch.zeros_like(gradient[:, ~allowed]))
    index = (1, N, U)
    assert abs(float(gradient[index])) > 1e-5
    step = 1e-5
    plus, minus = raw.detach().clone(), raw.detach().clone()
    plus[index] += step
    minus[index] -= step
    numerical = (risk_loss(plus) - risk_loss(minus)) / (2 * step)
    torch.testing.assert_close(gradient[index], numerical, rtol=1e-7, atol=1e-9)
    trace = transition_log_probs(raw, active)
    first_objective = first_hit_log_masses(trace, born_closed)[-1]
    (first_gradient,) = torch.autograd.grad(first_objective, raw)
    assert torch.isfinite(first_gradient).all()

    long_size = 1_024
    transition = transition_log_probs(torch.zeros((long_size, 4, 4), dtype=torch.float64), mask([True] * long_size))
    emissions = torch.full((long_size, 4), -3.0, dtype=torch.float64)
    long_nll = trace_nll(transition, born_closed, emissions, mask([True] * long_size))
    close(long_nll, real(3 * long_size))
    close(trace_log_marginals(transition, born_closed).logsumexp(dim=-1), torch.zeros(long_size, dtype=torch.float64))
    close(first_hit_log_masses(transition, born_closed).logsumexp(dim=-1), real(0))

    tracks, tokens, prefix = 8, 8_192, 2_047
    random_logits = torch.randn((tracks, tokens, 4, 4), generator=torch.Generator().manual_seed(424242))
    visible = torch.arange(tokens)[None, :] >= torch.arange(tracks)[:, None] * 91
    with torch.inference_mode():
        for dtype, tolerance in ((torch.float32, 1e-4), (torch.float64, 1e-10)):
            initial = born_closed.expand(tracks, -1).to(dtype)
            full = transition_log_probs(random_logits.to(dtype), visible)
            marginal = trace_log_marginals(full, initial)
            arrivals = first_hit_log_masses(full, initial)
            short = full[:, :prefix]
            assert torch.equal(marginal[:, :prefix], trace_log_marginals(short, initial))
            assert torch.equal(arrivals[:, :prefix], first_hit_log_masses(short, initial)[:, :-1])
            assert marginal.logsumexp(dim=-1).abs().max() < tolerance
            assert arrivals.logsumexp(dim=-1).abs().max() < tolerance

    invalid = (
        lambda: transition_log_probs(raw, mask([True])),
        lambda: trace_log_marginals(transition, born_closed.unsqueeze(0)),
        lambda: trace_nll(transition, born_closed, emissions, mask([True])),
        lambda: first_hit_log_masses(transition[:2], real([0.0, 0.0, 0.0, 1.0]).log()),
    )
    for operation in invalid:
        try:
            operation()
        except ValueError:
            continue
        raise AssertionError("An invalid model input was accepted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-only", action="store_true", help="Check the independent path oracle without Torch")
    args = parser.parse_args()
    verify_reference()
    if not args.reference_only:
        verify_torch()
    print("Independent trajectory checks passed." if not args.reference_only else "Reference enumerator checks passed.")


if __name__ == "__main__":
    main()
