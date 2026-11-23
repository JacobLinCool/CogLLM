"""Fit the frozen-feature QASPER 2×2 and seal question-conditioned forecasts."""

import argparse
from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from research.core_features.fit import canonical_stops, write_rows
from research.stage3.probe import _jsonl
from research.stage3.run import save_json
from .joint_model import interval_nll
from .qasper_fit import (
    PENALTY, PROJECTION_SEED, STEPS, WIDTH, assemble, digest, load_states, pack, projection, unit, validate,
)


RANK_WEIGHT = 0.5
METHODS = ('position', 'question_only', 'pooled_interval', 'pooled_rank', 'token_interval', 'token_rank')
LOCAL_START = 1 + 3 + 3 * WIDTH
LOCAL_END = LOCAL_START + WIDTH
BODY_PARAMETERS = LOCAL_END + 2
SQRT_WIDTH = np.sqrt(np.float32(WIDTH))


def load_token_states(path, paragraph_file, documents, matrix, paper_ids):
    """Map each paper to projected states and exclusive causal stop indices."""
    with np.load(paragraph_file, allow_pickle=False) as paragraphs:
        keys = list(zip(paragraphs['ids'].tolist(), paragraphs['stops'].tolist(), strict=True))
    with np.load(path, allow_pickle=False) as dense:
        if set(dense.files) != {'ids', 'indptr', 'offsets', 'stop_counts', 'states', 'projection'}:
            raise ValueError('Unexpected dense QASPER feature fields')
        identifiers = dense['ids'].tolist()
        indptr, offsets, states = dense['indptr'], dense['offsets'], dense['states']
        stop_counts = dense['stop_counts']
        if not np.array_equal(dense['projection'], matrix):
            raise ValueError('Dense and pooled states must use the same frozen projection')
    if len(identifiers) != len(paper_ids) or set(identifiers) != paper_ids:
        raise ValueError('Dense features must cover every paper once')
    if (indptr.shape != (len(identifiers) + 1,) or indptr.dtype.kind not in 'iu' or indptr[0] != 0
            or np.any(indptr[1:] <= indptr[:-1]) or indptr[-1] != len(states)
            or offsets.shape != (len(states), 2) or offsets.dtype.kind not in 'iu'
            or states.shape != (len(states), WIDTH) or states.dtype != np.float32
            or stop_counts.shape != (len(keys),) or stop_counts.dtype.kind not in 'iu'):
        raise ValueError('Invalid ragged token geometry or projected state array')
    lengths = np.linalg.norm(states, axis=-1)
    if not np.all(np.isfinite(lengths) & (np.abs(lengths - 1.0) < 2e-5)):
        raise ValueError('Expected finite unit-normalized projected token states')

    counts = dict(zip(keys, stop_counts.tolist(), strict=True))
    result = {}
    for index, paper in enumerate(identifiers):
        left, right = indptr[index:index + 2]
        spans = offsets[left:right]
        char_stops = np.asarray(documents[paper]['stops'])
        stops = np.array([counts[paper, end] for end in char_stops], dtype=np.int64)
        if (stops[0] <= 0 or np.any(np.diff(stops) <= 0) or stops[-1] != len(spans)
                or np.any(spans[:, 0] < 0) or np.any(spans[:, 0] > spans[:, 1])
                or np.any(spans[:, 1] > len(documents[paper]['text']))
                or np.any(np.maximum.accumulate(spans[:, 1])[stops - 1] > char_stops)):
            raise ValueError(f'Token stops must expose exactly the historical prefix for {paper!r}')
        result[paper] = states[left:right], stops
    return result


def question_features(documents, means, matrix, rows):
    result = {}
    for row in rows:
        question = row['question_id']
        stop, = documents[question]['stops']
        result[question] = unit(means[question, stop] @ matrix)
    return result


def question_only_features(geometries, rows, questions, center, scale):
    """Use the supplied question and paragraph clock with no article text or lengths."""
    clocks = [geometry[:, :1] for geometry in geometries]
    features = []
    for clock, row in zip(clocks, rows, strict=True):
        q = questions[row['question_id']]
        static = np.broadcast_to(q, (len(clock), WIDTH))
        normalized_clock = (clock - center[0]) / scale[0]
        features.append(np.column_stack((static, normalized_clock * static)))
    return pack(clocks, features, center[:1], scale[:1])


def token_interactions(states, question):
    """Coordinatewise interaction between unit token states and a unit question vector."""
    return states * question * SQRT_WIDTH


@dataclass(frozen=True)
class TokenDesign:
    interactions: torch.Tensor
    lengths: torch.Tensor
    group: torch.Tensor
    paragraph_slots: torch.Tensor

    def log_mean_exp(self, weights, paragraph_shape):
        scores = self.interactions @ weights
        maxima = torch.segment_reduce(scores.detach(), 'max', lengths=self.lengths, unsafe=True)
        shifted = torch.exp(scores - maxima[self.group])
        masses = torch.segment_reduce(shifted, 'sum', lengths=self.lengths, unsafe=True)
        values = maxima + torch.log(masses / self.lengths)
        return torch.zeros(paragraph_shape, device=scores.device, dtype=scores.dtype).flatten().index_copy(
            0, self.paragraph_slots, values,
        ).view(paragraph_shape)


def token_design(rows, paper_tokens, questions, paragraphs, device, *, training):
    interactions, lengths, placements = [], [], []
    for index, row in enumerate(rows):
        states, stops = paper_tokens[row['paper_id']]
        if training:
            stops = stops[:row['upper']]
        interactions.append(token_interactions(states[:stops[-1]], questions[row['question_id']]))
        lengths.extend(np.diff(stops, prepend=0))
        placements.extend(index * paragraphs + np.arange(len(stops)))
    sizes = np.asarray(lengths, dtype=np.int64)
    groups = np.repeat(np.arange(len(sizes)), sizes)
    return TokenDesign(torch.as_tensor(np.concatenate(interactions), device=device),
                       torch.as_tensor(sizes, device=device), torch.as_tensor(groups, device=device),
                       torch.as_tensor(placements, dtype=torch.int64, device=device))


def paragraph_logits(weights, design, tokens=None):
    if tokens is None:
        return design @ weights
    # Match the pooled head's parameter count by replacing its local question–segment term.
    common = design[..., :LOCAL_START] @ weights[:LOCAL_START] + design[..., LOCAL_END:] @ weights[LOCAL_END:]
    return common + tokens.log_mean_exp(weights[LOCAL_START:LOCAL_END], common.shape)


def same_cutoff_pairs(train_rows):
    by_paper = {}
    for index, row in enumerate(train_rows):
        by_paper.setdefault(row['paper_id'], []).append((row['upper'], row['question_id'], index))
    result = [(early_index, late_index, early_stop - 1)
              for questions in by_paper.values()
              for (early_stop, _, early_index), (late_stop, _, late_index) in combinations(sorted(questions), 2)
              if early_stop < late_stop]
    if not result:
        raise ValueError('Expected strictly ordered, same-paper training pairs')
    return np.asarray(result, dtype=np.int64)


def ranking_nll(logits, pairs):
    """Negative log-probability of the observed evidence order at one shared cutoff."""
    early, late, cutoff = pairs.unbind(-1)
    cumulative_intensity = F.softplus(logits.double()).cumsum(dim=-1)
    selected = cumulative_intensity[early, cutoff], cumulative_intensity[late, cutoff]
    # If F = 1 - exp(-a), logit(F) = a + log(1 - exp(-a)).
    odds = [a + torch.log(-torch.expm1(-a)) for a in selected]
    return F.softplus(odds[1] - odds[0]).mean()


def fit(design, bounds, pairs=None, tokens=None):
    weights = torch.zeros(design.shape[-1], dtype=design.dtype, device=design.device, requires_grad=True)
    lower, upper = bounds.unbind(-1)
    mean_position = upper.double().mean().item()
    with torch.no_grad():
        weights[0] = -float(np.log(mean_position - 1)) if mean_position > 1 else 0
    optim = torch.optim.LBFGS([weights], max_iter=STEPS, line_search_fn='strong_wolfe',
                              tolerance_grad=1e-7, tolerance_change=1e-9)

    def losses():
        logits = paragraph_logits(weights, design, tokens)
        nll = interval_nll(logits, lower, upper).mean()
        rank = ranking_nll(logits, pairs) if pairs is not None else logits.new_zeros(())
        return nll, rank

    def closure():
        optim.zero_grad()
        nll, rank = losses()
        loss = nll + RANK_WEIGHT * rank + PENALTY * weights[1:].square().sum() / 2
        loss.backward()
        return loss

    optim.step(closure)
    cost = closure()
    with torch.no_grad():
        nll, rank = losses()
    if not torch.isfinite(cost).item() or not torch.isfinite(weights.grad).all().item():
        raise ArithmeticError('Nonfinite factorial hazard fit')
    return weights.detach(), {
        'train_interval_nll_nats': nll.item(), 'train_rank_nll_nats': rank.item() if pairs is not None else None,
        'objective': cost.item(), 'gradient_max': weights.grad.abs().max().item(), 'parameters': weights.numel(),
        'lbfgs_iterations': optim.state[weights]['n_iter'], 'function_evaluations': optim.state[weights]['func_evals'],
    }


def run(stimuli, features, token_features, train, evaluation, output, *, device):
    started = perf_counter()
    documents = canonical_stops(stimuli)
    train_rows, eval_rows = list(_jsonl(train)), list(_jsonl(evaluation))
    training_papers, heldout_papers = validate(train_rows, eval_rows, documents)
    extraction = json.loads(features.with_name('extraction.json').read_text(encoding='utf-8'))
    if extraction['input_sha256'] != digest(stimuli):
        raise ValueError('Frozen states were extracted from a different stimulus file')
    token_sha = digest(token_features)
    if extraction.get('projected_tokens', {}).get('sha256') != token_sha:
        raise ValueError('Dense states differ from the extraction manifest')
    states, means = load_states(features, documents)
    matrix = projection()
    paper_tokens = load_token_states(token_features, features, documents, matrix, training_papers | heldout_papers)
    questions = question_features(documents, means, matrix, train_rows + eval_rows)
    tr_pos, tr_body = assemble(train_rows, documents, states, means, matrix)
    ev_pos, ev_body = assemble(eval_rows, documents, states, means, matrix)
    geometry = np.concatenate([rows[:row['upper']] for row, rows in zip(train_rows, tr_pos, strict=True)])
    center, scale = geometry.mean(0), np.maximum(0.1, geometry.std(0))

    bounds = torch.as_tensor([[row['lower'], row['upper']] for row in train_rows], device=device)
    pairs = torch.as_tensor(same_cutoff_pairs(train_rows), device=device)
    tr_packed = torch.as_tensor(pack(tr_pos, tr_body, center, scale), device=device)
    ev_packed = torch.as_tensor(pack(ev_pos, ev_body, center, scale), device=device)
    if tr_packed.shape[-1] != BODY_PARAMETERS or ev_packed.shape[-1] != BODY_PARAMETERS:
        raise ValueError('Unexpected pooled design width')
    tr_tokens = token_design(train_rows, paper_tokens, questions, tr_packed.shape[1], device, training=True)
    ev_tokens = token_design(eval_rows, paper_tokens, questions, ev_packed.shape[1], device, training=False)
    tr_questions = torch.as_tensor(question_only_features(tr_pos, train_rows, questions, center, scale), device=device)
    ev_questions = torch.as_tensor(question_only_features(ev_pos, eval_rows, questions, center, scale), device=device)

    designs = (
        (tr_packed[..., :4], ev_packed[..., :4], None, None, None),
        (tr_questions, ev_questions, None, None, None),
        (tr_packed, ev_packed, None, None, None),
        (tr_packed, ev_packed, pairs, None, None),
        (tr_packed, ev_packed, None, tr_tokens, ev_tokens),
        (tr_packed, ev_packed, pairs, tr_tokens, ev_tokens),
    )
    result = [{**row, 'stops': documents[row['paper_id']]['stops']} for row in eval_rows]
    fits, checkpoints = {}, {}
    output.mkdir(parents=True, exist_ok=False)
    for name, (tr_design, ev_design, pair_indices, train_tokens, eval_tokens) in zip(METHODS, designs, strict=True):
        weights, fits[name] = fit(tr_design, bounds, pair_indices, train_tokens)
        checkpoints[name] = weights.cpu().numpy()
        with torch.inference_mode():
            hazards = torch.sigmoid(paragraph_logits(weights, ev_design, eval_tokens).double()).cpu().numpy()
            if not np.all(np.isfinite(hazards) & (0 < hazards) & (hazards < 1)):
                raise ArithmeticError(f'Expected finite interior hazards for {name}')
        for row, estimates in zip(result, hazards, strict=True):
            row[name] = estimates[:len(row['stops'])].tolist()
        print(f'Fitted QASPER {name}', flush=True)

    predictions, heads = output / 'predictions.jsonl', output / 'heads.npz'
    write_rows(predictions, result)
    np.savez(heads, **checkpoints, projection=matrix, position_mean=center, position_scale=scale)
    save_json(output / 'fit-manifest.json', {
        'device': str(device), 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'torch': torch.__version__, 'predictions_sha256': digest(predictions), 'heads_sha256': digest(heads),
        'train_questions': len(train_rows), 'heldout_questions': len(eval_rows),
        'train_papers': len(training_papers), 'heldout_papers': len(heldout_papers),
        'train_same_cutoff_pairs': len(pairs), 'same_cutoff_pair_reduction': 'all strict pairs equally weighted',
        'methods': fits, 'fit_wall_s': perf_counter() - started, 'max_lbfgs_steps': STEPS, 'penalty': PENALTY,
        'rank_weight': RANK_WEIGHT, 'projection_seed': PROJECTION_SEED, 'token_log_mean_exp_temperature': 1.0,
        'token_readout': 'Current revealed segment, including initial title/abstract context; normalized log-sum-exp '
                         'of learned question–token scores replaces the pooled local interaction at each paragraph stop. '
                         'Contextual endpoint terms are shared by design; earlier segments enter cumulative hazard.',
        'question_only_readout': 'Paragraph index and fixed question projection, with a question × index interaction.',
        'source_sha256': {
            'stimuli': digest(stimuli), 'train': digest(train), 'eval': digest(evaluation),
            'features': digest(features), 'token_features': token_sha,
            'extraction': digest(features.with_name('extraction.json')),
        },
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('stimuli', 'features', 'token_features', 'train', 'evaluation', 'output'):
        parser.add_argument(f'--{field.replace("_", "-")}', type=Path, required=True)
    args = parser.parse_args()
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != 'NVIDIA L4':
        raise RuntimeError('Expected one NVIDIA L4 for the planned fitting run')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    run(**vars(args), device=torch.device('cuda'))


if __name__ == '__main__':
    main()
