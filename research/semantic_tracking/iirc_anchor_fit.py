"""Fit article-only localization of the visible span selected for an authored IIRC question."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
from time import perf_counter

import numpy as np

from research.stage2.records import write_jsonl
from research.stage3.probe import _jsonl
from research.stage3.run import MODEL, REVISION, save_json
from .iirc_prepare import digest


WORD = re.compile(r"[^\W_]+(?:[’'\-][^\W_]+)*")
MAX_WORDS = 8
TOP_SPANS = 5
METHODS = ('form', 'qwen')
FORM_FEATURES = (
    'log_words', 'log_characters', 'relative_start', 'relative_start_squared', 'initial_uppercase',
    'uppercase_word_share', 'digit_character_share', 'punctuation_character_share',
)
RIDGE = 0.002
MAX_ITERATIONS = 50


@dataclass(frozen=True)
class PassageDesign:
    words: np.ndarray
    word_tokens: np.ndarray
    endpoints: np.ndarray
    forms: np.ndarray


def passage_design(text, offsets):
    """Generate the common span support and its Unicode and token alignments from plain text."""
    words = np.asarray([match.span() for match in WORD.finditer(text)], dtype=np.int64).reshape(-1, 2)
    if not len(words):
        return PassageDesign(words, np.empty(0, dtype=np.int64), words.copy(),
                             np.empty((0, len(FORM_FEATURES)), dtype=np.float32))
    offsets = np.asarray(offsets)
    if (offsets.ndim != 2 or offsets.shape[1] != 2 or offsets.dtype.kind not in 'iu'
            or np.any(offsets[:, 0] < 0) or np.any(offsets[:, 1] > len(text))
            or np.any(offsets[:, 0] > offsets[:, 1])):
        raise ValueError('Expected half-open tokenizer offsets within the source text')
    retained = np.flatnonzero(offsets[:, 0] < offsets[:, 1])
    starts, ends = offsets[retained].T
    if np.any(np.diff(starts) < 0):
        raise ValueError('Expected tokenizer spans sorted by source start')
    last = np.searchsorted(starts, words[:, 1], side='left') - 1
    if np.any(last < 0) or np.any(ends[last] <= words[:, 0]):
        raise ValueError('A lexical run does not intersect any token')

    word_count = len(words)
    widths = np.minimum(MAX_WORDS, word_count - np.arange(word_count))
    left = np.repeat(np.arange(word_count), widths)
    right = left + np.arange(len(left)) - np.repeat(np.cumsum(widths) - widths, widths)
    endpoints = np.column_stack((left, right))
    start, end = words[left, 0], words[right, 1]
    word_width = right - left + 1
    char_width = end - start
    relative_start = start / max(len(text), 1)
    uppercase = np.fromiter((text[index].isupper() for index in words[:, 0]), dtype=np.int64)
    uppercase_prefix = np.r_[0, uppercase.cumsum()]
    digit_prefix = np.r_[0, np.fromiter(map(str.isdigit, text), dtype=np.int64).cumsum()]
    punct_prefix = np.r_[0, np.fromiter((not (c.isalnum() or c.isspace()) for c in text), dtype=np.int64).cumsum()]
    forms = np.column_stack((
        np.log1p(word_width) / np.log1p(MAX_WORDS), np.log1p(char_width) / 6.0, relative_start, relative_start ** 2,
        uppercase[left], (uppercase_prefix[right + 1] - uppercase_prefix[left]) / word_width,
        (digit_prefix[end] - digit_prefix[start]) / char_width, (punct_prefix[end] - punct_prefix[start]) / char_width,
    )).astype(np.float32)
    return PassageDesign(words, retained[last], endpoints, forms)


def candidate_spans(design):
    left, right = design.endpoints.T
    return np.column_stack((design.words[left, 0], design.words[right, 1]))


def candidate_logits(design, states, form, start=None, end=None):
    """Evaluate a checkpoint without importing Torch in the serving runtime."""
    logits = design.forms @ form
    if start is not None and end is not None:
        word_states = states[design.word_tokens]
        left, right = design.endpoints.T
        logits = logits + (word_states @ start)[left] + (word_states @ end)[right]
    return logits


def top_spans(design, logits):
    """Return the highest-probability spatially distinct spans, conditional on a response."""
    if not len(logits):
        return []
    if not np.isfinite(logits).all():
        raise ValueError('Expected finite span logits')
    spans = candidate_spans(design)
    probabilities = np.exp(np.asarray(logits, dtype=np.float64) - float(np.max(logits)))
    probabilities /= probabilities.sum()
    selected = []
    # Candidates are built in ascending (start, end) order, which also breaks ties.
    for index in np.argsort(-probabilities, kind='stable'):
        start, end = map(int, spans[index])
        if any(start < span['end'] and span['start'] < end for span in selected):
            continue
        selected.append({'start': start, 'end': end, 'score': float(probabilities[index])})
        if len(selected) == TOP_SPANS:
            break
    return selected


def predict_passage(text, offsets, states, weights):
    design = passage_design(text, offsets)
    return {
        'form': top_spans(design, candidate_logits(design, states, weights['form'])),
        'qwen': top_spans(design, candidate_logits(design, states, weights['qwen_form'],
                                                  weights['qwen_start'], weights['qwen_end'])),
    }


def read_stimuli(path):
    records = {}
    for row in _jsonl(path):
        identifier, text = row['id'], row['text']
        if (row.keys() != {'id', 'text', 'stops'} or not isinstance(identifier, str) or not identifier
                or identifier in records or not isinstance(text, str) or not text.strip() or row['stops'] != [len(text)]):
            raise ValueError('Expected unique paragraph identities with a single end-of-paragraph stop')
        records[identifier] = text
    if not records:
        raise ValueError('No stimulus paragraphs')
    return records


def read_labels(path, texts, evaluation_ids):
    reports = defaultdict(list)
    for row in _jsonl(path):
        paragraph, start, end = row['paragraph_id'], row['start'], row['end']
        if (row.keys() != {'paragraph_id', 'start', 'end'} or paragraph not in texts or paragraph in evaluation_ids
                or type(start) is not int or type(end) is not int or not 0 <= start < end <= len(texts[paragraph])):
            raise ValueError('Expected one valid training-only span for each authored question')
        reports[paragraph].append((start, end))
    if not reports:
        raise ValueError('No training labels')
    return reports


@dataclass(frozen=True)
class PackedDesign:
    forms: np.ndarray
    word_states: np.ndarray
    endpoints: np.ndarray
    lengths: np.ndarray
    labels: np.ndarray
    paragraph_label_counts: np.ndarray
    dispositions: dict[str, int]


def pack_training(texts, reports, feature_slices, offsets, states):
    forms, word_states, endpoints, lengths, labels, label_counts = [], [], [], [], [], []
    word_base, candidate_base = 0, 0
    unmatched, omitted_paragraphs = 0, 0
    for paragraph, spans in reports.items():
        feature_slice = feature_slices[paragraph]
        text = texts[paragraph]
        design = passage_design(text, offsets[feature_slice])
        candidates = {tuple(bounds): index for index, bounds in enumerate(candidate_spans(design))}
        matched = []
        for start, end in spans:
            source = text[start:end]
            bounds = start + len(source) - len(source.lstrip()), start + len(source.rstrip())
            index = candidates.get(bounds)
            if index is not None:
                matched.append(candidate_base + index)
        unmatched += len(spans) - len(matched)
        if not matched:
            omitted_paragraphs += 1
            continue
        forms.append(design.forms)
        word_states.append(states[feature_slice][design.word_tokens])
        endpoints.append(design.endpoints + word_base)
        lengths.append(len(candidates))
        labels.extend(matched)
        label_counts.append(len(matched))
        word_base += len(design.words)
        candidate_base += len(candidates)
    if not labels:
        raise ValueError('No training annotations align with the fixed lexical span support')
    return PackedDesign(np.concatenate(forms), np.concatenate(word_states), np.concatenate(endpoints),
                        np.asarray(lengths), np.asarray(labels), np.asarray(label_counts),
                        {'observed_reports': len(labels) + unmatched, 'used_reports': len(labels),
                         'excluded_span_out_of_support': unmatched, 'used_paragraphs': len(lengths),
                         'excluded_paragraphs_without_supported_reports': omitted_paragraphs})


def fit(design, device):
    import torch

    forms = torch.as_tensor(design.forms, device=device)
    word_states = torch.as_tensor(design.word_states, device=device)
    left, right = torch.as_tensor(design.endpoints.T, device=device)
    lengths = torch.as_tensor(design.lengths, device=device)
    group = torch.arange(len(lengths), device=device).repeat_interleave(lengths, output_size=len(forms))
    labels = torch.as_tensor(design.labels, device=device)
    counts = torch.as_tensor(design.paragraph_label_counts, dtype=forms.dtype, device=device)
    width = len(FORM_FEATURES)
    state_width = word_states.shape[1]
    weights, diagnostics = {}, {}
    for name in METHODS:
        use_states = name == 'qwen'
        parameter = torch.zeros(width + (2 * state_width if use_states else 0), device=device, requires_grad=True)
        optimizer = torch.optim.LBFGS([parameter], max_iter=MAX_ITERATIONS, line_search_fn='strong_wolfe',
                                       tolerance_grad=1e-5, tolerance_change=1e-7)

        def data_loss():
            logits = forms @ parameter[:width]
            if use_states:
                word_scores = word_states @ parameter[width:].view(2, state_width).T
                logits = logits + word_scores[:, 0][left] + word_scores[:, 1][right]
            maxima = torch.segment_reduce(logits.detach(), 'max', lengths=lengths, unsafe=True)
            total = torch.segment_reduce((logits - maxima[group]).exp(), 'sum', lengths=lengths, unsafe=True)
            log_normalizers = maxima + total.log()
            return (log_normalizers @ counts - logits[labels].sum()) / len(labels)

        def closure():
            optimizer.zero_grad()
            loss = data_loss() + 0.5 * RIDGE * parameter.square().sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        loss = closure()
        if not torch.isfinite(loss).item() or not torch.isfinite(parameter.grad).all().item():
            raise ArithmeticError(f'Nonfinite conditional span fit in {name}')
        vector = parameter.detach().cpu().numpy()
        diagnostics[name] = {'loss': loss.item(), 'gradient_max': parameter.grad.abs().max().item(),
                             'function_evaluations': optimizer.state[parameter]['func_evals']}
        if use_states:
            weights.update(qwen_form=vector[:width], qwen_start=vector[width:width + state_width],
                           qwen_end=vector[width + state_width:])
        else:
            weights['form'] = vector
        print(f'Fitted {name}: {diagnostics[name]}', flush=True)
    return weights, diagnostics


def run(stimuli, token_features, train_spans, evaluation, output, device):
    import torch
    from .qasper_fit import projection as fixed_projection

    start = perf_counter()
    texts = read_stimuli(stimuli)
    eval_texts = read_stimuli(evaluation)
    if any(texts.get(key) != text for key, text in eval_texts.items()):
        raise ValueError('Evaluation must be an exact plain-text subset of the combined stimuli')
    reports = read_labels(train_spans, texts, eval_texts.keys())
    extraction = token_features.with_name('extraction.json')
    provenance = json.loads(extraction.read_text(encoding='utf-8'))
    stimulus_sha, dense_sha = digest(stimuli), digest(token_features)
    if (provenance['model'] != MODEL or provenance['revision'] != REVISION
            or provenance['input_sha256'] != stimulus_sha
            or provenance['projected_tokens']['sha256'] != dense_sha):
        raise ValueError('Causal feature file and fixed model provenance do not match')
    with np.load(token_features, allow_pickle=False) as dense:
        if set(dense.files) != {'ids', 'indptr', 'offsets', 'states', 'projection'}:
            raise ValueError('Unexpected dense IIRC feature schema')
        identifiers, indptr = dense['ids'].tolist(), dense['indptr']
        offsets, states, projection = dense['offsets'], dense['states'], dense['projection']
    if (len(identifiers) != len(texts) or set(identifiers) != set(texts)
            or indptr.shape != (len(identifiers) + 1,) or indptr.dtype.kind not in 'iu'
            or indptr[0] != 0 or np.any(np.diff(indptr) <= 0) or indptr[-1] != len(states)
            or offsets.shape != (len(states), 2) or offsets.dtype.kind not in 'iu'
            or states.shape != (len(states), 128) or states.dtype != np.float32 or not np.isfinite(states).all()
            or not np.array_equal(projection, fixed_projection())):
        raise ValueError('Dense IIRC states disagree with the certified stimulus and projection inventory')
    feature_slices = {key: slice(indptr[index], indptr[index + 1]) for index, key in enumerate(identifiers)}
    design = pack_training(texts, reports, feature_slices, offsets, states)
    weights, diagnostics = fit(design, torch.device(device))
    dispositions = design.dispositions
    del design
    result = []
    for key, text in eval_texts.items():
        sl = feature_slices[key]
        result.append({'paragraph_id': key, 'methods': predict_passage(text, offsets[sl], states[sl], weights)})
    output.mkdir(parents=True, exist_ok=False)
    heads = output / 'heads.npz'
    predictions = output / 'predictions.jsonl'
    np.savez(heads, **weights, projection=projection)
    write_jsonl(predictions, result)
    save_json(output / 'fit-manifest.json', {
        'model': MODEL, 'revision': REVISION, 'stimuli_sha256': stimulus_sha, 'token_features_sha256': dense_sha,
        'train_spans_sha256': digest(train_spans), 'evaluation_sha256': digest(evaluation),
        'heads_sha256': digest(heads), 'predictions_sha256': digest(predictions), 'fit': diagnostics,
        'torch': torch.__version__, 'device': device, 'ridge': RIDGE, 'maximum_iterations': MAX_ITERATIONS,
        'methods': list(METHODS), 'max_words': MAX_WORDS, 'form_features': list(FORM_FEATURES),
        'top_spans': TOP_SPANS, 'suppression': 'greedy exact character nonoverlap, ties ascending source offsets',
        'probability_scope': 'One selected lexical span conditional on an observed authored question and in-support target',
        'training': dispositions, 'evaluation_paragraphs': len(eval_texts), 'elapsed_s': perf_counter() - start,
    })


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('stimuli', 'token-features', 'train-spans', 'evaluation', 'output'):
        parser.add_argument(f'--{field}', type=Path, required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    if args.device == 'cuda' and (torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != 'NVIDIA L4'):
        raise RuntimeError('Expected one NVIDIA L4 CUDA device')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    run(**vars(args))


if __name__ == '__main__':
    main()
