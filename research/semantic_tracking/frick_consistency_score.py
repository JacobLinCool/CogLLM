"""Cross-fit a frozen causal readout for Frick and Schüler consistency reports."""

import argparse
from collections import Counter, defaultdict
from hashlib import file_digest, sha256
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from unicodedata import normalize

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit

from research.semantic_tracking.gum_bridge_extract import CANDIDATE, TOKENIZER_SHA256
from research.semantic_tracking.rodger_belief_readout import block_ranges, feature_matrix, standardization


RIDGE = 0.10
WIDTH = 2048
METHODS = ('surface', 'standalone', 'contextual')
TASKS = ('current', 'first')
PROTOCOL = Path(__file__).with_name('frick-consistency-protocol.md')
CHECKPOINT = Path(__file__).resolve().parents[1] / 'results' / 'gradio-candidate-checkpoint' / 'checkpoint.json'
CONDITIONS = ('01', '02')
STORIES = 20
READERS = 180
TRIALS = 1800
SCALAR_WIDTH = 5
SCALAR_FEATURES = ('log1p_sentence_characters', 'log1p_sentence_words', 'log1p_sentence_ordinal',
                   'log1p_prior_words', 'prior_overlap_of_current_types')
MAX_PREFIX_ERROR = 0.002
DRAWS = 10_000
SEED = 735972
MARGIN = 0.10
TYPE = re.compile(r'[^\W_]+')


def binary_loss(parameters, design, hit, miss):
    intercept, coefficients = parameters[0], parameters[1:]
    logits = intercept + design @ coefficients
    loss = hit @ np.logaddexp(0, -logits) + miss @ np.logaddexp(0, logits)
    loss += RIDGE * 0.5 * (coefficients @ coefficients)
    residual = miss * expit(logits) - hit * expit(-logits)
    gradient = np.concatenate(([residual.sum()], design.T @ residual + RIDGE * coefficients))
    return float(loss), gradient


def fit(design, hits, trials, blocks=()):
    """Fit an exact grouped Bernoulli likelihood with training-only feature scales."""
    matrix = feature_matrix(design)
    hits, trials = map(np.asarray, (hits, trials))
    if (hits.shape != (len(matrix),) or trials.shape != hits.shape or hits.dtype.kind not in 'iu'
            or trials.dtype.kind not in 'iu' or np.any(trials < 0) or np.any(hits < 0) or np.any(hits > trials)):
        raise ValueError('Group counts must be aligned integers with 0 <= hits <= trials')
    keep = trials > 0
    if not np.all(keep):
        matrix, hits, trials = matrix[keep], hits[keep], trials[keep]
    total = float(trials.sum(dtype=np.float64))
    positives = float(hits.sum(dtype=np.float64))
    if not 0 < positives < total:
        raise ValueError('A Bernoulli fit with a free intercept needs both outcomes')
    hit = hits / total
    miss = (trials - hits) / total
    standardized, center, scale = standardization(matrix, trials / total, block_ranges(matrix.shape[1], blocks))
    initial = np.zeros(matrix.shape[1] + 1)
    initial[0] = np.log(positives) - np.log(total - positives)
    result = minimize(binary_loss, initial, args=(standardized, hit, miss), jac=True, method='L-BFGS-B',
                      options={'maxiter': 500, 'ftol': 1e-13, 'gtol': 1e-9})
    if not result.success or not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise RuntimeError(f'Bernoulli readout failed to converge: {result.message}')
    return {'center': center, 'scale': scale, 'intercept': float(result.x[0]), 'coefficients': result.x[1:],
            'iterations': int(result.nit), 'train_rows': len(matrix), 'train_trials': int(total),
            'train_hits': int(positives), 'regularized_nll_nats': float(result.fun)}


def predict_logits(design, fitted):
    matrix = feature_matrix(design)
    center, scale, coefficients = (np.asarray(fitted[key], dtype=np.float64)
                                   for key in ('center', 'scale', 'coefficients'))
    if (any(item.shape != (matrix.shape[1],) or not np.isfinite(item).all()
            for item in (center, scale, coefficients)) or np.any(scale <= 0) or not np.isfinite(fitted['intercept'])):
        raise ValueError('Frozen Bernoulli readout and design do not align')
    logits = fitted['intercept'] + ((matrix - center) / scale) @ coefficients
    if not np.isfinite(logits).all():
        raise ValueError('Frozen Bernoulli readout produced nonfinite logits')
    return logits


def digest(path):
    with path.open('rb') as handle:
        return file_digest(handle, 'sha256').hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_rows(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle]


def write_json(path, value):
    with path.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def indexed(rows, filename):
    lookup = {row['id']: row for row in rows}
    if len(lookup) != len(rows) or not rows or any(type(key) is not str or not key for key in lookup):
        raise ValueError(f'Invalid or duplicate stimulus identifier in {filename}')
    return lookup


def load_layout(assay):
    audit_path = assay / 'source-audit.json'
    audit = read_json(audit_path)
    for name in ('stimuli.jsonl', 'metadata.jsonl'):
        if digest(assay / name) != audit['output_sha256'][name]:
            raise ValueError(f'Frick source audit checksum mismatch for {name}')
    stimuli = indexed(read_rows(assay / 'stimuli.jsonl'), 'stimuli.jsonl')
    metadata = indexed(read_rows(assay / 'metadata.jsonl'), 'metadata.jsonl')
    if len(metadata) != 2 * STORIES or not metadata.keys() <= stimuli.keys():
        raise ValueError('Expected all 40 two-condition story versions')
    families, controls, texts = defaultdict(dict), set(), {}
    for key, info in metadata.items():
        story, condition = info['story'], info['condition']
        count = len(info['sentences'])
        body, stops = stimuli[key]['text'], stimuli[key]['stops']
        if (condition not in CONDITIONS or condition in families[story] or count < 2
                or not 0 <= info['changed_index'] < info['target_index'] < count
                or [item['stop'] for item in info['sentences']] != stops or stops[-1] != len(body)):
            raise ValueError(f'Invalid story version or annotated display layout for {key}')
        families[story][condition] = key
        left, sentences = 0, []
        for i, sentence in enumerate(info['sentences']):
            start, stop, control = sentence['start'], sentence['stop'], sentence['single_id']
            if (not (start == 0 if i == 0 else left < start and body[left:start].isspace())
                    or not left <= start < stop <= len(body) or control in metadata or control not in stimuli):
                raise ValueError(f'Invalid original display sentence coordinates for {key}')
            content = body[start:stop]
            if stimuli[control]['text'] != content or stimuli[control]['stops'] != [len(content)]:
                raise ValueError(f'Standalone control does not match its original displayed sentence in {key}')
            controls.add(control)
            sentences.append(content)
            left = stop
        texts[key] = tuple(sentences)
    if (len(families) != STORIES or any(set(pair) != set(CONDITIONS) for pair in families.values())
            or controls != stimuli.keys() - metadata.keys()
            or len({stimuli[key]['text'] for key in controls}) != len(controls)):
        raise ValueError('Incomplete story pairs or duplicate standalone sentences')
    for pair in families.values():
        left_id, right_id = map(pair.__getitem__, CONDITIONS)
        left, right = metadata[left_id], metadata[right_id]
        change, target = left['changed_index'], left['target_index']
        differences = [i for i, (a, b) in enumerate(zip(texts[left_id], texts[right_id], strict=True)) if a != b]
        if (change != right['changed_index'] or target != right['target_index'] or differences != [change]
                or stimuli[left_id]['text'][:left['sentences'][change]['start']]
                != stimuli[right_id]['text'][:right['sentences'][change]['start']]):
            raise ValueError('A registered story pair changes more than its single earlier sentence')
    return audit, stimuli, metadata, dict(families), texts


def word_types(sentence):
    return {token for token in TYPE.findall(normalize('NFKC', sentence).casefold()) if len(token) >= 4}


def load_states(assay, extraction, packet, protocol, stimuli, metadata, families):
    source = assay / 'source-audit.json'
    stimulus_sha = digest(assay / 'stimuli.jsonl')
    packet_path, preflight_path = packet / 'preflight.json', packet / 'extractor-preflight.json'
    packet_receipt, preflight = read_json(packet_path), read_json(preflight_path)
    stops = {key: row['stops'] for key, row in stimuli.items()}
    if (packet_receipt['source_audit_sha256'] != digest(source) or packet_receipt['candidate_id'] != CANDIDATE
            or packet_receipt['documents'] != stops
            or any(row['tokenizer_sha256'] != TOKENIZER_SHA256 or row['stimuli_sha256'] != stimulus_sha
                   for row in (packet_receipt, preflight))
            or {row['id']: row['stops'] for row in preflight['documents']} != stops
            or preflight['certified_stops'] != sum(map(len, stops.values()))
            or preflight['input_documents'] != len(stops) or preflight['encoded_documents'] != len(stops)):
        raise ValueError('Candidate, source and preflight do not certify the complete stimulus endpoint inventory')

    seal_path = extraction / 'seal.json'
    seal = read_json(seal_path)
    checkpoint = read_json(CHECKPOINT)
    backbone = checkpoint['backbone']
    anchor = CHECKPOINT.parent / 'anchor' / 'heads.npz'
    if checkpoint['checkpoint_sha256'] != CANDIDATE or digest(anchor) != checkpoint['files']['anchor/heads.npz']:
        raise ValueError('Unexpected frozen app checkpoint or reference projection weights')
    with np.load(anchor, allow_pickle=False) as weights:
        projection_sha = sha256(weights['projection'].tobytes(order='C')).hexdigest()
    scope = {'assay_sha256': digest(packet_path), 'preflight_sha256': digest(preflight_path),
             'protocol_sha256': digest(protocol), 'stimuli_sha256': stimulus_sha, 'candidate_id': CANDIDATE,
             'tokenizer_sha256': TOKENIZER_SHA256, 'anchor_heads_sha256': checkpoint['files']['anchor/heads.npz'],
             'projection_sha256': projection_sha, 'content_forward_per_document': 1, 'states_dtype': 'float32',
             **{key: backbone[key] for key in ('model', 'revision', 'transformers')}}
    if any(seal[key] != value for key, value in scope.items()):
        raise ValueError('State extraction is not bound to this exact assay, protocol and frozen app candidate')
    artifacts = ('coordinates.jsonl', 'states.npy', 'causality.json')
    for name in artifacts:
        if digest(extraction / name) != seal['file_sha256'][name]:
            raise ValueError(f'Frozen extraction digest mismatch for {name}')
    checks = read_json(extraction / 'causality.json')['checks']
    if len(checks) != 12 or not all(row['passed'] and row['projected_tokens']['passed'] for row in checks):
        raise ValueError('The archived causal prefix checks did not pass')
    coordinates = indexed(read_rows(extraction / 'coordinates.jsonl'), 'coordinates.jsonl')
    if (any(set(row) != {'id', 'stops'} for row in coordinates.values())
            or {key: row['stops'] for key, row in coordinates.items()} != stops):
        raise ValueError('An expected exact sentence-end coordinate was excluded')
    h = np.load(extraction / 'states.npy', mmap_mode='r', allow_pickle=False)
    expected_shape = (sum(map(len, stops.values())), WIDTH)
    if (h.shape != expected_shape or seal['states_shape'] != list(expected_shape) or h.dtype != np.float32
            or not np.isfinite(h).all()):
        raise ValueError('Frozen states must exactly align with all certified endpoints as FP32 rows')
    by_id, offset = {}, 0
    for key, row in coordinates.items():
        n = len(row['stops'])
        by_id[key] = h[offset:offset + n]
        offset += n
    errors = []
    for pair in families.values():
        left, right = map(pair.__getitem__, CONDITIONS)
        count = metadata[left]['changed_index']
        reference = by_id[left][:count].astype(np.float64)
        compared = by_id[right][:count].astype(np.float64)
        norm = np.linalg.norm(reference, axis=1)
        if np.any(norm <= 0):
            raise ValueError('Malformed frozen reference state at a shared causal prefix')
        errors.extend(np.linalg.norm(reference - compared, axis=1) / norm)
    if not errors:
        raise ValueError('The study does not contain any same-prefix causal checks')
    maximum = float(max(errors))
    if maximum >= MAX_PREFIX_ERROR:
        raise ValueError(f'Matched prefixes before the changed sentence differ by {maximum:.6f}')
    return by_id, {**scope, 'state_extraction_sha256': digest(seal_path), 'source_audit_sha256': digest(source),
                   'shared_prechange_prefixes': len(errors), 'maximum_relative_l2': maximum,
                   'extractor_causality_checks': len(checks), 'states_shape': list(expected_shape)}


def design_rows(stimuli, metadata, texts, states):
    normalized = {}
    for key, rows in states.items():
        rows = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(rows, axis=1)
        if not np.all(np.isfinite(norms) & (norms > 0)):
            raise ValueError(f'Invalid sentence state norm in {key}')
        normalized[key] = rows / norms[:, None]
    total = sum(len(info['sentences']) - 1 for info in metadata.values())
    matrix = np.empty((total, SCALAR_WIDTH + 2 * WIDTH), dtype=np.float32)
    slots, row = {}, 0
    for key, info in sorted(metadata.items()):
        start = row
        prior_words, prior_types = 0, set()
        for j, (text, sentence) in enumerate(zip(texts[key], info['sentences'], strict=True)):
            words = len(text.split())
            types = word_types(text)
            if j > 0:
                overlap = len(prior_types & types) / len(types) if types else 0.0
                matrix[row, :SCALAR_WIDTH] = (*np.log1p((len(text), words, j + 1, prior_words)), overlap)
                local = normalized[sentence['single_id']][0]
                contextual = normalized[key][j]
                matrix[row, SCALAR_WIDTH:SCALAR_WIDTH + WIDTH] = local
                matrix[row, SCALAR_WIDTH + WIDTH:] = contextual
                row += 1
            prior_words += words
            prior_types.update(types)
        slots[key] = slice(start, row)
        if row - start != len(stimuli[key]['stops']) - 1:
            raise RuntimeError('Sentence design does not agree with verified source boundaries')
    return matrix, slots


def load_outcomes(assay, audit, metadata, slots):
    labels = assay / 'outcomes.jsonl'
    if digest(labels) != audit['output_sha256'][labels.name]:
        raise ValueError('Frick outcomes checksum does not match the registered source')
    rows = read_rows(labels)
    size = sum(slot.stop - slot.start for slot in slots.values())
    hits = np.zeros((len(TASKS), size), dtype=np.int64)
    totals = np.zeros_like(hits)
    seen, reader_conditions, variant_trials = set(), Counter(), Counter()
    target_by_condition = {condition: Counter() for condition in CONDITIONS}
    trials = []
    for row in rows:
        key, reader = row['id'], row['reader_id']
        info = metadata[key]
        identity = reader, info['story']
        condition = info['condition']
        keys, times = row['keys'], row['rt_seconds']
        if (identity in seen or not type(reader) is str or not reader or len(keys) != len(info['sentences'])
                or len(times) != len(keys) or any(item not in ('k', 'd') for item in keys)
                or any(type(value) not in (int, float) or not np.isfinite(value) or value < 0 for value in times)):
            raise ValueError('Unexpected duplicate reader/story or incomplete raw judgment trajectory')
        seen.add(identity)
        reader_conditions[reader, condition] += 1
        variant_trials[key] += 1
        observed = np.array([item == 'd' for item in keys[1:]])
        first = int(observed.argmax()) if observed.any() else None
        slot = slots[key]
        hits[0, slot] += observed
        totals[0, slot] += 1
        if first is None:
            totals[1, slot] += 1
        else:
            totals[1, slot.start:slot.start + first + 1] += 1
            hits[1, slot.start + first] += 1
        target = info['target_index']
        target_hit = keys[target] == 'd'
        included = times[target] * 1000 + 100 >= 500
        target_by_condition[condition].update(uncut_n=1, uncut_d=target_hit,
                                             author_filtered_n=included, author_filtered_d=included and target_hit)
        trials.append((key, reader, observed, first))
    readers = sorted({reader for reader, _ in reader_conditions})
    if (len(rows) != TRIALS or len(readers) != READERS or set(variant_trials) != metadata.keys()
            or reader_conditions != {(reader, condition): 5 for reader in readers for condition in CONDITIONS}):
        raise ValueError('Prepared outcomes do not cover all 180 readers, 40 versions and 1,800 traces')
    inventory = {'stories': len(metadata) // 2, 'variants': len(metadata), 'readers': len(readers),
                 'trials': len(trials), 'current_judgments': int(totals[0].sum()),
                 'min_trials_per_variant': min(variant_trials.values()),
                 'max_trials_per_variant': max(variant_trials.values()),
                 'first_at_risk_judgments': int(totals[1].sum()), 'first_d': int(hits[1].sum()),
                 'right_censored': len(trials) - int(hits[1].sum()),
                 'target_observations': {key: dict(value) for key, value in target_by_condition.items()}}
    if inventory['current_judgments'] != 26_340:
        raise ValueError('Prepared outcomes do not contain the registered post-first-sentence judgment count')
    return trials, readers, hits, totals, inventory


def crossfit(matrix, slots, families, hits, totals):
    logits = np.full((len(TASKS), len(METHODS), len(matrix)), np.nan)
    fits, weights = [], {}
    scalar = matrix[:, :SCALAR_WIDTH]
    standalone = matrix[:, :SCALAR_WIDTH + WIDTH]
    contextual = np.column_stack((scalar, matrix[:, SCALAR_WIDTH + WIDTH:]))
    for fold_number, (story, pair) in enumerate(sorted(families.items())):
        held_out = np.zeros(len(matrix), dtype=bool)
        for key in pair.values():
            held_out[slots[key]] = True
        for t, task in enumerate(TASKS):
            for m, (method, design) in enumerate(zip(METHODS, (scalar, standalone, contextual), strict=True)):
                dim = design.shape[1]
                blocks = tuple((start, start + WIDTH) for start in range(SCALAR_WIDTH, dim, WIDTH))
                fitted = fit(design[~held_out], hits[t, ~held_out], totals[t, ~held_out], blocks)
                logits[t, m, held_out] = predict_logits(design[held_out], fitted)
                prefix = f'fold{fold_number}_{task}_{method}'
                for field in ('center', 'scale', 'coefficients'):
                    weights[f'{prefix}_{field}'] = fitted.pop(field)
                fits.append({'held_out_story': story, 'task': task, 'method': method, 'weights_prefix': prefix,
                             'vector_blocks': blocks, **fitted})
    if not np.isfinite(logits).all():
        raise ValueError('Not every held-out sentence received both task probabilities')
    return logits, fits, weights


def first_distribution(logits):
    """Log probability of each first event and of no event through the last sentence."""
    log_no_event = -np.logaddexp(0, logits)
    survival = np.cumsum(log_no_event, axis=1)
    prior_survival = np.column_stack((np.zeros(len(logits)), survival[:, :-1]))
    first = prior_survival - np.logaddexp(0, -logits)
    complete = np.column_stack((first, survival[:, -1]))
    if not np.allclose(np.exp(complete).sum(axis=1), 1, rtol=0, atol=1e-12):
        raise ValueError('First-event probabilities and the survivor mass must sum to one')
    return complete


def write_forecasts(path, logits, slots, stimuli):
    with path.open('x', encoding='utf-8') as handle:
        for key, slot in slots.items():
            for row, stop in zip(range(slot.start, slot.stop), stimuli[key]['stops'][1:], strict=True):
                record = {'id': key, 'stop': stop, 'methods': {}}
                for m, method in enumerate(METHODS):
                    current, hazard = logits[:, m, row]
                    record['methods'][method] = {'current_logit': float(current), 'first_logit': float(hazard),
                                                 'current_d_probability': float(expit(current)),
                                                 'first_d_hazard_probability': float(expit(hazard))}
                handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')


def read_forecasts(path, slots, stimuli):
    size = sum(slot.stop - slot.start for slot in slots.values())
    logits = np.full((len(TASKS), len(METHODS), size), np.nan)
    rows = read_rows(path)
    expected = ((key, stop) for key in slots for stop in stimuli[key]['stops'][1:])
    for position, (row, identity) in enumerate(zip(rows, expected, strict=True)):
        if (set(row) != {'id', 'stop', 'methods'} or (row['id'], row['stop']) != identity
                or set(row['methods']) != set(METHODS)):
            raise ValueError('Held-out probability records do not match the certified sentence stops')
        for m, method in enumerate(METHODS):
            entries = row['methods'][method]
            if (set(entries) != {'current_logit', 'first_logit', 'current_d_probability', 'first_d_hazard_probability'}
                    or expit(entries['current_logit']) != entries['current_d_probability']
                    or expit(entries['first_logit']) != entries['first_d_hazard_probability']):
                raise ValueError('Stored forecast is missing a primary task')
            logits[:, m, position] = [entries[f'{task}_logit'] for task in TASKS]
    probs = expit(logits)
    if not np.all((0 < probs) & (probs < 1)):
        raise ValueError('Stored held-out logits must give strictly interior Bernoulli probabilities')
    return logits


def trial_statistics(logits, slots, families, metadata, trials, readers):
    reader_idx = {reader: i for i, reader in enumerate(readers)}
    story_idx = {story: i for i, story in enumerate(sorted(families))}
    shape = (len(story_idx), len(CONDITIONS), len(readers))
    present = np.zeros(shape, dtype=bool)
    nll = np.zeros((*shape, len(TASKS), len(METHODS)))
    target = np.zeros((*shape, len(TASKS)))
    predicted_target = np.zeros((*shape[:2], len(TASKS), len(METHODS)))
    phase_count = np.zeros((*shape[:2], 4), dtype=np.int64)
    phase_predicted = np.zeros((*shape[:2], len(METHODS), 4))
    prepared = {}
    for key, slot in slots.items():
        info = metadata[key]
        s, c = story_idx[info['story']], CONDITIONS.index(info['condition'])
        position = info['target_index'] - 1
        current = logits[0, :, slot]
        first = first_distribution(logits[1, :, slot])
        target_support = np.exp(first)
        predicted_target[s, c, 0] = expit(current[:, position])
        predicted_target[s, c, 1] = target_support[:, position]
        phase_predicted[s, c] = np.column_stack((target_support[:, :position].sum(axis=1),
                                                target_support[:, position], target_support[:, position + 1:-1].sum(axis=1),
                                                target_support[:, -1]))
        prepared[key] = s, c, position, current, first
    for key, reader, observed, first_seen in trials:
        s, c, position, current, first = prepared[key]
        idx = s, c, reader_idx[reader]
        classifier_nll = np.logaddexp(0, np.where(observed[None, :], -current, current)).mean(axis=1)
        nll[idx] = np.stack((classifier_nll, -first[:, -1 if first_seen is None else first_seen]))
        target[idx] = observed[position], first_seen == position
        present[idx] = True
        phase = 3 if first_seen is None else 0 if first_seen < position else 1 if first_seen == position else 2
        phase_count[s, c, phase] += 1
    return present, nll, target, predicted_target, phase_count, phase_predicted


def averaged_cells(reader_weights, present, nll, target):
    dimensions = present.shape[:2]
    _, _, readers = present.shape
    denominator = reader_weights @ present.reshape(-1, readers).T
    if np.any(denominator <= 0):
        raise ValueError('Bootstrap draw leaves a story condition without a sampled reader')
    channels = len(TASKS) * len(METHODS)
    stream = np.concatenate((nll.reshape(*present.shape, channels), target), axis=-1)
    flattened = stream.transpose(0, 1, 3, 2).reshape(-1, readers)
    values = (reader_weights @ flattened.T).reshape(len(reader_weights), *dimensions, channels + len(TASKS))
    values /= denominator.reshape(len(reader_weights), *dimensions, 1)
    return (values[..., :channels].reshape(len(reader_weights), *dimensions, len(TASKS), len(METHODS)),
            values[..., channels:])


def contrasts(nll, human, model_effect, story_weights):
    weights = story_weights / story_weights.sum(axis=1, keepdims=True)
    nll_difference = (nll[..., METHODS.index('contextual')] - nll[..., METHODS.index('standalone')]).mean(axis=2)
    human_difference = human[:, :, 1] - human[:, :, 0]
    contrast_nll = np.einsum('bs,bst->bt', weights, nll_difference)
    contrast_human = np.einsum('bs,bst->bt', weights, human_difference)
    contrast_model = weights @ model_effect
    return np.stack((contrast_nll, contrast_model, contrast_model - contrast_human, contrast_human), axis=-1)


def interval(draws):
    return [float(value) for value in np.quantile(draws, (0.025, 0.975))]


def summarize(logits, slots, families, metadata, trials, readers):
    present, nll, target, predicted, phase_count, phase_predicted = trial_statistics(
        logits, slots, families, metadata, trials, readers)
    n_stories, _, n_readers = present.shape
    model_effects = predicted[:, 1] - predicted[:, 0]
    modeled_primary = model_effects[..., METHODS.index('contextual')]
    obs_nll, obs_target = averaged_cells(np.ones((1, n_readers)), present, nll, target)
    point = contrasts(obs_nll, obs_target, modeled_primary, np.ones((1, n_stories)))[0]

    rng = np.random.default_rng(SEED)
    reader_draws = rng.multinomial(n_readers, np.full(n_readers, 1 / n_readers), size=DRAWS)
    story_draws = rng.multinomial(n_stories, np.full(n_stories, 1 / n_stories), size=DRAWS)
    crossed, readers_only = [], []
    for start in range(0, DRAWS, 200):
        stop = min(start + 200, DRAWS)
        draw_nll, draw_target = averaged_cells(reader_draws[start:stop], present, nll, target)
        crossed.append(contrasts(draw_nll, draw_target, modeled_primary, story_draws[start:stop]))
        readers_only.append(contrasts(draw_nll, draw_target, modeled_primary, np.ones((stop - start, n_stories))))
    crossed = np.concatenate(crossed)
    readers_only = np.concatenate(readers_only)

    comparisons = ('nll_contextual_minus_standalone', 'contextual_target_effect', 'target_model_minus_human',
                   'human_target_effect')
    endpoint = {}
    for t, task in enumerate(TASKS):
        estimates = {name: {'estimate': float(point[t, i]), 'crossed_story_reader_ci95': interval(crossed[:, t, i]),
                            'fixed_story_reader_only_ci95': interval(readers_only[:, t, i])}
                     for i, name in enumerate(comparisons)}
        nll_ci, effect_ci, equivalent_ci = [interval(crossed[:, t, i]) for i in range(3)]
        gates = {'contextual_nll_better': nll_ci[1] < 0, 'contextual_target_effect_positive': effect_ci[0] > 0,
                 'model_human_effect_gap_equivalent': -MARGIN < equivalent_ci[0] and equivalent_ci[1] < MARGIN}
        endpoint[task] = {
            'nll_unit': 'nats_per_sentence' if task == 'current' else 'nats_per_story_trial',
            'human_target_probability_by_condition': {value: float(obs_target[0, :, c, t].mean())
                                                      for c, value in enumerate(CONDITIONS)},
            'methods': {method: {'nll': float(obs_nll[0, :, :, t, m].mean()),
                                 'target_effect': float(model_effects[:, t, m].mean()),
                                 'target_probability_by_condition': {value: float(predicted[:, c, t, m].mean())
                                                                     for c, value in enumerate(CONDITIONS)}}
                        for m, method in enumerate(METHODS)},
            'comparisons': estimates, 'gates': gates, 'pass': all(gates.values()),
        }
    phases = ('before_target', 'at_target', 'after_target', 'no_d')
    observed_phase = phase_count / present.sum(axis=2)[..., None]
    onset = {value: {'human_counts': dict(zip(phases, phase_count[:, c].sum(axis=0).tolist(), strict=True)),
                     'human_probabilities': dict(zip(phases, observed_phase[:, c].mean(axis=0).tolist(), strict=True)),
                     'methods': {method: dict(zip(phases, phase_predicted[:, c, m].mean(axis=0).tolist(), strict=True))
                                 for m, method in enumerate(METHODS)}}
             for c, value in enumerate(CONDITIONS)}
    return endpoint, onset


def run(assay, extraction, packet, protocol, output):
    if output.exists():
        raise FileExistsError(output)
    audit, stimuli, metadata, families, texts = load_layout(assay)
    states, provenance = load_states(assay, extraction, packet, protocol, stimuli, metadata, families)
    design, slots = design_rows(stimuli, metadata, texts, states)
    del states
    trials, readers, hits, totals, inventory = load_outcomes(assay, audit, metadata, slots)
    logits, fits, weights = crossfit(design, slots, families, hits, totals)
    del design
    with TemporaryDirectory(dir=output.parent, prefix='.frick-score-') as scratch:
        stage = Path(scratch) / 'assay'
        stage.mkdir()
        forecasts = stage / 'forecasts.jsonl'
        write_forecasts(forecasts, logits, slots, stimuli)
        write_json(stage / 'fits.json', {'ridge': RIDGE, 'scalar_features': SCALAR_FEATURES, 'folds': fits,
                                         'vector_features': {'surface': [], 'standalone': ['unit_standalone_state'],
                                                             'contextual': ['unit_causal_contextual_state']}})
        np.savez_compressed(stage / 'fit_weights.npz', **weights)
        del weights
        provenance.update({'outcomes_sha256': audit['output_sha256']['outcomes.jsonl'],
                           'scorer_sha256': digest(Path(__file__)),
                           'standardization_sha256': digest(Path(standardization.__code__.co_filename)),
                           'numpy': np.__version__, 'scipy': scipy.__version__})
        names = ('forecasts.jsonl', 'fits.json', 'fit_weights.npz')
        seal = {'schema': 1, 'provenance': provenance, 'forecast_rows': logits.shape[-1],
                'fits': len(fits), 'file_sha256': {name: digest(stage / name) for name in names}}
        seal_path = stage / 'forecast-seal.json'
        write_json(seal_path, seal)
        if digest(forecasts) != read_json(seal_path)['file_sha256']['forecasts.jsonl']:
            raise ValueError('Held-out forecast serialization changed after sealing')
        frozen = read_forecasts(forecasts, slots, stimuli)
        if not np.array_equal(frozen, logits):
            raise ValueError('Held-out forecast round-trip is not exact')
        endpoints, onset = summarize(frozen, slots, families, metadata, trials, readers)
        write_json(stage / 'results.json', {
            'schema': 1, 'forecast_seal_sha256': digest(seal_path), 'inventory': inventory,
            'provenance': provenance, 'endpoints': endpoints, 'first_d_by_condition': onset,
            'bootstrap': {'draws': DRAWS, 'seed': SEED, 'effect_margin': MARGIN,
                          'weighting': 'equal stories and conditions, crossed resampling of stories and readers'},
        })
        stage.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('assay', 'extraction', 'packet', 'output'):
        parser.add_argument(f'--{name}', required=True, type=Path)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    run(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
