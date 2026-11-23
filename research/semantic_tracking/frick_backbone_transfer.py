"""Evaluate matched backbone readouts against Frick's sealed sentence judgments."""

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np
from scipy.special import expit

from research.semantic_tracking import frick_consistency_score as frick


TRANSFER = Path(__file__).resolve().parents[1] / 'backbone_transfer'
MODELS = TRANSFER / 'models.json'
ALIASES = ('qwen', 'lfm3', 'gemma4')
VIEWS = ('unicode_end', 'separator_end')


def source_views(stimuli, metadata):
    texts, layouts = {}, {}
    for key, row in stimuli.items():
        if key not in metadata:
            continue
        exact = row['stops']
        successors = metadata[key]['sentences'][1:]
        if any(row['text'][stop:item['start']] != '\n\n' or item['start'] != stop + 2
               for stop, item in zip(exact[:-1], successors, strict=True)):
            raise ValueError(f'Sentence separators do not match the archived source join for {key}')
        separator = [*(item['start'] for item in successors), exact[-1]]
        texts[key] = {**row, 'stops': sorted(set(exact) | set(separator))}
        layouts[key] = {'id': key, 'unicode_end': exact, 'separator_end': separator}
    return texts, layouts


def write_rows(path, rows):
    with path.open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n')


def packet_manifest(assay, prepared, texts, layouts):
    return {
        'source_audit_sha256': frick.digest(assay / 'source-audit.json'),
        'original_stimuli_sha256': frick.digest(assay / 'stimuli.jsonl'),
        'stimuli_sha256': frick.digest(prepared / 'stimuli.jsonl'),
        'views_sha256': frick.digest(prepared / 'views.jsonl'),
        'documents': len(texts),
        'union_stops': sum(len(row['stops']) for row in texts.values()),
        'view_stops': {view: sum(len(row[view]) for row in layouts.values()) for view in VIEWS},
        'verified_separator_joins': sum(len(row['unicode_end']) - 1 for row in layouts.values()),
    }


def prepare(assay, output):
    _, stimuli, metadata, _, _ = frick.load_layout(assay)
    texts, layouts = source_views(stimuli, metadata)
    if output.exists():
        raise FileExistsError(output)
    with TemporaryDirectory(dir=output.parent, prefix='.frick-prepare-') as temporary:
        stage = Path(temporary) / 'prepared'
        stage.mkdir()
        write_rows(stage / 'stimuli.jsonl', texts.values())
        write_rows(stage / 'views.jsonl', layouts.values())
        frick.write_json(stage / 'manifest.json', packet_manifest(assay, stage, texts, layouts))
        stage.rename(output)


def load_packet(assay, prepared, stimuli, metadata):
    expected, layouts = source_views(stimuli, metadata)
    path = prepared / 'stimuli.jsonl'
    views = prepared / 'views.jsonl'
    records = frick.indexed(frick.read_rows(path), path)
    actual_layouts = frick.indexed(frick.read_rows(views), views)
    manifest = packet_manifest(assay, prepared, expected, layouts)
    if (records != expected or actual_layouts != layouts
            or frick.read_json(prepared / 'manifest.json') != manifest):
        raise ValueError('Frick model inputs or exact annotation endpoint mapping differ from the source')
    return manifest, expected, layouts


def check_combined(path, expected):
    seen, observed = set(), {}
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            key = row['id']
            if key in seen:
                raise ValueError(f'Combined inputs repeat stimulus {key}')
            seen.add(key)
            if key in expected:
                observed[key] = row
    if observed != expected:
        raise ValueError('Combined extraction does not contain every exact Frick passage')
    return seen, frick.digest(path)


def surface_rows(texts):
    features, slots = [], {}
    for key, sentences in sorted(texts.items()):
        start = len(features)
        prior_words, prior_types = 0, set()
        for j, text in enumerate(sentences):
            words = len(text.split())
            types = frick.word_types(text)
            if j:
                overlap = len(prior_types & types) / len(types) if types else 0.0
                features.append((*np.log1p((len(text), words, j + 1, prior_words)), overlap))
            prior_words += words
            prior_types.update(types)
        slots[key] = slice(start, len(features))
    return np.asarray(features, dtype=np.float32), slots


def contextual_rows(surface, slots, states):
    width = next(iter(states.values())).shape[1]
    design = np.empty((len(surface), surface.shape[1] + width), dtype=np.float32)
    design[:, :surface.shape[1]] = surface
    for key, span in slots.items():
        rows = states[key][1:]
        norms = np.linalg.norm(rows, axis=1)
        if not np.all(np.isfinite(norms) & (norms > 0)):
            raise ValueError(f'Invalid sentence state norm for {key}')
        design[span, surface.shape[1]:] = rows / norms[:, None]
    return design


def load_states(alias, directory, spec, combined_ids, combined_sha, stimuli, layouts, metadata, families,
                protocol_sha):
    path = directory / 'seal.json'
    seal = frick.read_json(path)
    pins = {'candidate': alias, 'model': spec['repo'], 'revision': spec['revision'],
            'tokenizer_sha256': spec['sha256']['tokenizer.json'], 'hidden_width': spec['hidden_width'],
            'model_files_sha256': spec['sha256'], 'transformers': spec['transformers'],
            'stimuli_sha256': combined_sha, 'protocol_sha256': protocol_sha,
            'states_dtype': 'float32', 'gpu': 'NVIDIA L4', 'content_forward_per_document': 1}
    if (any(seal[key] != value for key, value in pins.items())
            or re.fullmatch(r'[0-9a-f]{64}', seal['model_manifest_sha256']) is None):
        raise ValueError('State extraction does not bind this exact layout and protocol')
    for name in ('coordinates.jsonl', 'states.npy', 'causality.json', 'preflight.json'):
        if frick.digest(directory / name) != seal['file_sha256'][name]:
            raise ValueError(f'Extraction artifact digest mismatch for {name}')
    checks = frick.read_json(directory / 'causality.json')['checks']
    if (not checks or len(checks) != seal['independent_causal_checks']
            or not all(check['passed'] is True for check in checks)):
        raise ValueError('The sealed extraction lacks passing independent prefix checks')
    coordinates = frick.indexed(frick.read_rows(directory / 'coordinates.jsonl'), 'coordinates.jsonl')
    preflight = frick.read_json(directory / 'preflight.json')
    geometry = frick.indexed(preflight['details'], 'preflight.details')
    if (coordinates.keys() != combined_ids or geometry.keys() != combined_ids
            or any(preflight[key] != pins[key]
                   for key in ('candidate', 'model', 'revision', 'tokenizer_sha256', 'stimuli_sha256',
                               'protocol_sha256'))
            or any(coordinates[key] != {'id': key, 'stops': row['stops']} for key, row in geometry.items())):
        raise ValueError('Coordinates do not bind the complete grade-free preflight')
    states = np.load(directory / 'states.npy', mmap_mode='r', allow_pickle=False)
    expected_shape = (sum(len(row['stops']) for row in coordinates.values()), spec['hidden_width'])
    if (states.shape != expected_shape or seal['states_shape'] != list(expected_shape)
            or states.dtype != np.float32 or not states.flags.c_contiguous):
        raise ValueError('Expected certified states as contiguous FP32 rows')
    offset, coordinate_indices, missing_reasons = 0, {}, {}
    expected = {key: set(stimuli[key]['stops']) for key in metadata}
    for key, row in coordinates.items():
        count = len(row['stops'])
        if key not in metadata:
            offset += count
            continue
        stops = row['stops']
        details = geometry[key]
        tokens, length = details['token_indices'], details['input_tokens']
        unavailable = details['unavailable']
        if (type(length) is not int or len(tokens) != len(stops)
                or any(type(i) is not int or not 0 <= i < length for i in tokens)
                or any(a >= b for a, b in zip(tokens, tokens[1:]))
                or any(type(stop) is not int or stop not in expected[key] for stop in stops)
                or any(a >= b for a, b in zip(stops, stops[1:]))
                or len(unavailable) != len(expected[key]) - count
                or {item['stop'] for item in unavailable} != expected[key] - set(stops)
                or any(item['reason'] not in {'no_token_end', 'unstable_prefix'} for item in unavailable)):
            raise ValueError(f'Invalid preflight record for published Frick sentences in {key}')
        coordinate_indices[key] = {stop: offset + i for i, stop in enumerate(stops)}
        missing_reasons[key] = {item['stop']: item['reason'] for item in unavailable}
        offset += count
    provenance = {'seal_sha256': frick.digest(path), 'states_shape': list(states.shape),
                  'model': spec['repo'], 'revision': spec['revision'],
                  'independent_causal_checks': len(checks), 'gpu': seal['gpu'],
                  'compute_dtype': seal['compute_dtype'], 'transformers': seal['transformers'],
                  'model_manifest_sha256': seal['model_manifest_sha256']}
    views, availability = {}, {}
    for view in VIEWS:
        missing = {key: [missing_reasons[key][stop] for stop in layout[view] if stop in missing_reasons[key]]
                   for key, layout in layouts.items()}
        reasons = Counter(reason for values in missing.values() for reason in values)
        if reasons:
            availability[view] = {'status': 'unavailable', 'rejected_stops': sum(reasons.values()),
                                  'incomplete_documents': sum(bool(row) for row in missing.values()),
                                  'unavailable_stops_by_reason': dict(reasons)}
            continue
        indexed = {key: states[[coordinate_indices[key][stop] for stop in layout[view]]]
                   for key, layout in layouts.items()}
        if any(not np.isfinite(values).all() for values in indexed.values()):
            raise ValueError('A sealed Frick state has nonfinite values')
        views[view] = indexed
        availability[view] = {'status': 'scored', 'rejected_stops': 0,
                              **check_shared_prefixes(indexed, metadata, families)}
    return views, {**provenance, 'views': availability}


def check_shared_prefixes(indexed, metadata, families):
    errors = []
    for family in families.values():
        left, right = map(family.__getitem__, frick.CONDITIONS)
        count = metadata[left]['changed_index']
        h_left = indexed[left][:count].astype(np.float64)
        h_right = indexed[right][:count].astype(np.float64)
        norms = np.linalg.norm(h_left, axis=1)
        if np.any(norms <= 0):
            raise ValueError('Malformed state at a common causal prefix')
        errors.extend((np.linalg.norm(h_left - h_right, axis=1) / norms).tolist())
    max_error = max(errors)
    if max_error >= frick.MAX_PREFIX_ERROR:
        raise ValueError(f'Matched prefixes before the intervention differed by {max_error:.6f}')
    return {'shared_prechange_prefixes': len(errors), 'maximum_relative_l2': max_error}


def crossfit(design, slots, families, hits, totals):
    logits = np.full_like(hits, np.nan, dtype=np.float64)
    blocks = () if design.shape[1] == frick.SCALAR_WIDTH else ((frick.SCALAR_WIDTH, design.shape[1]),)
    summaries, weights = [], {}
    for number, (story, pair) in enumerate(sorted(families.items())):
        held = np.zeros(len(design), dtype=bool)
        for key in pair.values():
            held[slots[key]] = True
        train_x, test_x = design[~held], design[held]
        for i, task in enumerate(frick.TASKS):
            fitted = frick.fit(train_x, hits[i, ~held], totals[i, ~held], blocks)
            logits[i, held] = frick.predict_logits(test_x, fitted)
            prefix = f'fold{number}_{task}'
            for name in ('center', 'scale', 'coefficients'):
                weights[f'{prefix}_{name}'] = fitted.pop(name)
            summaries.append({'held_out_story': story, 'task': task, 'weights_prefix': prefix,
                              'vector_blocks': blocks, **fitted})
    if not np.isfinite(logits).all():
        raise ValueError('A fold omitted certified sentence stops')
    return logits, summaries, weights


def write_forecasts(path, logits, methods, slots, layouts, view):
    with path.open('x', encoding='utf-8') as stream:
        for key, span in slots.items():
            for row, stop, model_stop in zip(range(span.start, span.stop), layouts[key]['unicode_end'][1:],
                                             layouts[key][view][1:], strict=True):
                values = {
                    method: dict(zip(frick.TASKS, logits[:, m, row].tolist(), strict=True))
                    for m, method in enumerate(methods)
                }
                stream.write(json.dumps({'id': key, 'stop': stop, 'model_stop': model_stop, 'logits': values},
                                        allow_nan=False) + '\n')


def read_forecasts(path, methods, slots, layouts, view):
    expected = ((key, row, stop, model_stop) for key, span in slots.items()
                for row, stop, model_stop in zip(range(span.start, span.stop), layouts[key]['unicode_end'][1:],
                                                 layouts[key][view][1:], strict=True))
    logits = np.empty((len(frick.TASKS), len(methods), sum(s.stop - s.start for s in slots.values())))
    for item, (key, index, stop, model_stop) in zip(frick.read_rows(path), expected, strict=True):
        if (set(item) != {'id', 'stop', 'model_stop', 'logits'}
                or (item['id'], item['stop'], item['model_stop']) != (key, stop, model_stop)
                or set(item['logits']) != set(methods)):
            raise ValueError('Malformed or reordered sealed forecast')
        for m, method in enumerate(methods):
            vote = item['logits'][method]
            if set(vote) != set(frick.TASKS):
                raise ValueError('A sealed forecast omitted a judgment task')
            logits[:, m, index] = [vote[task] for task in frick.TASKS]
    if not np.isfinite(logits).all() or not np.all((expit(logits) > 0) & (expit(logits) < 1)):
        raise ValueError('All sealed logits must give strictly interior finite probabilities')
    return logits


def trial_arrays(logits, slots, families, metadata, trials, readers):
    story_indices = {story: i for i, story in enumerate(sorted(families))}
    reader_indices = {reader: i for i, reader in enumerate(readers)}
    trials_shape = (len(families), len(frick.CONDITIONS), len(readers))
    present = np.zeros(trials_shape, dtype=bool)
    outcome = np.zeros((*trials_shape, len(frick.TASKS)))
    loss = np.zeros((*outcome.shape, logits.shape[1]))
    predicted = np.zeros((*trials_shape[:2], len(frick.TASKS), logits.shape[1]))
    onset_model = np.zeros((*trials_shape[:2], logits.shape[1], 4))
    onset_counts = np.zeros((*trials_shape[:2], 4), dtype=np.int64)
    passages = {}
    for key, span in slots.items():
        info = metadata[key]
        story, condition = story_indices[info['story']], frick.CONDITIONS.index(info['condition'])
        target = info['target_index'] - 1
        current = logits[0, :, span]
        first = frick.first_distribution(logits[1, :, span])
        distribution = np.exp(first)
        onset_model[story, condition] = np.column_stack((distribution[:, :target].sum(axis=1),
                                                        distribution[:, target],
                                                        distribution[:, target + 1:-1].sum(axis=1),
                                                        distribution[:, -1]))
        predicted[story, condition] = (expit(current[:, target]), distribution[:, target])
        passages[key] = story, condition, target, current, first
    for key, reader, observed, first_seen in trials:
        story, condition, target, current, first = passages[key]
        index = (story, condition, reader_indices[reader])
        present[index] = True
        outcome[index] = observed[target], first_seen == target
        category = 3 if first_seen is None else int(first_seen >= target) + int(first_seen > target)
        onset_counts[story, condition, category] += 1
        loss[index] = (np.logaddexp(0, np.where(observed[None, :], -current, current)).mean(axis=1),
                       -first[:, -1 if first_seen is None else first_seen])
    return present, loss, outcome, predicted, onset_model, onset_counts


def summarize(logits, methods, slots, families, metadata, trials, readers):
    present, loss, outcome, predicted, onset, onset_counts = trial_arrays(
        logits, slots, families, metadata, trials, readers)
    stories, conditions, n_readers = present.shape
    tasks, n_methods = logits.shape[:2]
    channels = tasks * n_methods
    packed = np.concatenate((loss.reshape(*present.shape, channels), outcome), axis=-1)
    packed = packed.transpose(0, 1, 3, 2).reshape(-1, n_readers)
    model_effect = predicted[:, 1] - predicted[:, 0]

    def draw(reader_weights, story_weights):
        denominator = reader_weights @ present.reshape(-1, n_readers).T
        if np.any(denominator <= 0):
            raise ValueError('Bootstrap draw leaves a story condition without a reader')
        cells = (reader_weights @ packed.T).reshape(-1, stories, conditions, channels + tasks)
        cells /= denominator.reshape(-1, stories, conditions, 1)
        nll = cells[..., :channels].reshape(-1, stories, conditions, tasks, n_methods).mean(axis=2)
        weights = story_weights / stories
        human = np.einsum('bs,bst->bt', weights, cells[:, :, 1, channels:] - cells[:, :, 0, channels:])
        return np.einsum('bs,bstm->btm', weights, nll), human, np.einsum('bs,stm->btm', weights, model_effect)

    nll, human, model = (item[0] for item in draw(np.ones((1, n_readers)), np.ones((1, stories))))
    rng = np.random.default_rng(frick.SEED)
    rw = rng.multinomial(n_readers, np.full(n_readers, 1 / n_readers), size=frick.DRAWS)
    sw = rng.multinomial(stories, np.full(stories, 1 / stories), size=frick.DRAWS)
    batches = [draw(rw[i:i + 200], sw[i:i + 200]) for i in range(0, frick.DRAWS, 200)]
    draw_nll, draw_human, draw_model = map(np.concatenate, zip(*batches, strict=True))

    def estimate(point, draws):
        return {'estimate': float(point), 'crossed_story_reader_ci95': frick.interval(draws)}

    endpoints = {}
    for i, task in enumerate(frick.TASKS):
        record = {'nll_unit': 'nats_per_sentence' if task == 'current' else 'nats_per_story_trial',
                  'human_target_effect': estimate(human[i], draw_human[:, i]), 'methods': {}, 'comparisons': {}}
        for m, method in enumerate(methods):
            gap = estimate(model[i, m] - human[i], draw_model[:, i, m] - draw_human[:, i])
            effect = estimate(model[i, m], draw_model[:, i, m])
            result = {'nll': estimate(nll[i, m], draw_nll[:, i, m]), 'target_effect': effect,
                      'target_model_minus_human': gap}
            if method != 'surface':
                improvement = estimate(nll[i, m] - nll[i, 0], draw_nll[:, i, m] - draw_nll[:, i, 0])
                lo, hi = gap['crossed_story_reader_ci95']
                result['nll_minus_surface'] = improvement
                result['gates'] = {
                    'nll_better': improvement['crossed_story_reader_ci95'][1] < 0,
                    'effect_positive': effect['crossed_story_reader_ci95'][0] > 0,
                    'effect_gap_equivalent': -frick.MARGIN < lo and hi < frick.MARGIN,
                }
                result['passed'] = all(result['gates'].values())
            record['methods'][method] = result
        for a, b in combinations(range(1, len(methods)), 2):
            record['comparisons'][f'{methods[b]}_minus_{methods[a]}'] = {
                'nll': estimate(nll[i, b] - nll[i, a], draw_nll[:, i, b] - draw_nll[:, i, a]),
                'absolute_human_effect_error': estimate(
                    abs(model[i, b] - human[i]) - abs(model[i, a] - human[i]),
                    np.abs(draw_model[:, i, b] - draw_human[:, i]) -
                    np.abs(draw_model[:, i, a] - draw_human[:, i])),
            }
        endpoints[task] = record
    phases = ('before_target', 'at_target', 'after_target', 'no_d')
    observed = onset_counts / present.sum(axis=2)[..., None]
    onset_report = {
        value: {'human': dict(zip(phases, observed[:, c].mean(axis=0).tolist(), strict=True)),
                'methods': {method: dict(zip(phases, onset[:, c, m].mean(axis=0).tolist(), strict=True))
                            for m, method in enumerate(methods)}}
        for c, value in enumerate(frick.CONDITIONS)
    }
    return endpoints, onset_report


def score(assay, prepared, combined_stimuli, protocol, models, extractions, output):
    if output.exists():
        raise FileExistsError(output)
    audit, stimuli, metadata, families, texts = frick.load_layout(assay)
    manifest, expected, layouts = load_packet(assay, prepared, stimuli, metadata)
    combined_ids, combined_sha = check_combined(combined_stimuli, expected)
    protocol_sha = frick.digest(protocol)
    registry = frick.read_json(models)
    model_manifest_sha = frick.digest(models)
    representations, provenance = {view: {} for view in VIEWS}, {}
    for method in ALIASES:
        pins = {'transformers': registry['transformers'], **registry['models'][method]}
        states, provenance[method] = load_states(
            method, extractions[method], pins, combined_ids, combined_sha, expected, layouts, metadata, families,
            protocol_sha)
        for view, rows in states.items():
            representations[view][method] = rows
    surface, slots = surface_rows(texts)
    trials, readers, hits, totals, inventory = frick.load_outcomes(assay, audit, metadata, slots)
    baseline = crossfit(surface, slots, families, hits, totals)
    inputs = {'prepared_manifest_sha256': frick.digest(prepared / 'manifest.json'),
              'views_sha256': manifest['views_sha256'], 'source_audit_sha256': manifest['source_audit_sha256'],
              'outcomes_sha256': audit['output_sha256']['outcomes.jsonl'], 'protocol_sha256': protocol_sha,
              'combined_stimuli_sha256': combined_sha, 'model_pins_sha256': model_manifest_sha,
              'scorer_sha256': frick.digest(Path(__file__)), 'original_scorer_sha256': frick.digest(Path(frick.__file__)),
              'extractions': provenance}
    with TemporaryDirectory(dir=output.parent, prefix='.frick-matched-score-') as temporary:
        stage = Path(temporary) / 'assay'
        stage.mkdir()
        receipts = {}
        for view, states in representations.items():
            fitted = {'surface': baseline}
            for method, rows in states.items():
                fitted[method] = crossfit(contextual_rows(surface, slots, rows), slots, families, hits, totals)
            receipts[view] = score_view(stage / view, view, inputs, fitted, layouts, slots, families,
                                        metadata, trials, readers, inventory)
        frick.write_json(stage / 'results.json', {'views': receipts, 'inventory': inventory, 'extractions': provenance,
                                                 'primary_endpoint': 'first'})
        stage.rename(output)


def score_view(folder, view, inputs, fitted, layouts, slots, families, metadata, trials, readers, inventory):
    folder.mkdir()
    methods = tuple(fitted)
    logits = np.stack([entry[0] for entry in fitted.values()], axis=1)
    weights = {f'{method}_{key}': value for method, entry in fitted.items() for key, value in entry[2].items()}
    forecasts = folder / 'forecasts.jsonl'
    write_forecasts(forecasts, logits, methods, slots, layouts, view)
    frick.write_json(folder / 'fits.json', {'folds': {method: entry[1] for method, entry in fitted.items()},
                                          'ridge': frick.RIDGE, 'scalar_features': frick.SCALAR_FEATURES})
    np.savez_compressed(folder / 'fit_weights.npz', **weights)
    seal = folder / 'forecast-seal.json'
    frick.write_json(seal, {'view': view, 'inputs': inputs,
                            'file_sha256': {name: frick.digest(folder / name)
                                            for name in ('forecasts.jsonl', 'fits.json', 'fit_weights.npz')}})
    frozen = read_forecasts(forecasts, methods, slots, layouts, view)
    if not np.array_equal(frozen, logits):
        raise ValueError('Sealed forecast round trip differs')
    endpoints, onset = summarize(frozen, methods, slots, families, metadata, trials, readers)
    result = folder / 'results.json'
    frick.write_json(result, {
        'view': view, 'primary_endpoint': 'first', 'inventory': inventory, 'endpoints': endpoints,
        'first_d_by_condition': onset, 'forecast_seal_sha256': frick.digest(seal),
        'bootstrap': {'draws': frick.DRAWS, 'seed': frick.SEED, 'margin': frick.MARGIN},
    })
    return {'methods': list(methods), 'results_sha256': frick.digest(result),
            'forecast_seal_sha256': frick.digest(seal)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest='command', required=True)
    prepare_parser = subcommands.add_parser('prepare')
    score_parser = subcommands.add_parser('score')
    for command in (prepare_parser, score_parser):
        command.add_argument('--assay', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
    score_parser.add_argument('--prepared', type=Path, required=True)
    score_parser.add_argument('--combined-stimuli', type=Path, required=True)
    score_parser.add_argument('--protocol', type=Path, required=True)
    score_parser.add_argument('--models', type=Path, default=MODELS)
    score_parser.add_argument('--extraction', action='append', required=True, metavar='NAME=PATH')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.assay, args.output)
        return
    extractions = {}
    for item in args.extraction:
        name, equal, path = item.partition('=')
        if not equal or not path or not name.isidentifier() or name in extractions or name == 'surface':
            parser.error('Each extraction must have a unique, non-surface NAME=PATH')
        extractions[name] = Path(path)
    if set(extractions) != set(ALIASES):
        parser.error('Expected exactly qwen, lfm3 and gemma4 backbone extractions')
    score(args.assay, args.prepared, args.combined_stimuli, args.protocol, args.models, extractions, args.output)


if __name__ == '__main__':
    main()
