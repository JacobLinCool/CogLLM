"""Freeze Blott 2021 cue-update readouts and replay the previously used 2023 cohort."""

import argparse
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.special import expit

from research.semantic_tracking import blott_format_score as previous
from research.semantic_tracking import blott_update_fit as readout
from research.semantic_tracking import frick_consistency_score as records
from research.stage6 import blott_score as early_source


PROTOCOL = Path(__file__).with_name('blott_update_protocol.md')
PROTOCOL_SHA256 = 'd11f1d13d68ec63fc14bc4935fa0d33df7c9c13383067108d28439be0852553c'
RESULTS = PROTOCOL.parents[1] / 'results'
TRAIN_RECEIPT = RESULTS / 'blott-update-2021-source-audit.json'
BASELINE = RESULTS / 'blott-format-transfer-l4'
PREVIOUS_SEAL_SHA256 = '74d208ca26b501f79c1f57ca0da17610ec7033c946d430d3596484a46e9637cc'
CONTRASTS = {'effect_vs_terminal': ('joint_effect', 'terminal'),
             'effect_vs_fixed_grade': ('joint_effect', 'fixed_grade'),
             'effect_vs_joint': ('joint_effect', 'joint'),
             'joint_vs_terminal': ('joint', 'terminal'),
             'update_vs_terminal': ('update', 'terminal'),
             'terminal_vs_surface': ('terminal', 'surface')}
ARM_ORDER = ('Amb', 'UA')


def load_vectors(stimuli, layout_file, packet, extraction, protocol_sha):
    input_file = packet / 'stimuli.jsonl'
    receipt = records.read_json(packet / 'preflight.json')
    preflight = records.read_json(packet / 'extractor-preflight.json')
    seal_path = extraction / 'seal.json'
    seal = records.read_json(seal_path)
    scope = {'candidate_id': early_source.CANDIDATE_ID, 'protocol_sha256': protocol_sha,
             'stimuli_sha256': records.digest(input_file), 'preflight_sha256': records.digest(packet / 'extractor-preflight.json'),
             'assay_sha256': records.digest(packet / 'preflight.json'), 'raw_stops': True, 'states_dtype': 'float32',
             'content_forward_per_document': 1, 'independent_causal_checks': 12, 'gpu': 'NVIDIA L4'}
    if (receipt['original_stimuli_sha256'] != records.digest(stimuli)
            or any(seal.get(k) != v for k, v in scope.items())
            or any(records.digest(extraction / n) != seal['file_sha256'][n] for n in
                   ('states.npy', 'coordinates.jsonl', 'causality.json'))):
        raise ValueError('Blott cue states are not bound to this exact registered inference packet')
    checks = records.read_json(extraction / 'causality.json')['checks']
    if len(checks) != 12 or not all(test['passed'] and test['projected_tokens']['passed'] for test in checks):
        raise ValueError('Blott cue extraction failed the registered causal tests')

    source = records.indexed(records.read_rows(stimuli), 'Blott original stimuli')
    layout = records.indexed(records.read_rows(layout_file), 'Blott source layout')
    packed = records.indexed(records.read_rows(input_file), 'Blott certified stimuli')
    coordinates = records.indexed(records.read_rows(extraction / 'coordinates.jsonl'), 'Blott states')
    certified = {item['id']: item['stops'] for item in preflight['documents']}
    if (source.keys() != layout.keys() or source.keys() != packed.keys() or source.keys() != coordinates.keys()
            or receipt['documents'] != certified or certified != {key: row['stops'] for key, row in coordinates.items()}):
        raise ValueError('All published versions must have the exact certified coordinate inventory')
    states = np.load(extraction / 'states.npy', mmap_mode='r', allow_pickle=False)
    expected = (sum(map(len, certified.values())), 2048)
    if states.shape != expected or seal['states_shape'] != list(expected) or states.dtype != np.float32:
        raise ValueError('Unexpected archived Qwen raw state geometry')
    selected = {}
    offset = 0
    for key, coordinate in coordinates.items():
        text, stops = source[key]['text'], coordinate['stops']
        item = layout[key]
        if (packed[key]['text'] != text or packed[key]['stops'] != stops or not set(stops) <= set(source[key]['stops'])
                or stops[-1] != item['sentence_stop'] or stops[-1] != len(text)):
            raise ValueError('Certified states do not correspond to the original displayed text')
        chunk = states[offset:offset + len(stops)]
        offset += len(stops)
        if item['cue_span'] is None:
            continue
        left, right = item['cue_span']
        earlier = item['before_cue_stop']
        if not 0 < earlier < left < right <= len(text) or text[earlier:left].strip() or len(text[left:right].split()) != 1:
            raise ValueError('An annotated Blott cue does not match an exact lexical source span')
        if earlier not in stops or right not in stops:
            continue
        vectors = chunk[[stops.index(pos) for pos in (earlier, right, len(text))]].astype(np.float64)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or np.any(norms <= 0):
            raise ValueError('Expected finite nonzero states at all three source stops')
        pre, post, end = vectors / norms
        selected[key] = {'length': np.log([len(text), len(text.split())]), 'terminal': end, 'update': post - pre}
    return layout, selected, records.digest(seal_path)


def ordered_pairs(layout, feature_map, training):
    families = defaultdict(dict)
    for key, item in layout.items():
        arm, family = item['ambiguity'], item['family']
        if training and item['condition'] != f'Plaus{arm}':
            continue
        if arm not in ARM_ORDER or arm in families[family]:
            raise ValueError('Expected each coherent ambiguity condition once per family')
        families[family][arm] = key
    pairs, rejected = {}, []
    for family, variants in sorted(families.items()):
        keys = tuple(variants[arm] for arm in ARM_ORDER)
        if all(key in feature_map for key in keys):
            pairs[family] = keys
        else:
            rejected.append(family)
    return pairs, rejected


def build_features(pairs, feature_map):
    ordered = [feature_map[key] for keys in pairs.values() for key in keys]
    return {column: np.vstack([row[column] for row in ordered]) for column in ('length', 'terminal', 'update')}


def training_counts(old_sources, pairs, layout):
    stimuli = old_sources / 'model_input/stimuli.jsonl'
    design, trials = (old_sources / ('scoring/' + n + '.jsonl') for n in ('design', 'human_trials'))
    if {'stimuli': records.digest(stimuli), 'design': records.digest(design), 'human_trials': records.digest(trials)} != early_source.INPUTS:
        raise ValueError('Training trials and identities must match the previously certified 2021 corpus')
    _, cases, _ = early_source.read_inputs(stimuli, design)
    observed, positive = early_source.human_matrices(trials, cases)
    if ({row['id']: (row['family'], row['condition']) for row in records.read_rows(design)}
            != {key: (row['family'], row['condition']) for key, row in layout.items()}):
        raise ValueError('Original behavioral arms must agree with the independently reconstructed cue coordinates')
    index = {key: i for i, key in enumerate(cases.ravel())}
    chosen = [index[key] for keys in pairs.values() for key in keys]
    return positive[:, chosen].sum(axis=0), observed[:, chosen].sum(axis=0)


def word_sorted_training(pairs, layout, stimuli):
    texts = {row['id']: row['text'] for row in records.read_rows(stimuli)}
    entries = []
    for family, keys in pairs.items():
        ambiguous, _ = keys
        start, stop = layout[ambiguous]['noun_span']
        noun = texts[ambiguous][start:stop].casefold()
        if not noun or not noun.strip():
            raise ValueError('Every training family needs its visible ambiguous noun for the locked fold ordering')
        entries.append((f'{noun}:{family}', family, keys))
    ordered = sorted(entries)
    return {family: keys for _, family, keys in ordered}, tuple(name for name, _, _ in ordered)


def baseline_forecasts():
    seal_path = BASELINE / 'forecast-seal.json'
    path = BASELINE / 'forecasts.json'
    seal = records.read_json(seal_path)
    if (records.digest(seal_path) != PREVIOUS_SEAL_SHA256 or records.digest(path) != seal['forecast_sha256']
            or seal['source_audit_sha256'] != previous.SOURCE_AUDIT_SHA256):
        raise ValueError('The comparator must use the exact previously sealed grade-only forecasts')
    return records.read_json(path)


def historical_evaluation(predictions, hits, totals):
    human = hits[1::2] / totals[1::2] - hits[::2] / totals[::2]
    loss = (totals[:, None] * np.logaddexp(0, predictions) - hits[:, None] * predictions).sum(axis=0) / totals.sum()
    probability = expit(predictions)
    effect = (probability[1::2] - probability[::2]).mean(axis=0)
    return {'human_effect': float(human.mean()), 'methods': {
        name: {'nll': float(nll), 'effect': float(delta)} for name, nll, delta in zip(readout.ARMS, loss, effect, strict=True)
    }}


def evaluate_external(prepared, predictions, original):
    terminal = records.indexed(predictions, 'cue-update predictions')
    values = {}
    for name, (model, control) in CONTRASTS.items():
        rows = []
        for row in original:
            key = row['id']
            if row['stop'] == terminal[key]['stop']:
                logits = terminal[key]['logits']
                rows.append({**row, 'logits': {'qwen': logits[model], 'surface': logits[control]}})
            else:
                rows.append(row)
        scored = previous.evaluate(prepared, rows)
        summary = scored['summary']
        values[name] = {'nll_model': summary['qwen_nll'], 'nll_control': summary['surface_nll'],
                        'nll_gain': summary['qwen_nll_gain'], 'human_effect': summary['human_effect'],
                        'model_effect': summary['qwen_effect'], 'control_effect': summary['surface_effect'],
                        'effect_error': summary['qwen_minus_human_effect'], 'gates': scored['gates']}
    gates = {'nll_gain_over_terminal': values['effect_vs_terminal']['gates']['nll_gain'],
             'nll_gain_over_fixed_grade': values['effect_vs_fixed_grade']['gates']['nll_gain'],
             'effect_direction': values['effect_vs_terminal']['gates']['effect_direction'],
             'effect_equivalence': values['effect_vs_terminal']['gates']['effect_equivalence']}
    return {'readers': scored['readers'], 'families': scored['families'], 'matched_pairs': scored['complete_pairs'],
            'comparisons': values, 'gates': gates, 'pass': all(gates.values()), 'retrospective': True}


def run(training, training_states, old_sources, transfer, transfer_packet, output):
    if output.exists():
        raise FileExistsError(output)
    if records.digest(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError('Cannot change the registered cue-update behavior assay')
    receipt = records.read_json(TRAIN_RECEIPT)
    if (any(records.digest(training / name) != signature for name, signature in receipt['artifact_sha256'].items())
            or records.digest(transfer / 'source-audit.json') != previous.SOURCE_AUDIT_SHA256):
        raise ValueError('The registered training or transfer material inventory changed')
    transfer_audit = records.read_json(transfer / 'source-audit.json')
    for name in ('stimuli.jsonl', 'layout.jsonl'):
        if records.digest(transfer / name) != transfer_audit['prepared_sha256'][name]:
            raise ValueError(f'Updated transfer material differs from the prior intake in {name}')

    training_layout, train_map, training_seal = load_vectors(
        training / 'prepared' / 'stimuli.jsonl', training / 'prepared' / 'layout.jsonl', training / 'packet',
        training_states, PROTOCOL_SHA256,
    )
    transfer_layout, transfer_map, transfer_seal = load_vectors(
        transfer / 'stimuli.jsonl', transfer / 'layout.jsonl', transfer_packet, BASELINE / 'projection',
        previous.calibration.PROTOCOL_SHA256,
    )
    pairs, excluded = ordered_pairs(training_layout, train_map, training=True)
    transfer_pairs, rejected = ordered_pairs(transfer_layout, transfer_map, training=False)
    if len(pairs) != 46 or len(transfer_pairs) != 44 or rejected:
        raise ValueError('Blott tokenizer-certified denominators differ from the registered source intake')
    pairs, fold_names = word_sorted_training(pairs, training_layout, training / 'prepared' / 'stimuli.jsonl')
    training_vectors = build_features(pairs, train_map)
    transfer_vectors = build_features(transfer_pairs, transfer_map)
    hits, totals = training_counts(old_sources, pairs, training_layout)
    if totals.sum() != 2208 or not np.all(totals == 24):
        raise ValueError('The cue-update training set requires all 2,208 matched-condition observations')
    historical, fold_receipts = readout.crossfit(training_vectors, hits, totals, fold_names)
    fits = {name: readout.fit_model(training_vectors, hits, totals, name) for name in readout.ARMS}
    forecasts = np.column_stack([fitted.predict(transfer_vectors) for fitted in fits.values()])
    original = baseline_forecasts()
    original_end = {row['id']: row for row in original if row['stop'] == transfer_layout[row['id']]['sentence_stop']}
    keys = [key for variants in transfer_pairs.values() for key in variants]
    if set(keys) != original_end.keys():
        raise ValueError('Transfer forecasts would compare different study versions')
    external = [
        {'id': key, 'stop': original_end[key]['stop'], 'logits': {
            **dict(zip(readout.ARMS, logits.tolist(), strict=True)),
            'fixed_grade': original_end[key]['logits']['qwen'],
        }} for key, logits in zip(keys, forecasts, strict=True)
    ]
    historical_forecasts = [
        {'id': key, 'logits': dict(zip(readout.ARMS, values.tolist(), strict=True))}
        for key, values in zip((key for variants in pairs.values() for key in variants), historical, strict=True)
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.blott-update-') as temporary:
        stage = Path(temporary) / 'scored'
        stage.mkdir()
        records.write_json(stage / 'historical-forecasts.json', historical_forecasts)
        records.write_json(stage / 'forecasts.json', external)
        records.write_json(stage / 'fits.json', {'fits': {name: fit.receipt() for name, fit in fits.items()},
                                                'historical_folds': fold_receipts, 'ridge': readout.RIDGE,
                                                'excluded_training_families': excluded})
        seal = stage / 'forecast-seal.json'
        records.write_json(seal, {'candidate_id': early_source.CANDIDATE_ID, 'protocol_sha256': PROTOCOL_SHA256,
                                  'training_extraction_sha256': training_seal, 'transfer_extraction_sha256': transfer_seal,
                                  'old_trial_sha256': early_source.INPUTS['human_trials'], 'training_source_sha256': records.digest(TRAIN_RECEIPT),
                                  'transfer_source_sha256': previous.SOURCE_AUDIT_SHA256, 'baseline_forecast_seal_sha256': PREVIOUS_SEAL_SHA256,
                                  'fit_code_sha256': records.digest(Path(readout.__file__)), 'scorer_sha256': records.digest(Path(__file__)),
                                  'file_sha256': {n: records.digest(stage / n) for n in
                                                  ('fits.json', 'forecasts.json', 'historical-forecasts.json')}})
        external_scores = evaluate_external(transfer, records.read_json(stage / 'forecasts.json'), original)
        records.write_json(stage / 'results.json', {'forecast_seal_sha256': records.digest(seal),
                           'training_families': len(pairs), 'training_responses': int(totals.sum()),
                           'historical_crossfit': historical_evaluation(historical, hits, totals), 'transfer': external_scores})
        stage.rename(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('training', 'training-states', 'old-sources', 'transfer', 'transfer-packet', 'output'):
        parser.add_argument('--' + field, type=Path, required=True)
    run(**vars(parser.parse_args()))
