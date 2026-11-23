"""Test a frozen cue ranker and verb-disjoint response readouts on Czech garden paths."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.special import softmax

from research.gradio_app.checkpoint import DEFAULT_CHECKPOINT, load_checkpoint
from research.gradio_app.word_cues import CueReadout, RELEASE_SHA256, TOKEN, word_updates
from research.semantic_tracking.blott_word_fit import RankFit
from research.semantic_tracking import czech_garden_fit as readouts
from research.semantic_tracking import czech_garden_metrics as metrics
from research.semantic_tracking import frick_consistency_score as records


PROTOCOL = Path(__file__).with_name('czech_garden_protocol.md')
PROTOCOL_SHA256 = 'c7cfbeccb810a4c43a9012c67c8668aadc3501990dec975a4dec2f2231d40ba7'
SOURCE = PROTOCOL.parents[1] / 'results' / 'czech-garden-prepared-audit.json'
SOURCE_SHA256 = 'f74718a86ebe511511b6661013dd5d3855f22e2fb86a7e980313966531822929'
SUBSETS = ('constant', 'surface', 'terminal')
CONDITIONS = ('gp', 'nongp')
ANIMACY = ('animate', 'inanimate')
REGIONS = tuple(range(2, 10))
DRAWS = 10_000
SEED = 735972


def sealed_material(work, extraction, tokenizer):
    from tokenizers import Tokenizer

    if records.digest(SOURCE) != SOURCE_SHA256:
        raise ValueError('The audited Czech source manifest changed')
    receipt = records.read_json(SOURCE)
    if any(records.digest(work / name) != signature for name, signature in receipt['file_sha256'].items()
           if name != 'prepared/outcomes.jsonl'):
        raise ValueError('Czech material, layout, or tokenizer packet changed before scoring')
    packet = work / 'packet'
    seal = records.read_json(extraction / 'seal.json')
    certification = receipt['certification']
    qwen_tokenizer = certification['tokenizer_sha256']
    expected = {'protocol_sha256': PROTOCOL_SHA256, 'candidate_id': certification['candidate_id'],
                'stimuli_sha256': records.digest(packet / 'stimuli.jsonl'), 'tokenizer_sha256': qwen_tokenizer,
                'assay_sha256': records.digest(packet / 'preflight.json'), 'gpu': 'NVIDIA L4', 'raw_stops': True,
                'preflight_sha256': records.digest(packet / 'extractor-preflight.json'), 'states_dtype': 'float32',
                'independent_causal_checks': 12, 'content_forward_per_document': 1}
    if (records.digest(tokenizer) != qwen_tokenizer or any(seal.get(k) != v for k, v in expected.items())
            or any(records.digest(extraction / name) != seal['file_sha256'][name]
                   for name in ('states.npy', 'coordinates.jsonl', 'causality.json'))):
        raise ValueError('Czech FP32 word-state archive differs from the sealed extraction')
    checks = records.read_json(extraction / 'causality.json')['checks']
    if len(checks) != 12 or not all(check['passed'] and check['projected_tokens']['passed'] for check in checks):
        raise ValueError('Independent causal word-state checks did not pass')
    corpus = records.indexed(records.read_rows(packet / 'stimuli.jsonl'), 'Czech grade-free source')
    layout = records.indexed(records.read_rows(work / 'prepared' / 'layout.jsonl'), 'Czech source design')
    order = records.indexed(records.read_rows(extraction / 'coordinates.jsonl'), 'Czech state coordinates')
    if set(layout) != set(corpus) or set(corpus) != set(order) or len(order) != 192:
        raise ValueError('Expected all 192 experimental Czech sentences')
    states = np.load(extraction / 'states.npy', mmap_mode='r', allow_pickle=False)
    expected_shape = (certification['certified_stops'], 2048)
    if states.shape != expected_shape or list(expected_shape) != seal['states_shape'] or states.dtype != np.float32:
        raise ValueError('Czech extraction does not have the expected raw FP32 vectors')
    return receipt, corpus, layout, order, states, Tokenizer.from_file(str(tokenizer))


def unit_rows(raw):
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError('Word-state normalization requires finite, nonzero causal vectors')
    return raw / norms


def make_rows(corpus, layout, order, states, tokenizer, english):
    head = english.ranker
    suffix = sum(len(row) for row in head.centers[:2])
    state_head = RankFit(head.blocks[2:], head.centers[2:], head.scales[2:], head.coefficients[suffix:], {})
    features, local_scores, offset = {}, {}, 0
    for key, coordinate in order.items():
        text, stops = corpus[key]['text'], coordinate['stops']
        spans = tuple(tuple(span) for span in layout[key]['words'])
        expected_stops = tuple(end for _, end in spans)
        if stops != corpus[key]['stops'] or stops[-1] != len(text) or len(spans) != 11 or not set(expected_stops) <= set(stops):
            raise ValueError('Czech state coordinates do not cover the eleven original study words')
        raw = states[offset:offset + len(stops)].astype(np.float64)
        offset += len(stops)
        unit = unit_rows(raw)
        states_for_stop = dict(zip(stops, unit, strict=True))
        encoded = tokenizer.encode(text, add_special_tokens=False)
        token_ends = {end: i for i, (start, end) in enumerate(encoded.offsets) if start < end}
        if not set(stops) <= token_ends.keys():
            raise ValueError('Czech stop is missing its exact Qwen source token')
        updates = word_updates(text, tuple(match.span() for match in TOKEN.finditer(text)), token_ends, states_for_stop)
        full = head.score(updates.features)
        state = state_head.score(updates.features)
        lookup = {span: i for i, span in enumerate(updates.spans)}
        cue = lookup[spans[6]]
        classified = {number: lookup[spans[number - 1]] for number in REGIONS}
        original = [text[left:right] for left, right in spans]
        labels = [layout[key]['condition'] == 'gp', layout[key]['animacy'] == 'animate']
        features[key] = {'length': np.log([len(text), len(original[3]), len(original[6])]),
                         'word4_ending': original[3].casefold()[-2:],
                         'terminal': states_for_stop[len(text)],
                         'cue': states_for_stop[expected_stops[6]] - states_for_stop[expected_stops[5]],
                         'pooled': softmax(full) @ updates.features['delta'],
                         'labels': [*map(float, labels), float(all(labels))]}
        local_scores[key] = {'id': key, 'scores': {str(n): {'state': float(state[i]), 'surface': float(full[i] - state[i]),
                                                          'raw': float(full[i]), 'token': updates.token_indices[i],
                                                          'span': updates.spans[i]}
                                                  for n, i in classified.items()},
                             'top1_cue': int(np.argmax(full)) == cue,
                             'cue_score_nll': float(np.logaddexp.reduce(full) - full[cue])}
    if offset != len(states):
        raise ValueError('Unexpected unreferenced raw word states')
    return features, local_scores


def family_pairs(corpus, layout):
    materials = defaultdict(dict)
    for key, row in layout.items():
        cell = (row['animacy'], row['condition'])
        words = [corpus[key]['text'][left:right] for left, right in row['words']]
        if cell in materials[row['family']]:
            raise ValueError('Repeated Czech item, animacy and condition')
        materials[row['family']][cell] = (key, words)
    groups = defaultdict(list)
    by_family = {}
    for family, variants in materials.items():
        verbs = {words[2] for _, words in variants.values()}
        if set(variants) != {(a, c) for a in ANIMACY for c in CONDITIONS} or len(verbs) != 1:
            raise ValueError('Expected complete same-verb Czech family')
        verb, = verbs
        group = sha256(('cogllm-czech-garden-v1:' + verb).encode()).hexdigest()
        groups[group].append(family)
        by_family[family] = group
        for animacy in ANIMACY:
            gp, control = (variants[(animacy, condition)][1] for condition in CONDITIONS)
            if [i for i, (left, right) in enumerate(zip(gp, control, strict=True)) if left != right] != [3]:
                raise ValueError('Each Czech pair must vary exactly the noun at word 4')
            prefixes = [corpus[variants[(animacy, condition)][0]]['text'][:layout[variants[(animacy, condition)][0]]['words'][2][1]]
                        for condition in CONDITIONS]
            if prefixes[0] != prefixes[1]:
                raise ValueError('Shared Czech prefix negative control differs in visible source text')
    if len(groups) != 24 or any(len(group) != 2 for group in groups.values()):
        raise ValueError('Expected 24 repeated-verb groups of two families')
    keys = sorted(groups)
    training = {family for group in keys[:12] for family in groups[group]}
    testing = {family for group in keys[12:] for family in groups[group]}
    return materials, groups, by_family, training, testing


def interval(values, draws):
    return {'estimate': float(values.mean()),
            'ci95': np.quantile(values[draws].mean(axis=1), [.025, .975]).tolist()}


def evaluate_cues(materials, groups, data):
    pair_results = {}
    channels = ('state', 'surface', 'raw')
    for family, mat in materials.items():
        versions = []
        for animacy in ANIMACY:
            left, right = (data[mat[(animacy, condition)][0]]['scores'] for condition in CONDITIONS)
            versions.append([[left[str(n)][c] - right[str(n)][c] for n in REGIONS] for c in channels])
        pair_results[family] = np.asarray(versions)
    unchanged_prefix = max(float(np.max(np.abs(value[:, :, :2]))) for value in pair_results.values())
    if unchanged_prefix > .001:
        raise ValueError('Different future words changed Czech scores at shared prefix words 2–3')
    estimates = np.stack([np.concatenate([pair_results[family] for family in fams]).mean(axis=0)
                          for _, fams in sorted(groups.items())])
    sample = np.random.default_rng(SEED).integers(0, len(estimates), (DRAWS, len(estimates)))
    contrasts = {name: {str(n): interval(estimates[:, idx, n - 2], sample) for n in REGIONS}
                 for idx, name in enumerate(channels)}
    combined = interval(estimates[:, 0, 5] - (estimates[:, 0, 3] + estimates[:, 0, 4]) / 2, sample)
    return {'families': len(materials), 'verb_clusters': len(groups), 'stimulus_pairs': len(materials) * 2,
            'max_shared_prefix_difference': unchanged_prefix, 'word_7_ordinal_oracle_top1': 1.,
            'actual_top1': float(np.mean([item['top1_cue'] for item in data.values()])),
            'actual_cue_nll_nats': float(np.mean([item['cue_score_nll'] for item in data.values()])),
            'gp_minus_control': contrasts, 'state_word7_minus_mean_word5_word6': combined,
            'primary_pass': contrasts['state']['7']['ci95'][0] > 0 and combined['ci95'][0] > 0}


def qmis_rows(raw, layout, experiment, families):
    comparison = {'gp': 'gp', 'nongp': 'control'}
    result = []
    for item in raw:
        source = layout[item['id']]
        if item['experiment'] == experiment and item['question'] == 'qmis' and source['family'] in families:
            result.append({'reader_id': item['reader'], 'family': source['family'], 'animacy': source['animacy'],
                           'condition': comparison[source['condition']], 'id': source['id'],
                           'response': None if item['response'] == 'missing' else item['response']})
    return result


def fit_forecast(work, receipt, layout, features, training, testing):
    responses = work / 'prepared' / 'outcomes.jsonl'
    if records.digest(responses) != receipt['file_sha256']['prepared/outcomes.jsonl']:
        raise ValueError('Czech source cohort changed before fitting')
    with responses.open(encoding='utf-8') as stream:
        training_rows = (row for row in map(json.loads, stream) if row['experiment'] == 2)
        train = qmis_rows(training_rows, layout, 2, training)
    train_ids = sorted({key for key, row in layout.items() if row['family'] in training})
    test_ids = sorted({key for key, row in layout.items() if row['family'] in testing})
    counts = {key: np.zeros(3, dtype=np.int64) for key in train_ids}
    for row in train:
        if row['response'] is not None:
            counts[row['id']][readouts.RESPONSES.index(row['response'])] += 1
    heads = {name: readouts.fit([features[key] for key in train_ids], np.stack([counts[key] for key in train_ids]), name)
             for name in readouts.ARMS}
    predict_data = [features[key] for key in test_ids]
    probabilities = {name: head.predict(predict_data) for name, head in heads.items()}
    rows = []
    for i, key in enumerate(test_ids):
        meta = layout[key]
        rows.append({'id': key, 'family': meta['family'], 'animacy': meta['animacy'],
                     'condition': 'gp' if meta['condition'] == 'gp' else 'control',
                     'probabilities': {name: values[i].tolist() for name, values in probabilities.items()}})
    return heads, rows


def run(work, extraction, tokenizer, output):
    if output.exists():
        raise FileExistsError(output)
    if records.digest(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError('Czech analysis protocol changed')
    receipt, corpus, layout, order, states, qwen_tokenizer = sealed_material(work, extraction, tokenizer)
    head = CueReadout(load_checkpoint(DEFAULT_CHECKPOINT))
    materials, groups, by_family, training, testing = family_pairs(corpus, layout)
    features, cue_forecasts = make_rows(corpus, layout, order, states, qwen_tokenizer, head)
    heads, behavior = fit_forecast(work, receipt, layout, features, training, testing)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.czech-garden-') as temporary:
        stage = Path(temporary) / 'scored'
        stage.mkdir()
        files = {'cue-forecasts.json': cue_forecasts, 'response-forecasts.json': behavior,
                 'fits.json': {name: result.receipt() for name, result in heads.items()},
                 'split.json': {'training_families': sorted(training), 'held_families': sorted(testing),
                                'verb_clusters_sha256': {k: sorted(v) for k, v in sorted(groups.items())}}}
        for name, value in files.items():
            records.write_json(stage / name, value)
        seal_path = stage / 'forecast-seal.json'
        records.write_json(seal_path, {'protocol_sha256': PROTOCOL_SHA256, 'source_sha256': records.digest(SOURCE),
                                       'cue_release_sha256': RELEASE_SHA256, 'scorer_sha256': records.digest(Path(__file__)),
                                       'fit_sha256': records.digest(Path(readouts.__file__)),
                                       'metrics_sha256': records.digest(Path(metrics.__file__)),
                                       'extraction_sha256': records.digest(extraction / 'seal.json'),
                                       'file_sha256': {name: records.digest(stage / name) for name in files}})
        responses = records.read_rows(work / 'prepared' / 'outcomes.jsonl')
        forecasts = records.read_json(stage / 'response-forecasts.json')
        comparisons = tuple(name for name in readouts.ARMS if name != 'unmarked')
        results = {}
        for experiment in (2, 3):
            rows = qmis_rows(responses, layout, experiment, testing)
            results[str(experiment)] = metrics.evaluate(rows, forecasts, primary='unmarked', comparators=comparisons,
                                                        family_groups={key: by_family[key] for key in testing})
        main = results['3']['gates']
        criteria = {'effect_positive': main['positive_effect'], 'effect_equivalent': main['effect_equivalent'],
                    **{f'lower_nll_vs_{k}': results['3']['primary_nll_minus_comparator_nats'][k]['ci95'][1] < 0
                       for k in SUBSETS}}
        records.write_json(stage / 'results.json', {'response_gates': criteria, 'response_pass': all(criteria.values()),
                          'responses': results, 'localization': evaluate_cues(materials, groups,
                                                                           records.read_json(stage / 'cue-forecasts.json')),
                          'forecast_seal_sha256': records.digest(seal_path)})
        stage.rename(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('work', 'extraction', 'tokenizer', 'output'):
        parser.add_argument('--' + argument, type=Path, required=True)
    run(**vars(parser.parse_args()))
