"""Freeze and score the Kerwer article assay using sealed serving forecasts."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from .kerwer_holdout import ARMS, DOI_PREFIX, STUDIES, digest


CANDIDATE_ID = '74c26ff02569b2b3bc28ed291bf15fd2f85db5b4c93a3ad7bfc34f6a5d333ea0'
STIMULI = 'model_input/stimuli.jsonl'
DESIGN = 'scoring/design.jsonl'
OUTCOMES = 'scoring/public-study-ratings.jsonl'
INPUTS = {
    STIMULI: 'fd43d9d5768ec1dfb66b9025d9ad777639346e458161fd2f0a918e3487264b58',
    DESIGN: '05357eb388007ca1d831385029f5e9e905f52c46f16a185451ed58462a72c72c',
    OUTCOMES: '78503ca22bfa1fb8b06dab91919864eb667202629effdf164ff609539648f5a2',
}
LENGTH_HEAD = '7ca3cbf11b44e2d17a8a76a3d34baed49351412e34d1a2946c7e2a1b915c25e6'
ALIGNMENT = '3bc68d53462d8f82477e5280a9424c4b4a1ca90d8d8bb7b92778ef8cda270fb3'
LEVELS = np.arange(1, 6)
TARGETS = ('ease', 'understanding')
CONTRASTS = {'rewrite': (1, 0), 'headings': (2, 1)}
PROTOCOL = Path(__file__).with_name('kerwer-article-holdout.md')
METRICS = ('human_effect', 'candidate_effect', 'length_effect', 'candidate_minus_human',
           'candidate_absolute_error', 'length_absolute_error', 'candidate_error_reduction',
           'candidate_direction_agreement', 'length_direction_agreement')
READERS, OBSERVATIONS = 166, 1947
DRAWS, SEED, BATCH = 10_000, 20261121, 128
MAGNITUDE_MARGIN = 0.5 / 7


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, item):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(item, stream, indent=2, allow_nan=False)
        stream.write('\n')


def rows(path):
    with path.open(encoding='utf-8') as stream:
        yield from map(json.loads, stream)


def fingerprint(item):
    return sha256(json.dumps(item, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def rules():
    return {'candidate_id': CANDIDATE_ID, 'protocol_sha256': digest(PROTOCOL), 'scorer_sha256': digest(Path(__file__)),
            'input_sha256': {'stimuli': INPUTS[STIMULI], 'design': INPUTS[DESIGN], 'human_trials': INPUTS[OUTCOMES]},
            'length_control_sha256': LENGTH_HEAD, 'alignment_sha256': ALIGNMENT,
            'contrasts': {name: [list(ARMS.values())[a], list(ARMS.values())[b]] for name, (a, b) in CONTRASTS.items()},
            'readout': 'mean of shipped ordinal perceived understanding over the locked text coordinates',
            'human_response_scale': [1, 8], 'model_scale': [1, 5],
            'magnitude_margin_source_scale': 0.5, 'equivalence_interval': 0.90,
            'reader_and_family_bootstrap': {'draws': DRAWS, 'seed': SEED, 'batch': BATCH}}


def read_study(root):
    if any(digest(root / name) != expected for name, expected in INPUTS.items()):
        raise ValueError('The Kerwer source bytes differ from the frozen holdout')
    documents = {row['id']: row for row in rows(root / STIMULI)}
    design = {(row['study'], row['arm']): (row['id'], row['doi']) for row in rows(root / DESIGN)}
    cases = []
    for number, (doi, _, _) in enumerate(STUDIES, 1):
        family = [design[str(number), arm] for arm in ARMS.values()]
        if any(published != DOI_PREFIX + doi for _, published in family):
            raise ValueError('Study family differs from the published DOI key')
        cases.append([identifier for identifier, _ in family])
    if len(design) != len(documents) or set(documents) != {key for family in cases for key in family}:
        raise ValueError('The full three-arm design must cover the stimulus inventory')
    return documents, np.asarray(cases)


def frozen_geometry(documents, cases, checkpoint_path, tokenizer_path):
    from tokenizers import Tokenizer

    from research.gradio_app.checkpoint import load_checkpoint
    from research.gradio_app.plaba import TOKENIZER_SHA256, sentence_spans
    from research.stage3.run import sentence_position

    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint.identifier != CANDIDATE_ID or digest(tokenizer_path) != TOKENIZER_SHA256:
        raise ValueError('The serving checkpoint or tokenizer differs from the frozen release candidate')
    segmenter = checkpoint.segmenter()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    served, paragraphs, bodies = {}, {}, {}
    for key, row in documents.items():
        text = row['text']
        full = tokenizer.encode(text, add_special_tokens=False)
        served[key] = []
        for start, end in sentence_spans(text, segmenter):
            position = sentence_position(text, {'start': start, 'end': end, 'id': key}, full.offsets)
            prefix = tokenizer.encode(text[:end], add_special_tokens=False)
            if full.ids[:position + 1] != prefix.ids:
                raise ValueError(f'A served sentence lacks an exact causal stop: {key}/{end}')
            served[key].append([start, end, position])
        paragraphs[key] = [text[left:right] for left, right in row['body_spans']]
        bodies[key] = {(paragraph, start, end): [left + start, left + end]
                       for paragraph, (left, right) in enumerate(row['body_spans'])
                       for start, end in sentence_spans(text[left:right], segmenter)}

    def matched(key):
        coordinates = {tuple(span[:2]) for span in served[key]}
        return {relative for relative, absolute in bodies[key].items() if tuple(absolute) in coordinates}

    selection = {'rewrite': {}, 'headings': {}}
    for abstract, plain, headings in cases:
        for key in (abstract, plain):
            if set(bodies[key]) != matched(key):
                raise ValueError('A primary arm is missing an exact body sentence')
            selection['rewrite'][key] = list(bodies[key].values())
        exact_headings = matched(headings)
        if (paragraphs[plain] != paragraphs[headings] or bodies[plain].keys() != bodies[headings].keys()
                or set(bodies[headings]) - exact_headings != {min(bodies[headings])}):
            raise ValueError('Heading exclusion differs from the single merged first sentence per family')
        for key in (headings, plain):
            selection['headings'][key] = [absolute for relative, absolute in bodies[key].items()
                                          if relative in exact_headings]
    coverage = {'served_sentences': sum(map(len, served.values())),
                'all_body_sentences': sum(map(len, bodies.values())),
                'served_exact_body_sentences': sum(len(matched(key)) for key in documents),
                'primary_scored_sentences': sum(map(len, selection['rewrite'].values())),
                'heading_comparison_sentences_per_arm': sum(len(selection['headings'][key]) for key in cases[:, 2]),
                'heading_comparison_exclusions_per_arm': len(cases)}
    if tuple(coverage.values()) != (514, 470, 458, 277, 181, 12):
        raise ValueError('Serving segmentation or paired exclusion coverage differs from the preflight')
    return {'served': served, 'eligible': selection, 'coverage': coverage}


def freeze(holdout, checkpoint, tokenizer, length_heads, output):
    documents, cases = read_study(holdout)
    if digest(length_heads) != LENGTH_HEAD:
        raise ValueError('Length baseline differs from the frozen PLABA transfer weights')
    alignment = frozen_geometry(documents, cases, checkpoint, tokenizer)
    if fingerprint(alignment) != ALIGNMENT:
        raise ValueError('Serving alignment differs from the validated tokenizer preflight')
    write_json(output, {**rules(), 'alignment': alignment})


def read_forecasts(filename, seal_file, served):
    seal = read_json(seal_file)
    if (seal['candidate_id'] != CANDIDATE_ID or seal['input_sha256'] != INPUTS[STIMULI]
            or seal['predictions_sha256'] != digest(filename)
            or seal['documents'] != len(served) or seal['forwards'] != len(served)):
        raise ValueError('Candidate, stimulus, forecast bytes or forward count differs from its serving seal')
    forecasts = {}
    for row in rows(filename):
        key = row['id']
        sentences = row['sentences']
        if key in forecasts or key not in served or len(sentences) != len(served[key]):
            raise ValueError(f'Unexpected or incomplete served sentence inventory: {key}')
        forecasts[key] = {}
        for sentence, expected in zip(sentences, served[key], strict=True):
            coordinates = [sentence[axis] for axis in ('start', 'end', 'token')]
            if any(type(value) is not int for value in coordinates) or coordinates != expected:
                raise ValueError(f'Forecast sentence coordinates differ from the serving route: {key}')
            for target in TARGETS:
                probability = np.asarray(sentence[f'probability_{target}'], dtype=np.float64)
                if (type(sentence[target]) not in (int, float) or probability.shape != LEVELS.shape
                        or not np.isfinite(probability).all()
                        or np.any((probability < 0) | (probability > 1))
                        or not math.isclose(float(probability.sum()), 1.0, abs_tol=1e-6)
                        or not math.isclose(sentence[target], float(probability @ LEVELS), abs_tol=1e-6)):
                    raise ValueError(f'Invalid ordinal probability or expectation: {key}/{target}')
            forecasts[key][tuple(expected[:2])] = sentence['understanding']
    if forecasts.keys() != served.keys():
        raise ValueError('One or more frozen articles have no forecast')
    return forecasts


def controls(path, documents, selection):
    if digest(path) != LENGTH_HEAD:
        raise ValueError('Length baseline differs from the frozen PLABA transfer weights')
    with np.load(path, allow_pickle=False) as archive:
        labels = tuple(archive['targets'])
        weight = archive['length_only_weights'][labels.index('perceived_understanding')]
        center, scale = archive['length_mean'], archive['length_scale']
        if weight.shape != (3, len(LEVELS)) or not np.array_equal(archive['ratings'], LEVELS):
            raise ValueError('Frozen length-only readout has unexpected targets')
    result = {}
    for case, spans in selection.items():
        texts = [documents[case]['text'][start:end] for start, end in spans]
        lengths = (np.log1p([(len(text), len(text.split())) for text in texts]) - center) / scale / np.sqrt(2)
        logits = np.column_stack((np.ones(len(spans)), lengths)) @ weight
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        result[case] = float((exp @ LEVELS / exp.sum(axis=1)).mean())
    return result


def human_matrices(path, cases):
    all_rows = list(rows(path))
    readers = {reader: index for index, reader in enumerate(sorted({row['reader'] for row in all_rows}))}
    columns = {case: index for index, case in enumerate(cases.ravel())}
    observed = np.zeros((len(readers), len(columns)), dtype=np.float64)
    for row in all_rows:
        reader, case = readers[row['reader']], columns[row['id']]
        value = row['comprehensibility']
        if observed[reader, case] or type(value) is not int or not 1 <= value <= 8:
            raise ValueError('Duplicate trial or invalid observed comprehensibility')
        observed[reader, case] = value
    if (len(readers) != READERS or len(all_rows) != OBSERVATIONS
            or np.any((observed > 0).reshape(READERS, *cases.shape).sum(axis=-1) > 1)):
        raise ValueError('Unexpected reader inventory or repeated exposure to the same family')
    return observed > 0, observed


def effects(human, model):
    candidate = np.broadcast_to(model[..., 0], human.shape)
    length = np.broadcast_to(model[..., 1], human.shape)
    difference = candidate - human
    candidate_error, length_error = np.abs(difference), np.abs(length - human)
    return np.stack((human, candidate, length, difference, candidate_error, length_error,
                     length_error - candidate_error, (1 + np.sign(candidate) * np.sign(human)) / 2,
                     (1 + np.sign(length) * np.sign(human)) / 2), axis=-1)


def human_effects(totals, counts, reader_weights, families):
    draws = reader_weights @ counts
    if (draws == 0).any():
        raise ValueError('A frozen bootstrap replicate omitted all readers from a study arm')
    means = ((reader_weights @ totals) / draws).reshape(-1, families, 3)
    left, right = zip(*CONTRASTS.values())
    return (means[:, :, left] - means[:, :, right]) / 7


def bootstrap(recorded, observed, model):
    readers, families = len(recorded), len(model)
    point = effects(human_effects(observed, recorded, np.ones((1, readers)), families), model)[0]
    reader_rng, family_rng = (np.random.default_rng(seed) for seed in np.random.SeedSequence(SEED).spawn(2))
    crossed, readers_only = [], []
    for start in range(0, DRAWS, BATCH):
        size = min(BATCH, DRAWS - start)
        reader_weights = reader_rng.multinomial(readers, np.full(readers, 1 / readers), size)
        family_weights = family_rng.multinomial(families, np.full(families, 1 / families), size)
        draw_metrics = effects(human_effects(observed, recorded, reader_weights, families), model)
        crossed.append(np.einsum('bf,bfcm->bcm', family_weights / families, draw_metrics))
        readers_only.append(draw_metrics.mean(axis=1))
    return point, np.concatenate(crossed), np.concatenate(readers_only)


def metric_report(point, draws, readers_only, case, metric):
    family = point[:, case, metric]
    return {'estimate': float(family.mean()), 'crossed_ci95': np.quantile(draws[:, case, metric], [0.025, 0.975]).tolist(),
            'fixed_families_ci95': np.quantile(readers_only[:, case, metric], [0.025, 0.975]).tolist(),
            'leave_one_family_out': ((family.sum() - family) / (len(family) - 1)).tolist()}


def score(holdout, protocol, predictions, seal, length_heads, output):
    contract = read_json(protocol)
    alignment = contract.pop('alignment')
    if fingerprint(alignment) != ALIGNMENT or contract != rules():
        raise ValueError('The frozen scoring specification or alignment has changed')
    documents, cases = read_study(holdout)
    forecasts = read_forecasts(predictions, seal, alignment['served'])
    model = np.empty((len(cases), len(CONTRASTS), 2))
    for index, (name, (left, right)) in enumerate(CONTRASTS.items()):
        selection = alignment['eligible'][name]
        candidate = {key: float(np.mean([forecasts[key][tuple(span)] for span in spans]))
                     for key, spans in selection.items()}
        length = controls(length_heads, documents, selection)
        for method, values in enumerate((candidate, length)):
            model[:, index, method] = [(values[row[left]] - values[row[right]]) / 4 for row in cases]
    recorded, observed = human_matrices(holdout / OUTCOMES, cases)
    point, draws, readers_only = bootstrap(recorded, observed, model)
    results = {}
    for index, name in enumerate(CONTRASTS):
        effects_summary = {metric: metric_report(point, draws, readers_only, index, position)
                           for position, metric in enumerate(METRICS)}
        effects_summary['human_effect']['source_scale_change'] = float(point[:, index, 0].mean() * 7)
        for method in ('candidate_effect', 'length_effect'):
            effects_summary[method]['source_scale_range_equivalent'] = effects_summary[method]['estimate'] * 7
        results[name] = effects_summary
    primary = results['rewrite']
    human_interval = primary['human_effect']['crossed_ci95']
    interval90 = np.quantile(draws[:, 0, METRICS.index('candidate_minus_human')], [0.05, 0.95]).tolist()
    sign = np.sign(primary['human_effect']['estimate'])
    direction = bool(sign != 0 and np.sign(primary['candidate_effect']['estimate']) == sign)
    human_nonzero = bool(human_interval[0] > 0 or human_interval[1] < 0)
    equivalence = bool(-MAGNITUDE_MARGIN < interval90[0] <= interval90[1] < MAGNITUDE_MARGIN)
    write_json(output, {'candidate_id': CANDIDATE_ID, 'protocol_sha256': digest(protocol),
                        'forecasts_sha256': digest(predictions), 'forecasts_seal_sha256': digest(seal),
                        'coverage': alignment['coverage'], 'human_readers': READERS, 'human_observations': OBSERVATIONS,
                        'macro_effects': results,
                        'primary': {'human_ci95_excludes_zero': human_nonzero, 'direction_agrees': direction,
                                    'candidate_minus_human_crossed_ci90': interval90, 'margin': MAGNITUDE_MARGIN,
                                    'difference_within_margin': equivalence,
                                    'effect_gate_passes': direction and human_nonzero and equivalence,
                                    'improves_on_length': primary['candidate_error_reduction']['crossed_ci95'][0] > 0},
                        'families': [{'doi': DOI_PREFIX + article[0],
                                      'normalized': {name: dict(zip(METRICS, point[family, index].tolist()))
                                                     for index, name in enumerate(CONTRASTS)}}
                                     for family, article in enumerate(STUDIES)]})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command, paths in (('freeze', ('checkpoint', 'tokenizer')), ('score', ('protocol', 'predictions', 'seal'))):
        stage = commands.add_parser(command)
        for field in ('holdout', 'length_heads', 'output', *paths):
            stage.add_argument('--' + field.replace('_', '-'), type=Path, required=True)
    arguments = vars(parser.parse_args())
    action = {'freeze': freeze, 'score': score}[arguments.pop('command')]
    action(**arguments)
