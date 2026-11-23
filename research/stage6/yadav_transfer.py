"""Run a sealed sentence-rank transfer diagnostic on Yadav Experiment 1."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from research.core_features.readback_score import paired_material_interval
from research.gradio_app.causal_plaba import CHECKPOINT, CausalPlabaReader, digest
from research.gradio_app.combined import ExtractionTokenizer
from research.stage3.probe import _fit_ridge, _length_standardization, _load_plaba
from research.stage3.run import save_json

from .yadav_audit import CONDITIONS, LANGUAGES, RELEASES, build_stimulus_lookup


METHODS = ('plaba_length', 'causal_plaba')
STIMULI = 'experiment1-stimuli.txt'
TRIALS = 'experiment1-slider.rda'


def source(raw, name):
    path = raw / name
    if digest(path) != RELEASES[name][1]:
        raise ValueError(f'Unexpected Yadav release bytes: {name}')
    return path


def lexical(texts):
    return np.log1p([(len(text), len(text.split())) for text in texts])


def predict(raw, plaba, model, output):
    _, sentences, ratings, _, _ = _load_plaba(plaba)
    lengths = lexical(text for _, text in sentences.values())
    center, scale = _length_standardization(lengths)
    ratings = np.array([np.mean(ratings[key], axis=0)[0] for key in sentences])
    baseline, intercept = _fit_ridge((lengths - center) / scale, ratings)
    stimulus_path = source(raw, STIMULI)
    stimuli = build_stimulus_lookup(stimulus_path)
    reader = CausalPlabaReader(model)
    reader.backbone.tokenizer = ExtractionTokenizer(model)
    output.mkdir(parents=True, exist_ok=False)
    predictions = []
    for (language, item, level, paraphrase), text in sorted(stimuli.items()):
        end = len(text.rstrip())
        text = text[:end]
        encoding = reader.backbone.encode(text)
        position = reader.position(encoding, 0, end)
        if position is None:
            raise ValueError(f'Uncertified isolated stimulus at {(language, item, level, paraphrase)}')
        estimate = reader.score(encoding.states[[position]], [text])[0, 0]
        reference = np.clip((lexical([text])[0] - center) / scale @ baseline + intercept, 1, 5)
        predictions.append({'language': language, 'item': item, 'level': level, 'paraphrase': paraphrase,
                            'plaba_length': float(reference), 'causal_plaba': float(estimate)})
    destination = output / 'predictions.jsonl'
    with destination.open('w', encoding='utf-8', newline='\n') as stream:
        for row in predictions:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    save_json(output / 'seal.json', {
        'stimulus_sha256': digest(stimulus_path), 'forecast_sha256': digest(destination),
        'head_sha256': digest(CHECKPOINT / 'head.npz'), 'code_sha256': digest(Path(__file__)),
        'plaba_data_sha256': {name: digest(plaba / name) for name in ('stimuli.jsonl', 'trials.jsonl', 'splits.json')},
        'length_baseline': {'weights': baseline.tolist(), 'intercept': float(intercept),
                            'center': center.tolist(), 'scale': scale.tolist()},
        'model_calls': len(stimuli), 'runtime': 'MLX FP32 with pinned Transformers tokenizer',
    })


def score(raw, run, output):
    import rdata

    path = run / 'predictions.jsonl'
    seal = json.loads((run / 'seal.json').read_text(encoding='utf-8'))
    stimulus_path = source(raw, STIMULI)
    if (digest(path) != seal['forecast_sha256'] or seal['stimulus_sha256'] != digest(stimulus_path)
            or digest(Path(__file__)) != seal['code_sha256']
            or digest(CHECKPOINT / 'head.npz') != seal['head_sha256']):
        raise ValueError('Forecast, source or scoring rules differ from the pre-scoring seal')

    stimuli = build_stimulus_lookup(stimulus_path)
    predictions = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        key = row['language'], row['item'], row['level'], row['paraphrase']
        if key in predictions or any(not 1 <= row[method] <= 5 for method in METHODS):
            raise ValueError(f'Invalid forecast: {key}')
        predictions[key] = row
    if predictions.keys() != stimuli.keys():
        raise ValueError('Forecasts do not cover the complete stimulus inventory')

    trials_path = source(raw, TRIALS)
    data = rdata.read_rda(trials_path)['dat.slider']
    observed = defaultdict(list)
    for row in data.loc[data['condition'] != 'F'].itertuples(index=False):
        key = row.Language, int(row.item), row.Level, row.Paraphrase
        if (row.stimulus.rstrip() != stimuli[key].rstrip() or CONDITIONS[row.condition] != (row.Level, row.Paraphrase)
                or not 1 <= row.response <= 9):
            raise ValueError(f'Unmatched human observation: {key}')
        observed[key].append((float(row.response) - 1) * 12.5)
    if len(observed) != 189 or sum(map(len, observed.values())) != 1701:
        raise ValueError('Unexpected Yadav Experiment 1 observation counts')

    details = []
    for level in ('Level 2', 'Level 3'):
        for item in range(1, 25):
            keys = {language: ((language, item, level, 'No'), (language, item, level, 'Yes')) for language in LANGUAGES}
            if not all(key in observed for pair in keys.values() for key in pair):
                continue
            for language, (embedded, paraphrase) in keys.items():
                human_cost = float(np.mean(observed[embedded]) - np.mean(observed[paraphrase]))
                effects = {method: predictions[paraphrase][method] - predictions[embedded][method]
                           for method in METHODS}
                agreement = {method: (0.5 if abs(human_cost) <= 1e-12 or abs(cost) <= 1e-12
                                      else float(np.sign(cost) == np.sign(human_cost)))
                             for method, cost in effects.items()}
                details.append({'item': item, 'language': language, 'level': level, 'human_cost_0_to_100': human_cost,
                                'model_ease_cost_1_to_5': effects, 'direction_agreement': agreement})

    summary = {}
    for level in ('Level 2', 'Level 3'):
        for language in LANGUAGES:
            selected = [row for row in details if (row['level'], row['language']) == (level, language)]
            summary[f'{language} {level}'] = {
                'paired_items': len(selected),
                'mean_human_cost_0_to_100': float(np.mean([row['human_cost_0_to_100'] for row in selected])),
                'mean_model_cost_1_to_5': {method: float(np.mean([row['model_ease_cost_1_to_5'][method]
                                                               for row in selected])) for method in METHODS},
                'direction_agreement': {method: float(np.mean([row['direction_agreement'][method]
                                                             for row in selected])) for method in METHODS},
            }
    compare = paired_material_interval(np.array([row['direction_agreement']['causal_plaba']
                                                - row['direction_agreement']['plaba_length'] for row in details]),
                                       [{'group_id': str(row['item'])} for row in details])
    save_json(output, {
        'scope': 'Experiment 1 isolated native Dutch/German sentence difficulty, with a frozen English PLABA head.',
        'trials': sum(map(len, observed.values())), 'matched_items_language_levels': len(details),
        'forecasts_sha256': digest(path), 'seal_sha256': digest(run / 'seal.json'), 'trials_sha256': digest(trials_path),
        'direction_agreement_by_method': {method: float(np.mean([row['direction_agreement'][method]
                                                                for row in details])) for method in METHODS},
        'direction_agreement_causal_plaba_minus_length': compare,
        'diagnostic_superior_to_length': compare['descriptive_paired_material_95_ci'][0] > 0,
        'conditions': summary, 'paired_observations': details,
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='phase', required=True)
    for phase, arguments in (('predict', ('raw', 'plaba', 'model', 'output')), ('score', ('raw', 'run', 'output'))):
        sub = actions.add_parser(phase)
        for name in arguments:
            sub.add_argument(f'--{name}', type=Path, required=True)
    args = vars(parser.parse_args())
    action = {'predict': predict, 'score': score}[args.pop('phase')]
    action(**args)
