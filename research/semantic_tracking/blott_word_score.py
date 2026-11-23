"""Train a text-only word ranker and test known-cue versus causal word pooling."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.special import softmax

from research.semantic_tracking import blott_format_score as behavior
from research.semantic_tracking import blott_update_fit as button_fit
from research.semantic_tracking import blott_update_transfer as old
from research.semantic_tracking import blott_word_fit as ranking
from research.semantic_tracking import blott_word_metrics as metrics
from research.semantic_tracking import blott_word_states as states
from research.semantic_tracking import frick_consistency_score as records


PROTOCOL = Path(__file__).with_name('blott_word_protocol.md')
PROTOCOL_SHA256 = '213d033770b117a18b933c813b610f48362a5556d06a9e927ee63f3593c83488'
OLD_UPDATE = PROTOCOL.parents[1] / 'results' / 'blott-update-transfer-l4'
OLD_UPDATE_SEAL = 'd11c5c58fd7e87e9a730185588d82798eb1df600e52da9e3f5749a600a378749'
BLOCKS = {'position': ('position',), 'surface': ('position', 'surface'),
          'magnitude': ('position', 'surface', 'change_norm'),
          'state': ('position', 'surface', 'change_norm', 'delta')}
FOLDS = 5


def sources(early, late, updates):
    receipt = records.read_json(states.SOURCE_RECEIPT)['frozen_source_sha256']
    layouts = []
    for year, directory in (('2021', early), ('2023', late)):
        file = directory / 'layout.jsonl'
        if (records.digest(file) != receipt[year]['design']
                or records.digest(directory / 'stimuli.jsonl') != receipt[year]['stimuli']):
            raise ValueError('Ground truth must come from the exact published cue source inventory')
        layouts.append(records.indexed(records.read_rows(file), f'Blott {year} cue labeling'))
    cohorts = []
    for idx, layout in enumerate(layouts):
        training = idx == 0
        eligible = {key: row for key, row in layout.items() if row['cue_span'] is not None}
        pairs, _ = old.ordered_pairs(layout, eligible, training=training)
        if training:
            pairs, names = old.word_sorted_training(pairs, layout, early / 'stimuli.jsonl')
        else:
            names = tuple(pairs)
        families = [(name, keys) for name, keys in zip(names, pairs.values(), strict=True)]
        gold = {key: updates[key].spans.index(tuple(eligible[key]['cue_span'])) for _, keys in families for key in keys}
        cohorts.append((families, gold))
    if [len(families) for families, _ in cohorts] != [46, 44]:
        raise ValueError('The complete word-ranking comparison requires the locked Blott cohorts')
    return cohorts


def train_models(families, target, updates):
    keys = [key for _, variants in families for key in variants]
    inputs = [updates[key].features for key in keys]
    gold = np.array([target[key] for key in keys])
    return {name: ranking.fit_ranker(inputs, gold, blocks=blocks) for name, blocks in BLOCKS.items()}


def score_cohort(families, fits, updates, fold):
    forecasts = []
    for family, members in families:
        for key in members:
            item = updates[key]
            logits = {'uniform': [0.] * len(item.ordinals)}
            for name, fitted in fits.items():
                raw = fitted.score(item.features)
                for pos in range(len(item.ordinals)):
                    prefix = {block: values[:pos + 1] for block, values in item.features.items()}
                    if not np.array_equal(fitted.score(prefix), raw[:pos + 1]):
                        raise ValueError('A raw ranker score changes when an unread suffix is removed')
                logits[name] = raw.tolist()
            forecasts.append({'id': key, 'family': family, 'fold': fold, 'ordinals': item.ordinals,
                              'spans': item.spans, 'token_indices': item.token_indices, 'logits': logits})
    return forecasts


def fit_rankers(early, late, updates):
    families, gold = early
    historical, folds = [], []
    for fold in range(FOLDS):
        training = [pair for index, pair in enumerate(families) if index % FOLDS != fold]
        heldout = families[fold::FOLDS]
        fits = train_models(training, gold, updates)
        historical.extend(score_cohort(heldout, fits, updates, fold))
        folds.append({'held_out_families': [key for key, _ in heldout],
                      'fits': {name: fit.diagnostics for name, fit in fits.items()}})
    full = train_models(families, gold, updates)
    return historical, score_cohort(late[0], full, updates, fold=0), {
        'folds_2021': folds, 'fits_for_2023': {name: fit.receipt() for name, fit in full.items()},
    }


def frozen_button_model():
    path = OLD_UPDATE / 'forecast-seal.json'
    seal = records.read_json(path)
    fit_path = OLD_UPDATE / 'fits.json'
    archived = OLD_UPDATE / 'forecasts.json'
    if (records.digest(path) != OLD_UPDATE_SEAL or records.digest(fit_path) != seal['file_sha256']['fits.json']
            or records.digest(archived) != seal['file_sha256']['forecasts.json']):
        raise ValueError('Sentence evaluation requires the exact previously sealed known-cue readout')
    model = records.read_json(fit_path)['fits']['update']
    if model['blocks'] != ['length', 'update']:
        raise ValueError('The frozen readout must use only length and the selected update vector')
    fit = button_fit.Fit(tuple(model['blocks']), tuple(map(np.asarray, model['centers'])),
                         tuple(map(np.asarray, model['scales'])), np.asarray(model['coefficients']), {})
    baselines = records.indexed(records.read_json(archived), 'earlier known-cue forecasts')
    return fit, baselines


def pool_forecasts(rows, updates):
    fit, _ = frozen_button_model()
    result = []
    for row in rows:
        key = row['id']
        candidate = updates[key]
        delta = candidate.features['delta']
        methods = tuple(row['logits'])
        pooled = np.vstack([softmax(row['logits'][name]) @ delta for name in methods])
        lengths = np.broadcast_to(np.log([len(candidate.text), len(candidate.text.split())]), (len(methods), 2))
        v = fit.predict({'length': lengths, 'update': pooled})
        result.append({'id': key, 'stop': len(candidate.text), 'logits': dict(zip(methods, v.tolist(), strict=True))})
    return result


def evaluate_buttons(late, rows, updates, targets):
    data = records.indexed(rows, 'word-pooled sentence predictions')
    reference = old.baseline_forecasts()
    fit, marked = frozen_button_model()
    known = {}
    for key in data:
        item = updates[key]
        selected = item.features['delta'][targets[key]][None, :]
        length = np.log([[len(item.text), len(item.text.split())]])
        known[key] = float(fit.predict({'length': length, 'update': selected})[0])
    maximum_replay = max(abs(logit - marked[key]['logits']['update']) for key, logit in known.items())
    if maximum_replay > 5e-5:
        raise ValueError('The full word-stops export does not reproduce the previously marked-cue logits')
    results, gates = {}, {}
    for arm in ('uniform', 'position', 'surface', 'magnitude', 'marked_cue'):
        rekeyed = []
        for row in reference:
            key = row['id']
            if row['stop'] != data[key]['stop']:
                rekeyed.append(row)
                continue
            logits = data[key]['logits']
            control = known[key] if arm == 'marked_cue' else logits[arm]
            rekeyed.append({**row, 'logits': {'qwen': logits['state'], 'surface': control}})
        scored = behavior.evaluate(late, rekeyed)
        summary = scored['summary']
        results[arm] = {k: summary[k] for k in ('qwen_nll_gain', 'qwen_nll', 'surface_nll', 'human_effect',
                                                'qwen_effect', 'surface_effect', 'qwen_minus_human_effect')}
        gates[arm] = scored['gates']['nll_gain']
    return {'comparisons': results, 'primary_nll_pass': gates['uniform'] and gates['surface'],
            'observational_pairs': scored['complete_pairs'], 'families': scored['families'],
            'readers': scored['readers'], 'marked_cue_max_logit_replay_error': maximum_replay,
            'retrospective': True}


def run(work, extraction, tokenizer, early, late, output):
    if output.exists():
        raise FileExistsError(output)
    if records.digest(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError('The text-only cue study protocol has changed')
    updates, provenance = states.load_updates(work, extraction, tokenizer, PROTOCOL_SHA256)
    early_cohort, late_cohort = sources(early, late, updates)
    early_rows, late_rows, fit_receipt = fit_rankers(early_cohort, late_cohort, updates)
    pooled = pool_forecasts(late_rows, updates)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.blott-word-') as temporary:
        stage = Path(temporary) / 'scored'
        stage.mkdir()
        files = {'word-2021-forecasts.json': early_rows, 'word-2023-forecasts.json': late_rows,
                 'button-2023-forecasts.json': pooled, 'fits.json': fit_receipt}
        for name, data in files.items():
            records.write_json(stage / name, data)
        seal = stage / 'forecast-seal.json'
        records.write_json(seal, {**provenance, 'protocol_sha256': PROTOCOL_SHA256,
                                  'frozen_button_seal_sha256': OLD_UPDATE_SEAL,
                                  'scorer_sha256': records.digest(Path(__file__)),
                                  'ranker_sha256': records.digest(Path(ranking.__file__)),
                                  'metric_code_sha256': records.digest(Path(metrics.__file__)),
                                  'features_sha256': records.digest(Path(states.__file__)),
                                  'file_sha256': {name: records.digest(stage / name) for name in files}})
        word_results = {}
        for year, (_, targets) in [('2021', early_cohort), ('2023', late_cohort)]:
            forecasts = records.read_json(stage / f'word-{year}-forecasts.json')
            scored_rows = [{**row, 'gold': targets[row['id']]} for row in forecasts]
            word_results[year] = metrics.evaluate(scored_rows, primary=year == '2021')
        buttons = evaluate_buttons(late, records.read_json(stage / 'button-2023-forecasts.json'), updates, late_cohort[1])
        records.write_json(stage / 'results.json', {'forecast_seal_sha256': records.digest(seal),
                           'word_localization': word_results, 'button_prediction_2023': buttons})
        stage.rename(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('work', 'extraction', 'tokenizer', 'early', 'late', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    run(**vars(parser.parse_args()))
