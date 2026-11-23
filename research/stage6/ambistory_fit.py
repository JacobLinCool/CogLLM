"""Fit AmbiStory on the official training families and seal test predictions."""

import argparse
from collections import defaultdict
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unicodedata

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax

from research.semantic_tracking.iirc_prepare import digest
from research.stage2.records import write_jsonl
from research.stage3 import causal_ordinal
from research.stage3.probe import _jsonl, _length_standardization
from research.stage3.run import MODEL, REVISION, TRANSFORMERS_VERSION, save_json


PREPARATION_SHA256 = '0364a1942796740435c060da2d86a26774d6c794827dfd0cecae8037fafd3253'
PROJECTION_SHA256 = '8b0fcbc13c1b64ca0784741816e04fdf758d83e5fda95ee30e1d753c518dea91'
DESIGN_PROTOCOL = Path(__file__).with_name('ambistory-context-update.md')
SPLITS = ('train', 'dev', 'test')
METHODS = ('form_control', 'common_prefix', 'ending')
GRADES = (1, 2, 3, 4, 5)
WIDTH, PREFIX_TOLERANCE = 128, 0.002
STIMULI, DESIGN, OUTCOMES = 'model_input/stimuli.jsonl', 'design/stops.jsonl', 'scoring/ordinal-ratings.jsonl'
FILES = {'feature_manifest': 'extraction.json', 'story_features': 'story-states.npz',
         'sense_features': 'sense-states.npz', 'causality': 'causality.json', 'extraction_seal': 'seal.json'}
EXTRACTION_SOURCES = {'research/stage6/ambistory_extract.py', 'research/core_features/extract.py',
                      'research/core_features/fit.py', 'research/semantic_tracking/joint_model.py',
                      'research/semantic_tracking/qasper_fit.py', 'research/stage3/probe.py', 'research/stage3/run.py'}
PACKED_INPUTS = {
    'train/stimuli.jsonl': '089178c26651b2d401a1d7ea186169bd247d3e24827e715422a6f784d68713cd',
    'dev/stimuli.jsonl': '139e3e068326f8c88b145dd131e51d6544fb42ebebea5dd8e31cc7dbaf892588',
    'test/stimuli.jsonl': 'ba24280b10b5015721ad6a78ede79103efe80986e9ae0c7b372b5716dfa6c47b',
    'senses.jsonl': '439f8d59b0609c1e12d9fbd211af46dccd6a706e55f4b0ae153b47e1ff0fdb58',
    'shared-prefixes.jsonl': '860221cdd2574aea9c8f43a48752b85c3796d481e90658b0e348c7ab66894fd8',
}
KINDS = ('definition', 'example')
FORM_FEATURES = ('log1p_target_characters', 'log1p_common_prefix_words', 'log1p_definition_words', 'log1p_example_words',
                 'non_target_definition_type_coverage_by_common_prefix', 'non_target_example_type_coverage_by_common_prefix',
                 'target_word_appears_in_definition', 'target_word_appears_in_example')
CAUSAL_CHECKS = {'substring', 'right_padding', 'suffix_replacement', 'changed_suffix_positive_control'}
READY = 'fit_and_forecasts_locked_before_scoring'
CONTEXT_MAX_ITERATIONS = 3000
AMENDMENT = Path(__file__).with_name('ambistory-numerical-amendment.md')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def protocol():
    return {
        'status': 'prespecified_before_ambistory_feature_extraction_and_fitting',
        'input_preparation_sha256': PREPARATION_SHA256, 'design_protocol_sha256': digest(DESIGN_PROTOCOL),
        'fit_code_sha256': digest(Path(__file__)), 'softmax_implementation_sha256': digest(Path(causal_ordinal.__file__)),
        'numerical_amendment_sha256': digest(AMENDMENT),
        'length_scaler_sha256': digest(Path(_length_standardization.__code__.co_filename)),
        'fit_split': 'train', 'forecast_split': 'test', 'methods': list(METHODS), 'grades': list(GRADES),
        'feature_contract': {'model': MODEL, 'revision': REVISION, 'transformers': TRANSFORMERS_VERSION, 'device': 'NVIDIA L4',
                             'dtype': 'float32', 'projected_width': WIDTH, 'projection_sha256': PROJECTION_SHA256,
                             'one_forward_per_story': True, 'complete_certified_coordinates': True,
                             'packed_inputs_sha256': PACKED_INPUTS},
        'common_prefix': {'reference': 'no-ending first four sentences',
                          'validated_stops': ['before_target', 'target_end', 'setup_end'],
                          'alternative_vs_reference_maximum_relative_rms': PREFIX_TOLERANCE,
                          'same_forecast_reused_across_all_versions': True},
        'form_features': list(FORM_FEATURES),
        'form_scaling': 'official training item mean and population SD; replace zero SD by 1 and divide by sqrt(8)',
        'sense_vector': 'L2-normalized sum of projected unit states of the standalone definition and standalone example',
        'form_head': 'softmax([1,z,v] @ W_form)',
        'shared_context_head': 'softmax([1,z,v,h,sqrt(128)*(v*h)] @ W_context)',
        'context_fit_stop': 'setup for the base; final published ending stop for both suffixed versions',
        'prediction_stops': {'common_prefix': 'canonical setup', 'ending': 'actual final stop, or setup for base'},
        'objective': {'loss': 'summed ordinal-distribution cross-entropy, each training item equally weighted',
                      'votes': 'all released 1–5 judgments including votes flagged nonsensical',
                      'l2_non_intercept': causal_ordinal.PENALTY, 'solver': 'L-BFGS-B from zero',
                      'form_maxiter': 300, 'context_maxiter': CONTEXT_MAX_ITERATIONS, 'ftol': 1e-12, 'gtol': 1e-7},
    }


def locked_rows(prepared, receipt, split, name):
    relative = f'{split}/{name}'
    path = prepared / relative
    if digest(path) != receipt['files_sha256'][relative]:
        raise ValueError(f'Prepared material differs from the frozen AmbiStory release: {relative}')
    return list(_jsonl(path))


def read_stories(prepared, receipt):
    inventories, senses_seen, homonyms_seen = {}, set(), set()
    for split in SPLITS:
        stimuli = locked_rows(prepared, receipt, split, STIMULI)
        designs = locked_rows(prepared, receipt, split, DESIGN)
        documents = {row['id']: row for row in stimuli}
        families = defaultdict(list)
        for row in designs:
            stimulus = documents[row['id']]
            start, end = row['target']
            pre, setup, post = (row[name] for name in ('target_pre_stop', 'setup_stop', 'post_stop'))
            if (not 0 < pre <= start < end <= setup <= post or post != len(stimulus['text'])
                    or row['has_ending'] is not (setup < post)
                    or stimulus['stops'] != sorted({pre, end, setup, post})
                    or stimulus['text'][start:end].casefold() != row['homonym']
                    or len(row['senses']) != 2 or len({sense['id'] for sense in row['senses']}) != 2):
                raise ValueError(f'Target, sense or causal stop inventory disagrees for {split}/{row["id"]}')
            families[row['family']].append(row)
        expected = receipt['splits'][split]
        if (len(documents) != len(stimuli) or len(designs) != len(documents)
                or {row['id'] for row in designs} != documents.keys() or len(families) != expected['story_families']
                or len(documents) != expected['stories'] or 2 * len(documents) != expected['sense_samples']):
            raise ValueError(f'Unexpected story or sense inventory for {split}')
        homonyms = {row['homonym'] for row in designs}
        if homonyms & homonyms_seen:
            raise ValueError('A homonym crosses the official post-exclusion split')
        homonyms_seen.update(homonyms)
        for variants in families.values():
            bases = [row for row in variants if not row['has_ending']]
            if len(variants) != 3 or len(bases) != 1:
                raise ValueError('Expected one unsuffixed story and two alternatives per family')
            base, = bases
            prefix = documents[base['id']]['text']
            for row in variants:
                if (row['senses'] != base['senses'] or row['homonym'] != base['homonym'] or row['target'] != base['target']
                        or row['target_pre_stop'] != base['target_pre_stop'] or row['setup_stop'] != len(prefix)
                        or documents[row['id']]['text'][:len(prefix)] != prefix):
                    raise ValueError('Story family does not preserve its prefix, target and judged senses')
            for sense in base['senses']:
                if sense['id'] in senses_seen:
                    raise ValueError('A sense identifier belongs to multiple story families')
                senses_seen.add(sense['id'])
        inventories[split] = documents, dict(families)
    return inventories


def read_projected(path, coordinate_names, expected):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {*coordinate_names, 'states'}:
            raise ValueError(f'Expected sparse projected states: {path.name}')
        first, second = (archive[name] for name in coordinate_names)
        states = archive['states']
    size = len(expected)
    if (first.shape != (size,) or first.dtype.kind != 'U' or second.shape != (size,)
            or second.dtype.kind not in ('iu' if coordinate_names[1] == 'stops' else 'U')
            or states.shape != (size, WIDTH) or states.dtype != np.float32 or not np.isfinite(states).all()):
        raise ValueError(f'Unexpected projected array shape/dtype: {path.name}')
    coordinates = list(zip(first.tolist(), second.tolist(), strict=True))
    norms = np.linalg.norm(states.astype(np.float64), axis=1)
    if set(coordinates) != expected or len(set(coordinates)) != size or not np.allclose(norms, 1, atol=2e-5, rtol=0):
        raise ValueError(f'Sparse coordinates or normalized projected states differ: {path.name}')
    return dict(zip(coordinates, states, strict=True))


def read_features(folder, preparation, receipt, inventories):
    paths = {key: folder / name for key, name in FILES.items()}
    hashes = {key: digest(path) for key, path in paths.items()}
    extraction = read_json(paths['feature_manifest'])
    seal = read_json(paths['extraction_seal'])
    fixed = {'model': MODEL, 'revision': REVISION, 'projection_sha256': PROJECTION_SHA256}
    if any(seal.get(key) != value for key, value in {
        **fixed, 'preparation_sha256': digest(preparation), 'story_states_sha256': hashes['story_features'],
        'sense_states_sha256': hashes['sense_features'], 'extraction_sha256': hashes['feature_manifest'],
        'causality_sha256': hashes['causality'], 'telemetry_sha256': digest(folder / 'telemetry.jsonl'),
    }.items()):
        raise ValueError('Feature, provenance, or projection bytes differ from their extraction seal')
    packed_inputs, extraction_sources = seal['inputs_sha256'], seal['source_sha256']
    repository = Path(__file__).resolve().parents[2]
    if (packed_inputs != PACKED_INPUTS or extraction_sources.keys() != EXTRACTION_SOURCES
            or any(digest(repository / name) != checksum for name, checksum in extraction_sources.items())):
        raise ValueError('Extraction recipe or packed source differs from the frozen stimulus and worker bytes')
    documents = {key: row for split in SPLITS for key, row in inventories[split][0].items()}
    expected_stops = {(case, end) for case, row in documents.items() for end in row['stops']}
    expected_senses = {(sense['id'], kind) for split in SPLITS for variants in inventories[split][1].values()
                       for sense in variants[0]['senses'] for kind in KINDS}
    expected = {**fixed, 'transformers': TRANSFORMERS_VERSION, 'gpu': 'NVIDIA L4', 'dtype': 'float32',
                'projection_width': WIDTH, 'stories': len(documents), 'story_stops': len(expected_stops),
                'story_splits': {split: len(inventories[split][0]) for split in SPLITS},
                'story_tokens': sum(receipt['splits'][split]['tokenizer_preflight']['tokens'] for split in SPLITS),
                'sense_definitions': len(expected_senses) // 2, 'sense_examples': len(expected_senses) // 2}
    if any(extraction.get(key) != value for key, value in expected.items()):
        raise ValueError('Expected the complete frozen L4 story and standalone-sense extraction inventory')
    controls = read_json(paths['causality'])
    comparisons = defaultdict(list)
    for row in controls['checks']:
        key = row['id']
        if (key not in documents or row['passed'] is not True or row['projected_tokens']['passed'] is not True
                or not 0 < row['stop'] < len(documents[key]['text'])):
            raise ValueError('Missing passing projected causal or padding invariant')
        comparisons[key, row['stop']].append(row['comparison'])
    if (controls['control_stops'] != 'independently certified interior boundaries of the input text'
            or len(comparisons) != 3 or any(len(items) != 4 or set(items) != CAUSAL_CHECKS for items in comparisons.values())):
        raise ValueError('Incomplete causal, padding, and suffix controls')
    return (read_projected(paths['story_features'], ('ids', 'stops'), expected_stops),
            read_projected(paths['sense_features'], ('ids', 'kinds'), expected_senses), hashes, extraction['common_prefix'])


def compare_prefixes(inventories, states, *, extraction_reference=False):
    benchmark = {}
    for split, (_, families) in inventories.items():
        errors, maximum_absolute = [], 0.0
        for variants in families.values():
            base = (min(variants, key=lambda row: row['id']) if extraction_reference
                    else next(row for row in variants if not row['has_ending']))
            for row in variants:
                if row['id'] != base['id']:
                    for stop in (base['target_pre_stop'], base['target'][1], base['setup_stop']):
                        reference = states[base['id'], stop].astype(np.float64)
                        difference = states[row['id'], stop] - reference
                        errors.append(float(np.linalg.norm(difference) / np.linalg.norm(reference)))
                        maximum_absolute = max(maximum_absolute, float(np.abs(difference).max()))
        maximum = max(errors)
        if len(errors) != 6 * len(families) or maximum > PREFIX_TOLERANCE:
            raise ValueError(f'Projected shared prefix was not preserved across story variants: {split}')
        benchmark[split] = {'families': len(families), 'comparisons': len(errors), 'max_relative_rms': maximum,
                            'max_absolute': maximum_absolute}
    return benchmark


def words(text):
    return re.findall(r'[^\W_]+', unicodedata.normalize('NFKC', text).casefold())


def form_vector(prefix, target, sense):
    common, focal, definition, example = (words(text) for text in (prefix, target, sense['definition'], sense['example']))
    focal = set(focal)
    content = set(common) - focal

    def coverage(desc):
        terms = set(desc) - focal
        return len(content & terms) / max(1, len(terms))

    return [*np.log1p([len(target), len(common), len(definition), len(example)]), coverage(definition), coverage(example),
            float(bool(focal & set(definition))), float(bool(focal & set(example)))]


def assemble(inventory, stories, meanings):
    documents, families = inventory
    keys, forms, sense_vectors, prefixes, finals, suffixed = [], [], [], [], [], []
    for variants in families.values():
        base = next(row for row in variants if not row['has_ending'])
        prefix = documents[base['id']]['text']
        target = prefix[slice(*base['target'])]
        base_state = stories[base['id'], base['setup_stop']]
        for sense in base['senses']:
            form = form_vector(prefix, target, sense)
            sense_state = sum(meanings[sense['id'], kind].astype(np.float64) for kind in KINDS)
            length = np.linalg.norm(sense_state)
            if not np.isfinite(length) or length <= 0:
                raise ValueError('Definition and example did not form a finite, nonzero sense vector')
            sense_state /= length
            for row in variants:
                keys.append((row['id'], sense['id']))
                forms.append(form)
                sense_vectors.append(sense_state)
                prefixes.append(base_state)
                finals.append(stories[row['id'], row['post_stop']] if row['has_ending'] else base_state)
                suffixed.append(row['has_ending'])
    return keys, *(np.asarray(items, dtype=np.float64) for items in (forms, sense_vectors, prefixes, finals)), np.asarray(suffixed)


def read_train_outcomes(prepared, receipt, keys):
    observed = {}
    for row in locked_rows(prepared, receipt, 'train', OUTCOMES):
        key = row['id'], row['sense_id']
        grades = row['ratings']
        if (key in observed or not grades or any(type(grade) is not int or grade not in GRADES for grade in grades)):
            raise ValueError('Unexpected training sense item or grade')
        observed[key] = ((np.asarray(grades)[:, None] == GRADES).mean(axis=0), len(grades))
    if set(keys) != observed.keys() or sum(count for _, count in observed.values()) != receipt['splits']['train']['votes']:
        raise ValueError('Training rating inventory differs from the official post-exclusion split')
    return np.stack([observed[key][0] for key in keys])


def base_design(form, meanings, center, scale):
    scaled = (form - center) / scale / np.sqrt(form.shape[1])
    return np.column_stack((np.ones(len(form)), scaled, meanings))


def with_context(base, meanings, state):
    return np.column_stack((base, state, np.sqrt(state.shape[1]) * state * meanings))


def fit_context(design, observed):
    shape = design.shape[1], len(GRADES)

    def loss(flat):
        weights = flat.reshape(shape)
        log_p = log_softmax(design @ weights, axis=-1)
        penalty = causal_ordinal.PENALTY
        value = float(-np.sum(observed * log_p) + (penalty / 2) * np.sum(weights[1:] ** 2))
        gradient = design.T @ (np.exp(log_p) - observed)
        gradient[1:] += penalty * weights[1:]
        return value, gradient.ravel()

    result = minimize(loss, np.zeros(np.prod(shape)), method='L-BFGS-B', jac=True,
                      options={'maxiter': CONTEXT_MAX_ITERATIONS, 'ftol': 1e-12, 'gtol': 1e-7})
    if not (result.success and np.isfinite(result.x).all() and np.isfinite(result.fun) and np.isfinite(result.jac).all()):
        raise ArithmeticError(f'AmbiStory context fit failed: {result.message}')
    diagnostics = {'iterations': int(result.nit), 'evaluations': int(result.nfev), 'objective': float(result.fun),
                   'max_abs_gradient': float(np.max(np.abs(result.jac))), 'message': str(result.message)}
    return result.x.reshape(shape), diagnostics


def learn_and_predict(train, test, outcomes):
    train_keys, train_form, train_meanings, _, train_end, _ = train
    test_keys, test_form, test_meanings, test_common, test_end, suffixed = test
    if len(train_keys) != len(outcomes) or train_form.shape[1] != len(FORM_FEATURES):
        raise ValueError('Unexpected training response or form-feature width')
    center, scale = _length_standardization(train_form)
    train_base = base_design(train_form, train_meanings, center, scale)
    form_weights = causal_ordinal.fit_softmax(train_base, outcomes)
    context_weights, diagnostics = fit_context(with_context(train_base, train_meanings, train_end), outcomes)
    test_base = base_design(test_form, test_meanings, center, scale)
    # Each sense has a contiguous triple from assemble. Compute its common outputs once.
    shared_form = test_base[::3] @ form_weights
    shared_context = with_context(test_base[::3], test_meanings[::3], test_common[::3]) @ context_weights
    form_logits = np.repeat(shared_form, 3, axis=0)
    prefix_logits = np.repeat(shared_context, 3, axis=0)
    ending_logits = prefix_logits.copy()
    ending_logits[suffixed] = with_context(test_base[suffixed], test_meanings[suffixed], test_end[suffixed]) @ context_weights
    logits = np.stack((form_logits, prefix_logits, ending_logits), axis=1)
    probability = np.exp(log_softmax(logits, axis=-1))
    if (not np.isfinite(probability).all() or np.any(probability <= 0)
            or not np.allclose(probability.sum(axis=-1), 1, atol=1e-12, rtol=0)):
        raise ValueError('Failed to produce normalized finite ordinal forecasts')
    weights = {'form_weights': form_weights, 'context_weights': context_weights, 'form_center': center, 'form_scale': scale}
    return weights, dict(zip(test_keys, probability, strict=True)), diagnostics


def validate_invariance(inventory, forecasts):
    _, families = inventory
    for variants in families.values():
        base = next(row for row in variants if not row['has_ending'])
        for sense in base['senses']:
            baseline = forecasts[base['id'], sense['id']]
            if not np.array_equal(baseline[1], baseline[2]):
                raise ValueError('Base ending must equal its common-prefix forecast')
            for row in variants:
                if not np.array_equal(forecasts[row['id'], sense['id']][:2], baseline[:2]):
                    raise ValueError('Form and common-prefix forecasts must be identical across all three versions')


def fit(prepared, preparation, extraction, fit_protocol, output):
    if output.exists():
        raise FileExistsError(output)
    if digest(preparation) != PREPARATION_SHA256 or read_json(fit_protocol) != protocol():
        raise ValueError('Preparation, fit specification, or implementation differs from the preseal')
    receipt = read_json(preparation)
    inventories = read_stories(prepared, receipt)
    stories, meanings, input_hashes, attested = read_features(extraction, preparation, receipt, inventories)
    benchmark = compare_prefixes(inventories, stories)
    replay = compare_prefixes(inventories, stories, extraction_reference=True)
    if (attested['families'] != sum(row['families'] for row in replay.values())
            or attested['comparisons'] != sum(row['comparisons'] for row in replay.values())
            or not np.isclose(attested['max_relative_rms'], max(row['max_relative_rms'] for row in replay.values()),
                              atol=1e-6, rtol=0)
            or not np.isclose(attested['max_absolute'], max(row['max_absolute'] for row in replay.values()), atol=1e-6, rtol=0)):
        raise ValueError('Independent common-prefix replay differs from the extraction receipt')
    train = assemble(inventories['train'], stories, meanings)
    test = assemble(inventories['test'], stories, meanings)
    outcomes = read_train_outcomes(prepared, receipt, train[0])
    weights, forecasts, diagnostics = learn_and_predict(train, test, outcomes)
    validate_invariance(inventories['test'], forecasts)
    if any(digest(extraction / filename) != input_hashes[key] for key, filename in FILES.items()):
        raise ValueError('Extraction artifacts changed during the fit')
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.ambistory-fit-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'fit'
        stage.mkdir(mode=0o700)
        weights_path, prediction_path = stage / 'heads.npz', stage / 'predictions.jsonl'
        np.savez_compressed(weights_path, **weights)
        write_jsonl(prediction_path, ({'id': key, 'sense': sense, **dict(zip(METHODS, matrix.tolist(), strict=True))}
                                      for (key, sense), matrix in forecasts.items()))
        source_hashes = receipt['files_sha256']
        save_json(stage / 'fit-seal.json', {
            'status': READY, 'protocol_sha256': digest(fit_protocol), 'fit_code_sha256': digest(Path(__file__)),
            'input_sha256': {'preparation': PREPARATION_SHA256, **input_hashes,
                             'train_design': source_hashes[f'train/{DESIGN}'], 'test_design': source_hashes[f'test/{DESIGN}'],
                             'train_outcomes': source_hashes[f'train/{OUTCOMES}'], 'test_outcomes': source_hashes[f'test/{OUTCOMES}']},
            'methods': list(METHODS), 'grades': list(GRADES), 'fit_split': 'train', 'forecast_split': 'test',
            'fit_samples': len(train[0]), 'forecast_samples': len(forecasts), 'prefix_benchmark': benchmark,
            'context_optimizer': diagnostics,
            'predictions_sha256': digest(prediction_path), 'weights_sha256': digest(weights_path),
            'numeric_runtime': {'numpy': np.__version__, 'scipy': version('scipy')},
        })
        stage.rename(output)


def preseal(output):
    with output.open('x', encoding='utf-8') as target:
        json.dump(protocol(), target, indent=2, allow_nan=False)
        target.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command, paths in (('preseal', ('output',)),
                           ('fit', ('prepared', 'preparation', 'extraction', 'fit_protocol', 'output'))):
        subcommand = commands.add_parser(command)
        for name in paths:
            subcommand.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    args = vars(parser.parse_args())
    {'preseal': preseal, 'fit': fit}[args.pop('command')](**args)
