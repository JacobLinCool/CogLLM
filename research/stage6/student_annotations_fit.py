"""Fit the preregistered crossed-fold StudentAnnotations readouts on frozen states."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

# Bound the standalone fit process before NumPy or SciPy initializes BLAS.
if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from scipy.special import log_softmax

from research.semantic_tracking.iirc_prepare import digest
from research.stage2.records import write_jsonl
from research.stage3 import causal_ordinal
from research.stage3.probe import _jsonl, _length_standardization
from research.stage3.run import MODEL, REVISION, TRANSFORMERS_VERSION, save_json


PREPARATION_SHA256 = 'e9245766647a5ae24fb5347e39bea4d167da534380e44f7419d4464d7639fe97'
PREPARATION = 'preparation-receipt.json'
STIMULI = 'model_input/stimuli.jsonl'
INPUTS = {'trial_design': 'local/trial-design.jsonl', 'readers': 'local/readers.jsonl',
          'sentences': 'local/sentences.jsonl', 'outcomes': 'local/outcomes.jsonl'}
DESIGN_PROTOCOL = Path(__file__).with_name('student-annotations-background.md')
METHODS = ('length_group', 'qwen_additive', 'qwen_interaction', 'blank_background')
GRADES = (1, 2, 3, 4, 5)
FOLDS, HIDDEN_WIDTH, GROUPS = 4, 2048, 3
GROUP_CODE = np.asarray([[-1 / np.sqrt(2), -1 / np.sqrt(6)],
                         [1 / np.sqrt(2), -1 / np.sqrt(6)],
                         [0, 2 / np.sqrt(6)]])
GROUP_CODES = {'german': 0, 'german_and_another_language': 1, 'another_language': 2}
CAUSAL_CHECKS = {'substring', 'right_padding', 'suffix_replacement', 'changed_suffix_positive_control'}
READY = 'fit_and_forecasts_locked_before_scoring'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def protocol():
    return {
        'status': 'prespecified_before_student_annotations_features_and_fitting',
        'design_protocol_sha256': digest(DESIGN_PROTOCOL), 'preparation_sha256': PREPARATION_SHA256,
        'fit_code_sha256': digest(Path(__file__)), 'softmax_implementation_sha256': digest(Path(causal_ordinal.__file__)),
        'length_scaler_sha256': digest(Path(_length_standardization.__code__.co_filename)),
        'methods': list(METHODS), 'grades': list(GRADES), 'language_codes': GROUP_CODES,
        'heldout': {'reader_folds': FOLDS, 'normalized_sentence_folds': FOLDS,
                    'fit': 'reader fold differs AND sentence fold differs',
                    'predict': 'reader fold matches AND sentence fold matches'},
        'objective': {'label_encoding': 'one-hot per training observation', 'loss': 'sum of categorical negative log likelihoods',
                      'l2_non_intercept': causal_ordinal.PENALTY, 'solver': 'L-BFGS-B from zero',
                      'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-7},
        'length_transform': 'log1p Unicode characters and whitespace-delimited words at the visible sentence stop; '
                            'unique fit stimulus mean and population SD (replace zero SD by 1), divided by sqrt(2)',
        'group_contrast_code': GROUP_CODE.tolist(),
        'fixed_hidden_transform': 'sentence-end FP32 state converted to float64 and row L2-normalized',
        'interaction': 'two centered group-code coordinates multiplied by the normalized hidden state; shared '
                       'length, group, and hidden main effects as in the additive readout',
        'blank_background': 'arithmetic mixture of additive probability vectors using empirical fit-trial group frequencies',
        'preparation_and_extraction_checks': {'model': MODEL, 'revision': REVISION, 'transformers': TRANSFORMERS_VERSION,
                                             'device': 'NVIDIA L4', 'dtype': 'float32', 'all_planned_stops_required': True,
                                             'interior_causal_controls_required': True},
    }


def verify_inputs(prepared):
    receipt_path = prepared / PREPARATION
    receipt = read_json(receipt_path)
    if (digest(receipt_path) != PREPARATION_SHA256 or receipt['protocol_sha256'] != digest(DESIGN_PROTOCOL)
            or receipt['language_code'] != GROUP_CODES):
        raise ValueError('Preparation or crossed-fold design differs from its frozen receipt')
    for name in (STIMULI, *INPUTS.values(), 'model_input/extraction-plan.json', 'model_input/certified-stops.jsonl'):
        if digest(prepared / name) != receipt['sha256'][name]:
            raise ValueError(f'Prepared input differs from its receipt: {name}')
    return receipt


def distinct_rows(path, key, fields):
    result = {}
    for row in _jsonl(path):
        if row.keys() != fields or not isinstance(row[key], str) or not row[key] or row[key] in result:
            raise ValueError(f'Unexpected fields or duplicate key in {path.name}')
        result[row[key]] = row
    return result


def read_design(prepared, receipt):
    documents = distinct_rows(prepared / STIMULI, 'id', {'id', 'text', 'stops'})
    readers = distinct_rows(prepared / INPUTS['readers'], 'reader', {'reader', 'fold', 'language'})
    sentences = distinct_rows(prepared / INPUTS['sentences'], 'id', {'id', 'group', 'fold'})
    trials = distinct_rows(prepared / INPUTS['trial_design'], 'trial', {'trial', 'id', 'reader'})
    counts = receipt['counts']
    if (documents.keys() != sentences.keys() or len(documents) != counts['sentences']
            or len(readers) != counts['readers'] or len(trials) != counts['trials']):
        raise ValueError('The prepared document, reader, or trial inventory differs from the receipt')
    groups = {}
    for row in sentences.values():
        if (type(row['fold']) is not int or not 0 <= row['fold'] < FOLDS
                or groups.setdefault(row['group'], row['fold']) != row['fold']):
            raise ValueError('A normalized sentence group spans held-out sentence folds')
    if (len(groups) != counts['normalized_sentence_groups']
            or any(type(row['language']) is not int or not 0 <= row['language'] < GROUPS
                   or type(row['fold']) is not int or not 0 <= row['fold'] < FOLDS for row in readers.values())):
        raise ValueError('Unexpected reader fold, reader profile, or normalized sentence group count')
    if len({(row['reader'], row['id']) for row in trials.values()}) != len(trials):
        raise ValueError('Duplicate reader-stimulus observation within the prepared design')
    return documents, sentences, readers, trials


def read_states(folder, prepared, receipt, documents):
    plan = read_json(prepared / 'model_input/extraction-plan.json')
    manifest = read_json(folder / 'extraction.json')
    expected = {'model': MODEL, 'revision': REVISION, 'transformers': TRANSFORMERS_VERSION, 'dtype': 'float32',
                'gpu': 'NVIDIA L4', 'segment_means': False, 'input_sha256': receipt['sha256'][STIMULI],
                'input_texts': len(documents), 'encoded_texts': len(documents),
                'unique_certified_stops': len(documents), 'excluded_stops': [], 'tokens': plan['total_input_tokens']}
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise ValueError(f'Expected pinned sparse L4 extraction field: {name}')
    control = read_json(folder / 'causality.json')
    checks = {}
    for row in control['checks']:
        if (row['id'] not in documents or row['passed'] is not True or type(row['stop']) is not int
                or not 0 < row['stop'] < documents[row['id']]['stops'][0]):
            raise ValueError('Expected passing independent interior causal controls')
        checks.setdefault((row['id'], row['stop']), []).append(row['comparison'])
    if (control.get('control_stops') != 'independently certified interior boundaries of the input text'
            or len(checks) != 3 or any(len(keys) != 4 or set(keys) != CAUSAL_CHECKS for keys in checks.values())):
        raise ValueError('Incomplete interior causal control inventory')

    with np.load(folder / 'features.npz', allow_pickle=False) as archive:
        if set(archive.files) != {'ids', 'stops', 'states'}:
            raise ValueError('Expected sparse features with one state per prepared sentence')
        ids, stops, states = archive['ids'], archive['stops'], archive['states']
    n = len(documents)
    if (ids.shape != (n,) or ids.dtype.kind != 'U' or set(ids.tolist()) != documents.keys()
            or len(set(ids.tolist())) != n or stops.shape != (n,) or stops.dtype.kind not in 'iu'
            or any(documents[case]['stops'] != [stop] for case, stop in zip(ids.tolist(), stops.tolist(), strict=True))
            or states.shape != (n, HIDDEN_WIDTH) or states.dtype != np.float32 or not np.isfinite(states).all()):
        raise ValueError('Feature inventory does not match the complete frozen sentence stop set')
    by_id = {case: index for index, case in enumerate(ids)}
    state = states[[by_id[key] for key in documents]].astype(np.float64)
    length = np.linalg.norm(state, axis=1)
    if np.any(length <= 0) or not np.isfinite(length).all():
        raise ValueError('Expected finite nonzero sentence-end states')
    state /= length[:, None]
    return state


class TrainingOutcomes:
    """Decode ratings by fit indices only, using prepared trial-order byte offsets."""

    def __init__(self, path, trial_ids):
        self.path = path
        self.trial_ids = trial_ids
        self.offsets = []
        with path.open('rb') as stream:
            offset = 0
            for line in stream:
                self.offsets.append(offset)
                offset += len(line)
        if len(self.offsets) != len(trial_ids):
            raise ValueError('Outcome line count differs from the prepared trial inventory')

    def __call__(self, fit_indices):
        grades = []
        with self.path.open('rb') as source:
            for index in fit_indices:
                source.seek(self.offsets[index])
                row = json.loads(source.readline())
                if (row.keys() != {'trial', 'grade'} or row['trial'] != self.trial_ids[index]
                        or type(row['grade']) is not int or row['grade'] not in GRADES):
                    raise ValueError('Unexpected training outcome key or ordinal grade')
                grades.append(row['grade'])
        if set(grades) != set(GRADES):
            raise ValueError('The crossed training cell lacks one or more grade categories')
        return (np.asarray(grades)[:, None] == GRADES).astype(np.float64)


def design_matrix(method, lengths, states, language):
    group = GROUP_CODE[language]
    base = np.column_stack((np.ones(len(group)), lengths, group))
    if method == 'length_group':
        return base
    if method == 'qwen_additive':
        return np.column_stack((base, states))
    if method != 'qwen_interaction':
        raise ValueError(f'Unknown StudentAnnotations model: {method}')
    boundary, hidden_width = base.shape[1], states.shape[1]
    matrix = np.empty((len(base), boundary + 3 * hidden_width))
    matrix[:, :boundary] = base
    matrix[:, boundary:boundary + hidden_width] = states
    for column, code in enumerate(group.T, start=1):
        start = boundary + column * hidden_width
        np.multiply(states, code[:, None], out=matrix[:, start:start + hidden_width])
    return matrix


def predict(method, lengths, states, language, weights):
    return np.exp(log_softmax(design_matrix(method, lengths, states, language) @ weights, axis=-1))


def fit_crossed(lengths, hidden, text_indices, reader_folds, sentence_folds, languages, train_outcomes, counts):
    predictions = np.full((len(text_indices), len(METHODS), len(GRADES)), np.nan)
    fold_weights, folds = {}, []
    for reader_fold in range(FOLDS):
        for sentence_fold in range(FOLDS):
            same_reader = reader_folds == reader_fold
            same_sentence = sentence_folds == sentence_fold
            fit = np.flatnonzero(~same_reader & ~same_sentence)
            test = np.flatnonzero(same_reader & same_sentence)
            if (len(fit) != counts['fit_counts'][reader_fold][sentence_fold]
                    or len(test) != counts['test_counts'][reader_fold][sentence_fold]):
                raise ValueError('Crossed holdout sizes differ from the preparation receipt')
            fit_text, test_text = text_indices[fit], text_indices[test]
            fit_stimuli = np.unique(fit_text)
            center, scale = _length_standardization(lengths[fit_stimuli])
            fit_lengths = (lengths[fit_text] - center) / scale / np.sqrt(2)
            test_lengths = (lengths[test_text] - center) / scale / np.sqrt(2)
            fit_states, test_states = hidden[fit_text], hidden[test_text]
            fit_languages, test_languages = languages[fit], languages[test]
            group_counts = np.bincount(fit_languages, minlength=GROUPS)
            if np.any(group_counts == 0):
                raise ValueError('At least one background group is absent from a fit cell')
            outcomes = train_outcomes(fit)
            for method_index, method in enumerate(METHODS[:-1]):
                weights = causal_ordinal.fit_softmax(design_matrix(method, fit_lengths, fit_states, fit_languages), outcomes)
                fold_weights[f'r{reader_fold}_s{sentence_fold}_{method}'] = weights
                predictions[test, method_index] = predict(method, test_lengths, test_states, test_languages, weights)
                if method == 'qwen_additive':
                    mixture = np.zeros((len(test), len(GRADES)))
                    for group, fraction in enumerate(group_counts / group_counts.sum()):
                        mixture += fraction * predict(method, test_lengths, test_states, np.full(len(test), group), weights)
                    predictions[test, -1] = mixture
            folds.append({'reader_fold': reader_fold, 'sentence_fold': sentence_fold, 'fit_trials': len(fit),
                          'test_trials': len(test), 'fit_stimuli': len(fit_stimuli),
                          'group_fit_counts': group_counts.tolist(), 'length_center': center.tolist(), 'length_scale': scale.tolist()})
    if (not np.isfinite(predictions).all() or np.any(predictions <= 0)
            or not np.allclose(predictions.sum(axis=-1), 1, atol=1e-12, rtol=0)):
        raise ValueError('Incomplete, nonfinite, or non-normalized out-of-fold forecast probabilities')
    return predictions, folds, fold_weights


def fit(prepared, extraction, fit_protocol, output):
    if output.exists():
        raise FileExistsError(output)
    if read_json(fit_protocol) != protocol():
        raise ValueError('The fit specification or implementation differs from its preseal')
    receipt = verify_inputs(prepared)
    documents, sentences, readers, trials = read_design(prepared, receipt)
    raw_artifacts = {key: extraction / filename for key, filename in (
        ('extraction', 'extraction.json'), ('features', 'features.npz'), ('causality', 'causality.json'))}
    hashes = {key: digest(path) for key, path in raw_artifacts.items()}
    hidden = read_states(extraction, prepared, receipt, documents)
    visible = [row['text'][:row['stops'][0]] for row in documents.values()]
    lengths = np.log1p([(len(text), len(text.split())) for text in visible])
    positions = {case: index for index, case in enumerate(documents)}
    text_indices = np.array([positions[row['id']] for row in trials.values()])
    reader_folds = np.array([readers[row['reader']]['fold'] for row in trials.values()])
    sentence_folds = np.array([sentences[row['id']]['fold'] for row in trials.values()])
    languages = np.array([readers[row['reader']]['language'] for row in trials.values()])
    train_outcomes = TrainingOutcomes(prepared / INPUTS['outcomes'], tuple(trials))
    predictions, folds, weights = fit_crossed(lengths, hidden, text_indices, reader_folds, sentence_folds,
                                              languages, train_outcomes, receipt['crossed_four_fold'])
    if any(digest(path) != hashes[key] for key, path in raw_artifacts.items()):
        raise ValueError('Extraction artifacts changed during fitting')
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.student-annotations-fit-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'fit'
        stage.mkdir(mode=0o700)
        weights_path = stage / 'fold-heads.npz'
        forecast_path = stage / 'predictions.jsonl'
        np.savez_compressed(weights_path, **weights)
        write_jsonl(forecast_path, ({'trial': trial, **dict(zip(METHODS, matrix.tolist(), strict=True))}
                                    for trial, matrix in zip(trials, predictions, strict=True)))
        save_json(stage / 'fit-seal.json', {
            'status': READY, 'protocol_sha256': digest(fit_protocol), 'design_protocol_sha256': receipt['protocol_sha256'],
            'fit_code_sha256': digest(Path(__file__)), 'methods': list(METHODS), 'grades': list(GRADES),
            'input_sha256': {'preparation_receipt': PREPARATION_SHA256, **hashes,
                             **{key: receipt['sha256'][name] for key, name in INPUTS.items()}},
            'forecasts_schema': 'five ordered ordinal probabilities per method, keyed by prepared trial',
            'predictions_sha256': digest(forecast_path), 'fold_weights_sha256': digest(weights_path),
            'forecast_trials': len(trials), 'folds': folds, 'numeric_runtime': {'numpy': np.__version__, 'scipy': version('scipy')},
        })
        stage.rename(output)


def preseal(output):
    with output.open('x', encoding='utf-8') as target:
        json.dump(protocol(), target, indent=2, allow_nan=False)
        target.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command, paths in (('preseal', ('output',)), ('fit', ('prepared', 'extraction', 'fit_protocol', 'output'))):
        commands_parser = commands.add_parser(command)
        for field in paths:
            commands_parser.add_argument('--' + field.replace('_', '-'), type=Path, required=True)
    arguments = vars(parser.parse_args())
    {'preseal': preseal, 'fit': fit}[arguments.pop('command')](**arguments)
