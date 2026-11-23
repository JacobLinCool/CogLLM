"""Cross-fit single-article Frick forecasts with training-only paired effect supervision."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import scipy
from scipy.special import expit

from research.semantic_tracking import frick_consistency_score as base
from research.semantic_tracking import frick_pair_evaluate as evaluation
from research.semantic_tracking import frick_pair_fit as solver


PROTOCOL = Path(__file__).with_name('frick-pair-protocol.md')
REFERENCE = Path(__file__).resolve().parents[1] / 'results' / 'frick-consistency-l4'
METHODS = ('global', 'target_weighted', 'paired_effect')
FEATURES = base.SCALAR_FEATURES + ('pair_direction', 'pair_direction_change')


def normalize_states(states):
    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != base.WIDTH:
        raise ValueError('Expected aligned full-article Qwen states at each sentence end')
    scale = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.all(np.isfinite(scale) & (scale > 0)):
        raise ValueError('Every Qwen state must have a finite and positive norm')
    return values / scale


def sentence_forms(sentences):
    rows, prior_types, prior_words = [], set(), 0
    for ordinal, sentence in enumerate(sentences, 1):
        words = len(sentence.split())
        types = base.word_types(sentence)
        if ordinal > 1:
            overlap = len(types & prior_types) / len(types) if types else 0.0
            rows.append((*np.log1p((len(sentence), words, ordinal, prior_words)), overlap))
        prior_types.update(types)
        prior_words += words
    return np.asarray(rows, dtype=np.float64)


def project_story(unit_states, direction):
    """Use only consecutive causal sentence states from one article."""
    scores = unit_states @ direction
    return np.column_stack((scores[1:], np.diff(scores)))


def paired_direction(unit_states, metadata, families, excluded):
    deltas = []
    for story, pair in sorted(families.items()):
        if story == excluded:
            continue
        left, right = (pair[key] for key in base.CONDITIONS)
        target = metadata[left]['target_index']
        deltas.append(unit_states[right][target] - unit_states[left][target])
    differences = np.asarray(deltas)
    norms = np.linalg.norm(differences, axis=1, keepdims=True)
    if len(differences) != len(families) - 1 or not np.all(np.isfinite(norms) & (norms > 0)):
        raise ValueError('A training pair has no usable target direction')
    axis = (differences / norms).mean(axis=0)
    scale = np.linalg.norm(axis)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Training pair directions have no finite nonzero mean')
    return axis / scale


def training_targets(families, excluded, metadata, slots, training_index, hits, totals, task):
    pairs = []
    for story, pair in sorted(families.items()):
        if story == excluded:
            continue
        paths, human = [], []
        for condition in base.CONDITIONS:
            key = pair[condition]
            start = slots[key].start
            end = start + metadata[key]['target_index']
            path = training_index[start:end]
            if not len(path) or np.any(path < 0):
                raise ValueError('Matched training targets must exclude both held-out versions')
            count = totals[0, end - 1]
            if count <= 0:
                raise ValueError('Missing original reader denominator at the training target')
            # In the first-event task the numerator counts first D, but the
            # target contrast uses all readers in that version as its denominator.
            human.append(float(hits[task, end - 1] / count))
            paths.append(path)
        pairs.append((*paths, human[1] - human[0]))
    return pairs


def crossfit(unit_states, lexical, metadata, slots, families, hits, totals):
    count = sum(len(array) for array in lexical.values())
    logits = np.full((len(base.TASKS), len(METHODS), count), np.nan, dtype=np.float64)
    fits, directions = [], {}
    for story, pair in sorted(families.items()):
        axis = paired_direction(unit_states, metadata, families, excluded=story)
        directions[f'story_{story}'] = axis
        design = np.empty((count, len(FEATURES)), dtype=np.float64)
        for key, slot in slots.items():
            design[slot] = np.column_stack((lexical[key], project_story(unit_states[key], axis)))
        mask = np.ones(count, dtype=bool)
        for key in pair.values():
            mask[slots[key]] = False
        train, test = design[mask], design[~mask]
        training_index = np.full(count, -1, dtype=np.int64)
        training_index[mask] = np.arange(mask.sum())
        for t, task in enumerate(base.TASKS):
            pairs = training_targets(families, story, metadata, slots, training_index, hits, totals, t)
            for m, method in enumerate(METHODS):
                fitted = solver.fit(train, hits[t, mask], totals[t, mask], pairs, task, method)
                logits[t, m, ~mask] = base.predict_logits(test, fitted)
                fit_values = {key: value.tolist() if isinstance(value, np.ndarray) else value
                              for key, value in fitted.items()}
                fits.append({**fit_values, 'held_out_story': story, 'direction_key': f'story_{story}'})
    if not np.isfinite(logits).all():
        raise ValueError('Every stop must receive one finite held-out prediction per readout')
    return logits, fits, directions


def forecast_rows(logits, slots, stimuli):
    for key, slot in slots.items():
        for i, stop in enumerate(stimuli[key]['stops'][1:], start=slot.start):
            methods = {}
            for m, method in enumerate(METHODS):
                current, first = (float(v) for v in logits[:, m, i])
                methods[method] = {'current_logit': current, 'first_logit': first,
                                   'current_d_probability': float(expit(current)),
                                   'first_d_hazard_probability': float(expit(first))}
            yield {'id': key, 'stop': stop, 'methods': methods}


def run(assay, extraction, packet, protocol, reference, output):
    if output.exists():
        raise FileExistsError(output)
    audit, stimuli, metadata, families, texts = base.load_layout(assay)
    states, provenance = base.load_states(assay, extraction, packet, base.PROTOCOL, stimuli, metadata, families)
    reference_seal = reference / 'forecast-seal.json'
    original = base.read_json(reference_seal)
    original_forecasts = reference / 'forecasts.jsonl'
    original_results = base.read_json(reference / 'results.json')
    if (base.digest(original_forecasts) != original['file_sha256']['forecasts.jsonl']
            or base.digest(reference_seal) != original_results['forecast_seal_sha256']
            or any(original['provenance'][key] != provenance[key]
                   for key in ('state_extraction_sha256', 'protocol_sha256', 'source_audit_sha256'))):
        raise ValueError('Reference forecasts do not match the same sealed Gradio states and labels')

    lexical, unit_states, slots = {}, {}, {}
    offset = 0
    for key in sorted(metadata):
        lexical[key] = sentence_forms(texts[key])
        unit_states[key] = normalize_states(states[key])
        limit = offset + len(lexical[key])
        slots[key] = slice(offset, limit)
        offset = limit
    reference_logits = base.read_forecasts(original_forecasts, slots, stimuli)
    trials, readers, hits, totals, inventory = base.load_outcomes(assay, audit, metadata, slots)
    logits, fits, directions = crossfit(unit_states, lexical, metadata, slots, families, hits, totals)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.frick-pair-') as work:
        stage = Path(work) / 'evaluation'
        stage.mkdir()
        forecasts = stage / 'forecasts.jsonl'
        with forecasts.open('x', encoding='utf-8') as stream:
            for row in forecast_rows(logits, slots, stimuli):
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
        base.write_json(stage / 'fits.json', {'features': FEATURES, 'methods': METHODS, 'folds': fits})
        np.savez_compressed(stage / 'directions.npz', **directions)
        names = ('forecasts.jsonl', 'fits.json', 'directions.npz')
        seal = stage / 'forecast-seal.json'
        provenance.update({
            'parent_protocol_sha256': provenance['protocol_sha256'],
            'protocol_sha256': base.digest(protocol),
            'reference_forecast_seal_sha256': base.digest(reference_seal),
            'outcomes_sha256': audit['output_sha256']['outcomes.jsonl'],
            'scorer_sha256': base.digest(Path(__file__)),
            'readout_sha256': base.digest(Path(solver.__file__)),
            'evaluation_sha256': base.digest(Path(evaluation.__file__)),
            'numpy': np.__version__, 'scipy': scipy.__version__,
        })
        base.write_json(seal, {'provenance': provenance, 'file_sha256': {name: base.digest(stage / name) for name in names}})
        restored = np.empty_like(logits)
        for i, (declared, row) in enumerate(zip(forecast_rows(logits, slots, stimuli), base.read_rows(forecasts), strict=True)):
            if declared != row:
                raise ValueError('The frozen forecast rows differ from cross-fit probabilities')
            for m, method in enumerate(METHODS):
                scores = row['methods'][method]
                restored[:, m, i] = scores['current_logit'], scores['first_logit']
        report = evaluation.evaluate(restored, slots, families, metadata, trials, readers, reference_logits)
        base.write_json(stage / 'results.json', {
            **report, 'inventory': inventory, 'retrospective': True,
            'forecast_seal_sha256': base.digest(seal),
        })
        stage.rename(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('assay', 'extraction', 'packet', 'output'):
        parser.add_argument(f'--{option}', required=True, type=Path)
    parser.add_argument('--reference', default=REFERENCE, type=Path)
    parser.add_argument('--protocol', default=PROTOCOL, type=Path)
    args = vars(parser.parse_args())
    run(**args)
