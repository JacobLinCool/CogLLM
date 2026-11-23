"""Evaluate the sealed App reading-ease forecasts on independent CLEAR families."""

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import file_digest, sha256
import json
from math import isfinite, log
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unicodedata

import numpy as np
from scipy.stats import rankdata


PROTOCOL = Path(__file__).with_name('clear-reading-ease-protocol.md')
CANDIDATE = '74c26ff02569b2b3bc28ed291bf15fd2f85db5b4c93a3ad7bfc34f6a5d333ea0'
WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
FEATURES = ('log_words', 'words_per_sentence', 'letters_per_word', 'long_word_fraction', 'app_ease')
METHODS = ('app', 'surface', 'surface_app')
CONTRASTS = {'surface_app_minus_surface': (2, 1), 'app_minus_surface': (0, 1)}
SEED = 731503
RESAMPLES = 5000
PENALTY = 0.01
MIN_FAMILIES = 100


@dataclass(frozen=True)
class Passage:
    family: str
    split: str
    genre: str
    soft_overlap: bool
    input_tokens: int
    stops: int
    features: tuple[float, ...]
    target: float


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def jsonl(path):
    with path.open(encoding='utf-8') as stream:
        yield from map(json.loads, stream)


def sealed_inputs(prepared, forecasts, seal):
    assay_path = prepared / 'assay.json'
    assay = json.loads(assay_path.read_text(encoding='utf-8'))
    extraction = json.loads(seal.read_text(encoding='utf-8'))
    provenance = {
        'candidate_id': CANDIDATE,
        'assay_sha256': digest(assay_path),
        'extraction_sha256': digest(seal),
        'protocol_sha256': digest(PROTOCOL),
        'scorer_sha256': digest(Path(__file__)),
        'forecasts_sha256': digest(forecasts),
    }
    packet_files = ('stimuli.jsonl', 'materials.jsonl', 'outcomes.jsonl')
    for name in packet_files:
        actual = digest(prepared / name)
        if actual != assay['file_sha256'][name]:
            raise ValueError(f'CLEAR assay file differs from its receipt: {name}')
        provenance[f'{Path(name).stem}_sha256'] = actual
    if (assay['candidate_id'] != CANDIDATE
            or assay['protocol_sha256'] != provenance['protocol_sha256']
            or any(extraction[key] != provenance[key] for key in (
                'candidate_id', 'protocol_sha256', 'assay_sha256', 'stimuli_sha256'))
            or extraction['file_sha256']['forecasts.jsonl'] != provenance['forecasts_sha256']):
        raise ValueError('Expected sealed forecasts from the locked CLEAR assay and App candidate')
    return assay, extraction, provenance


def valid_number(value):
    return type(value) in (int, float) and isfinite(value)


def passage_features(text, spans, sentence_tokens, stops, input_tokens, sentences):
    if not spans or len(spans) != len(stops) or type(input_tokens) is not int or input_tokens <= 0:
        raise ValueError('Invalid certified CLEAR sentence inventory')
    previous_end = 0
    previous_token = -1
    total_words = 0
    weighted_ease = 0.0
    for bounds, token, stop, sentence in zip(spans, sentence_tokens, stops, sentences, strict=True):
        start, end = bounds
        if (set(sentence) != {'start', 'end', 'token', 'ease'}
                or any(type(value) is not int for value in (start, end, token, stop, sentence['start'],
                                                            sentence['end'], sentence['token']))
                or not previous_end <= start < end <= len(text) or stop != end
                or not previous_token < token < input_tokens
                or (sentence['start'], sentence['end'], sentence['token']) != (start, end, token)
                or not valid_number(sentence['ease'])):
            raise ValueError('Forecast must match each exact certified CLEAR sentence coordinate')
        word_count = len(text[start:end].split())
        if word_count == 0 or text[previous_end:start].strip():
            raise ValueError('Expected nonempty CLEAR sentences covering all visible text')
        total_words += word_count
        weighted_ease += word_count * sentence['ease']
        previous_end, previous_token = end, token
    if text[previous_end:].strip() or not isfinite(weighted_ease):
        raise ValueError('Expected full-text coverage and finite CLEAR sentence forecasts')
    body_words = len(text.split())
    letter_counts = [len(word.replace("'", ''))
                     for word in WORD.findall(unicodedata.normalize('NFKC', text).casefold())]
    if not letter_counts:
        raise ValueError('Expected at least one English word in each certified CLEAR passage')
    return (log(body_words), total_words / len(spans), sum(letter_counts) / len(letter_counts),
            sum(length >= 7 for length in letter_counts) / len(letter_counts), weighted_ease / total_words)


def load_passages(prepared, forecasts):
    paths = (prepared / 'stimuli.jsonl', prepared / 'materials.jsonl', prepared / 'outcomes.jsonl', forecasts)
    grouped = {'calibration': [], 'holdout': []}
    id_inventory = set()
    family_splits = {}
    for stimulus, material, outcome, forecast in zip(*(jsonl(path) for path in paths), strict=True):
        identifier, text = stimulus['id'], stimulus['text']
        if (set(stimulus) != {'id', 'text', 'stops'} or set(forecast) != {'id', 'sentences'}
                or any(row['id'] != identifier for row in (material, outcome, forecast))
                or sha256(text.encode('utf-8')).hexdigest() != identifier or identifier in id_inventory):
            raise ValueError('The four CLEAR streams must match each unique literal passage in order')
        id_inventory.add(identifier)
        family, split, genre = material['family'], material['split'], material['genre']
        soft_overlap = material['soft_overlap']
        expected_split = 'holdout' if int(sha256(f'clear-ease-v1:{family}'.encode()).hexdigest()[:8], 16) % 5 < 2 else 'calibration'
        target, error = outcome['bt_easiness'], outcome['standard_error']
        if (not isinstance(family, str) or not family or split != expected_split or genre not in ('Lit', 'Info')
                or family_splits.setdefault(family, split) != split or type(soft_overlap) is not bool
                or not valid_number(target) or not valid_number(error) or error < 0):
            raise ValueError('Unexpected CLEAR family, split, genre, overlap or teacher outcome')
        features = passage_features(text, material['spans'], material['sentence_tokens'], stimulus['stops'],
                                    material['input_tokens'], forecast['sentences'])
        grouped[split].append(Passage(family, split, genre, soft_overlap, material['input_tokens'],
                                      len(stimulus['stops']), features, target))
    return grouped


def inventory(rows):
    return {
        'passages': len(rows),
        'families': len({row.family for row in rows}),
        'input_tokens': sum(row.input_tokens for row in rows),
        'stops': sum(row.stops for row in rows),
        'by_genre': dict(Counter(row.genre for row in rows)),
    }


def arrays(rows):
    return (np.asarray([row.features for row in rows], dtype=np.float64).reshape(-1, len(FEATURES)),
            np.asarray([row.target for row in rows], dtype=np.float64))


def fit_ridge(calibration):
    design, target = arrays(calibration)
    if len(design) == 0:
        raise ValueError('Empty CLEAR calibration partition')
    varying = np.max(design, axis=0) > np.min(design, axis=0)
    active = np.flatnonzero(varying)
    averages = design[:, active].mean(axis=0)
    scales = design[:, active].std(axis=0)
    standardized = (design[:, active] - averages) / scales
    intercept = float(target.mean())
    rhs = standardized.T @ (target - intercept) / len(target)
    gram = standardized.T @ standardized / len(target)
    fits = {}
    for name, positions in (('surface', np.flatnonzero(active < len(FEATURES) - 1)),
                            ('surface_app', np.arange(len(active)))):
        block = gram[np.ix_(positions, positions)] + PENALTY * np.eye(len(positions))
        coefficients = np.linalg.solve(block, rhs[positions])
        fits[name] = {
            'columns': [FEATURES[index] for index in active[positions]],
            'mean': averages[positions].tolist(), 'std': scales[positions].tolist(),
            'intercept': intercept, 'slopes': coefficients.tolist(),
        }
    return fits


def predict(fits, design):
    values = [design[:, -1]]
    for name in METHODS[1:]:
        fit = fits[name]
        columns = [FEATURES.index(column) for column in fit['columns']]
        standardized = (design[:, columns] - fit['mean']) / fit['std']
        values.append(fit['intercept'] + standardized @ fit['slopes'])
    return np.column_stack(values)


def spearman(values):
    if len(values) < 2:
        return np.full(len(METHODS), np.nan)
    ranks = rankdata(values, axis=0)
    centered = ranks - ranks.mean(axis=0)
    length = np.linalg.norm(centered, axis=0)
    denom = length[0] * length[1:]
    return np.divide(centered[:, 0] @ centered[:, 1:], denom,
                     out=np.full(len(METHODS), np.nan), where=denom > 0)


def bootstrap(values, families):
    # Sampling the same family duplicates every passage in it. Rerank the sampled
    # passages with average ranks for ties, using one draw for all methods.
    groups, inverse = np.unique(families, return_inverse=True)
    simulations = np.full((RESAMPLES, len(METHODS)), np.nan)
    if len(groups) < 2:
        return simulations
    rng = np.random.default_rng(SEED)
    probability = np.full(len(groups), 1 / len(groups))
    rows = np.arange(len(values))
    for start in range(0, RESAMPLES, 128):
        counts = rng.multinomial(len(groups), probability, size=min(128, RESAMPLES - start))
        for i, draw in enumerate(counts, start=start):
            simulations[i] = spearman(values[np.repeat(rows, draw[inverse])])
    return simulations


def optional_float(value):
    return float(value) if np.isfinite(value) else None


def compare(holdout, fits):
    design, target = arrays(holdout)
    predictions = predict(fits, design)
    values = np.column_stack((target, predictions))
    estimates = spearman(values)
    profiles = {method: {'spearman': optional_float(estimates[index])}
                for index, method in enumerate(METHODS)}
    if len(target):
        rmse = np.sqrt(np.mean((predictions[:, 1:] - target[:, None]) ** 2, axis=0))
        for method, value in zip(METHODS[1:], rmse, strict=True):
            profiles[method]['rmse'] = float(value)
    draws = bootstrap(values, [row.family for row in holdout])
    contrasts = {}
    for name, (a, b) in CONTRASTS.items():
        delta = draws[:, a] - draws[:, b]
        finite = delta[np.isfinite(delta)]
        contrasts[name] = {
            'point': optional_float(estimates[a] - estimates[b]),
            'ci95': np.quantile(finite, (0.025, 0.975)).tolist() if len(finite) else None,
            'valid_resamples': len(finite),
        }
    return {**inventory(holdout), 'methods': profiles, 'contrasts': contrasts}


def score(prepared, forecasts, seal, output):
    if output.exists():
        raise FileExistsError(output)
    assay, extraction, provenance = sealed_inputs(prepared, forecasts, seal)
    grouped = load_passages(prepared, forecasts)
    calibration, holdout = grouped['calibration'], grouped['holdout']
    if ({name: inventory(rows) for name, rows in grouped.items()} != assay['split']
            or len(calibration) + len(holdout) != extraction['documents']
            or extraction['content_forwards'] != extraction['documents']
            or sum(row.stops for row in calibration + holdout) != extraction['certified_sentences']):
        raise ValueError('CLEAR source and extraction inventories do not match complete scored passages')

    filtered_calibration = [row for row in calibration if not row.soft_overlap]
    filtered_holdout = [row for row in holdout if not row.soft_overlap]
    fits = fit_ridge(calibration)
    filtered_fits = fit_ridge(filtered_calibration)
    main = compare(holdout, fits)
    interval = main['contrasts']['surface_app_minus_surface']['ci95']
    eligible = main['families'] >= MIN_FAMILIES
    calibration_receipt = {
        'provenance': provenance, 'penalty': PENALTY,
        'all': {'inventory': inventory(calibration), 'fits': fits},
        'no_soft_overlap': {'inventory': inventory(filtered_calibration), 'fits': filtered_fits},
    }
    metrics = {
        'provenance': provenance,
        'bootstrap': {'resamples': RESAMPLES, 'seed': SEED, 'unit': 'source_family',
                      'fits_refit_per_draw': False},
        'holdout': main,
        'holdout_by_genre': {genre: compare([row for row in holdout if row.genre == genre], fits)
                             for genre in ('Info', 'Lit')},
        'holdout_no_soft_overlap': compare(filtered_holdout, filtered_fits),
        'removed_soft_overlap': {'calibration': len(calibration) - len(filtered_calibration),
                                 'holdout': len(holdout) - len(filtered_holdout)},
        'gate': {'min_holdout_families': MIN_FAMILIES, 'eligible': eligible,
                 'passes': bool(eligible and interval is not None and interval[0] > 0)},
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.clear-scoring-') as temporary:
        staged = Path(temporary) / 'scored'
        staged.mkdir()
        calibration_path = staged / 'calibration.json'
        calibration_path.write_text(json.dumps(calibration_receipt, indent=2, sort_keys=True,
                                                allow_nan=False) + '\n', encoding='utf-8')
        metrics['calibration_sha256'] = digest(calibration_path)
        (staged / 'metrics.json').write_text(json.dumps(metrics, indent=2, sort_keys=True,
                                                        allow_nan=False) + '\n', encoding='utf-8')
        staged.rename(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('prepared', 'forecasts', 'seal', 'output'):
        parser.add_argument(f'--{option}', required=True, type=Path)
    score(**vars(parser.parse_args()))
