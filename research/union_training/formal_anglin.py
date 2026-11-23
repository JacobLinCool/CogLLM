"""Prepare and score the prospective, text-only Anglin Study 1 belief transfer."""

import argparse
from hashlib import sha256
from itertools import product
import math
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory

from research.backbone_transfer import extract
from research.backbone_transfer.extract import digest, read_json, stream_rows, write_json, write_rows


HOLDOUT = Path(__file__).with_name('holdout.json')
PROTOCOL = Path(__file__).with_name('formal-anglin-protocol.md')
DESIGN_CODE = Path(__file__).parents[1] / 'semantic_tracking' / 'anglin_audit.py'
HEAD = 'rodger_paired_effectiveness'
ANGLIN_CONTEXT_LIMIT = 4096
RASTER_SHA256 = (
    '6ce9c0771f24a7c314af793cd0e2ede58d326835c0486c3b08531a2aae06cd5d',
    'a876fffd678e2a3e57347835e4a387aca8fa525ea35a2ad30a8e57911527d604',
    'd4a243f5489e429ccf42abb7f4037835f745965ebc93f870e9519faf74262ad8',
    '2f77a83a252107a730aa911108b2c06d4348b7d7cc6af29e4e57d63b738dafe7',
)
ARM_NAMES = ('both_ineffective', 'both_effective', 'ineffective_first', 'effective_first')
PRIMARY = {
    'both_ineffective': ((0, 2, -1),),
    'both_effective': ((0, 2, 1),),
    'ineffective_first': ((0, 1, -1), (1, 2, 1)),
    'effective_first': ((0, 1, 1), (1, 2, -1)),
}
SCALE_TOLERANCE = 0.5
TABLE1_COMPLETE = (68, 93, 72, 79)


def pinned_file(source, name):
    family, = (row for row in read_json(HOLDOUT)['families'] if row['id'] == 'anglin-belief')
    pin, = (row for row in family['source_pins'] if Path(row['path']).name == name)
    path = source / name
    if digest(path) != pin['sha256']:
        raise ValueError(f'Anglin source differs from sealed file: {name}')
    return path


def chart_texts():
    """Manually verified numerical tables and captions from the four assigned raster pages."""
    longitudinal = (
        ('New Jersey', 4.0, 4.9), ('New York', 4.0, 4.8), ('New Mexico', 6.4, 7.2),
        ('Illinois', 4.7, 5.5), ('Connecticut', 2.3, 3.6), ('Maryland', 6.4, 6.3),
        ('Nebraska', 2.5, 3.3),
    )
    cross_sectional = (
        ('Montana', 3.5, 'North Dakota', 2.8), ('South Dakota', 3.7, 'Minnesota', 2.4),
        ('Missouri', 8.3, 'Iowa', 2.4), ('Indiana', 5.6, 'Illinois', 5.8),
        ('Ohio', 5.8, 'Michigan', 4.3), ('Kentucky', 4.7, 'West Virginia', 3.8),
        ('Virginia', 8.6, 'Maryland', 4.6), ('Delaware', 6.7, 'New Jersey', 4.1),
        ('Pennsylvania', 3.1, 'New York', 5.1), ('Arizona', 5.6, 'New Mexico', 4.5),
    )
    charts = {}
    for direction in (1, 2):
        supports = direction == 2
        rows = longitudinal if supports else tuple((state, after, before) for state, before, after in longitudinal)
        relation = 'increased' if supports else 'decreased'
        conclusion = 'support' if supports else 'oppose'
        charts['A', direction] = '\n'.join((
            'Murder Rates in Each State Before and After the Death Penalty Ban',
            'Murders per 100,000 people',
            'State | Year Before Death Penalty Ban | Year After Death Penalty Ban',
            *(f'{state} | {before:.1f} | {after:.1f}' for state, before, after in rows),
            f'In 6 of the 7 states, murder rates {relation} after the death penalty was removed. '
            f'These findings {conclusion} the deterrent effect of the death penalty.',
            'Bar chart (table values). Vertical axis: murders per 100,000 people, 0–10. '
            'Legend: Year Before Ban (black); Year After Ban (gray).',
        ))
        cross_rows = (tuple((a, y, b, x) for a, x, b, y in cross_sectional)
                      if supports else cross_sectional)
        comparison = 'lower' if supports else 'higher'
        charts['B', direction] = '\n'.join((
            'Murder Rate Comparisons between States with and without the Death Penalty',
            'Pair | State with Death Penalty | Murders per 100,000 people | '
            'State without Death Penalty | Murders per 100,000 people',
            *(f'Pair {i} | {a} | {x:.1f} | {b} | {y:.1f}'
              for i, (a, x, b, y) in enumerate(cross_rows, 1)),
            f'In 8 of the 10 pairs, murder rates were {comparison} in the state with the death penalty. '
            f'These findings {conclusion} the deterrent effect of the death penalty.',
            'Bar chart (table values). Vertical axis: murders per 100,000 people, 0–10. '
            'Legend: State with death penalty (black); State without death penalty (gray).',
        ))
    return charts


def between(paragraphs, start, end):
    left = paragraphs.index(start)
    right = paragraphs.index(end, left + 1)
    if right == left + 1:
        raise ValueError(f'Missing material between {start!r} and {end!r}')
    return paragraphs[left + 1:right]


def source_blocks(pdf):
    with TemporaryDirectory() as directory:
        raster = Path(directory) / 'chart'
        subprocess.run(['pdfimages', '-j', str(pdf), str(raster)], check=True, capture_output=True)
        images = sorted(Path(directory).glob('chart-*.jpg'))
        if tuple(digest(path) for path in images) != RASTER_SHA256:
            raise ValueError('The evidence chart images differ from the manually transcribed source')

    raw = subprocess.run(['pdftotext', '-layout', str(pdf), '-'], check=True, capture_output=True, text=True).stdout
    chrome = re.compile(r'^(?:[0-9]+/[0-9]+/[0-9]+\s+Qualtrics Survey Software|https://cmu\.ca1\.|'
                        r'These page timer metrics|First Click:|Last Click:|Page Submit:|Click Count:)')
    visible = (line.replace('\f', '').strip() for line in raw.split('\n'))
    text = '\n'.join(line for line in visible if not chrome.match(line))
    paragraphs = [' '.join(block.split()) for block in re.split(r'\n\s*\n', text) if block.strip()]
    first = between(paragraphs, 'belief_T1', 'instructions_overview')
    second = between(paragraphs, 'belief_T2', 'instructions - study B')
    last = between(paragraphs, 'belief_T3', 'instructions - part 2')
    question_end = 'Strongly deters murder'
    at_report = [block.index(question_end) + 1 for block in (first, second, last)]
    starting = first[:at_report[0]]
    continuing = first[at_report[0]:]
    first_instructions = between(paragraphs, 'instructions_overview', 'DP overview A')
    second_instructions = between(paragraphs, 'instructions - study B', 'instructions_details_B')
    detail_intro = {
        1: between(paragraphs, 'Instructions_detailsA', 'DP deters murder A - details'),
        2: between(paragraphs, 'instructions_details_B', 'belief_T3'),
    }
    overview = {
        'A': between(paragraphs, 'DP overview A', 'DP overview B'),
        'B': between(paragraphs, 'DP overview B', 'MC1'),
    }
    if any(len(block) != 4 or block[0] != 'Study Overview' or block[2] != block[0] for block in overview.values()):
        raise ValueError('Expected both printed orientation variants for each study overview')
    attention = {
        'A': between(paragraphs, 'MC1', 'MC2'),
        'B': between(paragraphs, 'MC2', 'Instructions_detailsA'),
    }
    details = {
        ('A', 1): between(paragraphs, 'DP promotes murder A - details', 'criticisms and rebuttals A - DP deters murder 1'),
        ('A', 2): between(paragraphs, 'DP deters murder A - details', 'DP promotes murder A - details'),
        ('B', 1): between(paragraphs, 'DP promotes murder B - details', 'DP deters murder B - details'),
        ('B', 2): between(paragraphs, 'DP deters murder B - details', 'criticisms and rebuttals B - DP promotes murder 2'),
    }
    criticisms = {
        ('A', 1): between(paragraphs, 'criticisms and rebuttals A - DP promotes murder 1', 'DP promotes murder B - details'),
        ('A', 2): between(paragraphs, 'criticisms and rebuttals A - DP deters murder 1', 'criticisms and rebuttals A - DP promotes murder 1'),
        ('B', 1): between(paragraphs, 'criticisms and rebuttals B - DP promotes murder 2', 'criticisms and rebuttals B - DP deters murder 2'),
        ('B', 2): between(paragraphs, 'criticisms and rebuttals B - DP deters murder 2', 'Evals A'),
    }
    evaluation = {
        'A': between(paragraphs, 'Evals A', 'Evals B'),
        'B': between(paragraphs, 'Evals B', 'belief_T2'),
    }
    charts = chart_texts()

    def read_study(study, direction, time):
        outline = overview[study][2 * (direction - 1):2 * direction]
        section = criticisms[study, direction]
        if sum(block.startswith('Criticism:') for block in section) != 4 or sum(
            block.startswith("Authors' rebuttal:") for block in section
        ) != 4:
            raise ValueError(f'Expected four pairs of critiques and rebuttals in {study}, {direction}')
        return [*outline, *attention[study], *detail_intro[time], *details[study, direction],
                charts[study, direction], *section, *evaluation[study]]

    for a, b, order in product((1, 2), repeat=3):
        first_study, second_study = ('A', 'B') if order == 1 else ('B', 'A')
        directions = {'A': a, 'B': b}
        middle = [*starting, *continuing, *first_instructions,
                  *read_study(first_study, directions[first_study], 1), *second[:at_report[1]]]
        ending = [*middle, *second[at_report[1]:], *second_instructions,
                  *read_study(second_study, directions[second_study], 2), *last[:at_report[2]]]
        pieces = ['\n\n'.join(block) for block in (starting, middle, ending)]
        if not pieces[2].startswith(pieces[1]) or not pieces[1].startswith(pieces[0]):
            raise ValueError('The immediate assessment histories are not nested prefixes')
        yield (a, b, order), pieces[2], [len(piece) for piece in pieces]


def prepare(source, tokenizer_path, output):
    from tokenizers import Tokenizer

    pdf = pinned_file(source, 'stimulus.pdf')
    model_files = read_json(extract.MODELS)['models']['qwen']['sha256']
    tokenizer_sha256 = model_files['tokenizer.json']
    if digest(tokenizer_path) != tokenizer_sha256:
        raise ValueError('Anglin must use the pinned Qwen tokenizer')
    config_path = tokenizer_path.with_name('config.json')
    if (digest(config_path) != model_files['config.json']
            or read_json(config_path)['text_config']['max_position_embeddings'] < ANGLIN_CONTEXT_LIMIT):
        raise ValueError('Anglin requires the pinned Qwen context configuration')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    stimuli = []
    design = []
    usual_limit = extract.MAX_INPUT_TOKENS
    extract.MAX_INPUT_TOKENS = ANGLIN_CONTEXT_LIMIT
    try:
        for (a, b, order), text, stops in source_blocks(pdf):
            identifier = f'anglin-a{a}-b{b}-o{order}'
            encoded = tokenizer.encode(text, add_special_tokens=False)
            geometry = extract.certify(identifier, text, stops, encoded, tokenizer)
            if list(geometry.certified) != stops:
                raise ValueError(f'Anglin timepoints are not stable token boundaries: {identifier}')
            stimuli.append({'id': identifier, 'text': text, 'stops': stops})
            design.append({'id': identifier, 'a': a, 'b': b, 'order': order, 'tokens': len(encoded.ids),
                           'text_sha256': sha256(text.encode()).hexdigest()})
    finally:
        extract.MAX_INPUT_TOKENS = usual_limit
    output.mkdir(parents=True, exist_ok=False)
    write_rows(output / 'stimuli.jsonl', stimuli)
    write_json(output / 'receipt.json', {
        'protocol_sha256': digest(PROTOCOL), 'analysis_code_sha256': digest(Path(__file__)),
        'design_code_sha256': digest(DESIGN_CODE),
        'stimulus_pdf_sha256': digest(pdf), 'tokenizer_sha256': tokenizer_sha256,
        'stimuli_sha256': digest(output / 'stimuli.jsonl'), 'histories': design, 'max_input_tokens': ANGLIN_CONTEXT_LIMIT,
    })
    print(f'{len(stimuli)} fixed Anglin histories, longest {max(row["tokens"] for row in design)} tokens')


def verify_prepared(path):
    receipt = read_json(path / 'receipt.json')
    if (receipt['protocol_sha256'] != digest(PROTOCOL) or receipt['analysis_code_sha256'] != digest(Path(__file__))
            or receipt['design_code_sha256'] != digest(DESIGN_CODE)
            or receipt['stimuli_sha256'] != digest(path / 'stimuli.jsonl')):
        raise ValueError('The Anglin materials are not bound to this prospective protocol')
    return receipt


def ordinal_probability(mean, spread):
    if not math.isfinite(mean) or not math.isfinite(spread) or spread <= 0:
        raise ValueError('Invalid frozen efficacy prediction')
    boundaries = [100 * (category + 0.5) / 7 for category in range(7)]
    cumulative = [0.0, *(0.5 * math.erfc((mean - bound) / (math.sqrt(2) * spread)) for bound in boundaries), 1.0]
    return [high - low for low, high in zip(cumulative[:-1], cumulative[1:], strict=True)]


def predict(prepared, projection, packet, heads, output):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from research.union_training.fit_observations import MIN_SCALE, load_projection
    from research.union_training.run_observations_l4 import verify_packet

    receipt = verify_prepared(prepared)
    extraction = read_json(projection / 'seal.json')
    if (extraction['candidate'] != 'qwen' or extraction['stimuli_sha256'] != receipt['stimuli_sha256']
            or extraction['tokenizer_sha256'] != receipt['tokenizer_sha256']
            or extraction['protocol_sha256'] != receipt['protocol_sha256']
            or extraction['content_forward_per_document'] != 1 or extraction['documents'] != len(receipt['histories'])
            or extraction['certified_stops'] != 3 * len(receipt['histories'])
            or any(digest(projection / file) != hash_ for file, hash_ in extraction['file_sha256'].items())):
        raise ValueError('Unverified Anglin representations or mismatched prospective protocol')
    verify_packet(packet)
    fitted = read_json(heads.parent / 'seal.json')
    if (fitted['files_sha256'][heads.name] != digest(heads) or fitted['packet_receipt_sha256'] != digest(packet / 'receipt.json')
            or fitted['encoder_revision'] != extraction['revision']):
        raise ValueError('Checkpoint does not match the training packet or backbone')
    rows = next(row for row in read_json(packet / 'index.json')['tasks'] if row['name'] == HEAD)
    original = np.asarray([row['outcome'] for row in stream_rows(packet / rows['file']) if row['role'] == 0], dtype=np.float64)
    center, scale = original.mean(), original.std()
    if scale <= 0 or rows['states'] != ['belief'] or rows['likelihood'] != 'paired_gaussian':
        raise ValueError('Rodger readout has an unrecognized state or scale')
    states, indexes = load_projection(projection, 'cpu')
    responses = []
    with np.load(heads, allow_pickle=False) as archive:
        def weight(name):
            return torch.from_numpy(archive[name])

        projection_weights = weight('shared.belief.weight'), weight('shared.belief.bias')
        measurement = weight(f'measurements.{HEAD}.linear.weight'), weight(f'measurements.{HEAD}.linear.bias')
        post_offset = weight(f'measurements.{HEAD}.post_offset').item()
        for row in stream_rows(prepared / 'stimuli.jsonl'):
            mapped = F.linear(states[[indexes[row['id'], stop] for stop in row['stops']]], *projection_weights).tanh()
            raw = F.linear(mapped, *measurement)
            # T1 is the 'before' member both times; T2/T3 consistently use the documented post offset.
            means = (raw[:, 0] + torch.tensor([0, post_offset, post_offset], dtype=raw.dtype)) * scale + center
            spreads = (F.softplus(raw[:, 1]) + MIN_SCALE) * scale
            probs = [ordinal_probability(mu.item(), sigma.item()) for mu, sigma in zip(means, spreads, strict=True)]
            responses.append({'id': row['id'], 'rodger_scale_means': means.tolist(), 'rodger_scale_sd': spreads.tolist(),
                              'probs': probs, 'mean_1_to_8': [sum(i * p for i, p in enumerate(time, 1)) for time in probs]})
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'predictions.json', {'prepared_receipt_sha256': digest(prepared / 'receipt.json'),
                                            'head_sha256': digest(heads), 'extraction_sha256': digest(projection / 'seal.json'),
                                            'responses': responses})
    write_json(output / 'seal.json', {'predictions_sha256': digest(output / 'predictions.json'),
                                      'protocol_sha256': digest(PROTOCOL), 'analysis_code_sha256': digest(Path(__file__))})


def score(source, prepared, predictions, output):
    import numpy as np
    import pyreadstat

    receipt = verify_prepared(prepared)
    sealed = read_json(predictions / 'seal.json')
    forecast_path = predictions / 'predictions.json'
    forecast = read_json(forecast_path)
    if (sealed['predictions_sha256'] != digest(forecast_path) or sealed['protocol_sha256'] != digest(PROTOCOL)
            or sealed['analysis_code_sha256'] != digest(Path(__file__))
            or forecast['prepared_receipt_sha256'] != digest(prepared / 'receipt.json')):
        raise ValueError('Seal the predictions against the protocol before loading participant outcomes')
    means = {row['id']: np.asarray(row['mean_1_to_8'], dtype=np.float64) for row in forecast['responses']}
    if set(means) != {row['id'] for row in receipt['histories']} or any(
        value.shape != (3,) or not np.isfinite(value).all() for value in means.values()
    ):
        raise ValueError('Sealed package has an invalid cohort forecast')

    from research.semantic_tracking.anglin_audit import BELIEF, DESIGN, get_design

    data_path = pinned_file(source, 'study1.sav')
    data, _ = pyreadstat.read_sav(str(data_path), usecols=[*DESIGN, *BELIEF], output_format='dict')
    a, b, order, condition, known = get_design(data)
    belief = np.column_stack([np.asarray(data[name], dtype=np.float64) for name in BELIEF])
    complete = known & np.isfinite(belief).all(axis=1)
    if tuple(int(np.sum(complete & (condition == index))) for index in range(1, 5)) != TABLE1_COMPLETE:
        raise ValueError('The eligible analytic sample differs from Study 1 Table 1')
    included = [identifier for identifier, keep in zip(data['ID'], complete, strict=True) if keep]
    if (any(identifier is None or identifier == '' or (isinstance(identifier, float) and not math.isfinite(identifier))
            for identifier in included)
            or len(set(included)) != len(included)
            or not np.all(np.isin(np.column_stack((a, b, order))[complete], (1, 2)))
            or not np.all(np.isin(belief[complete], np.arange(1, 9)))):
        raise ValueError('An included participant has an invalid rating, key, or exposure history')

    rng = np.random.default_rng(20261121)
    results = {}
    for condition_index, arm in enumerate(ARM_NAMES, 1):
        per_history = []
        for history in receipt['histories']:
            mask = complete & (condition == condition_index) & (a == history['a']) & (b == history['b']) & (order == history['order'])
            if mask.any():
                per_history.append((belief[mask, :3], means[history['id']]))
        total = sum(len(values) for values, _ in per_history)
        checks = []
        for before, after, direction in PRIMARY[arm]:
            predicted = sum(len(values) * (mu[after] - mu[before]) for values, mu in per_history) / total
            observed = sum((values[:, after] - values[:, before]).sum() for values, _ in per_history) / total
            replicates = np.zeros(10000)
            for values, _ in per_history:
                delta = values[:, after] - values[:, before]
                replicates += delta[rng.integers(0, len(delta), (len(replicates), len(delta)))].sum(axis=1) / total
            disagreement = predicted - replicates
            low, high = np.quantile(disagreement, [0.025, 0.975])
            checks.append({'before': before + 1, 'after': after + 1, 'n': total, 'published_direction': direction,
                           'predicted_change': predicted, 'human_change': observed, 'predicted_minus_human': predicted - observed,
                           'difference_ci95': [float(low), float(high)], 'direction_match': bool(direction * predicted > 0),
                           'within_tolerance': bool(low >= -SCALE_TOLERANCE and high <= SCALE_TOLERANCE)})
        results[arm] = checks
    all_checks = [aggregate for checks in results.values() for aggregate in checks]
    write_json(output, {
        'prediction_sha256': digest(forecast_path), 'subject_count': int(complete.sum()), 'primary': results,
        'alpha_per_two_sided_interval': 0.05, 'tolerance': SCALE_TOLERANCE,
        'directions_matched': sum(result['direction_match'] for result in all_checks),
        'within_tolerance': sum(result['within_tolerance'] for result in all_checks),
        'all_checks_passed': all(result['direction_match'] and result['within_tolerance'] for result in all_checks),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    for action, names in (
        ('prepare', ('source', 'tokenizer-path', 'output')),
        ('predict', ('prepared', 'projection', 'packet', 'heads', 'output')),
        ('score', ('source', 'prepared', 'predictions', 'output')),
    ):
        command = actions.add_parser(action)
        for name in names:
            command.add_argument(f'--{name}', required=True, type=Path)
    fields = vars(parser.parse_args())
    action = fields.pop('action')
    {'prepare': prepare, 'predict': predict, 'score': score}[action](**fields)


if __name__ == '__main__':
    main()
