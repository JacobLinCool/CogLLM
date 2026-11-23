"""Apply held-out token observations after the joint predictions are sealed."""

import argparse
from pathlib import Path

from tokenizers import Tokenizer

from research.backbone_transfer.extract import digest, read_json, write_json
from research.token_readout.data import TOKENIZER
from research.token_readout.score import evidence_summary, frick_summary, read_frick
from research.token_readout.transfer_data import EncodedText
from research.token_readout.transfer_score import (
    ASSAYS, ab_rows, ab_summary, compare, control_summary, controls_rows, evidence_rows, load_bundle, load_forecasts,
)
from research.union_training.verify_observations import verified_member


def run(prepared, joint_inputs, runs, seed_run, tokenizer_path, output):
    receipt, files, texts, design = load_bundle(prepared)
    seal = read_json(runs / 'seal.json')
    inputs = read_json(joint_inputs / 'receipt.json')
    if (seal['formal_labels_in_worker'] or seal['input_receipt_sha256'] != digest(joint_inputs / 'receipt.json')
            or inputs['source_token_receipt_sha256'] != digest(prepared / 'receipt.json')
            or digest(tokenizer_path) != receipt['tokenizer_sha256']['qwen']):
        raise ValueError('Joint token forecasts and locked comparison inputs differ')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    layout = {key: row for key, row in read_json(files['qwen.json'])['rows'].items() if row['split'] == 'test'}
    expected = {kind: {key for key, row in layout.items() if row['kind'] == kind} for kind in ASSAYS}
    encoded = {key: EncodedText.encode(texts[key], tokenizer) for key in layout}
    if any(value.offsets != tuple(map(tuple, layout[key]['offsets'])) for key, value in encoded.items()):
        raise ValueError('Token geometry no longer matches the verified test layout')
    frick = [row for row in design['frick'] if row['split'] == 'test']
    evidence = [row for row in design['evidence'] if row['split'] == 'test']
    stops = receipt['structural_alignment']['control_shared_stops']

    def apply(forecast):
        if (set(forecast) != set(expected) or any(set(forecast[kind]) != identities
                                                 for kind, identities in expected.items())):
            raise ValueError('Forecast includes unexpected or missing final stimulus IDs')
        return {
            'frick': read_frick(frick, (), False, forecast['frick']),
            'evidence': evidence_rows(evidence, encoded, layout, forecast['evidence']),
            'controls': controls_rows(design['controls'], encoded, layout, forecast['controls'], stops),
            'ab': ab_rows(design['ab'], encoded, layout, forecast['ab'], stops),
        }

    seed_seal = read_json(seed_run / 'seal.json')
    if seed_seal['files_sha256']['span-calibrated.pt'] != seal['seed_sha256']:
        raise ValueError('The comparison forecast is from another seed checkpoint')
    seed_forecasts, hashes = load_forecasts(seed_run, 'qwen', seed_seal['protocol_sha256'], expected)
    reference, initial = map(apply, (seed_forecasts['position'], seed_forecasts['span-calibrated']))
    arms, ordered = {}, {}
    for trial_seed in seal['seed_settings']['seeds']:
        for method in seal['seed_settings']['modes']:
            name = f'{method}-{trial_seed}'
            relative = f'{name}/token-forecast.json'
            graded = apply(read_json(verified_member(runs, Path(relative), seal['files_sha256'][relative])))
            ordered[name] = graded
            arms[name] = {**summary(graded, reference), 'vs_seed_checkpoint': compare(graded, initial)}
    contrasts = ({str(trial_seed): compare(ordered[f'adaptive-{trial_seed}'], ordered[f'fixed-{trial_seed}'])
                  for trial_seed in seal['seed_settings']['seeds']}
                 if {'adaptive', 'fixed'} <= set(seal['seed_settings']['modes']) else {})
    write_json(output, {
        'joint_seal_sha256': digest(runs / 'seal.json'), 'seed_forecast_hashes': hashes,
        'test_bundle_sha256': digest(prepared / 'receipt.json'), 'seed_checkpoint': summary(initial, reference),
        'position': summary(reference, reference), 'arms': arms, 'adaptive_minus_fixed': contrasts,
        'interpretation': {'frick': 'human first report at sentence presentation',
                           'evidence': 'annotator marked supporting text for supplied question',
                           'controls': 'authored early/late control, no human timings',
                           'ab': 'authored two-anchor control conditioned on anchors, no human timings'},
    })


def summary(values, reference):
    return {'frick': frick_summary(values['frick'], reference['frick']),
            'evidence': evidence_summary(values['evidence'], reference['evidence']),
            'controls': control_summary(values['controls']), 'ab': ab_summary(values['ab']),
            'details': values}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared', 'joint-inputs', 'runs', 'seed-run', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--tokenizer-path', type=Path, default=TOKENIZER)
    run(**vars(parser.parse_args()))
