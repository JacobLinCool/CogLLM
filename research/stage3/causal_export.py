"""Export the PLABA sentence-prefix rating head from the verified FP32 L4 cache."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .probe import TARGETS, _load_plaba, evaluate
from .run import MODEL, REVISION, save_json


RESULTS = Path(__file__).resolve().parents[1] / 'results'
STAGE3 = RESULTS / 'stage3-l4' / 'float32'
WIDTH = 2048
MAE_TOLERANCE = 1e-12


def sha256(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verified_inputs(features, data_dir, stage3_payload):
    verification = read_json(STAGE3 / 'verification.json')
    for filename in ('manifest.json', 'environment.json', 'causality.json'):
        if sha256(STAGE3 / filename) != verification['artifact_sha256'][filename]:
            raise ValueError(f'{filename} differs from the verified Stage 3 run')
    feature_digest = sha256(features)
    reference = read_json(STAGE3 / 'feature-ablation.json')
    if feature_digest != verification['cached_colab_features_sha256'] or feature_digest != reference['cached_sha256']:
        raise ValueError('Feature archive differs from the locked FP32 Stage 3 cache')
    environment = read_json(STAGE3 / 'environment.json')
    if (environment['model'], environment['revision'], environment['compute_dtype'], environment['readout_features']) != (
        MODEL, REVISION, 'float32', ['sentence_end_final_hidden', 'article_end_final_hidden'],
    ):
        raise ValueError('Unexpected FP32 checkpoint or feature order')
    causal_checks = read_json(STAGE3 / 'causality.json')['checks']
    if len(causal_checks) != 12 or not all(check['passed'] for check in causal_checks):
        raise ValueError('Expected the locked Stage 3 causality checks and positive controls to pass')

    baseline = read_json(STAGE3 / 'manifest.json')['data_sha256']
    current = {name: item['sha256'] for name, item in read_json(RESULTS / 'stage2-audit.json')['plaba']['artifacts'].items()}
    for directory, hashes in ((data_dir, current), (stage3_payload, baseline)):
        for filename in ('splits.json', 'stimuli.jsonl', 'trials.jsonl'):
            if sha256(directory / filename) != hashes[filename]:
                raise ValueError(f'{directory / filename} differs from its pinned source')
    if _load_plaba(data_dir) != _load_plaba(stage3_payload):
        raise ValueError('Updated PLABA records change the Stage 3 sentence ratings, readers or splits')
    return reference, {
        'features_sha256': feature_digest,
        'current_stage2_sha256': current,
        'archived_stage3_payload_sha256': baseline,
        'same_evaluated_records_across_data_versions': True,
        'source_manifest_sha256': {name: sha256(path) for name, path in (
            ('stage2', RESULTS / 'stage2-audit.json'), ('stage3', STAGE3 / 'manifest.json'),
            ('stage3_verification', STAGE3 / 'verification.json'), ('feature_ablation', STAGE3 / 'feature-ablation.json'),
        )},
    }


def run(features, data_dir, stage3_payload, output):
    reference, inputs = verified_inputs(features, data_dir, stage3_payload)
    with np.load(features, allow_pickle=False) as archive:
        ids = archive['ids'].tolist()
        hidden = archive['features']
    if hidden.shape != (len(ids), 2 * WIDTH) or hidden.dtype != np.float32:
        raise ValueError('Expected sentence-end and article-end hidden state blocks of width 2048')
    expected = {**reference['pooled_mae']['controls'], **reference['pooled_mae']['sentence_end']}
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.causal-plaba-') as temporary:
        candidate = Path(temporary) / 'checkpoint'
        metrics = evaluate(data_dir, ids, hidden[:, :WIDTH], candidate)
        error = max(abs(metrics['pooled_mae'][method][target] - expected[method][target])
                    for method in expected for target in TARGETS)
        if error > MAE_TOLERANCE:
            raise ArithmeticError(f'Sentence-end OOF replay differs by {error:.3g}')
        (candidate / 'predictions.jsonl').unlink()
        save_json(candidate / 'manifest.json', {
            'model': MODEL, 'revision': REVISION, 'source_feature_compute_dtype': 'float32',
            'source_feature_block': 'sentence_end_final_hidden', 'source_feature_slice': [0, WIDTH],
            'deployment_readout': 'qwen_plus_length',
            'model_exposure': 'Plain article tokens through the current sentence boundary',
            'human_rating_exposure': 'Full article shown, then sentence-level processing ease and understanding self-rating',
            'target_scale': [1, 5], 'targets': list(TARGETS),
            'input_transform': 'L2-normalized sentence-end state, then standardized log1p Unicode character and '
                               'whitespace-separated word counts divided by sqrt(2)',
            'scope': 'PLABA-EVAL English biomedical source articles and aligned plain-language rewrites',
            'data': inputs, 'numpy': np.__version__,
            'verification': {'all_data_head_separate_from_oof_fits': True, 'compared_mae_values': len(expected) * len(TARGETS),
                             'maximum_absolute_mae_difference': error, 'mae_absolute_tolerance': MAE_TOLERANCE},
            'head_sha256': sha256(candidate / 'head.npz'), 'metrics_sha256': sha256(candidate / 'metrics.json'),
        })
        candidate.rename(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('features', 'data-dir', 'stage3-payload', 'output'):
        parser.add_argument(f'--{field}', type=Path, required=True)
    run(**vars(parser.parse_args()))
