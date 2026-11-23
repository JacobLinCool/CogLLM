"""Package the frozen Gradio L4 verification inputs without reader outcome files."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile


REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / '.cache/cogllm'
MODEL = 'Qwen/Qwen3.5-2B'
REVISION = '15852e8c16360a2fea060d615a32b45270f8a8fc'
SOURCE_NAMES = (
    'gradio_app/app', 'gradio_app/causal_plaba', 'gradio_app/combined', 'gradio_app/cuda',
    'gradio_app/checkpoint', 'gradio_app/lexical', 'gradio_app/ordinal', 'gradio_app/plaba',
    'gradio_app/tokenizer', 'gradio_app/verify_causal_plaba',
    'gradio_app/verify_combined', 'gradio_app/pack_cuda_verification',
    'semantic_tracking/iirc_anchor_fit', 'semantic_tracking/iirc_anchor_score',
    'semantic_tracking/iirc_prepare', 'semantic_tracking/joint_score',
    'core_features/readback_score', 'stage2/records', 'stage3/probe', 'stage3/run',
    'stage5/complex_tokenization',
)
RESULT_NAMES = (
    'causal-plaba-checkpoint/head.npz', 'causal-plaba-checkpoint/manifest.json',
    'iirc-anchor-run/heads.npz', 'iirc-anchor-run/fit-manifest.json', 'iirc-anchor-run/predictions.jsonl',
    'iirc-anchor-run/scores.json', 'stage5-complex/fit/heads.npz', 'stage5-complex/fit/fitting.json',
)
PUNKT_FILES = ('README', 'english/abbrev_types.txt', 'english/collocations.tab',
               'english/ortho_context.tab', 'english/sent_starters.txt')
MODEL_FILES = ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'merges.txt', 'vocab.json',
               'model.safetensors.index.json', 'model.safetensors-00001-of-00001.safetensors')
DEPENDENCIES = {'gradio', 'nltk', 'numpy', 'transformers'}
MANIFEST = 'verification-payload.json'
PAYLOAD = 'gradio-l4-verification.tar.gz'


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def json_bytes(data):
    return (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')


def sources(args):
    paths = {f'research/{name}.py': REPO / f'research/{name}.py' for name in SOURCE_NAMES}
    paths.update({f'research/results/{name}': REPO / f'research/results/{name}' for name in RESULT_NAMES})
    paths.update({f'nltk_data/tokenizers/punkt_tab/{name}': args.nltk_data / f'tokenizers/punkt_tab/{name}'
                  for name in PUNKT_FILES})
    paths.update({'fixtures/iirc-stimuli.jsonl': args.iirc_stimuli,
                  'fixtures/plaba-features.npz': args.plaba_features,
                  'fixtures/plaba-stimuli.jsonl': args.plaba_stimuli,
                  'fixtures/iirc-dev/stimuli.jsonl': args.iirc_eval})
    return paths


def check_receipts(paths, hashes):
    plaba = load_json(paths['research/results/causal-plaba-checkpoint/manifest.json'])
    iirc_manifest = 'research/results/iirc-anchor-run/fit-manifest.json'
    iirc = load_json(paths[iirc_manifest])
    iirc_scores = load_json(paths['research/results/iirc-anchor-run/scores.json'])
    lexical = load_json(paths['research/results/stage5-complex/fit/fitting.json'])
    expected = {
        'fixtures/plaba-stimuli.jsonl': plaba['data']['current_stage2_sha256']['stimuli.jsonl'],
        'fixtures/plaba-features.npz': plaba['data']['features_sha256'],
        'fixtures/iirc-stimuli.jsonl': iirc['stimuli_sha256'],
        'fixtures/iirc-dev/stimuli.jsonl': iirc['evaluation_sha256'],
        'research/results/causal-plaba-checkpoint/head.npz': plaba['head_sha256'],
        'research/results/iirc-anchor-run/heads.npz': iirc['heads_sha256'],
        'research/results/iirc-anchor-run/predictions.jsonl': iirc['predictions_sha256'],
        'research/results/stage5-complex/fit/heads.npz': lexical['artifacts']['heads.npz']['sha256'],
    }
    differences = [name for name, value in expected.items() if hashes[name] != value]
    if differences:
        raise ValueError(f'Inputs differ from the frozen readout receipts: {differences}')
    if (any((receipt['model'], receipt['revision']) != (MODEL, REVISION) for receipt in (plaba, iirc))
            or not iirc_scores['acceptance']['all_conditions_passed']
            or iirc_scores['input_sha256']['heads'] != iirc['heads_sha256']
            or iirc_scores['input_sha256']['seal'] != hashes[iirc_manifest]):
        raise ValueError('Readout receipts disagree with the pinned model or acceptance results')


def tar_entry(archive, name, size, contents):
    member = tarfile.TarInfo(name)
    member.size = size
    archive.addfile(member, contents)


def pack(args):
    from .checkpoint import MANIFEST as CHECKPOINT_MANIFEST, MEMBERS, load_checkpoint

    checkpoint = load_checkpoint(args.checkpoint)
    paths = sources(args)
    paths.update({f'research/results/gradio-candidate-checkpoint/{name}': checkpoint.directory / name
                  for name in MEMBERS | {CHECKPOINT_MANIFEST}})
    hashes = {name: digest(path) for name, path in paths.items()}
    check_receipts(paths, hashes)
    for name in SOURCE_NAMES:
        path = paths[f'research/{name}.py']
        compile(path.read_bytes(), str(path), 'exec')
    requirement_lines = (REPO / 'research/gradio_app/requirements.txt').read_text(encoding='utf-8').splitlines()
    dependencies = sorted(line for line in requirement_lines if line.partition('==')[0] in DEPENDENCIES)
    if {line.partition('==')[0] for line in dependencies if '==' in line} != DEPENDENCIES:
        raise ValueError('CUDA verification requires pinned Gradio, NLTK, NumPy and Transformers')
    report = {
        'model': MODEL, 'revision': REVISION, 'model_files': MODEL_FILES, 'python_requirements': dependencies,
        'checkpoint_sha256': checkpoint.identifier,
        'device': 'NVIDIA L4', 'compute_dtype': 'float32', 'held_out_iirc_outcomes_packaged': False,
        'members': {name: {'sha256': hashes[name], 'bytes': path.stat().st_size} for name, path in paths.items()},
    }
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = args.output / MANIFEST
    manifest.write_bytes(json_bytes(report))
    compressed = args.output / PAYLOAD
    with compressed.open('wb') as raw, gzip.GzipFile(fileobj=raw, filename='', mode='wb', mtime=0) as stream:
        with tarfile.open(fileobj=stream, mode='w:') as archive:
            for name, path in sorted({**paths, MANIFEST: manifest}.items()):
                with path.open('rb') as contents:
                    tar_entry(archive, name, path.stat().st_size, contents)
    verify_archive(compressed, report)
    receipt = {'sha256': digest(compressed), 'bytes': compressed.stat().st_size, 'members': len(paths) + 1}
    (args.output / 'archive-receipt.json').write_bytes(json_bytes(receipt))
    print(json.dumps(receipt))


def verify_archive(path, data):
    expected = {**data['members'], MANIFEST: {'sha256': hashlib.sha256(json_bytes(data)).hexdigest(),
                                             'bytes': len(json_bytes(data))}}
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive:
            info = expected.pop(member.name)
            if not member.isfile() or member.size != info['bytes']:
                raise ValueError(f'Invalid archive member: {member.name}')
            with archive.extractfile(member) as contents:
                if hashlib.file_digest(contents, 'sha256').hexdigest() != info['sha256']:
                    raise ValueError(f'Member digest mismatch: {member.name}')
    if expected:
        raise ValueError(f'Missing archive members: {sorted(expected)}')


def verify(root):
    report = load_json(root / MANIFEST)
    archive = root / PAYLOAD
    if archive.is_file():
        verify_archive(archive, report)
        receipt = load_json(root / 'archive-receipt.json')
        if digest(archive) != receipt['sha256']:
            raise ValueError('Archive digest differs from the prepared receipt')
    else:
        for name, info in report['members'].items():
            if (root / name).stat().st_size != info['bytes'] or digest(root / name) != info['sha256']:
                raise ValueError(f'Extracted payload differs at {name}')
    print('Gradio L4 verification payload checksums verified')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    build = actions.add_parser('pack')
    build.add_argument('--output', type=Path, required=True)
    build.add_argument('--checkpoint', type=Path, default=REPO / 'research/results/gradio-candidate-checkpoint')
    build.add_argument('--nltk-data', type=Path, default=Path.home() / 'nltk_data')
    build.add_argument('--iirc-stimuli', type=Path, default=CACHE / 'iirc/anchor-inputs/stimuli.jsonl')
    build.add_argument('--iirc-eval', type=Path, default=CACHE / 'iirc/prepared/dev/stimuli.jsonl')
    build.add_argument('--plaba-stimuli', type=Path, default=CACHE / 'stage2/plaba/stimuli.jsonl')
    build.add_argument('--plaba-features', type=Path, default=CACHE /
                       'stage3/cogllm-stage3-fp32-20261120T141921Z-dd6129/outputs/features.npz')
    actions.add_parser('verify').add_argument('root', type=Path)
    args = parser.parse_args()
    if args.action == 'pack':
        pack(args)
    else:
        verify(args.root)


if __name__ == '__main__':
    main()
