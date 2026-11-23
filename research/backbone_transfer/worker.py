"""Execute one sealed frozen-backbone extraction on the owned L4 kernel."""

import argparse
from hashlib import file_digest
import json
import os
from pathlib import Path
from shutil import copyfileobj
import subprocess
import sys
import tarfile
from time import monotonic
import traceback


CHUNK_BYTES = 12 * 1024 * 1024
EXTRACTIONS = {
    'stops': ('research.backbone_transfer.extract', ('--targets', 'targets.jsonl')),
    'load-curve': ('research.semantic_tracking.load_curve_extract', ()),
    'expository-lora': ('research.expository_lora.run_l4', ()),
    'union-training': ('research.union_training.run_l4', ()),
    'union-observations': ('research.union_training.run_observations_l4', ()),
    'observation-formal-stops': ('research.union_training.formal_extract', ()),
    'token-readout': ('research.token_readout.fit', ('--training', 'training.json', '--layout', 'layout.json',
                                                   '--settings', 'settings.json')),
    'token-readout-ui': ('research.token_readout.live', ('--checkpoint', 'checkpoint.pt')),
    'portable-token-readout': ('research.token_readout.transfer_fit', ('--learning', 'learning.json',
                                                                      '--layout', 'layout.json',
                                                                      '--settings', 'settings.json')),
    'portable-token-ui': ('research.token_readout.portable_live', ('--checkpoint', 'checkpoint.pt')),
    'adaptive-joint': ('research.adaptive_joint.fit', ('--settings', 'settings.json')),
    'reading-validation': ('research.reading_validation.identity_run_l4', ()),
    'reading-validation-memory': ('research.reading_validation.memory_l4', ()),
    'reading-validation-architecture': ('research.reading_validation.architecture_l4', ()),
    'reading-validation-retrieval': ('research.reading_validation.retrieval_l4', ()),
    'reading-validation-scientific-clues': ('research.reading_validation.clue_bundle_l4', ()),
    'reading-validation-exact-word': ('research.reading_validation.exact_replay_bundle_l4', ()),
    'reading-validation-kerwer-headings': ('research.reading_validation.kerwer_replay_bundle_l4', ()),
    'reading-validation-muse': ('research.reading_validation.muse_bundle_l4', ()),
    'reading-validation-summary-writer': ('research.reading_validation.summary_writer_bundle_l4', ()),
    'reading-validation-summary-specialist': ('research.reading_validation.seahorse_comparator', ()),
    'reading-validation-t5-decision': ('research.reading_validation.t5_decision_comparator', ()),
    'reading-validation-reader-history': ('research.reading_validation.reader_history_bundle_l4', ()),
    'reading-validation-summary-choices': ('research.reading_validation.seahorse_native_l4', ()),
    'reading-validation-population-serving': ('research.reading_validation.population_serving_l4', ()),
    'reading-validation-expert-choices': ('research.reading_validation.reader_choice_l4', ()),
    'reading-validation-cloze-population': ('research.reading_validation.cloze_population_l4', ()),
    'reading-validation-author-context': ('research.reading_validation.author_context_l4', ()),
    'reading-validation-physiology': ('research.reading_validation.physiology_portability_l4', ()),
    'reading-validation-reason-quality': ('research.reading_validation.reason_quality_l4', ()),
    'reading-validation-transport-mandarin': ('research.reading_validation.transport_mandarin_l4', ()),
    'reading-validation-reader-followup': ('research.reading_validation.reader_followup_l4', ()),
    'reading-validation-game-recovery': ('research.reading_validation.game_recovery_l4', ()),
    'reading-validation-symbols': ('research.reading_validation.symbol_l4', ()),
    'reading-validation-t5-adapter': ('research.reading_validation.t5_adapter_comparator', ()),
    'reading-validation-retrieval-news': ('research.reading_validation.guardian_retrieval_l4', ()),
    'reading-validation-narrative-writer': ('research.reading_validation.narrative_writer_l4', ()),
    'reading-validation-narrative': ('research.reading_validation.narrative_course_l4', ()),
    'reading-validation-direct-game': ('research.reading_validation.direct_qwen_l4', ()),
    'reading-validation-direct': ('research.reading_validation.direct_l4', ()),
    'reading-validation-t5': ('research.reading_validation.t5_comparator', ()),
    'reading-validation-definitions': ('research.reading_validation.definition_l4', ()),
    'reading-validation-review': ('research.reading_validation.identity_review_l4', ()),
    'reading-validation-causal': ('research.reading_validation.identity_causal_l4', ()),
    'reading-validation-transfer': ('research.reading_validation.identity_followup_l4', ()),
}


def digest(path):
    with path.open('rb') as source:
        return file_digest(source, 'sha256').hexdigest()


def write_json(path, value):
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    pending.replace(path)


def unpack(path, destination, expected):
    with tarfile.open(path, 'r:gz') as archive:
        files = archive.getmembers()
        if (len(files) != len(expected) or {item.name for item in files} != expected.keys()
                or any(not item.isfile() for item in files)):
            raise ValueError('The model bundle contains unexpected paths or non-files')
        for item in files:
            folder = destination / item.name
            folder.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(item) as source, folder.open('xb') as target:
                copyfileobj(source, target)
            if digest(folder) != expected[item.name]:
                raise ValueError(f'Model bundle checksum mismatch: {item.name}')


def archive_results(root, kind):
    artifact = root / 'results.tar.gz'
    files = [path for path in (root / 'extraction').rglob('*') if path.is_file()]
    names = {item.relative_to(root).as_posix() for item in files}
    required = {'seal.json', 'causality.json'}
    if kind == 'expository-lora':
        required.update(('predictions.jsonl', 'convergence.json', 'checkpoints.npz'))
    elif kind == 'union-training':
        required.update(('heads.npz', 'predictions.npz', 'fitting.json', 'extraction.json', 'development-effects.json'))
    elif kind == 'union-observations':
        required.update(('heads.npz', 'predictions.npz', 'fitting.json', 'extraction.json'))
    elif kind == 'token-readout':
        settings = json.loads((root / 'settings.json').read_text(encoding='utf-8'))
        if not settings['smoke']:
            required.update(('frozen.pt', 'forecast-position.json', 'forecast-frozen.json', 'fitting.json'))
    elif kind in ('token-readout-ui', 'portable-token-ui'):
        required.update(('verification.json', 'preview.html'))
    elif kind == 'portable-token-readout':
        required.update(('point.pt', 'span.pt', 'span-calibrated.pt', 'fitting.json',
                         'forecast-position.json', 'forecast-point.json', 'forecast-span.json',
                         'forecast-span-calibrated.json', 'point-history.json', 'span-history.json'))
    elif kind == 'adaptive-joint':
        settings = json.loads((root / 'settings.json').read_text(encoding='utf-8'))
        for seed in settings['seeds']:
            for mode in settings['modes']:
                required.update(f'{mode}-{seed}/{name}' for name in ('curve.pt', 'adapter_weight.npy', 'heads.npz',
                                                                     'fitting.json', 'predictions.npz',
                                                                     'token-forecast.json'))
    elif kind not in ('observation-formal-stops', 'reading-validation') and not (
        kind in EXTRACTIONS and kind.startswith('reading-validation-')
    ):
        required.update(('states.npy', 'coordinates.jsonl'))
    if not {f'extraction/{name}' for name in required} <= names:
        raise ValueError('Incomplete extraction output cannot be collected')
    with tarfile.open(artifact, 'x:gz', compresslevel=2) as output:
        for path in sorted(files):
            output.add(path, arcname=path.relative_to(root), recursive=False)
    pieces = []
    with artifact.open('rb') as stream:
        for i, data in enumerate(iter(lambda: stream.read(CHUNK_BYTES), b'')):
            name = f'part-{i:03}.bin'
            path = root / name
            with path.open('xb') as output:
                output.write(data)
            pieces.append({'name': name, 'bytes': len(data), 'sha256': digest(path)})
    return {'parts': pieces, 'archive_sha256': digest(artifact), 'archive_bytes': artifact.stat().st_size,
            'members_sha256': {path.relative_to(root).as_posix(): digest(path) for path in sorted(files)}}


def run(root, seconds, payload_sha256, worker_sha256):
    if digest(Path(__file__)) != worker_sha256 or digest(root.with_suffix('.tar.gz')) != payload_sha256:
        raise ValueError('Received a different worker or model bundle')
    plan = json.loads(root.with_suffix('.plan.json').read_text(encoding='utf-8'))
    if plan['run_id'] != root.name or plan['payload_sha256'] != payload_sha256 or plan['worker_sha256'] != worker_sha256:
        raise ValueError('Plan does not identify this owned job and its exact input')
    deadline = monotonic() + seconds
    root.mkdir(mode=0o700)
    status = {'run_id': root.name, 'candidate': plan['candidate'], 'payload_sha256': payload_sha256,
              'protocol_sha256': plan['protocol_sha256'], 'finished': False}

    def update(phase):
        status['phase'] = phase
        write_json(root / 'status.json', status)

    def invoke(*cmd):
        remaining = deadline - monotonic()
        if remaining < 10:
            raise TimeoutError('Owned Colab worker exhausted its deadline')
        with (root / 'worker.log').open('ab') as log:
            subprocess.run([sys.executable, '-u', *map(str, cmd)], cwd=root, env=env, check=True,
                           stdout=log, stderr=subprocess.STDOUT, timeout=remaining)

    env = {**os.environ, 'HF_HUB_DISABLE_TELEMETRY': '1', 'HF_HUB_DISABLE_XET': '1',
           'HF_HUB_DOWNLOAD_TIMEOUT': '120', 'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '2'}
    try:
        unpack(root.with_suffix('.tar.gz'), root, plan['payload_members_sha256'])
        registry = json.loads((root / 'research/backbone_transfer/models.json').read_text(encoding='utf-8'))
        if plan['extraction_kind'] in ('reading-validation-t5','reading-validation-t5-adapter','reading-validation-t5-decision'):
            if plan['candidate']!='t5' or plan['model']!='google-t5/t5-large':
                raise ValueError('Unexpected noncausal external comparison')
            version=registry['transformers']
        elif plan['extraction_kind'] == 'reading-validation-summary-specialist':
            if plan['candidate'] != 'seahorse' or plan['model'] != 'google/seahorse-large-q1':
                raise ValueError('Unexpected SEAHORSE comparator')
            version = registry['transformers']
        else:
            version = registry['models'][plan['candidate']].get('transformers', registry['transformers'])
        update('dependencies')
        dependencies = [f'transformers=={version}', 'accelerate==1.13.0']
        if plan['extraction_kind'] in ('token-readout-ui', 'portable-token-ui'):
            dependencies.append('gradio==6.17.3')
        if plan['candidate'] in ('t5', 'seahorse'):
            dependencies.append('sentencepiece==0.2.1')
        invoke('-m', 'pip', 'install', '--quiet', *dependencies)
        update('model-download')
        invoke('-c', '''import json
from pathlib import Path
import sys
from huggingface_hub import snapshot_download
if sys.argv[1]=='t5':
    from research.reading_validation.transition_t5_l4 import REPO, REVISION, SMALL
    spec={'repo':REPO,'revision':REVISION,'files':[ *SMALL,'model.safetensors']}
elif sys.argv[1]=='seahorse':
    from research.reading_validation.transition_seahorse_specialist_l4 import MODEL, REV
    spec={'repo':MODEL, 'revision':REV, 'files':['config.json','pytorch_model.bin','generation_config.json','spiece.model','special_tokens_map.json','tokenizer.json','tokenizer_config.json']}
else:
    registry = json.loads(Path('research/backbone_transfer/models.json').read_text(encoding='utf-8'))
    spec = registry['models'][sys.argv[1]]
location = snapshot_download(repo_id=spec['repo'], revision=spec['revision'], allow_patterns=spec['files'],
                             max_workers=6, token=False)
Path('snapshot-path.txt').write_text(location, encoding='utf-8')
''', plan['candidate'])
        snapshot = (root / 'snapshot-path.txt').read_text(encoding='utf-8')
        update('extraction')
        module, extra = EXTRACTIONS[plan['extraction_kind']]
        stimuli = 'token/stimuli.jsonl' if plan['extraction_kind'] == 'adaptive-joint' else 'stimuli.jsonl'
        invoke('-m', module, '--candidate', plan['candidate'], '--model', snapshot,
               '--stimuli', stimuli, '--output', 'extraction',
               '--protocol-sha256', plan['protocol_sha256'], *extra)
        update('archive')
        status['result'] = archive_results(root, plan['extraction_kind'])
        status['finished'] = True
        update('finished')
    except BaseException as error:
        with (root / 'worker.log').open('a', encoding='utf-8') as log:
            traceback.print_exc(file=log)
        status['error'] = type(error).__name__
        status['finished'] = True
        update('failed')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--seconds', required=True, type=int)
    parser.add_argument('--payload-sha256', required=True)
    parser.add_argument('--worker-sha256', required=True)
    run(**vars(parser.parse_args()))
