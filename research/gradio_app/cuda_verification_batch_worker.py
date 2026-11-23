"""Run the serving and external validation checks sequentially on one Colab L4."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
from time import monotonic
import traceback


RUN_ID = re.compile(r'cogllm-cuda-checks-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}')
V4_STAGES = ('setup', 'gradio', 'iirc-later', 'cwi-transfer', 'plaba-span')
STAGES = (*V4_STAGES, 'textcomplexityde')
TEXTDE_STAGES = ('setup', 'textcomplexityde')
GRADIO_STAGES = ('setup', 'gradio')
FORECAST_STAGES = (*GRADIO_STAGES, 'forecast')
STUDENT_STAGES = ('setup', 'student-annotations')
AMBISTORY_STAGES = ('setup', 'ambistory')
READING_EXTRACTION_STAGES = (*STUDENT_STAGES, 'ambistory')
MEDREADME_STAGES = ('setup', 'medreadme')
ARTICLE_ROUTE_STAGES = ('setup', 'article-route')
ONESTOP_STAGES = ('setup', 'onestop')
EXPMRC_STAGES = ('setup', 'expmrc')
NQ_EVIDENCE_STAGES = ('setup', 'nq-evidence')
QASPER_TAIL_STAGES = ('setup', 'qasper-tail')
GUM_BRIDGE_STAGES = ('setup', 'gum-bridge')
BEUKERS_STAGES = ('setup', 'beukers')
PROJECTION_STAGES = ('setup', 'projection')
FRICK_SIGNALS_STAGES = ('setup', 'frick-signals')
CLEAR_EASE_STAGES = ('setup', 'clear-ease')
FIRST_READBACK_STAGES = ('setup', 'first-readback')
CONTEXT_READBACK_STAGES = ('setup', 'context-readback')
V4_BUNDLE = frozenset({'bundle.json', 'gradio.tar.gz', 'later-model-inputs.tar.gz', 'later-source.tar.gz',
                       'cwi-model-inputs.tar.gz', 'cwi-source.tar.gz', 'plaba-model-inputs.tar.gz', 'plaba-source.tar.gz'})
BUNDLE = V4_BUNDLE | {'textde-model-inputs.tar.gz', 'textde-source.tar.gz'}
TEXTDE_BUNDLE = frozenset({'bundle.json', 'textde-model-inputs.tar.gz', 'textde-source.tar.gz'})
GRADIO_BUNDLE = frozenset({'bundle.json', 'gradio.tar.gz'})
FORECAST_INPUT_ARCHIVE = 'forecast-inputs.tar.gz'
FORECAST_BUNDLE = GRADIO_BUNDLE | {FORECAST_INPUT_ARCHIVE}
STUDENT_DATA_ARCHIVE = 'student-model-inputs.tar.gz'
STUDENT_SOURCE_ARCHIVE = 'student-source.tar.gz'
STUDENT_BUNDLE = frozenset({'bundle.json', STUDENT_DATA_ARCHIVE, STUDENT_SOURCE_ARCHIVE})
AMBISTORY_DATA_ARCHIVE = 'ambistory-model-inputs.tar.gz'
AMBISTORY_SOURCE_ARCHIVE = 'ambistory-source.tar.gz'
AMBISTORY_BUNDLE = frozenset({'bundle.json', AMBISTORY_DATA_ARCHIVE, AMBISTORY_SOURCE_ARCHIVE})
READING_EXTRACTION_BUNDLE = STUDENT_BUNDLE | AMBISTORY_BUNDLE
MEDREADME_ARCHIVE = 'medreadme-inputs.tar.gz'
MEDREADME_BUNDLE = frozenset({'bundle.json', MEDREADME_ARCHIVE})
MEDREADME_STIMULI = 'holdout/stimuli.jsonl'
ARTICLE_ROUTE_ARCHIVE = 'article-route-inputs.tar.gz'
ARTICLE_ROUTE_BUNDLE = frozenset({'bundle.json', ARTICLE_ROUTE_ARCHIVE})
OVERVIEW_HTTP_ROUTE = 'overview_http_route'
WORD_CUE_ROUTE = 'word_cue_route'
WORD_CUE_HTML = 'word-cues.html'
WORD_CUE_PREVIEW = 'cue-preview.html'
ONESTOP_ARCHIVE = 'onestop-inputs.tar.gz'
ONESTOP_BUNDLE = frozenset({'bundle.json', ONESTOP_ARCHIVE})
ONESTOP_STIMULI = 'holdout/stimuli.jsonl'
EXPMRC_ARCHIVE = 'expmrc-inputs.tar.gz'
EXPMRC_BUNDLE = frozenset({'bundle.json', EXPMRC_ARCHIVE})
EXPMRC_HEADS = 'holdout/heads.npz'
EXPMRC_ARTICLES = 'holdout/articles.jsonl'
EXPMRC_QUERIES = 'holdout/queries.jsonl'
CONTRACTNLI_PACKET = 'holdout/contractnli'
CONTRACTNLI_MODULES = {
    'contractnli': 'research.semantic_tracking.contractnli_forecast',
    'contractnli_rerank': 'research.semantic_tracking.contractnli_rerank_forecast',
    'evidence_inference_transfer': 'research.semantic_tracking.evidence_inference_transfer_forecast',
}
NQ_EVIDENCE_ARCHIVE = 'nq-evidence-inputs.tar.gz'
NQ_EVIDENCE_BUNDLE = frozenset({'bundle.json', NQ_EVIDENCE_ARCHIVE})
QASPER_TAIL_ARCHIVE = 'qasper-tail-inputs.tar.gz'
QASPER_TAIL_BUNDLE = frozenset({'bundle.json', QASPER_TAIL_ARCHIVE})
GUM_BRIDGE_ARCHIVE = 'gum-bridge-inputs.tar.gz'
GUM_BRIDGE_BUNDLE = frozenset({'bundle.json', GUM_BRIDGE_ARCHIVE})
BEUKERS_ARCHIVE = 'beukers-inputs.tar.gz'
BEUKERS_BUNDLE = frozenset({'bundle.json', BEUKERS_ARCHIVE})
PROJECTION_ARCHIVE = 'projection-inputs.tar.gz'
PROJECTION_BUNDLE = frozenset({'bundle.json', PROJECTION_ARCHIVE})
PROJECTION_STIMULI = 'model_input/stimuli.jsonl'
FRICK_SIGNALS_ARCHIVE = 'frick-signals-inputs.tar.gz'
FRICK_SIGNALS_BUNDLE = frozenset({'bundle.json', FRICK_SIGNALS_ARCHIVE})
CLEAR_EASE_ARCHIVE = 'clear-ease-inputs.tar.gz'
CLEAR_EASE_BUNDLE = frozenset({'bundle.json', CLEAR_EASE_ARCHIVE})
FIRST_READBACK_ARCHIVE = 'first-readback-inputs.tar.gz'
FIRST_READBACK_BUNDLE = frozenset({'bundle.json', FIRST_READBACK_ARCHIVE})
FIRST_READBACK_INPUTS = {name: f'holdout/{name}.jsonl' for name in ('stimuli', 'targets', 'train')}
CONTEXT_READBACK_ARCHIVE = 'context-readback-inputs.tar.gz'
CONTEXT_READBACK_BUNDLE = frozenset({'bundle.json', CONTEXT_READBACK_ARCHIVE})
CONTEXT_READBACK_STIMULI = 'holdout/stimuli.jsonl'
CONTEXT_READBACK_HEADS = 'holdout/heads.npz'
GRADIO_CHECKPOINT = 'research/results/gradio-candidate-checkpoint'
FORECAST_SOURCE = 'research/gradio_app/forecast.py'
FORECAST_STIMULI = {dataset: f'holdout/{dataset}/stimuli.jsonl' for dataset in ('blott', 'kerwer')}
FORECAST_MEMBERS = frozenset({FORECAST_SOURCE, *FORECAST_STIMULI.values()})
LATER_DATA = frozenset({'assay.json', 'train/stimuli.jsonl', 'train/queries.jsonl', 'train/intervals.jsonl',
                        'dev/stimuli.jsonl', 'dev/queries.jsonl'})
CWI_DATA = frozenset({'assay.json', 'stimuli.jsonl', 'cwi-train.jsonl', 'cwi-eval.jsonl'})
PLABA_DATA = frozenset({'assay.json', 'stimuli.jsonl', 'sentences.jsonl', 'heads.npz'})
TEXTDE_DATA = frozenset({'LICENSE', 'assay.json', 'attribution.json', 'heads.npz', 'stimuli.jsonl'})
STUDENT_DATA = frozenset({'stimuli.jsonl', 'certified-stops.jsonl', 'extraction-plan.json'})
STUDENT_ARTIFACTS = {'features_sha256': 'features.npz', 'extraction_sha256': 'extraction.json',
                     'causality_sha256': 'causality.json', 'telemetry_sha256': 'telemetry.jsonl'}
AMBISTORY_ARTIFACTS = {'story_states_sha256': 'story-states.npz', 'sense_states_sha256': 'sense-states.npz',
                       'extraction_sha256': 'extraction.json', 'causality_sha256': 'causality.json',
                       'telemetry_sha256': 'telemetry.jsonl'}
MAX_BYTES = 16 * 1024 * 1024
MAX_SECONDS = 30 * 60
TEXTDE_SECONDS = 14 * 60
GRADIO_SECONDS = 14 * 60
FORECAST_SECONDS = 18 * 60
STUDENT_SECONDS = 12 * 60
AMBISTORY_SECONDS = 14 * 60
READING_EXTRACTION_SECONDS = 18 * 60
MEDREADME_SECONDS = 14 * 60
ARTICLE_ROUTE_SECONDS = 12 * 60
ARTICLE_OVERVIEW_HTML = 'reading-overview.html'
ONESTOP_SECONDS = 12 * 60
EXPMRC_SECONDS = 12 * 60
NQ_EVIDENCE_SECONDS = 14 * 60
QASPER_TAIL_SECONDS = 14 * 60
GUM_BRIDGE_SECONDS = 18 * 60
GUM_BRIDGE_MAX_BYTES = 80 * 1024 * 1024
BEUKERS_SECONDS = 18 * 60
BEUKERS_MAX_BYTES = 80 * 1024 * 1024
PROJECTION_SECONDS = 18 * 60
PROJECTION_MAX_BYTES = 80 * 1024 * 1024
FRICK_SIGNALS_SECONDS = 12 * 60
CLEAR_EASE_SECONDS = 12 * 60
FIRST_READBACK_SECONDS = 12 * 60
CONTEXT_READBACK_SECONDS = 12 * 60
OUTPUTS = {
    'setup': {'runtime.json', 'run.log'},
    'gradio': {'verification.json', 'replay-report.json', 'serving-seal.json', 'predictions.jsonl', 'run.log'},
    'forecast': {'run.log'} | {f'{dataset}/{name}' for dataset in FORECAST_STIMULI
                               for name in ('predictions.jsonl', 'seal.json')},
    'medreadme': {'run.log', 'predictions.jsonl', 'seal.json'},
    'article-route': {'run.log', 'verification.json'},
    'onestop': {'run.log', 'features.npz', 'seal.json', 'telemetry.jsonl'},
    'expmrc': {'run.log', 'predictions.jsonl', 'seal.json', 'extraction.json', 'causality.json', 'telemetry.jsonl'},
    'nq-evidence': {'run.log', 'predictions.jsonl', 'seal.json', 'extraction.json', 'causality.json', 'telemetry.jsonl'},
    'qasper-tail': {'run.log', 'predictions.jsonl', 'seal.json', 'extraction.json', 'causality.json', 'telemetry.jsonl'},
    'gum-bridge': {'run.log', 'seal.json', 'coordinates.jsonl', 'vectors.npy', 'causality.json', 'telemetry.jsonl'},
    'beukers': {'run.log', 'seal.json', 'coordinates.jsonl', 'vectors.npy', 'causality.json', 'telemetry.jsonl'},
    'projection': {'run.log', 'seal.json', 'coordinates.jsonl', 'vectors.npy', 'terminal_states.npy',
                   'causality.json', 'telemetry.jsonl'},
    'frick-signals': {'run.log', 'seal.json', 'coordinates.jsonl', 'token_features.npy', 'states.npy',
                      'causality.json', 'telemetry.jsonl'},
    'clear-ease': {'run.log', 'seal.json', 'forecasts.jsonl', 'causality.json', 'ui-verification.json',
                   ARTICLE_OVERVIEW_HTML},
    'first-readback': {'run.log', 'predictions.jsonl', 'heads.npz', 'seal.json', 'extraction.json',
                       'causality.json', 'telemetry.jsonl'},
    'context-readback': {'run.log', 'predictions.jsonl', 'seal.json', 'extraction.json', 'causality.json',
                         'telemetry.jsonl'},
    'student-annotations': {'run.log', 'extraction/seal.json'} | {f'extraction/{name}' for name in STUDENT_ARTIFACTS.values()},
    'ambistory': {'run.log', 'extraction/seal.json'} | {f'extraction/{name}' for name in AMBISTORY_ARTIFACTS.values()},
    'iirc-later': {'features/plan.json', 'features/extraction.json', 'features/causality.json', 'features/progress.json',
                   'fit/fit-manifest.json', 'fit/heads.npz', 'fit/predictions.jsonl', 'run.log'},
    'cwi-transfer': {'extraction/extraction.json', 'extraction/causality.json', 'extraction/telemetry.jsonl',
                     'fit/fit-manifest.json', 'fit/heads.npz', 'fit/lexicon.json', 'fit/cwi-predictions.jsonl', 'run.log'},
    'plaba-span': {'predictions.jsonl', 'forecast-manifest.json', 'extraction/extraction.json',
                   'extraction/causality.json', 'extraction/telemetry.jsonl', 'run.log'},
    'textcomplexityde': {'forecast/predictions.jsonl', 'forecast/seal.json', 'extraction/extraction.json',
                         'extraction/causality.json', 'extraction/telemetry.jsonl', 'run.log'},
}
TELEMETRY = re.compile(r'features/telemetry-shard-[0-9]{4}\.jsonl')


def digest(path):
    with path.open('rb') as contents:
        return hashlib.file_digest(contents, 'sha256').hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, data):
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def unpack(archive_path, expected, directory=None):
    with tarfile.open(archive_path, 'r:gz') as archive:
        members = archive.getmembers()
        names = {m.name for m in members}
        if names != set(expected) or len(names) != len(members) or not all(m.isfile() for m in members):
            raise ValueError(f'Unexpected members in {archive_path.name}')
        hashes = {}
        for member in members:
            with archive.extractfile(member) as contents:
                hashes[member.name] = hashlib.file_digest(contents, 'sha256').hexdigest()
        if hasattr(expected, 'items') and hashes != expected:
            raise ValueError(f'Member checksum mismatch in {archive_path.name}')
        if directory is not None:
            archive.extractall(directory, filter='data')
    return hashes


def prepare(root):
    manifest = read_json(root / 'bundle.json')
    for filename, receipt in manifest['archives'].items():
        path = root / filename
        if digest(path) != receipt['sha256']:
            raise ValueError(f'Nested archive checksum mismatch: {filename}')
        unpack(path, receipt['members'], root / receipt['directory'])
    return manifest


SNAPSHOT = '''
import json
from pathlib import Path
import torch
from importlib.metadata import version
from huggingface_hub import snapshot_download

if torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != 'NVIDIA L4':
    raise RuntimeError('Expected exactly one NVIDIA L4')
if not torch.cuda.is_bf16_supported(including_emulation=False):
    raise RuntimeError('Expected native BF16 support on the L4')
plan = json.loads(Path('bundle.json').read_text(encoding='utf-8'))
files = plan['model_files'] if 'model_files' in plan else json.loads(
    Path('gradio/verification-payload.json').read_text(encoding='utf-8'))['model_files']
snapshot = snapshot_download(plan['model'], revision=plan['revision'], allow_patterns=files)
packages = {requirement.split('==')[0] for requirement in plan['requirements']} | {'scipy'}
if 'gradio' in packages:
    packages.add('gradio-client')
record = dict(device=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda,
              vram_bytes=torch.cuda.get_device_properties(0).total_memory, snapshot=snapshot,
              packages={pkg: version(pkg) for pkg in sorted(packages)})
Path('runtime.json').write_text(json.dumps(record, indent=2) + '\\n', encoding='utf-8')
'''

FORECAST = '''
import json
from pathlib import Path
import sys

from research.gradio_app.forecast import export_forecasts, read_stimuli

plan = json.loads(Path('../bundle.json').read_text(encoding='utf-8'))['forecast']
stimuli = {name: read_stimuli(Path(spec['path']), spec['input_sha256']) for name, spec in plan.items()}

from research.gradio_app.combined import CombinedReader

reader = CombinedReader(Path(sys.argv[1]), checkpoint=Path(sys.argv[2]), cuda=True)
for name, documents in stimuli.items():
    seal = export_forecasts(reader, documents, Path('forecast') / name)
    print(json.dumps({'dataset': name, **seal}, sort_keys=True), flush=True)
'''


def invoke(directory, command, env, log, deadline, seconds):
    timeout = min(seconds, deadline - monotonic())
    if timeout <= 0:
        raise TimeoutError('Batch worker deadline reached')
    subprocess.run([sys.executable, *command], cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT,
                   timeout=timeout, check=True)


def setup(root, env, log, deadline, manifest):
    invoke(root, ('-m', 'pip', 'install', '--quiet', *manifest['requirements']), env, log, deadline, 140)
    invoke(root, ('-c', SNAPSHOT), env, log, deadline, 270)


def gradio(root, env, log, deadline, snapshot, *, candidate=False):
    directory = root / 'gradio'
    env = {**env, 'NLTK_DATA': str(directory / 'nltk_data'), 'GRADIO_ANALYTICS_ENABLED': 'False'}
    checkpoint = ('--checkpoint', GRADIO_CHECKPOINT) if candidate else ()
    invoke(directory, ('-m', 'research.gradio_app.verify_combined', '--cuda', '--check-ui', '--model', snapshot,
                       '--iirc-stimuli', 'fixtures/iirc-stimuli.jsonl', '--plaba-features', 'fixtures/plaba-features.npz',
                       '--plaba-stimuli', 'fixtures/plaba-stimuli.jsonl', '--iirc-dev', 'fixtures/iirc-dev',
                       '--iirc-export-only', '--output', 'outputs', *checkpoint), env, log, deadline, 325)


def forecast(root, env, log, deadline, snapshot):
    invoke(root / 'gradio', ('-c', FORECAST, snapshot, GRADIO_CHECKPOINT), env, log, deadline, 360)


def medreadme(root, env, log, deadline, snapshot, manifest):
    invoke(root / 'medreadme', ('-m', 'research.stage6.medreadme_lexical_forecast', '--model', snapshot,
                                '--checkpoint', GRADIO_CHECKPOINT, '--stimuli', MEDREADME_STIMULI,
                                '--stimuli-sha256', manifest['medreadme']['stimuli_sha256'], '--output', 'outputs'),
           env, log, deadline, 400)


def article_route(root, env, log, deadline, snapshot, route):
    env = {**env, 'GRADIO_ANALYTICS_ENABLED': 'False'}
    directory = root / 'article-route'
    if route == WORD_CUE_ROUTE:
        (directory / 'outputs').mkdir(parents=True, exist_ok=True)
    module, mode = {
        None: ('verify_article_route', ('--cuda',)),
        OVERVIEW_HTTP_ROUTE: ('verify_overview_http', ('--cuda',)),
        WORD_CUE_ROUTE: ('verify_word_cues', ('http', '--port', '7861', '--save-markup', f'outputs/{WORD_CUE_HTML}',
                                              '--markup-output', f'outputs/{WORD_CUE_PREVIEW}')),
    }[route]
    invoke(directory, ('-m', f'research.gradio_app.{module}', *mode, '--model', snapshot,
                       '--checkpoint', GRADIO_CHECKPOINT, '--output', 'outputs/verification.json'),
           env, log, deadline, 180)


def onestop(root, env, log, deadline, snapshot, manifest):
    invoke(root / 'onestop', ('-m', 'research.stage6.onestop_rereading_extract', '--model', snapshot,
                             '--checkpoint', GRADIO_CHECKPOINT, '--stimuli', ONESTOP_STIMULI,
                             '--stimuli-sha256', manifest['onestop']['stimuli_sha256'], '--output', 'outputs'),
           env, log, deadline, 240)


def evidence_forecast(root, env, log, deadline, snapshot, plan, stage):
    if stage == 'expmrc' and plan.get('experiment') in CONTRACTNLI_MODULES:
        invoke(root / stage, (
            '-m', CONTRACTNLI_MODULES[plan['experiment']], 'forecast', '--packet', CONTRACTNLI_PACKET,
            '--model', snapshot, '--heads', EXPMRC_HEADS, '--assay-sha256', plan['assay_sha256'],
            '--protocol-sha256', plan['protocol_sha256'], '--output', 'outputs',
        ), env, log, deadline, 420)
        return
    invoke(root / stage, ('-m', 'research.semantic_tracking.expmrc_forecast', '--model', snapshot,
                          '--checkpoint', GRADIO_CHECKPOINT, '--heads', EXPMRC_HEADS, '--articles', EXPMRC_ARTICLES,
                          '--queries', EXPMRC_QUERIES, '--articles-sha256', plan['articles_sha256'],
                          '--queries-sha256', plan['queries_sha256'], '--assay-sha256', plan['assay_sha256'],
                          '--protocol-sha256', plan['protocol_sha256'], '--output', 'outputs'),
           env, log, deadline, 250)


def first_readback(root, env, log, deadline, snapshot, manifest):
    plan = manifest['first_readback']
    inputs = tuple(value for name, path in FIRST_READBACK_INPUTS.items()
                   for value in (f'--{name}', path, f'--{name}-sha256', plan[f'{name}_sha256']))
    invoke(root / 'first-readback', ('-m', 'research.semantic_tracking.first_readback_transfer',
                                    '--model', snapshot, '--checkpoint', GRADIO_CHECKPOINT, *inputs,
                                    '--assay-sha256', plan['assay_sha256'],
                                    '--protocol-sha256', plan['protocol_sha256'], '--output', 'outputs'),
           env, log, deadline, 250)


def context_readback(root, env, log, deadline, snapshot, manifest):
    plan = manifest['context_readback']
    hashes = tuple(value for name in ('stimuli', 'targets', 'assay', 'protocol')
                   for value in (f'--{name}-sha256', plan[f'{name}_sha256']))
    invoke(root / 'context-readback', ('-m', 'research.semantic_tracking.context_readback_forecast',
                                      '--model', snapshot, '--checkpoint', GRADIO_CHECKPOINT,
                                      '--stimuli', CONTEXT_READBACK_STIMULI, '--heads', CONTEXT_READBACK_HEADS,
                                      *hashes, '--output', 'outputs'), env, log, deadline, 250)


def later(root, env, log, deadline):
    directory = root / 'iirc-later'
    invoke(directory, ('-m', 'research.semantic_tracking.iirc_linked_extract', '--inputs', 'inputs',
                       '--output', 'features'), env, log, deadline, 180)
    invoke(directory, ('-m', 'research.semantic_tracking.iirc_seed_later_assay', 'fit', '--inputs', 'inputs',
                       '--features', 'features/features.npz', '--extraction', 'features/extraction.json',
                       '--output', 'fit'), env, log, deadline, 85)


def cwi(root, env, log, deadline):
    directory = root / 'cwi-transfer'
    invoke(directory, ('-m', 'research.core_features.extract', '--stimuli', 'inputs/stimuli.jsonl',
                       '--output', 'extraction'), env, log, deadline, 130)
    invoke(directory, ('-m', 'research.core_features.background_transfer_fit', '--inputs', 'inputs',
                       '--features', 'extraction', '--output', 'fit'), env, log, deadline, 120)


def plaba(root, env, log, deadline):
    invoke(root / 'plaba-span', ('-m', 'research.semantic_tracking.plaba_span_transfer', 'forecast',
                                 '--inputs', 'inputs', '--output', 'outputs'), env, log, deadline, 105)


def textde(root, env, log, deadline):
    directory = root / 'textcomplexityde'
    invoke(directory, ('-m', 'research.core_features.extract', '--stimuli', 'inputs/stimuli.jsonl',
                       '--output', 'extraction', '--certified-token-controls'), env, log, deadline, 125)
    invoke(directory, ('-m', 'research.stage6.textcomplexityde_eval', 'forecast', '--inputs', 'inputs',
                       '--features', 'extraction', '--output', 'forecast'), env, log, deadline, 50)


def student_annotations(root, env, log, deadline, snapshot, manifest):
    directory = root / 'student-annotations'
    student = manifest['student_annotations']
    tokenizer_sha256 = digest(Path(snapshot) / 'tokenizer.json')
    if tokenizer_sha256 != student['tokenizer_sha256']:
        raise ValueError('StudentAnnotations requires the tokenizer certified against its sentence stops')
    invoke(directory, ('-m', 'research.core_features.extract', '--stimuli', 'inputs/stimuli.jsonl',
                       '--output', 'extraction', '--certified-token-controls'), env, log, deadline, 175)
    extraction = directory / 'extraction'
    write_json(extraction / 'seal.json', {
        'model': manifest['model'], 'revision': manifest['revision'], 'tokenizer_sha256': tokenizer_sha256,
        'preparation_sha256': student['preparation_sha256'],
        'inputs_sha256': manifest['archives'][STUDENT_DATA_ARCHIVE]['members'],
        'source_sha256': manifest['archives'][STUDENT_SOURCE_ARCHIVE]['members'],
        **{field: digest(extraction / name) for field, name in STUDENT_ARTIFACTS.items()},
    })


def ambistory(root, env, log, deadline, snapshot, manifest):
    directory = root / 'ambistory'
    spec = manifest['ambistory']
    if any(digest(Path(snapshot) / name) != expected for name, expected in spec['tokenizer_files_sha256'].items()):
        raise ValueError('AmbiStory requires the tokenizer certified against all source stops')
    invoke(directory, ('-m', 'research.stage6.ambistory_extract', '--inputs', 'inputs', '--model', snapshot,
                       '--output', 'extraction'), env, log, deadline, 260)
    extraction = directory / 'extraction'
    summary = read_json(extraction / 'extraction.json')
    if summary['projection_sha256'] != spec['projection_sha256']:
        raise ValueError('AmbiStory extraction used a different projection matrix')
    write_json(extraction / 'seal.json', {
        'model': manifest['model'], 'revision': manifest['revision'], 'projection_sha256': summary['projection_sha256'],
        'preparation_sha256': spec['preparation_sha256'],
        'inputs_sha256': manifest['archives'][AMBISTORY_DATA_ARCHIVE]['members'],
        'source_sha256': manifest['archives'][AMBISTORY_SOURCE_ARCHIVE]['members'],
        **{field: digest(extraction / name) for field, name in AMBISTORY_ARTIFACTS.items()},
    })


def output_members(stage, *, raw_stops=False, combined_article_route=False, word_cue_route=False):
    names = OUTPUTS[stage]
    if raw_stops:
        names = names | {'states.npy'}
    if stage == 'article-route' and combined_article_route:
        names = names | {ARTICLE_OVERVIEW_HTML}
    if stage == 'article-route' and word_cue_route:
        names = names | {WORD_CUE_HTML, WORD_CUE_PREVIEW}
    return names


def seal_stage(root, stage, status, callback, *, raw_stops=False, combined_article_route=False, word_cue_route=False):
    started = monotonic()
    output = root / 'return' / stage
    output.mkdir()
    receipt = {'state': 'running'}
    status['stages'][stage] = receipt
    status['phase'] = stage
    write_json(root / 'status.json', status)
    with (output / 'run.log').open('w', encoding='utf-8') as log:
        try:
            callback(log)
            receipt['state'] = 'complete'
        except Exception as error:
            receipt.update(state='failed', error_type=type(error).__name__)
            traceback.print_exc(file=log)

    if stage in ('gradio', 'plaba-span', 'medreadme', 'article-route', 'onestop', 'expmrc', 'nq-evidence', 'qasper-tail', 'first-readback',
                 'context-readback', 'gum-bridge', 'beukers', 'projection', 'frick-signals', 'clear-ease'):
        directory = root / stage / 'outputs'
    elif stage == 'forecast':
        directory = root / 'gradio' / 'forecast'
    elif stage in ('iirc-later', 'cwi-transfer', 'textcomplexityde', 'student-annotations', 'ambistory'):
        directory = root / stage
    else:
        directory = root
    required = output_members(stage, raw_stops=raw_stops, combined_article_route=combined_article_route,
                              word_cue_route=word_cue_route)
    members = {name: directory / name for name in required - {'run.log'} if (directory / name).is_file()}
    if stage == 'iirc-later':
        members.update({str(path.relative_to(directory)): path for path in directory.glob('features/telemetry-shard-*.jsonl')})
    members['run.log'] = output / 'run.log'
    if not required <= members.keys() and receipt['state'] == 'complete':
        receipt.update(state='failed', error_type='MissingStageArtifacts')
    archive = output.with_suffix('.tar.gz')
    pending = archive.with_suffix('.pending')
    with tarfile.open(pending, 'w:gz', compresslevel=1) as package:
        for name, path in sorted(members.items()):
            package.add(path, arcname=name, recursive=False)
    if pending.stat().st_size > archive_budget(stage):
        raise ValueError(f'Stage results exceed the download budget: {stage}')
    pending.replace(archive)
    receipt.update(elapsed_s=monotonic() - started, archive_sha256=digest(archive), archive_bytes=archive.stat().st_size,
                   members={name: digest(path) for name, path in members.items()})
    write_json(root / 'status.json', status)


def archive_budget(stage):
    return {'gum-bridge': GUM_BRIDGE_MAX_BYTES, 'beukers': BEUKERS_MAX_BYTES,
            'projection': PROJECTION_MAX_BYTES}.get(stage, MAX_BYTES)


def run(root, payload_sha256, worker_sha256, seconds):
    os.umask(0o077)
    if not RUN_ID.fullmatch(root.name) or root.parent != Path('/content') or not 0 < seconds <= MAX_SECONDS:
        raise ValueError('Invalid owned verification batch configuration')
    payload = root.with_suffix('.tar.gz')
    if digest(payload) != payload_sha256 or digest(Path(__file__)) != worker_sha256:
        raise ValueError('Remote payload or worker differs from the local seal')
    with tarfile.open(payload, 'r:gz') as archive:
        names = frozenset(archive.getnames())
    scopes = {BUNDLE: (STAGES, MAX_SECONDS), TEXTDE_BUNDLE: (TEXTDE_STAGES, TEXTDE_SECONDS),
              GRADIO_BUNDLE: (GRADIO_STAGES, GRADIO_SECONDS), FORECAST_BUNDLE: (FORECAST_STAGES, FORECAST_SECONDS),
              STUDENT_BUNDLE: (STUDENT_STAGES, STUDENT_SECONDS), AMBISTORY_BUNDLE: (AMBISTORY_STAGES, AMBISTORY_SECONDS),
              READING_EXTRACTION_BUNDLE: (READING_EXTRACTION_STAGES, READING_EXTRACTION_SECONDS),
              MEDREADME_BUNDLE: (MEDREADME_STAGES, MEDREADME_SECONDS),
              ARTICLE_ROUTE_BUNDLE: (ARTICLE_ROUTE_STAGES, ARTICLE_ROUTE_SECONDS),
              ONESTOP_BUNDLE: (ONESTOP_STAGES, ONESTOP_SECONDS),
              EXPMRC_BUNDLE: (EXPMRC_STAGES, EXPMRC_SECONDS), NQ_EVIDENCE_BUNDLE: (NQ_EVIDENCE_STAGES, NQ_EVIDENCE_SECONDS),
              QASPER_TAIL_BUNDLE: (QASPER_TAIL_STAGES, QASPER_TAIL_SECONDS),
              GUM_BRIDGE_BUNDLE: (GUM_BRIDGE_STAGES, GUM_BRIDGE_SECONDS),
              BEUKERS_BUNDLE: (BEUKERS_STAGES, BEUKERS_SECONDS),
              PROJECTION_BUNDLE: (PROJECTION_STAGES, PROJECTION_SECONDS),
              FRICK_SIGNALS_BUNDLE: (FRICK_SIGNALS_STAGES, FRICK_SIGNALS_SECONDS),
              CLEAR_EASE_BUNDLE: (CLEAR_EASE_STAGES, CLEAR_EASE_SECONDS),
              FIRST_READBACK_BUNDLE: (FIRST_READBACK_STAGES, FIRST_READBACK_SECONDS),
              CONTEXT_READBACK_BUNDLE: (CONTEXT_READBACK_STAGES, CONTEXT_READBACK_SECONDS)}
    if names not in scopes or seconds > scopes[names][1]:
        raise ValueError('Unexpected verification bundle membership or scope duration')
    stages, _ = scopes[names]
    root.mkdir()
    (root / 'return').mkdir()
    status = {'run_id': root.name, 'payload_sha256': payload_sha256, 'worker_sha256': worker_sha256,
              'phase': 'preflight', 'finished': False, 'stages': {stage: {'state': 'pending'} for stage in stages}}
    write_json(root / 'status.json', status)
    deadline = monotonic() + seconds
    try:
        unpack(payload, names, root)
        manifest = prepare(root)
        raw_stops = manifest['projection'].get('raw_stops', False) if 'projection' in stages else False
        if type(raw_stops) is not bool:
            raise ValueError('Expected a boolean raw-stops export option')
        env = {**os.environ, 'HF_HUB_DISABLE_TELEMETRY': '1', 'HF_HUB_DISABLE_PROGRESS_BARS': '1',
               'HF_HUB_CACHE': str(root / 'hf-cache'), 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
               'MKL_NUM_THREADS': '2', 'PYTHONUNBUFFERED': '1'}
        env.pop('HF_HUB_OFFLINE', None)
        seal_stage(root, 'setup', status, lambda log: setup(root, env, log, deadline, manifest))
        if status['stages']['setup']['state'] == 'complete':
            runtime = read_json(root / 'runtime.json')
            env['HF_HUB_OFFLINE'] = '1'
            if 'gradio' in stages:
                seal_stage(root, 'gradio', status, lambda log: gradio(root, env, log, deadline, runtime['snapshot'],
                                                                    candidate=names in (GRADIO_BUNDLE, FORECAST_BUNDLE)))
            if 'forecast' in stages and status['stages']['gradio']['state'] == 'complete':
                seal_stage(root, 'forecast', status, lambda log: forecast(root, env, log, deadline, runtime['snapshot']))
            if stages == STAGES:
                seal_stage(root, 'iirc-later', status, lambda log: later(root, env, log, deadline))
                seal_stage(root, 'cwi-transfer', status, lambda log: cwi(root, env, log, deadline))
                seal_stage(root, 'plaba-span', status, lambda log: plaba(root, env, log, deadline))
            if 'textcomplexityde' in stages:
                seal_stage(root, 'textcomplexityde', status, lambda log: textde(root, env, log, deadline))
            if 'student-annotations' in stages:
                seal_stage(root, 'student-annotations', status,
                           lambda log: student_annotations(root, env, log, deadline, runtime['snapshot'], manifest))
            if 'ambistory' in stages:
                seal_stage(root, 'ambistory', status,
                           lambda log: ambistory(root, env, log, deadline, runtime['snapshot'], manifest))
            if 'medreadme' in stages:
                seal_stage(root, 'medreadme', status,
                           lambda log: medreadme(root, env, log, deadline, runtime['snapshot'], manifest))
            if 'article-route' in stages:
                route = next((name for name in (WORD_CUE_ROUTE, OVERVIEW_HTTP_ROUTE) if name in manifest), None)
                seal_stage(root, 'article-route', status,
                           lambda log: article_route(root, env, log, deadline, runtime['snapshot'], route),
                           combined_article_route=route == OVERVIEW_HTTP_ROUTE or 'combined_article_route' in manifest,
                           word_cue_route=route == WORD_CUE_ROUTE)
            if 'onestop' in stages:
                seal_stage(root, 'onestop', status,
                           lambda log: onestop(root, env, log, deadline, runtime['snapshot'], manifest))
            if 'expmrc' in stages:
                seal_stage(root, 'expmrc', status,
                           lambda log: evidence_forecast(root, env, log, deadline, runtime['snapshot'],
                                                         manifest['expmrc'], 'expmrc'))
            if 'nq-evidence' in stages:
                seal_stage(root, 'nq-evidence', status,
                           lambda log: evidence_forecast(root, env, log, deadline, runtime['snapshot'],
                                                         manifest['nq_evidence'], 'nq-evidence'))
            if 'qasper-tail' in stages:
                seal_stage(root, 'qasper-tail', status,
                           lambda log: evidence_forecast(root, env, log, deadline, runtime['snapshot'],
                                                         manifest['qasper_tail'], 'qasper-tail'))
            if 'gum-bridge' in stages:
                seal_stage(root, 'gum-bridge', status,
                           lambda log: invoke(root / 'gum-bridge', ('-m', 'research.semantic_tracking.gum_bridge_extract',
                                              'extract', '--inputs', 'model_input', '--model', runtime['snapshot'],
                                              '--checkpoint', GRADIO_CHECKPOINT, '--preflight', 'preflight.json',
                                              '--output', 'outputs'), env, log, deadline, 620))
            for stage in ('beukers', 'projection'):
                if stage in stages:
                    plan = manifest[stage]
                    stage_raw_stops = stage == 'projection' and raw_stops
                    feature_flags = ('--terminal',) if stage == 'projection' else ()
                    if stage_raw_stops:
                        feature_flags += ('--raw-stops',)
                    seal_stage(root, stage, status,
                               lambda log: invoke(root / stage, ('-m', 'research.semantic_tracking.beukers_extract',
                                                  'extract', '--stimuli', PROJECTION_STIMULI, '--model', runtime['snapshot'],
                                                  '--checkpoint', GRADIO_CHECKPOINT, '--preflight', 'preflight.json',
                                                  '--assay-sha256', plan['assay_sha256'], '--protocol-sha256', plan['protocol_sha256'],
                                                  '--output', 'outputs', *feature_flags), env, log, deadline, 620),
                               raw_stops=stage_raw_stops)
            if 'frick-signals' in stages:
                plan = manifest['frick_signals']
                seal_stage(root, 'frick-signals', status,
                           lambda log: invoke(root / 'frick-signals', ('-m', 'research.semantic_tracking.frick_signals_extract',
                                              '--stimuli', PROJECTION_STIMULI, '--model', runtime['snapshot'],
                                              '--checkpoint', GRADIO_CHECKPOINT, '--preflight', 'preflight.json',
                                              '--assay-sha256', plan['assay_sha256'], '--protocol-sha256', plan['protocol_sha256'],
                                              '--output', 'outputs'), env, log, deadline, 400))
            if 'clear-ease' in stages:
                seal_stage(root, 'clear-ease', status,
                           lambda log: invoke(root / 'clear-ease', ('-m', 'research.stage6.clear_reading_ease_forecast',
                                              '--model', runtime['snapshot'], '--checkpoint', GRADIO_CHECKPOINT,
                                              '--plan-path', 'plan.json', '--output', 'outputs'), env, log, deadline, 580))
            if 'first-readback' in stages:
                seal_stage(root, 'first-readback', status,
                           lambda log: first_readback(root, env, log, deadline, runtime['snapshot'], manifest))
            if 'context-readback' in stages:
                seal_stage(root, 'context-readback', status,
                           lambda log: context_readback(root, env, log, deadline, runtime['snapshot'], manifest))
        status['phase'] = 'stages-finished'
    except Exception as error:
        status.update(phase='worker-failed', error_type=type(error).__name__)
        raise
    finally:
        status['finished'] = True
        write_json(root / 'status.json', status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--payload-sha256', required=True)
    parser.add_argument('--worker-sha256', required=True)
    parser.add_argument('--seconds', type=int, required=True)
    run(**vars(parser.parse_args()))
