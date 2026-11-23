"""Package and retrieve sealed verification stages on a single owned Colab L4."""

import argparse
import ast
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
from io import BytesIO
import json
import math
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tarfile
from tempfile import TemporaryDirectory
from time import monotonic, sleep
from uuid import uuid4

from research.gradio_app import cuda_verification_batch_worker as worker
from research.gradio_app import pack_cuda_verification as gradio
from research.gradio_app.forecast import read_stimuli
from research.semantic_tracking.iirc_linked_colab import CLI_VERSION, Colab, manifest_path, run_lock, utc
from research.stage6 import ambistory_extract as ambistory


RELEASE_RESERVE_S = 175
FORMATS = {
    4: (worker.V4_BUNDLE, 27, worker.V4_STAGES),
    5: (worker.BUNDLE, worker.MAX_SECONDS // 60, worker.STAGES),
    6: (worker.TEXTDE_BUNDLE, worker.TEXTDE_SECONDS // 60, worker.TEXTDE_STAGES),
    7: (worker.GRADIO_BUNDLE, worker.GRADIO_SECONDS // 60, worker.GRADIO_STAGES),
    8: (worker.FORECAST_BUNDLE, worker.FORECAST_SECONDS // 60, worker.FORECAST_STAGES),
    9: (worker.STUDENT_BUNDLE, worker.STUDENT_SECONDS // 60, worker.STUDENT_STAGES),
    10: (worker.AMBISTORY_BUNDLE, worker.AMBISTORY_SECONDS // 60, worker.AMBISTORY_STAGES),
    11: (worker.READING_EXTRACTION_BUNDLE, worker.READING_EXTRACTION_SECONDS // 60, worker.READING_EXTRACTION_STAGES),
    12: (worker.MEDREADME_BUNDLE, worker.MEDREADME_SECONDS // 60, worker.MEDREADME_STAGES),
    13: (worker.ARTICLE_ROUTE_BUNDLE, worker.ARTICLE_ROUTE_SECONDS // 60, worker.ARTICLE_ROUTE_STAGES),
    14: (worker.ONESTOP_BUNDLE, worker.ONESTOP_SECONDS // 60, worker.ONESTOP_STAGES),
    15: (worker.EXPMRC_BUNDLE, worker.EXPMRC_SECONDS // 60, worker.EXPMRC_STAGES),
    16: (worker.FIRST_READBACK_BUNDLE, worker.FIRST_READBACK_SECONDS // 60, worker.FIRST_READBACK_STAGES),
    17: (worker.CONTEXT_READBACK_BUNDLE, worker.CONTEXT_READBACK_SECONDS // 60, worker.CONTEXT_READBACK_STAGES),
    18: (worker.NQ_EVIDENCE_BUNDLE, worker.NQ_EVIDENCE_SECONDS // 60, worker.NQ_EVIDENCE_STAGES),
    19: (worker.QASPER_TAIL_BUNDLE, worker.QASPER_TAIL_SECONDS // 60, worker.QASPER_TAIL_STAGES),
    20: (worker.GUM_BRIDGE_BUNDLE, worker.GUM_BRIDGE_SECONDS // 60, worker.GUM_BRIDGE_STAGES),
    21: (worker.BEUKERS_BUNDLE, worker.BEUKERS_SECONDS // 60, worker.BEUKERS_STAGES),
    22: (worker.PROJECTION_BUNDLE, worker.PROJECTION_SECONDS // 60, worker.PROJECTION_STAGES),
    23: (worker.FRICK_SIGNALS_BUNDLE, worker.FRICK_SIGNALS_SECONDS // 60, worker.FRICK_SIGNALS_STAGES),
    24: (worker.CLEAR_EASE_BUNDLE, worker.CLEAR_EASE_SECONDS // 60, worker.CLEAR_EASE_STAGES),
}
READING_EXTRACTION_INPUTS = {
    'student_annotations': (9, worker.STUDENT_BUNDLE - {'bundle.json'}),
    'ambistory': (10, worker.AMBISTORY_BUNDLE - {'bundle.json'}),
}
BACKBONE_FIELDS = ('model', 'revision', 'model_files', 'requirements')
GRADIO_FROZEN_HEADS = {
    'anchor/heads.npz': 'research/results/iirc-anchor-run/heads.npz',
    'sentence/head.npz': 'research/results/causal-plaba-checkpoint/head.npz',
    'lexical/heads.npz': 'research/results/stage5-complex/fit/heads.npz',
}
GRADIO_FIXTURES = {'fixtures/iirc-stimuli.jsonl', 'fixtures/iirc-dev/stimuli.jsonl',
                   'fixtures/plaba-features.npz', 'fixtures/plaba-stimuli.jsonl'}
GRADIO_ORDINAL_TOLERANCE = 0.005
FORECAST_INPUTS = {
    'blott': 'reading-trace-source-audit/blott-holdout/model_input/stimuli.jsonl',
    'kerwer': 'kerwer-2021/holdout/model_input/stimuli.jsonl',
}
FORECAST_PROTOCOLS = {'blott': 'blott-coherence-holdout.md', 'kerwer': 'kerwer-article-holdout.md'}
MEDREADME_PROTOCOL = 'research/stage6/medreadme-lexical-holdout.md'
MEDREADME_PREPARATION = 'research/stage6/medreadme_lexical_prepare.py'
MEDREADME_EXPORTER = 'research/stage6/medreadme_lexical_forecast.py'
WORD_HOLDOUT_STUDIES = {
    'medreadme': (MEDREADME_PROTOCOL, MEDREADME_PREPARATION),
    'cwi_news': ('research/core_features/cwi-news-holdout.md', 'research/core_features/cwi_news_prepare.py'),
    'cwi_wikinews': ('research/core_features/cwi-lexical-readout-protocol.md',
                     'research/core_features/cwi_wikinews_prepare.py'),
}
MEDREADME_APP_SOURCE = frozenset(f'research/{name}.py' for name in (
    'gradio_app/causal_plaba', 'gradio_app/checkpoint', 'gradio_app/combined', 'gradio_app/cuda',
    'gradio_app/lexical', 'gradio_app/ordinal', 'gradio_app/plaba', 'gradio_app/tokenizer',
    'semantic_tracking/iirc_anchor_fit', 'semantic_tracking/iirc_prepare', 'stage2/records', 'stage3/probe',
    'stage3/run', 'stage5/complex_tokenization',
))
GRADIO_APP = 'research/gradio_app/app.py'
ARTICLE_ROUTE_VERIFIER = 'research/gradio_app/verify_article_route.py'
ARTICLE_ROUTE_APP_SOURCE = MEDREADME_APP_SOURCE | {GRADIO_APP}
REFERENCE_ROUTE_VERIFIER = 'research/gradio_app/verify_reference.py'
REFERENCE_ROUTE_CODE = 'research/gradio_app/reference.py'
REFERENCE_ROUTE_COMBINED = 'research/gradio_app/combined.py'
REFERENCE_ROUTE_RELEASE = 'research/results/arrau-bridge-l4/reference-release.json'
REFERENCE_ROUTE_FIT = 'research/results/arrau-bridge-l4/fit.json'
REFERENCE_ROUTE_SCORES = 'research/results/arrau-bridge-l4/scores.json'
REFERENCE_ROUTE_ARTIFACTS = {REFERENCE_ROUTE_RELEASE, REFERENCE_ROUTE_FIT, REFERENCE_ROUTE_SCORES}
REFERENCE_ROUTE_SHIM = b'from .verify_reference import main\n\nif __name__ == "__main__":\n    main()\n'
REFERENCE_ROUTE_SOURCE = {
    GRADIO_APP, REFERENCE_ROUTE_COMBINED, REFERENCE_ROUTE_CODE, REFERENCE_ROUTE_VERIFIER,
}
REFERENCE_ROUTE_IDENTITY = {
    'source_code_sha256': REFERENCE_ROUTE_VERIFIER,
    'app_sha256': GRADIO_APP,
    'combined_sha256': REFERENCE_ROUTE_COMBINED,
    'reference_code_sha256': REFERENCE_ROUTE_CODE,
    'reference_release_sha256': REFERENCE_ROUTE_RELEASE,
}
ARTICLE_ROUTE_RECEIPT = 'research/results/gradio-article-route-l4/run-verification.json'
REFERENCE_ROUTE_RECEIPT = 'research/results/gradio-reference-l4/run-verification.json'
COMBINED_ARTICLE_RECEIPT = 'research/results/gradio-combined-overview-l4/run-verification.json'
COMBINED_ARTICLE_ROUTE = 'combined_article_route'
OVERVIEW_HTTP_ROUTE = worker.OVERVIEW_HTTP_ROUTE
OVERVIEW_HTTP_RECEIPT = 'research/results/gradio-overview-http-l4/run-verification.json'
OVERVIEW_HTTP_VERIFIER = 'research/gradio_app/verify_overview_http.py'
OVERVIEW_HTTP_NEW_SOURCES = {GRADIO_APP, REFERENCE_ROUTE_COMBINED, 'research/gradio_app/plaba.py', OVERVIEW_HTTP_VERIFIER}
OVERVIEW_HTTP_IDENTITY = {
    'source_code_sha256': OVERVIEW_HTTP_VERIFIER,
    'app_sha256': GRADIO_APP,
    'combined_sha256': REFERENCE_ROUTE_COMBINED,
    'plaba_sha256': 'research/gradio_app/plaba.py',
    'article_verifier_sha256': ARTICLE_ROUTE_VERIFIER,
    'reference_verifier_sha256': REFERENCE_ROUTE_VERIFIER,
    'reference_code_sha256': REFERENCE_ROUTE_CODE,
    'reference_release_sha256': REFERENCE_ROUTE_RELEASE,
}
OVERVIEW_HTTP_FIXTURE_SHA256 = 'b38f55bfc86e30caac82d469092ea3a078c76fa416ddde1885263cee483234da'
WORD_CUE_ROUTE = worker.WORD_CUE_ROUTE
WORD_CUE_VERIFIER = 'research/gradio_app/verify_word_cues.py'
WORD_CUE_RELEASE = 'research/results/blott-word-l4/word-cue-release.json'
WORD_CUE_RELEASE_SHA256 = '26b581e2754ed76540bb3dc337a427ba17aa556590d2ba2067b7eeb9f9dd65bd'
WORD_CUE_SCIPY = 'scipy==1.17.1'
WORD_CUE_PREFIX_TOLERANCE = 0.001
WORD_CUE_IDENTITY = {
    'source_code_sha256': WORD_CUE_VERIFIER,
    'app_sha256': GRADIO_APP,
    'combined_sha256': REFERENCE_ROUTE_COMBINED,
    'cue_release_sha256': WORD_CUE_RELEASE,
}
WORD_CUE_NEW_SOURCES = {GRADIO_APP, REFERENCE_ROUTE_COMBINED, WORD_CUE_VERIFIER} | {
    f'research/{name}.py' for name in (
        'gradio_app/word_cues', 'semantic_tracking/beukers_extract', 'semantic_tracking/blott_format_prepare',
        'semantic_tracking/blott_word_fit',
        'semantic_tracking/blott_word_states', 'semantic_tracking/frick_consistency_score',
        'semantic_tracking/gum_bridge_extract', 'semantic_tracking/rodger_belief_readout',
    )
}
CLEAR_EASE_PROTOCOL = 'research/stage6/clear-reading-ease-protocol.md'
CLEAR_EASE_RECEIPT = 'research/results/clear-reading-ease-preflight.json'
CLEAR_EASE_EXPORTER = 'research/stage6/clear_reading_ease_forecast.py'
ONESTOP_PROTOCOL = 'research/stage6/onestop-rereading-model.md'
ONESTOP_EXTRACTOR = 'research/stage6/onestop_rereading_extract.py'
EXPMRC_PROTOCOL = 'research/semantic_tracking/expmrc-transfer.md'
EXPMRC_EXPORTER = 'research/semantic_tracking/expmrc_forecast.py'
EXPMRC_SOURCE = frozenset(f'research/{name}.py' for name in (
    'core_features/extract', 'core_features/fit', 'semantic_tracking/joint_model', 'semantic_tracking/qasper_fit',
    'semantic_tracking/qasper_factorial_fit',
)) | {EXPMRC_EXPORTER}
CONTRACTNLI_EXPORTER = 'research/semantic_tracking/contractnli_forecast.py'
CONTRACTNLI_PROTOCOLS = {
    'contractnli': 'research/semantic_tracking/contractnli-evidence-protocol.md',
    'contractnli_rerank': 'research/semantic_tracking/contractnli-rerank-protocol.md',
    'evidence_inference_transfer': 'research/semantic_tracking/evidence-inference-transfer-protocol.md',
}
EVIDENCE_INFERENCE = 'evidence_inference_transfer'
CONTRACTNLI_INVENTORIES = {'contractnli': (47, 4318, 4207), 'contractnli_rerank': (23, 1739, 1684)}
CONTRACTNLI_REFERENCE = 'research/results/expmrc-transfer-l4/run-verification.json'
CONTRACTNLI_INPUTS = {f'{worker.CONTRACTNLI_PACKET}/{name}' for name in (
    'preflight.json', 'stimuli.jsonl', 'articles.jsonl',
)}
NQ_EVIDENCE_PROTOCOL = 'research/semantic_tracking/nq-evidence-transfer.md'
QASPER_TAIL_PROTOCOL = 'research/semantic_tracking/qasper-evidence-tail.md'
EVIDENCE_CALIBRATION = 'research/results/evidence-calibration-comparison.json'
EVIDENCE_CALIBRATION_SHA256 = 'efab56f7379a00b6b468a82c5a0ac8ad34aa8fe9d1189f434e995bd27bf1f944'
EVIDENCE_SCOPES = {
    18: ('nq-evidence', 'nq_evidence', worker.NQ_EVIDENCE_ARCHIVE, NQ_EVIDENCE_PROTOCOL, worker.NQ_EVIDENCE_SECONDS, 120_000),
    19: ('qasper-tail', 'qasper_tail', worker.QASPER_TAIL_ARCHIVE, QASPER_TAIL_PROTOCOL, worker.QASPER_TAIL_SECONDS, 150_000),
}
FIRST_READBACK_PROTOCOL = 'research/semantic_tracking/first-readback-transfer.md'
FIRST_READBACK_EXPORTER = 'research/semantic_tracking/first_readback_transfer.py'
FIRST_READBACK_SOURCE = MEDREADME_APP_SOURCE | {FIRST_READBACK_EXPORTER, 'research/core_features/extract.py',
                                               'research/core_features/fit.py'}
CONTEXT_READBACK_PROTOCOL = 'research/semantic_tracking/context-readback-transfer.md'
CONTEXT_READBACK_EXPORTER = 'research/semantic_tracking/context_readback_forecast.py'
CONTEXT_READBACK_SOURCE = FIRST_READBACK_SOURCE | {CONTEXT_READBACK_EXPORTER}
GUM_BRIDGE_PROTOCOL = 'research/semantic_tracking/gum-bridge-protocol.md'
GUM_BRIDGE_EXPORTER = 'research/semantic_tracking/gum_bridge_extract.py'
GUM_BRIDGE_SOURCE = MEDREADME_APP_SOURCE | {GUM_BRIDGE_EXPORTER, 'research/core_features/extract.py'}
GUM_BRIDGE_INPUTS = {f'model_input/{split}.jsonl' for split in ('train', 'dev', 'test')}
BEUKERS_PROTOCOL = 'research/semantic_tracking/beukers-background.md'
BEUKERS_EXPORTER = 'research/semantic_tracking/beukers_extract.py'
BEUKERS_SOURCE = GUM_BRIDGE_SOURCE | {BEUKERS_EXPORTER}
FRICK_SIGNALS_PROTOCOL = 'research/semantic_tracking/frick-signals-protocol.md'
FRICK_SIGNALS_EXPORTER = 'research/semantic_tracking/frick_signals_extract.py'
FRICK_PREVIOUS_PREFLIGHT = 'research/results/frick-consistency-l4/extractor-preflight.json'
FRICK_PREVIOUS_PREFLIGHT_SHA256 = 'c471c331de138b8fd9377f8f2c766187083a1c8efa931866faef9997b1e451b8'
FRICK_PREVIOUS_STIMULI_SHA256 = '12851b63014a17ba9ae8d62111ed42e1cdc275c3d0a711741d712392df01c531'
FRICK_SIGNAL_STIMULI_SHA256 = '1f0d2ae835429088fe6569366428226082397fb69f74660ac4ddcc0dacded46c'
FRICK_SIGNAL_PREFLIGHT_SHA256 = 'b8656522e71e72e6287fddaab421c0758bd661a16b19d523a9b82acb6bb50bb9'
FRICK_SIGNAL_INVENTORY = {'encoded_documents': 40, 'input_tokens': 9_386, 'certified_stops': 624}
CORE_VOCABULARY_PROTOCOL = 'research/semantic_tracking/core-vocabulary-qwen-protocol.md'
CORE_VOCABULARY_DISJOINT_PROTOCOL = 'research/semantic_tracking/core-vocabulary-disjoint-protocol.md'
ACT_PROTOCOL = 'research/semantic_tracking/act-transfer-protocol.md'
READER_PROJECTION_ASSAYS = {
    CORE_VOCABULARY_PROTOCOL: (
        'assay.json', '4fa6258798bc3f76410276491b985031f4c7e8a0d156063fd0908b1bb4451bdd', 6,
        {'encoded_documents': 768, 'input_tokens': 69_810, 'certified_stops': 4_608}, None,
    ),
    CORE_VOCABULARY_DISJOINT_PROTOCOL: (
        'preflight.json', '20f5e6dfbc9e5936d5ba6a5701e14a9810200d356314d94145f04030d02b6edf', 6,
        {'encoded_documents': 1024, 'input_tokens': 93_122, 'certified_stops': 6_144},
        '69dafdd439abf372455b7a29e6159769660bfbbb3155ce3f1b3cff4be025e8f1',
    ),
    ACT_PROTOCOL: (
        'preflight.json', 'f42050df403a163aeff0c5468443cd4edf0e738bdfa2fbb48fb3cd6aa45dd5ab', 15,
        {'encoded_documents': 20, 'input_tokens': 1_186, 'certified_stops': 300},
        'cad4e7293e348c6c7e703b413b274c593d527ed34896423cc8cce01624a6b19c',
    ),
}
PROJECTION_SCOPES = {
    'beukers': (21, worker.BEUKERS_ARCHIVE, worker.BEUKERS_SECONDS),
    'projection': (22, worker.PROJECTION_ARCHIVE, worker.PROJECTION_SECONDS),
}
LATER_SOURCE = frozenset(f'research/{name}.py' for name in (
    'core_features/extract', 'core_features/fit', 'semantic_tracking/iirc_linked_extract',
    'semantic_tracking/iirc_linked_fit', 'semantic_tracking/iirc_linked_pack', 'semantic_tracking/iirc_prepare',
    'semantic_tracking/iirc_seed_later', 'semantic_tracking/iirc_seed_later_assay',
    'semantic_tracking/joint_model', 'semantic_tracking/qasper_fit',
    'stage2/records', 'stage3/probe', 'stage3/run',
))
CWI_PROTOCOL = 'research/core-features-background-transfer.md'
CWI_SOURCE = frozenset(f'research/{name}.py' for name in (
    'core_features/background_prepare', 'core_features/background_transfer', 'core_features/background_transfer_fit',
    'core_features/extract', 'core_features/fit', 'core_features/localization_cwi_score',
    'semantic_tracking/iirc_linked_pack', 'semantic_tracking/iirc_prepare', 'stage2/records',
    'stage3/probe', 'stage3/run', 'stage5/complex_inputs', 'stage5/complex_tokenization',
))
PLABA_SOURCE = frozenset(f'research/{name}.py' for name in (
    'core_features/extract', 'core_features/readback_score', 'semantic_tracking/iirc_anchor_fit',
    'semantic_tracking/iirc_anchor_score', 'semantic_tracking/iirc_linked_pack', 'semantic_tracking/iirc_prepare',
    'semantic_tracking/joint_score', 'semantic_tracking/plaba_span_transfer', 'stage2/records',
    'stage3/probe', 'stage3/run',
))
PLABA_SOURCE_SHA256 = '6f393059baa58886af56154c7aa0d60c04e674ce9175866df6607ed38a5201b3'
PLABA_DATA_SHA256 = '8d1f8ca85cc6ec5ec3afa3a3afbf95638a29a003fc08b00b29d39717fe22c306'
TEXTDE_PROTOCOL = 'research/stage6/textcomplexityde-transfer.md'
TEXTDE_AMENDMENT = 'research/stage6/textcomplexityde-transfer-amendment.md'
TEXTDE_INTERIOR_CONTROLS = 'independently certified interior boundaries of the input text'
TEXTDE_REQUIREMENTS = ('numpy==2.4.6', 'scipy==1.16.3', 'transformers==5.17.0')
TEXTDE_SOURCE = frozenset(f'research/{name}.py' for name in (
    'core_features/extract', 'core_features/readback_score', 'semantic_tracking/iirc_linked_pack',
    'semantic_tracking/iirc_prepare', 'stage2/records', 'stage3/causal_export', 'stage3/causal_ordinal',
    'stage3/probe', 'stage3/run', 'stage6/textcomplexityde_eval', 'stage6/textcomplexityde_transfer',
)) | {TEXTDE_PROTOCOL, TEXTDE_AMENDMENT}
STUDENT_RECEIPT = 'research/results/student-annotations-preparation.json'
STUDENT_INVENTORY = {'documents': 1143, 'expected_certified_stops': 1143, 'total_input_tokens': 33745,
                     'max_input_tokens': 101}
# These exact files ran the verified TextComplexityDE certified-control extraction on L4.
STUDENT_SOURCE_SHA256 = {
    'research/core_features/extract.py': '8f7df0a8e994474bb6c6c64ed4df259827de809f713facc3050f16d10c53d320',
    'research/stage3/run.py': '3edc5740fa63c59b77aaacbeb809075f802623eff4c55d562cf855a9e822e282',
    'research/stage3/probe.py': 'bb2bbb41d34292ef0e67fc4a876cf833a64e3fafe180f9eecd70086b2109009e',
}
AMBISTORY_RECEIPT = 'research/results/ambistory-preparation.json'
AMBISTORY_SOURCE = frozenset({*STUDENT_SOURCE_SHA256, 'research/stage6/ambistory_extract.py',
                              'research/core_features/fit.py', 'research/semantic_tracking/joint_model.py',
                              'research/semantic_tracking/qasper_fit.py'})
OFF_PATH = {
    'gradio': {('research.gradio_app.plaba', 'research.train_plaba_probe'),
               ('research.semantic_tracking.iirc_anchor_fit', 'research.semantic_tracking.qasper_fit')},
    'expmrc': {('research.gradio_app.plaba', 'research.train_plaba_probe')},
    'contractnli': {('research.gradio_app.plaba', 'research.train_plaba_probe'),
                    ('research.semantic_tracking.contractnli_forecast', 'research.semantic_tracking.gum_bridge_extract')},
    'contractnli_rerank': {
        ('research.gradio_app.plaba', 'research.train_plaba_probe'),
        ('research.semantic_tracking.contractnli_forecast', 'research.semantic_tracking.gum_bridge_extract'),
        ('research.semantic_tracking.contractnli_rerank_forecast', 'research.semantic_tracking.gum_bridge_extract'),
    },
    EVIDENCE_INFERENCE: {
        ('research.gradio_app.plaba', 'research.train_plaba_probe'),
        ('research.semantic_tracking.contractnli_forecast', 'research.semantic_tracking.gum_bridge_extract'),
        ('research.semantic_tracking.evidence_inference_transfer_forecast', 'research.semantic_tracking.gum_bridge_extract'),
    },
    'later': {('research.semantic_tracking.iirc_seed_later_assay', name) for name in (
        'research.semantic_tracking.iirc_linked_prepare', 'research.semantic_tracking.iirc_linked_score',
        'research.semantic_tracking.joint_score',
    )},
    'cwi': set(),
    'plaba': {('research.semantic_tracking.iirc_anchor_fit', 'research.semantic_tracking.qasper_fit'),
              ('research.semantic_tracking.plaba_span_transfer', 'research.semantic_tracking.iirc_linked_prepare')},
    'textde': set(),
    'student': set(),
    'ambistory': set(),
}
OFF_PATH['word-cues'] = OFF_PATH['gradio'] | {
    ('research.semantic_tracking.beukers_extract', 'research.core_features.extract'),
    ('research.semantic_tracking.gum_bridge_extract', 'research.core_features.extract'),
}


def source_closure(package, selected, kind):
    modules = {name.removesuffix('.py').replace('/', '.') for name in selected if name.endswith('.py')}
    absent = set()
    for name, contents in selected.items():
        if not name.endswith('.py'):
            continue
        compile(contents, name, 'exec')
        module = name.removesuffix('.py').replace('/', '.')
        for node in ast.walk(ast.parse(contents)):
            if not isinstance(node, ast.ImportFrom):
                continue
            target = node.module or ''
            if node.level:
                target = importlib.util.resolve_name('.' * node.level + target, module.rsplit('.', 1)[0])
            if (target.startswith('research.') and target not in modules
                    and not any(f'{target}.{alias.name}' in modules for alias in node.names)):
                absent.add((module, target))
    if absent != OFF_PATH[kind]:
        raise ValueError(f'{package.name} source closure has unaccounted edges: {sorted(absent ^ OFF_PATH[kind])}')
    return {'modules': sorted(modules), 'off_path_imports': sorted(absent)}


def archive_code(path, allowed=None):
    with tarfile.open(path, 'r:gz') as package:
        included = [member for member in package if member.name.endswith('.py') and
                    (allowed is None or member.name in allowed)]
        names = {member.name for member in included}
        if len(names) != len(included) or (allowed is not None and names != allowed):
            raise ValueError(f'Unexpected source membership: {path.name}')
        return {member.name: package.extractfile(member).read() for member in included}


def write_archive(path, files):
    with path.open('wb') as destination, gzip.GzipFile(fileobj=destination, filename='', mode='wb', mtime=0) as stream:
        with tarfile.open(fileobj=stream, mode='w') as archive:
            for name, source in sorted(files.items()):
                member = tarfile.TarInfo(name)
                if isinstance(source, bytes):
                    member.size = len(source)
                    archive.addfile(member, BytesIO(source))
                else:
                    member.size = source.stat().st_size
                    with source.open('rb') as contents:
                        archive.addfile(member, contents)


def textde_packet(root):
    model = root / 'model-inputs.tar.gz'
    assay = worker.read_json(root / 'assay.json')
    receipt = worker.read_json(gradio.REPO / 'research/results/textcomplexityde-transfer-amendment.json')
    hashes = {**assay['model_input_hashes'], 'assay.json': worker.digest(root / 'assay.json')}
    if (set(hashes) != worker.TEXTDE_DATA or hashes != receipt['archive_members_sha256']
            or worker.digest(model) != receipt['archive_sha256']
            or worker.digest(gradio.REPO / TEXTDE_PROTOCOL) != assay['protocol_sha256']
            or worker.digest(gradio.REPO / TEXTDE_AMENDMENT) != assay['amendment_sha256']
            or worker.digest(gradio.REPO / 'research/stage6/textcomplexityde_transfer.py') != assay['preparation_code_sha256']):
        raise ValueError('TextComplexityDE grade-free input or source protocol differs from its amendment seal')
    worker.unpack(model, hashes)
    sources = {name: (gradio.REPO / name).read_bytes() for name in TEXTDE_SOURCE}
    closure = source_closure(model, sources, 'textde')
    return model, assay, hashes, sources, closure


def new_job(output):
    job = output / f'cogllm-cuda-checks-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}'
    job.mkdir(mode=0o700, parents=True)
    shutil.copyfile(worker.__file__, job / 'worker.py')
    return job


def seal_job(job, bundles, manifest, schema, runtime_seconds, **details):
    manifest['archives'] = {name: {'sha256': worker.digest(path), 'directory': directory, 'members': hashes}
                            for name, (path, directory, hashes) in bundles.items()}
    (job / 'bundle.json').write_bytes(gradio.json_bytes(manifest))
    payload = job / 'payload.tar.gz'
    write_archive(payload, {'bundle.json': job / 'bundle.json', **{name: details[0] for name, details in bundles.items()}})
    inventory = worker.unpack(payload, FORMATS[schema][0])
    data = {'run_id': job.name, 'schema_version': schema, 'phase': 'prepared', 'timeline': {'prepared': utc()},
            'payload_sha256': worker.digest(payload), 'worker_sha256': worker.digest(job / 'worker.py'),
            'payload_members_sha256': inventory, 'payload_bytes': payload.stat().st_size,
            'minutes_budget': runtime_seconds / 60, **details}
    worker.write_json(manifest_path(job), data)
    verify(job, data)
    print(job.resolve(), flush=True)


def prepare_textde(textde_root, output):
    model, assay, hashes, sources, closure = textde_packet(textde_root)
    if (assay['model'], assay['revision']) != (gradio.MODEL, gradio.REVISION):
        raise ValueError('TextComplexityDE inputs use a different pinned checkpoint')
    extractor = ast.parse(sources['research/core_features/extract.py'])
    if not any(isinstance(node, ast.Constant) and node.value == '--certified-token-controls' for node in ast.walk(extractor)):
        raise ValueError('TextComplexityDE requires the certified interior token controls flag')
    compile(worker.SNAPSHOT, '<snapshot-setup>', 'exec')
    job = new_job(output)
    source = job / 'textde-source.tar.gz'
    write_archive(source, sources)
    bundles = {'textde-model-inputs.tar.gz': (model, 'textcomplexityde/inputs', hashes),
               'textde-source.tar.gz': (source, 'textcomplexityde',
                                        {name: hashlib.sha256(code).hexdigest() for name, code in sources.items()})}
    manifest = {'model': gradio.MODEL, 'revision': gradio.REVISION, 'model_files': gradio.MODEL_FILES,
                'requirements': TEXTDE_REQUIREMENTS, 'source_closure': {'textde': closure},
                'textde': {key: assay[key] for key in ('cases', 'article_groups', 'forecast_methods', 'target_order')},
                'textde_control_stops': TEXTDE_INTERIOR_CONTROLS}
    seal_job(job, bundles, manifest, 6, worker.TEXTDE_SECONDS)


def student_preparation():
    path = gradio.REPO / STUDENT_RECEIPT
    receipt = worker.read_json(path)
    expected = {'protocol_sha256': 'research/stage6/student-annotations-background.md',
                'source_audit_sha256': 'research/results/student-annotations-audit.json',
                'source_selection_code_sha256': 'research/stage6/student_annotations_audit.py',
                'prep_code_sha256': 'research/stage6/student_annotations_prepare.py'}
    if (receipt['status'] != 'prepared_before_any_model_extraction_or_fitting'
            or receipt['counts']['sentences'] != STUDENT_INVENTORY['documents']
            or receipt['counts']['exact_causal_stops'] != STUDENT_INVENTORY['expected_certified_stops']
            or any(receipt[digest] != worker.digest(gradio.REPO / name) for digest, name in expected.items())):
        raise ValueError('StudentAnnotations preparation differs from its audited inventory or protocol')
    return (worker.digest(path), receipt['sha256']['model-inputs.tar.gz'],
            {name: receipt['sha256'][f'model_input/{name}'] for name in worker.STUDENT_DATA})


def student_inputs(materials, hashes):
    with materials.extractfile('extraction-plan.json') as stream:
        plan = json.load(stream)
    if ((plan['model'], plan['model_revision']) != (gradio.MODEL, gradio.REVISION)
            or f"transformers=={plan['transformers']}" not in TEXTDE_REQUIREMENTS
            or (plan['dtype'], plan['planned_accelerator']) != ('float32', 'NVIDIA L4')
            or plan['extractor'] != 'research.core_features.extract'
            or plan['runtime_causality_checks'] != 'certified_token_controls'
            or plan['record_fields'] != ['id', 'text', 'stops']
            or any(plan[key] != count for key, count in STUDENT_INVENTORY.items())
            or plan['sha256'] != {name: hashes[name] for name in ('stimuli.jsonl', 'certified-stops.jsonl')}):
        raise ValueError('StudentAnnotations extraction plan differs from the sealed sentence and tokenizer protocol')
    coordinates = {}
    total_tokens, max_tokens = 0, 0
    with materials.extractfile('stimuli.jsonl') as texts, materials.extractfile('certified-stops.jsonl') as stops:
        for text, proof in zip(map(json.loads, texts), map(json.loads, stops), strict=True):
            source, identifier, count = text['text'], text['id'], proof['tokens']
            if (text.keys() != {'id', 'text', 'stops'} or proof.keys() != {'id', 'stop', 'token', 'tokens'}
                    or identifier != proof['id'] or identifier in coordinates
                    or re.fullmatch(r'[0-9a-f]{24}', identifier) is None
                    or type(count) is not int or count <= 0 or proof['token'] != count - 1
                    or text['stops'] != [len(source.rstrip())] or text['stops'] != [proof['stop']] or not source.strip()):
                raise ValueError('StudentAnnotations input contains an invalid sentence-end stop')
            coordinates[identifier] = proof['stop']
            total_tokens += count
            max_tokens = max(max_tokens, count)
    if (len(coordinates), total_tokens, max_tokens) != (plan['documents'], plan['total_input_tokens'], plan['max_input_tokens']):
        raise ValueError('StudentAnnotations stop or token inventory differs from the frozen extraction plan')
    return plan, coordinates


def prepare_student_annotations(student_root, output):
    receipt_sha256, input_sha256, hashes = student_preparation()
    inputs = student_root / 'model-inputs.tar.gz'
    if worker.digest(inputs) != input_sha256:
        raise ValueError('StudentAnnotations model-only archive differs from its preparation receipt')
    worker.unpack(inputs, hashes)
    with tarfile.open(inputs, 'r:gz') as materials:
        plan, _ = student_inputs(materials, hashes)
    sources = {name: (gradio.REPO / name).read_bytes() for name in STUDENT_SOURCE_SHA256}
    if {name: hashlib.sha256(code).hexdigest() for name, code in sources.items()} != STUDENT_SOURCE_SHA256:
        raise ValueError('StudentAnnotations extraction requires the previously verified extractor source')
    closure = source_closure(inputs, sources, 'student')
    compile(worker.SNAPSHOT, '<snapshot-setup>', 'exec')
    job = new_job(output)
    source = job / worker.STUDENT_SOURCE_ARCHIVE
    write_archive(source, sources)
    archives = {worker.STUDENT_DATA_ARCHIVE: (inputs, 'student-annotations/inputs', hashes),
                worker.STUDENT_SOURCE_ARCHIVE: (source, 'student-annotations', STUDENT_SOURCE_SHA256)}
    student = {**STUDENT_INVENTORY, 'preparation_sha256': receipt_sha256,
               'tokenizer_sha256': plan['tokenizer_sha256'], 'state_width': 2048}
    bundle = {'model': gradio.MODEL, 'revision': gradio.REVISION, 'model_files': gradio.MODEL_FILES,
              'requirements': TEXTDE_REQUIREMENTS, 'source_closure': {'student': closure}, 'student_annotations': student}
    seal_job(job, archives, bundle, 9, worker.STUDENT_SECONDS, student_annotations=student)


def ambistory_preparation():
    path = gradio.REPO / AMBISTORY_RECEIPT
    report = worker.read_json(path)
    token = report['tokenizer']
    if (report['preparation_sha256'] != worker.digest(gradio.REPO / 'research/stage6/ambistory_prepare.py')
            or token['revision'] != gradio.REVISION or f"transformers=={token['transformers']}" not in TEXTDE_REQUIREMENTS
            or token['files_sha256'].keys() != {'tokenizer.json', 'tokenizer_config.json'}
            or report['splits'].keys() != set(ambistory.SPLITS)):
        raise ValueError('AmbiStory preparation or tokenizer differs from its frozen input receipt')
    inventory = {split: report['splits'][split]['tokenizer_preflight'] for split in ambistory.SPLITS}
    if any(values['rejected_stops'] for values in inventory.values()):
        raise ValueError('AmbiStory extraction requires every declared stop to have passed preflight')
    spec = {'preparation_sha256': worker.digest(path), 'tokenizer_files_sha256': token['files_sha256'],
            'projection_sha256': ambistory.PROJECTION_SHA256, 'splits': inventory,
            'families': sum(report['splits'][split]['story_families'] for split in ambistory.SPLITS)}
    return spec, report['files_sha256']


def ambistory_model_files(root, expected):
    files = {}
    senses = {}
    prefixes = []
    for split in ambistory.SPLITS:
        stimuli, design = f'{split}/model_input/stimuli.jsonl', f'{split}/design/stops.jsonl'
        if any(worker.digest(root / name) != expected[name] for name in (stimuli, design)):
            raise ValueError(f'AmbiStory {split} grade-free source differs from the frozen receipt')
        files[f'{split}/stimuli.jsonl'] = root / stimuli
        families = defaultdict(list)
        for row in ambistory.read_rows(root / design):
            common = sorted({row['target_pre_stop'], row['target'][1], row['setup_stop']})
            families[row['family']].append((row['id'], common))
            for sense in row['senses']:
                for kind in ambistory.KINDS:
                    key, text = (sense['id'], kind), sense[kind]
                    if senses.setdefault(key, text) != text:
                        raise ValueError(f'AmbiStory sense text varies among the {split} story variants')
        for variants in families.values():
            ids, stops = zip(*variants, strict=True)
            if len(ids) != 3 or any(stops[0] != shared for shared in stops):
                raise ValueError(f'AmbiStory {split} needs three matching shared-prefix variants per family')
            prefixes.append({'ids': sorted(ids), 'stops': stops[0]})
    sense_rows = [{'id': key, 'kind': kind, 'text': text} for (key, kind), text in sorted(senses.items())]
    for name, rows in {'senses.jsonl': sense_rows, 'shared-prefixes.jsonl': prefixes}.items():
        files[name] = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode('utf-8')
    return files


def prepare_ambistory(ambistory_root, output):
    spec, expected = ambistory_preparation()
    files = ambistory_model_files(ambistory_root, expected)
    hashes = {name: hashlib.sha256(value).hexdigest() if isinstance(value, bytes) else worker.digest(value)
              for name, value in files.items()}
    sources = {name: (gradio.REPO / name).read_bytes() for name in AMBISTORY_SOURCE}
    source_hashes = {name: hashlib.sha256(code).hexdigest() for name, code in sources.items()}
    if any(source_hashes[name] != reference for name, reference in STUDENT_SOURCE_SHA256.items()):
        raise ValueError('AmbiStory requires the verified causal state extraction and control source')
    compile(worker.SNAPSHOT, '<snapshot-setup>', 'exec')
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        inputs = root / worker.AMBISTORY_DATA_ARCHIVE
        write_archive(inputs, files)
        worker.unpack(inputs, hashes, root / 'inputs')
        stories, senses, prefixes, split_counts = ambistory.read_inputs(root / 'inputs')
        if (hashes.keys() != ambistory.INPUT_MEMBERS
                or split_counts != {split: row['stories'] for split, row in spec['splits'].items()}
                or sum(len(row['stops']) for row in stories.values()) !=
                sum(row['certified_stops'] for row in spec['splits'].values())
                or len(prefixes) != spec['families'] or len(senses) != spec['families'] * 4):
            raise ValueError('AmbiStory grade-free model archive differs from the prepared corpus inventory')
        closure = source_closure(inputs, sources, 'ambistory')
        job = new_job(output)
        source = job / worker.AMBISTORY_SOURCE_ARCHIVE
        write_archive(source, sources)
        archives = {worker.AMBISTORY_DATA_ARCHIVE: (inputs, 'ambistory/inputs', hashes),
                    worker.AMBISTORY_SOURCE_ARCHIVE: (source, 'ambistory', source_hashes)}
        bundle = {'model': gradio.MODEL, 'revision': gradio.REVISION, 'model_files': gradio.MODEL_FILES,
                  'requirements': TEXTDE_REQUIREMENTS, 'source_closure': {'ambistory': closure}, 'ambistory': spec}
        seal_job(job, archives, bundle, 10, worker.AMBISTORY_SECONDS, ambistory=spec)


def prepare_reading_extractions(student_job, ambistory_job, output):
    archives, source_runs, specs, closure = {}, {}, {}, {}
    backbone = None
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        for (dataset, (schema, names)), previous in zip(READING_EXTRACTION_INPUTS.items(),
                                                      (student_job, ambistory_job), strict=True):
            data = worker.read_json(manifest_path(previous))
            if data['schema_version'] != schema:
                raise ValueError(f'Expected sealed standalone schema {schema} for {dataset}')
            verify(previous, data)
            bundle = worker.read_json(previous / 'bundle.json')
            if bundle['archives'].keys() != names:
                raise ValueError(f'Unexpected archive inventory in the standalone {dataset} batch')
            config = {name: bundle[name] for name in BACKBONE_FIELDS}
            if backbone is not None and config != backbone:
                raise ValueError('The standalone extraction packets need different backbone setups')
            backbone = config
            specs[dataset] = bundle[dataset]
            closure.update(bundle['source_closure'])
            digests = data['payload_members_sha256']
            with tarfile.open(previous / 'payload.tar.gz', 'r:gz') as package:
                for name, archived in bundle['archives'].items():
                    path = root / name
                    with package.extractfile(name) as contents, path.open('wb') as destination:
                        shutil.copyfileobj(contents, destination)
                    if not worker.digest(path) == archived['sha256'] == digests[name]:
                        raise ValueError(f'Standalone nested archive digest mismatch: {name}')
                    worker.unpack(path, archived['members'])
                    archives[name] = (path, archived['directory'], archived['members'])
            source_runs[dataset] = {'run_id': data['run_id'], 'payload_sha256': data['payload_sha256'],
                                    'archives_sha256': {name: digests[name] for name in names}}

        job = new_job(output)
        manifest = {**backbone, **specs, 'source_closure': closure, 'source_runs': source_runs}
        seal_job(job, archives, manifest, 11, worker.READING_EXTRACTION_SECONDS, **specs, source_runs=source_runs)


def forecast_preregistration(dataset, checkpoint_id):
    research = gradio.REPO / 'research'
    path = research / 'results' / f'{dataset}-transfer-protocol.json'
    receipt = worker.read_json(path)
    if (receipt['candidate_id'] != checkpoint_id
            or receipt['protocol_sha256'] != worker.digest(research / 'stage6' / FORECAST_PROTOCOLS[dataset])
            or receipt['scorer_sha256'] != worker.digest(research / 'stage6' / f'{dataset}_score.py')):
        raise ValueError(f'{dataset} preregistration differs from the candidate checkpoint, protocol, or scorer')
    return receipt['input_sha256']['stimuli'], worker.digest(path)


def forecast_packet(checkpoint_id):
    files = {worker.FORECAST_SOURCE: gradio.REPO / worker.FORECAST_SOURCE}
    plan = {}
    for dataset, name in worker.FORECAST_STIMULI.items():
        path = gradio.CACHE / FORECAST_INPUTS[dataset]
        input_sha256, preregistration_sha256 = forecast_preregistration(dataset, checkpoint_id)
        documents = read_stimuli(path, input_sha256)
        files[name] = path
        plan[dataset] = {'path': name, 'input_sha256': input_sha256, 'documents': len(documents.documents),
                         'preregistration_sha256': preregistration_sha256}
    return files, plan


def prepare_gradio(gradio_root, output, *, with_forecasts=False):
    path = gradio_root / gradio.PAYLOAD
    packet = worker.read_json(gradio_root / gradio.MANIFEST)
    gradio.verify(gradio_root)
    if ((packet['model'], packet['revision']) != (gradio.MODEL, gradio.REVISION)
            or tuple(packet['model_files']) != gradio.MODEL_FILES
            or (packet['device'], packet['compute_dtype']) != ('NVIDIA L4', 'float32')
            or packet['held_out_iirc_outcomes_packaged'] is not False):
        raise ValueError('Gradio inputs differ from the pinned Qwen serving configuration')

    hashes = {name: value['sha256'] for name, value in packet['members'].items()}
    hashes[gradio.MANIFEST] = worker.digest(gradio_root / gradio.MANIFEST)
    prefix = f'{worker.GRADIO_CHECKPOINT}/'
    with tarfile.open(path, 'r:gz') as archive:
        checkpoint = json.load(archive.extractfile(f'{prefix}checkpoint.json'))
    identifier = checkpoint.pop('checkpoint_sha256')
    canonical = json.dumps(checkpoint, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    members = {name.removeprefix(prefix): sha for name, sha in hashes.items()
               if name.startswith(prefix) and name != f'{prefix}checkpoint.json'}
    expected = ({gradio.MANIFEST, f'{prefix}checkpoint.json'} | GRADIO_FIXTURES
                | {f'research/{name}.py' for name in gradio.SOURCE_NAMES}
                | {f'research/results/{name}' for name in gradio.RESULT_NAMES}
                | {f'nltk_data/tokenizers/punkt_tab/{name}' for name in gradio.PUNKT_FILES}
                | {f'{prefix}{name}' for name in checkpoint['files']})
    if (hashes.keys() != expected
            or identifier != packet['checkpoint_sha256'] or identifier != hashlib.sha256(canonical).hexdigest()
            or checkpoint['status'] != 'candidate' or checkpoint['contract']['input'] != 'article'
            or checkpoint['files'] != members
            or any(members[name] != hashes[source] for name, source in GRADIO_FROZEN_HEADS.items())):
        raise ValueError('Packed candidate checkpoint differs from its identity or verified readouts')
    ordinal = members['ordinal/head.npz']
    codes = archive_code(path)
    files, forecasts = forecast_packet(identifier) if with_forecasts else ({}, {})
    if with_forecasts:
        codes[worker.FORECAST_SOURCE] = files[worker.FORECAST_SOURCE].read_bytes()
        compile(worker.FORECAST, '<holdout-forecast>', 'exec')
    closure = source_closure(path, codes, 'gradio')
    compile(worker.SNAPSHOT, '<snapshot-setup>', 'exec')
    job = new_job(output)
    bundle = {'model': gradio.MODEL, 'revision': gradio.REVISION, 'model_files': gradio.MODEL_FILES,
              'requirements': packet['python_requirements'], 'source_closure': {'gradio': closure},
              'checkpoint_sha256': identifier}
    archives = {'gradio.tar.gz': (path, 'gradio', hashes)}
    details = {'checkpoint_sha256': identifier, 'ordinal_head_sha256': ordinal}
    if with_forecasts:
        inputs = job / worker.FORECAST_INPUT_ARCHIVE
        input_hashes = {worker.FORECAST_SOURCE: hashlib.sha256(codes[worker.FORECAST_SOURCE]).hexdigest(),
                        **{spec['path']: spec['input_sha256'] for spec in forecasts.values()}}
        write_archive(inputs, {**files, worker.FORECAST_SOURCE: codes[worker.FORECAST_SOURCE]})
        worker.unpack(inputs, input_hashes)
        archives[worker.FORECAST_INPUT_ARCHIVE] = (inputs, 'gradio', input_hashes)
        bundle['forecast'] = details['forecast'] = forecasts
    schema, seconds = (8, worker.FORECAST_SECONDS) if with_forecasts else (7, worker.GRADIO_SECONDS)
    seal_job(job, archives, bundle, schema, seconds, **details)


def word_holdout_inputs(root, study):
    receipt = root / 'assay.json'
    plan = worker.read_json(receipt)
    protocol, preparation = WORD_HOLDOUT_STUDIES[study]
    if (plan['protocol_sha256'] != worker.digest(gradio.REPO / protocol)
            or plan['preparation_code_sha256'] != worker.digest(gradio.REPO / preparation)):
        raise ValueError('Word holdout stimuli differ from the frozen protocol or preparation source')
    path = root / 'stimuli.jsonl'
    sample = read_stimuli(path, plan['stimuli_sha256'])
    with path.open(encoding='utf-8') as stream:
        if any(set(row) != {'id', 'text'} for row in map(json.loads, stream)):
            raise ValueError('Word holdout model inputs may contain only IDs and raw source text')
    if len(sample.documents) != plan['retained_sentences']:
        raise ValueError('Word holdout stimulus count differs from the registered freeze')
    reduced = {key: plan[key] for key in ('candidate_id', 'lexical_weights_sha256', 'protocol_sha256',
                                        'stimuli_sha256', 'retained_sentences')}
    return path, {**reduced, 'assay_sha256': worker.digest(receipt)}


def verified_gradio_packet(v8_job, app_sources, *, include_forecast=False):
    data = worker.read_json(manifest_path(v8_job))
    verify(v8_job, data)
    if (data['schema_version'] != 8 or data['phase'] != 'finished' or data.get('release_verified') is not True
            or not data['stages']['gradio']['verified'] or not data['stages']['setup']['verified']):
        raise ValueError('Expected a released V8 job with the evidence-verified Gradio candidate')
    verified = v8_job / 'outputs/gradio/verification.json'
    if worker.digest(verified) != data['stages']['gradio']['members']['verification.json']:
        raise ValueError('The local V8 Gradio report differs from its verified GPU return')
    packet = worker.read_json(v8_job / 'bundle.json')
    known = packet['archives']['gradio.tar.gz']['members']
    prefix = f'{worker.GRADIO_CHECKPOINT}/'
    checkpoint_members = {name for name in known if name.startswith(prefix)}
    inputs_by_archive = {'gradio.tar.gz': app_sources | checkpoint_members}
    if include_forecast:
        inputs_by_archive[worker.FORECAST_INPUT_ARCHIVE] = {worker.FORECAST_SOURCE}
    files = {}
    with tarfile.open(v8_job / 'payload.tar.gz', 'r:gz') as package:
        for name, selected in inputs_by_archive.items():
            with package.extractfile(name) as artifact:
                with tarfile.open(fileobj=BytesIO(artifact.read()), mode='r:gz') as inputs:
                    source_hashes = packet['archives'][name]['members']
                    for member in selected:
                        with inputs.extractfile(member) as stream:
                            files[member] = stream.read()
                        if hashlib.sha256(files[member]).hexdigest() != source_hashes[member]:
                            raise ValueError(f'V8 inference member differs from its verified bytes: {member}')
    report = worker.read_json(verified)
    checkpoint = json.loads(files[f'{prefix}checkpoint.json'])
    if (not checkpoint['checkpoint_sha256'] == report['checkpoint_sha256'] == packet['checkpoint_sha256']
            or report['device'] != 'NVIDIA L4 FP32' or report['model_forwards_per_combined_gradio_request'] != 1
            or checkpoint_members != {f'{prefix}{name}' for name in checkpoint['files'].keys() | {'checkpoint.json'}}
            or report['lexical_head_sha256'] != known[f'{prefix}lexical/heads.npz']):
        raise ValueError('The V8 checkpoint or lexical head differs from the verified serving route')
    return files, packet, data['run_id'], worker.digest(verified), report['lexical_head_sha256']


def verified_combined_reader_packet(route_job, *, include_forecast=False):
    data = worker.read_json(manifest_path(route_job))
    verify(route_job, data)
    receipt, released, route = combined_reader_release(data['run_id'])
    pins = ('run_id', 'payload_sha256', 'payload_members_sha256', 'stages', route)
    if (data['phase'] != 'finished' or data.get('release_verified') is not True
            or data['schema_version'] != 13 or any(data[key] != released[key] for key in pins)):
        raise ValueError('The experiment requires the latest verified combined article route')
    plan = data[route]
    bundle = worker.read_json(route_job / 'bundle.json')
    archived = bundle['archives'][worker.ARTICLE_ROUTE_ARCHIVE]['members']
    selected = MEDREADME_APP_SOURCE | {REFERENCE_ROUTE_CODE}
    prefix = f'{worker.GRADIO_CHECKPOINT}/'
    with tarfile.open(route_job / worker.ARTICLE_ROUTE_ARCHIVE, 'r:gz') as source:
        files = {name: source.extractfile(name).read() for name in selected | {name for name in archived if name.startswith(prefix)}}
    if any(files[name] != (gradio.REPO / name).read_bytes() for name in selected):
        raise ValueError('The packed readout must use the verified numerical Gradio sources')
    if include_forecast:
        files[worker.FORECAST_SOURCE] = (gradio.REPO / worker.FORECAST_SOURCE).read_bytes()
    return files, bundle, plan['candidate_id'], archived[f'{prefix}lexical/heads.npz'], {
        'combined_route_run_id': released['run_id'], 'combined_route_receipt_sha256': worker.digest(receipt),
    }


def combined_reader_release(run_id, checksum=None):
    releases = ((COMBINED_ARTICLE_RECEIPT, COMBINED_ARTICLE_ROUTE), (OVERVIEW_HTTP_RECEIPT, OVERVIEW_HTTP_ROUTE))
    for relative, route in releases:
        path = gradio.REPO / relative
        if path.is_file():
            receipt = worker.read_json(path)
            if (receipt['run_id'] == run_id and (checksum is None or worker.digest(path) == checksum)
                    and receipt['phase'] == 'finished' and receipt['release_verified'] is True
                    and receipt['stages']['article-route']['verified'] is True):
                return path, receipt, route
    raise ValueError('Expected a published and released article verification run')


def combined_reader_member_hashes(released, route):
    plan = released[route]
    if route == OVERVIEW_HTTP_ROUTE:
        base = worker.read_json(gradio.REPO / COMBINED_ARTICLE_RECEIPT)
        return {**combined_reader_member_hashes(base, COMBINED_ARTICLE_ROUTE), **plan['replaced_source_sha256']}
    return {**plan['preserved_input_sha256'], GRADIO_APP: plan['app_sha256'], ARTICLE_ROUTE_VERIFIER: plan['source_code_sha256']}


def prepare_word_holdout(root, reference_job, output, study):
    stimuli, plan = word_holdout_inputs(root, study)
    if study != 'medreadme':
        files, original, candidate_id, lexical_sha256, provenance = verified_combined_reader_packet(
            reference_job, include_forecast=True)
    else:
        files, original, run_id, report_sha256, lexical_sha256 = verified_gradio_packet(
            reference_job, MEDREADME_APP_SOURCE, include_forecast=True)
        if any(files[name] != (gradio.REPO / name).read_bytes() for name in MEDREADME_APP_SOURCE):
            raise ValueError('The current Gradio inference source differs from the route verified on V8')
        candidate_id = original['checkpoint_sha256']
        provenance = {'verified_gradio_run_id': run_id, 'verified_gradio_sha256': report_sha256}
    if (plan['candidate_id'], plan['lexical_weights_sha256']) != (candidate_id, lexical_sha256):
        raise ValueError('Word holdout is registered against a different served checkpoint or lexical head')
    exporter = (gradio.REPO / MEDREADME_EXPORTER).read_bytes()
    files[MEDREADME_EXPORTER] = exporter
    closure = source_closure(reference_job, files, 'gradio')
    files[worker.MEDREADME_STIMULI] = stimuli
    hashes = {name: hashlib.sha256(value).hexdigest() if isinstance(value, bytes) else worker.digest(value)
              for name, value in files.items()}
    plan.update(provenance, forecast_code_sha256=hashes[MEDREADME_EXPORTER])
    job = new_job(output)
    archive = job / worker.MEDREADME_ARCHIVE
    write_archive(archive, files)
    worker.unpack(archive, hashes)
    bundle = {field: original[field] for field in BACKBONE_FIELDS}
    bundle['requirements'] = [item for item in bundle['requirements'] if not item.startswith('gradio==')]
    bundle.update(medreadme=plan, source_closure={'medreadme': closure})
    seal_job(job, {worker.MEDREADME_ARCHIVE: (archive, 'medreadme', hashes)}, bundle, 12, worker.MEDREADME_SECONDS,
             medreadme=plan)


def combined_article_provenance():
    path = gradio.REPO / REFERENCE_ROUTE_RECEIPT
    released = worker.read_json(path)
    old_article = worker.read_json(gradio.REPO / ARTICLE_ROUTE_RECEIPT)['article_route']
    reference = released['reference_route']
    provenance = {
        **{key: old_article[key] for key in ('candidate_id', 'verified_gradio_run_id', 'verified_gradio_sha256',
                                            'fixture_sha256', 'characters', 'paragraphs')},
        'reference_route_run_id': released['run_id'],
        'reference_route_payload_sha256': released['payload_sha256'],
        'reference_route_archive_sha256': released['payload_members_sha256'][worker.ARTICLE_ROUTE_ARCHIVE],
        'reference_route_receipt_sha256': worker.digest(path),
        'reference_route_app_sha256': reference['app_sha256'],
    }
    return released, provenance


def prepare_article_route(v8_job, reference_job, output):
    from research.gradio_app.verify_article_route import ARTICLE, PARAGRAPHS, text_digest

    v8_files, v8, run_id, report_sha256, _ = verified_gradio_packet(v8_job, ARTICLE_ROUTE_APP_SOURCE)
    released, provenance = combined_article_provenance()
    baseline = worker.read_json(manifest_path(reference_job))
    verify(reference_job, baseline)
    recorded = ('run_id', 'schema_version', 'phase', 'release_verified', 'payload_sha256', 'worker_sha256',
                'payload_members_sha256', 'reference_route', 'stages')
    if (any(baseline[key] != released[key] for key in recorded) or baseline['schema_version'] != 13
            or baseline['phase'] != 'finished' or baseline['release_verified'] is not True
            or any(baseline['stages'][stage]['verified'] is not True for stage in worker.ARTICLE_ROUTE_STAGES)
            or baseline['reference_route']['candidate_id'] != provenance['candidate_id']
            or v8['checkpoint_sha256'] != provenance['candidate_id']
            or (run_id, report_sha256) != (provenance['verified_gradio_run_id'], provenance['verified_gradio_sha256'])
            or (text_digest(ARTICLE), len(ARTICLE), len(PARAGRAPHS))
            != (provenance['fixture_sha256'], provenance['characters'], provenance['paragraphs'])):
        raise ValueError('Expected the released reference route, original V8 checkpoint and fixed article fixture')
    reference_bundle = worker.read_json(reference_job / 'bundle.json')
    if any(reference_bundle[field] != v8[field] for field in BACKBONE_FIELDS):
        raise ValueError('Released reference and V8 routes use different serving environments')
    with tarfile.open(reference_job / worker.ARTICLE_ROUTE_ARCHIVE, 'r:gz') as package:
        files = {member.name: package.extractfile(member).read() for member in package}
    if any(files[name] != content for name, content in v8_files.items()
           if name not in {GRADIO_APP, REFERENCE_ROUTE_COMBINED}):
        raise ValueError('Released reference route changed the verified V8 checkpoint or shared source')
    replacements = {GRADIO_APP, ARTICLE_ROUTE_VERIFIER}
    preserved = files.keys() - replacements
    if any(files[name] != (gradio.REPO / name).read_bytes() for name in preserved if name.endswith('.py')):
        raise ValueError('The current serving dependencies differ from the released reference route')
    files.update({name: (gradio.REPO / name).read_bytes() for name in replacements})
    closure = source_closure(reference_job, files, 'gradio')
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}
    plan = {**provenance, 'source_code_sha256': hashes[ARTICLE_ROUTE_VERIFIER], 'app_sha256': hashes[GRADIO_APP],
            'preserved_input_sha256': {name: hashes[name] for name in preserved}}
    validate_combined_article_binding(plan, hashes)
    job = new_job(output)
    archive = job / worker.ARTICLE_ROUTE_ARCHIVE
    write_archive(archive, files)
    bundle = {field: v8[field] for field in BACKBONE_FIELDS}
    bundle.update({COMBINED_ARTICLE_ROUTE: plan, 'source_closure': {'article-route': closure}})
    seal_job(job, {worker.ARTICLE_ROUTE_ARCHIVE: (archive, 'article-route', hashes)}, bundle, 13,
             worker.ARTICLE_ROUTE_SECONDS, **{COMBINED_ARTICLE_ROUTE: plan})


def validate_combined_article_binding(plan, hashes):
    released, provenance = combined_article_provenance()
    reference = released['reference_route']
    changed = {'app_sha256': GRADIO_APP, 'source_code_sha256': ARTICLE_ROUTE_VERIFIER}
    preserved = {name: value for name, value in hashes.items() if name not in changed.values()}
    if (plan.keys() != provenance.keys() | changed.keys() | {'preserved_input_sha256'}
            or any(plan[key] != value for key, value in provenance.items())
            or any(plan[key] != hashes[name] for key, name in changed.items())
            or plan['app_sha256'] == provenance['reference_route_app_sha256']
            or plan['preserved_input_sha256'] != preserved
            or any(preserved[name] != reference[key] for key, name in REFERENCE_ROUTE_IDENTITY.items()
                   if name not in changed.values())):
        raise ValueError('Combined article packet differs from the registered replacement and reference provenance')


def validate_overview_http_binding(plan, members):
    receipt = gradio.REPO / COMBINED_ARTICLE_RECEIPT
    base = worker.read_json(receipt)
    expected_metadata = {
        'candidate_id': base[COMBINED_ARTICLE_ROUTE]['candidate_id'],
        'base_run_id': base['run_id'], 'base_receipt_sha256': worker.digest(receipt),
        'fixture_sha256': OVERVIEW_HTTP_FIXTURE_SHA256,
    }
    if (base['release_verified'] is not True or base['stages']['article-route']['verified'] is not True
            or plan.keys() != expected_metadata.keys() | {'replaced_source_sha256'}
            or any(plan[key] != expected for key, expected in expected_metadata.items())
            or plan['replaced_source_sha256'].keys() != OVERVIEW_HTTP_NEW_SOURCES
            or members != {**combined_reader_member_hashes(base, COMBINED_ARTICLE_ROUTE),
                           **plan['replaced_source_sha256']}):
        raise ValueError('Overview HTTP package must contain the verified app checkpoint and exact changed source files')


def prepare_overview_http(base_job, output):
    data = worker.read_json(manifest_path(base_job))
    verify(base_job, data)
    receipt, base, route = combined_reader_release(data['run_id'])
    pins = ('run_id', 'payload_sha256', 'payload_members_sha256', 'stages', COMBINED_ARTICLE_ROUTE)
    if (route != COMBINED_ARTICLE_ROUTE or data['phase'] != 'finished' or data['release_verified'] is not True
            or any(data[key] != base[key] for key in pins)):
        raise ValueError('Use the published, released combined article run as the overview HTTP source')
    original = worker.read_json(base_job / 'bundle.json')
    with tarfile.open(base_job / worker.ARTICLE_ROUTE_ARCHIVE, 'r:gz') as package:
        files = {member.name: package.extractfile(member).read() for member in package}
    for name in files.keys() - OVERVIEW_HTTP_NEW_SOURCES:
        if name.endswith('.py') and (gradio.REPO / name).read_bytes() != files[name]:
            raise ValueError(f'Unregistered source change in overview HTTP dependencies: {name}')
    files.update({name: (gradio.REPO / name).read_bytes() for name in OVERVIEW_HTTP_NEW_SOURCES})
    closure = source_closure(base_job, files, 'gradio')
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}
    plan = {'candidate_id': base[route]['candidate_id'], 'base_run_id': base['run_id'],
            'base_receipt_sha256': worker.digest(receipt), 'fixture_sha256': OVERVIEW_HTTP_FIXTURE_SHA256,
            'replaced_source_sha256': {name: hashes[name] for name in OVERVIEW_HTTP_NEW_SOURCES}}
    validate_overview_http_binding(plan, hashes)
    job = new_job(output)
    archive = job / worker.ARTICLE_ROUTE_ARCHIVE
    write_archive(archive, files)
    bundle = {field: original[field] for field in BACKBONE_FIELDS}
    bundle.update({OVERVIEW_HTTP_ROUTE: plan, 'source_closure': {'article-route': closure}})
    seal_job(job, {archive.name: (archive, 'article-route', hashes)}, bundle, 13, worker.ARTICLE_ROUTE_SECONDS,
             **{OVERVIEW_HTTP_ROUTE: plan})


def reference_fixture_digest(verifier):
    assignments = [node.value for node in ast.parse(verifier).body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'ARTICLE' for target in node.targets)]
    if len(assignments) != 1:
        raise ValueError('The reference verifier must declare one fixed article fixture')
    article = ast.literal_eval(assignments[0])
    if type(article) is not str or not article.strip():
        raise ValueError('The reference verifier fixture must be literal article text')
    return hashlib.sha256(article.encode('utf-8')).hexdigest()


def validate_word_cue_binding(plan, members, verifier):
    receipt = gradio.REPO / OVERVIEW_HTTP_RECEIPT
    base = worker.read_json(receipt)
    identity = {
        'candidate_id': base[OVERVIEW_HTTP_ROUTE]['candidate_id'],
        'base_run_id': base['run_id'], 'base_receipt_sha256': worker.digest(receipt),
        'fixtures': {'plain': reference_fixture_digest(verifier), 'unicode-crlf': OVERVIEW_HTTP_FIXTURE_SHA256},
        'cue_head_id': WORD_CUE_RELEASE_SHA256,
    }
    if (base['phase'] != 'finished' or base['release_verified'] is not True
            or base['stages']['article-route']['verified'] is not True
            or plan.keys() != identity.keys() | {'source_sha256'}
            or any(plan[key] != value for key, value in identity.items())
            or plan['source_sha256'].keys() != WORD_CUE_NEW_SOURCES
            or members != {**combined_reader_member_hashes(base, OVERVIEW_HTTP_ROUTE), **plan['source_sha256'],
                           WORD_CUE_RELEASE: WORD_CUE_RELEASE_SHA256}):
        raise ValueError('The word-cue route differs from the verified overview, cue readout or bound fixture code')


def prepare_word_cue_route(overview_job, output):
    data = worker.read_json(manifest_path(overview_job))
    verify(overview_job, data)
    receipt, base, route = combined_reader_release(data['run_id'])
    pins = ('run_id', 'payload_sha256', 'payload_members_sha256', 'stages', OVERVIEW_HTTP_ROUTE)
    if (route != OVERVIEW_HTTP_ROUTE or data['phase'] != 'finished' or data['release_verified'] is not True
            or any(data[key] != base[key] for key in pins)):
        raise ValueError('Use the published and released overview HTTP job as the word-cue baseline')
    original = worker.read_json(overview_job / 'bundle.json')
    with tarfile.open(overview_job / worker.ARTICLE_ROUTE_ARCHIVE, 'r:gz') as package:
        files = {member.name: package.extractfile(member).read() for member in package}
    for name in files.keys() - WORD_CUE_NEW_SOURCES:
        if name.endswith('.py') and (gradio.REPO / name).read_bytes() != files[name]:
            raise ValueError(f'Unregistered source change in the word-cue route: {name}')
    files.update({name: (gradio.REPO / name).read_bytes() for name in WORD_CUE_NEW_SOURCES | {WORD_CUE_RELEASE}})
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}
    plan = {'candidate_id': base[route]['candidate_id'], 'base_run_id': base['run_id'],
            'base_receipt_sha256': worker.digest(receipt),
            'fixtures': {'plain': reference_fixture_digest(files[WORD_CUE_VERIFIER]),
                         'unicode-crlf': OVERVIEW_HTTP_FIXTURE_SHA256},
            'cue_head_id': WORD_CUE_RELEASE_SHA256,
            'source_sha256': {name: hashes[name] for name in WORD_CUE_NEW_SOURCES}}
    validate_word_cue_binding(plan, hashes, files[WORD_CUE_VERIFIER])
    closure = source_closure(overview_job, files, 'word-cues')
    job = new_job(output)
    archive = job / worker.ARTICLE_ROUTE_ARCHIVE
    write_archive(archive, files)
    bundle = {field: original[field] for field in BACKBONE_FIELDS}
    bundle['requirements'] = [*bundle['requirements'], WORD_CUE_SCIPY]
    bundle.update({WORD_CUE_ROUTE: plan, 'source_closure': {'article-route': closure}})
    seal_job(job, {archive.name: (archive, 'article-route', hashes)}, bundle, 13, worker.ARTICLE_ROUTE_SECONDS,
             **{WORD_CUE_ROUTE: plan})


def prepare_reference_route(article_job, output):
    source = worker.read_json(manifest_path(article_job))
    verify(article_job, source)
    released = worker.read_json(gradio.REPO / ARTICLE_ROUTE_RECEIPT)
    stages = ('setup', 'article-route')
    fields = ('run_id', 'schema_version', 'phase', 'release_verified', 'payload_sha256',
              'payload_members_sha256', 'article_route', 'stages')
    if (any(source.get(key) != released.get(key) for key in fields) or source['schema_version'] != 13
            or source['phase'] != 'finished' or source['release_verified'] is not True
            or any(source['stages'][stage]['verified'] is not True for stage in stages)):
        raise ValueError('Expected the published, released V13 article-route job')
    v13 = worker.read_json(article_job / 'bundle.json')
    with tarfile.open(article_job / worker.ARTICLE_ROUTE_ARCHIVE, 'r:gz') as original:
        files = {member.name: original.extractfile(member).read() for member in original}
    files.update({name: (gradio.REPO / name).read_bytes() for name in REFERENCE_ROUTE_SOURCE | REFERENCE_ROUTE_ARTIFACTS})
    # Leave the old verifier in the original archive; the remote worker keeps its existing module entry point.
    files[ARTICLE_ROUTE_VERIFIER] = REFERENCE_ROUTE_SHIM
    closure = source_closure(article_job, files, 'gradio')
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}
    plan = {'candidate_id': v13['article_route']['candidate_id'],
            'previous_route_run_id': source['run_id'],
            'previous_route_payload_sha256': source['payload_sha256'],
            'previous_route_archive_sha256': source['payload_members_sha256'][worker.ARTICLE_ROUTE_ARCHIVE],
            'fixture_sha256': reference_fixture_digest(files[REFERENCE_ROUTE_VERIFIER]),
            **{key: hashes[name] for key, name in REFERENCE_ROUTE_IDENTITY.items()}}
    validate_reference_binding(plan, hashes, {name: json.loads(files[name]) for name in REFERENCE_ROUTE_ARTIFACTS})
    job = new_job(output)
    archive = job / worker.ARTICLE_ROUTE_ARCHIVE
    write_archive(archive, files)
    bundle = {field: v13[field] for field in BACKBONE_FIELDS}
    bundle.update(reference_route=plan, source_closure={'article-route': closure})
    seal_job(job, {worker.ARTICLE_ROUTE_ARCHIVE: (archive, 'article-route', hashes)}, bundle, 13,
             worker.ARTICLE_ROUTE_SECONDS, reference_route=plan)


def validate_reference_binding(plan, hashes, artifacts):
    reference = artifacts[REFERENCE_ROUTE_RELEASE]
    fit, scores = artifacts[REFERENCE_ROUTE_FIT], artifacts[REFERENCE_ROUTE_SCORES]
    projection = reference['projection_identity']
    if (any(plan[key] != hashes[name] for key, name in REFERENCE_ROUTE_IDENTITY.items())
            or hashes[ARTICLE_ROUTE_VERIFIER] != hashlib.sha256(REFERENCE_ROUTE_SHIM).hexdigest()
            or reference['fit_sha256'] != hashes[REFERENCE_ROUTE_FIT]
            or reference['scores_sha256'] != hashes[REFERENCE_ROUTE_SCORES]
            or scores['fit_sha256'] != hashes[REFERENCE_ROUTE_FIT] or scores['pear_primary']['pass'] is not True
            or fit['projection_identity'] != projection or projection['candidate_id'] != plan['candidate_id']
            or projection['anchor_heads_sha256'] != hashes[f'{worker.GRADIO_CHECKPOINT}/anchor/heads.npz']
            or (projection['model'], projection['revision']) != (gradio.MODEL, gradio.REVISION)):
        raise ValueError('Reference route differs from the sealed ARRAU readout and matching candidate')


def prepare_onestop(assay_root, preflight_path, v8_job, output):
    from research.stage6.onestop_rereading_extract import TOKENIZER_SHA256, read_stimuli

    assay_path = assay_root / 'assay.json'
    assay = worker.read_json(assay_path)
    stimulus = assay_root / 'stimuli.jsonl'
    preflight = worker.read_json(preflight_path)
    documents = read_stimuli(stimulus, assay['file_sha256']['stimuli.jsonl'])
    if (preflight['stimuli_sha256'] != assay['file_sha256']['stimuli.jsonl']
            or preflight['tokenizer_sha256'] != TOKENIZER_SHA256
            or not preflight['documents'] == assay['counts']['documents'] == len(documents)
            or not preflight['stops'] == assay['counts']['stops'] == sum(len(doc.stops) for doc in documents)
            or not re.fullmatch(r'[0-9a-f]{64}', preflight['positions_sha256'])
            or not 0 < preflight['input_tokens'] <= len(documents) * 8192):
        raise ValueError('OneStop tokenization preflight does not match the sealed stimuli')
    expected_files = {'stimuli.jsonl', 'development.jsonl', 'holdout.jsonl'}
    if (assay['file_sha256'].keys() != expected_files
            or any(worker.digest(assay_root / name) != sha for name, sha in assay['file_sha256'].items())):
        raise ValueError('OneStop assay inputs changed after source preparation')
    files, v8, run_id, report_sha256, _ = verified_gradio_packet(v8_job, MEDREADME_APP_SOURCE)
    if any(files[name] != (gradio.REPO / name).read_bytes() for name in MEDREADME_APP_SOURCE):
        raise ValueError('The current Qwen backend differs from the served V8 route')
    files[ONESTOP_EXTRACTOR] = (gradio.REPO / ONESTOP_EXTRACTOR).read_bytes()
    closure = source_closure(v8_job, files, 'gradio')
    files[worker.ONESTOP_STIMULI] = stimulus.read_bytes()
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}
    plan = {**preflight, 'candidate_id': v8['checkpoint_sha256'], 'verified_gradio_run_id': run_id,
            'verified_gradio_sha256': report_sha256, 'protocol_sha256': worker.digest(gradio.REPO / ONESTOP_PROTOCOL),
            'assay_sha256': worker.digest(assay_path), 'source_code_sha256': hashes[ONESTOP_EXTRACTOR]}
    job = new_job(output)
    archive = job / worker.ONESTOP_ARCHIVE
    write_archive(archive, files)
    bundle = {field: v8[field] for field in BACKBONE_FIELDS}
    bundle['requirements'] = [item for item in bundle['requirements'] if not item.startswith('gradio==')]
    bundle.update(onestop=plan, source_closure={'onestop': closure})
    seal_job(job, {worker.ONESTOP_ARCHIVE: (archive, 'onestop', hashes)}, bundle, 14, worker.ONESTOP_SECONDS,
             onestop=plan)


def prepare_expmrc(assay_root, preflight_path, v8_job, output):
    from research.semantic_tracking.expmrc_forecast import HEADS_SHA256, material

    assay_path = assay_root / 'assay.json'
    assay = worker.read_json(assay_path)
    preflight = worker.read_json(preflight_path)
    sources = assay['file_sha256']
    articles, queries = material(assay_root / 'articles.jsonl', assay_root / 'queries.jsonl',
                                 sources['articles.jsonl'], sources['queries.jsonl'])
    if (sources.keys() != {'articles.jsonl', 'queries.jsonl', 'outcomes.jsonl'}
            or any(worker.digest(assay_root / name) != value for name, value in sources.items())
            or not preflight['documents'] == assay['article_count'] == len(articles)
            or not preflight['queries'] == assay['query_count'] == len(queries)
            or (preflight['articles_sha256'], preflight['queries_sha256']) !=
               (sources['articles.jsonl'], sources['queries.jsonl'])
            or preflight['candidate_stops'] - preflight['certified_stops'] != preflight['coalesced_stops']):
        raise ValueError('ExpMRC preflight does not match the frozen article, query, and scoring files')
    heads = gradio.REPO / 'research/results/qasper-factorial-run/heads.npz'
    if worker.digest(heads) != HEADS_SHA256:
        raise ValueError('Expected the published QASPER factorial heads')
    files, v8, run_id, report_sha256, _ = verified_gradio_packet(v8_job, MEDREADME_APP_SOURCE)
    if any(files[name] != (gradio.REPO / name).read_bytes() for name in MEDREADME_APP_SOURCE):
        raise ValueError('The current serving code differs from the verified V8 route')
    files.update({name: (gradio.REPO / name).read_bytes() for name in EXPMRC_SOURCE})
    closure = source_closure(v8_job, files, 'expmrc')
    files.update({worker.EXPMRC_ARTICLES: (assay_root / 'articles.jsonl').read_bytes(),
                  worker.EXPMRC_QUERIES: (assay_root / 'queries.jsonl').read_bytes(),
                  worker.EXPMRC_HEADS: heads.read_bytes()})
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}
    plan = {**preflight, 'candidate_id': v8['checkpoint_sha256'], 'verified_gradio_run_id': run_id,
            'verified_gradio_sha256': report_sha256, 'protocol_sha256': worker.digest(gradio.REPO / EXPMRC_PROTOCOL),
            'assay_sha256': worker.digest(assay_path), 'source_code_sha256': hashes[EXPMRC_EXPORTER],
            'heads_sha256': HEADS_SHA256}
    job = new_job(output)
    archive = job / worker.EXPMRC_ARCHIVE
    write_archive(archive, files)
    bundle = {field: v8[field] for field in BACKBONE_FIELDS}
    bundle['requirements'] = [item for item in bundle['requirements'] if not item.startswith('gradio==')]
    bundle.update(expmrc=plan, source_closure={'expmrc': closure})
    seal_job(job, {worker.EXPMRC_ARCHIVE: (archive, 'expmrc', hashes)}, bundle, 15, worker.EXPMRC_SECONDS,
             expmrc=plan)


def prepare_contractnli(packet, assay, reference_job, output, experiment):
    from research.semantic_tracking.expmrc_forecast import HEADS_SHA256

    reference = worker.read_json(gradio.REPO / CONTRACTNLI_REFERENCE)
    published = worker.read_json(reference_job / 'bundle.json')
    packet_archive = reference_job / worker.EXPMRC_ARCHIVE
    expected = published['archives'][packet_archive.name]['members']
    if (reference_job.name != reference['run_id'] or reference['release_verified'] is not True
            or worker.digest(reference_job / 'bundle.json') != reference['payload_members_sha256']['bundle.json']
            or worker.digest(packet_archive) != reference['payload_members_sha256'][packet_archive.name]
            or expected[worker.EXPMRC_HEADS] != HEADS_SHA256):
        raise ValueError('The contract evidence run needs the verified ExpMRC readout and backbone bundle')
    worker.unpack(packet_archive, expected)
    with tarfile.open(packet_archive, 'r:gz') as previous:
        files = {name: previous.extractfile(name).read()
                 for name in expected.keys() - {worker.EXPMRC_ARTICLES, worker.EXPMRC_QUERIES}}
    exporter = worker.CONTRACTNLI_MODULES[experiment].replace('.', '/') + '.py'
    for source in {CONTRACTNLI_EXPORTER, exporter}:
        files[source] = (gradio.REPO / source).read_bytes()
    closure = source_closure(reference_job, files, experiment)
    files.update({name: (packet / Path(name).name).read_bytes() for name in CONTRACTNLI_INPUTS})
    hashes = {name: hashlib.sha256(contents).hexdigest() for name, contents in files.items()}
    preflight = worker.read_json(packet / 'preflight.json')
    if experiment == EVIDENCE_INFERENCE:
        audited = preflight['source_audit_sha256'] == worker.digest(assay)
    else:
        audited = preflight['source_sha256'] == {name: sha for name, sha in worker.read_json(assay)['output_sha256'].items()
                                                 if name in {'stimuli.jsonl', 'layout.jsonl', 'queries.jsonl'}}
    if not audited:
        raise ValueError('ContractNLI preflight must come from the audited public holdout')
    plan = {'experiment': experiment, 'candidate_id': published['expmrc']['candidate_id'],
            'reference_run_id': reference['run_id'], 'reference_bundle_sha256': worker.digest(reference_job / 'bundle.json'),
            'assay_sha256': worker.digest(assay), 'protocol_sha256': worker.digest(gradio.REPO / CONTRACTNLI_PROTOCOLS[experiment]),
            'heads_sha256': HEADS_SHA256, 'source_code_sha256': hashes[exporter],
            'packet_sha256': {Path(name).name: hashes[name] for name in CONTRACTNLI_INPUTS}}
    job = new_job(output)
    archive = job / worker.EXPMRC_ARCHIVE
    write_archive(archive, files)
    bundle = {field: published[field] for field in BACKBONE_FIELDS}
    bundle.update(expmrc=plan, source_closure={'expmrc': closure})
    seal_job(job, {worker.EXPMRC_ARCHIVE: (archive, 'expmrc', hashes)}, bundle, 15, worker.EXPMRC_SECONDS,
             expmrc=plan)


def prepare_nq_evidence(assay_root, preflight_path, snapshot, v8_job, output):
    from research.semantic_tracking.nq_evidence_prepare import CANDIDATE_INDEX_SHA256, CHECKPOINT, DATASET, SOURCE_REVISION

    assay_path = assay_root / 'assay.json'
    assay = worker.read_json(assay_path)
    preflight = worker.read_json(preflight_path)
    sources = assay['file_sha256']
    if (assay['dataset'] != DATASET or assay['source_revision'] != SOURCE_REVISION
            or assay['candidate_index_sha256'] != CANDIDATE_INDEX_SHA256 or assay['checkpoint'] != CHECKPOINT
            or sources.keys() != {'articles.jsonl', 'queries.jsonl', 'span-maps.jsonl', 'preflight.json'}
            or worker.digest(preflight_path) != sources['preflight.json']
            or worker.digest(assay_root / 'span-maps.jsonl') != sources['span-maps.jsonl']
            or assay['security'] != {'remote_inputs': ['articles.jsonl', 'queries.jsonl'],
                                    'local_geometry_only': ['span-maps.jsonl'], 'reference_escrow_opened': False}):
        raise ValueError('Wrong Natural Questions source, grade-free geometry or release plan')
    prepare_evidence_transfer(assay_root, snapshot, v8_job, output, preflight, assay, CHECKPOINT, 18)


def prepare_qasper_tail(assay_root, preflight_path, snapshot, v8_job, output):
    from research.semantic_tracking.nq_evidence_prepare import CHECKPOINT

    assay = worker.read_json(assay_root / 'assay.json')
    preflight = worker.read_json(preflight_path)
    if (worker.digest(preflight_path) != assay['file_sha256']['preflight.json']
            or any(worker.digest(assay_root / name) != assay['file_sha256'][name]
                   for name in ('development.jsonl', 'holdout.jsonl'))):
        raise ValueError('QASPER tail coordinates and local outcomes must match the frozen assay')
    prepare_evidence_transfer(assay_root, snapshot, v8_job, output, preflight, assay, CHECKPOINT, 19)


def prepare_evidence_transfer(assay_root, snapshot, v8_job, output, preflight, assay, checkpoint, format_version):
    from research.semantic_tracking.expmrc_forecast import HEADS_SHA256, material
    from research.semantic_tracking.nq_tokenizer_verification import audit_inputs

    stage, key, archive_name, protocol, seconds, runtime_budget = EVIDENCE_SCOPES[format_version]
    sources = assay['file_sha256']
    article_file, query_file = assay_root / 'articles.jsonl', assay_root / 'queries.jsonl'
    articles, queries = material(article_file, query_file, sources[article_file.name], sources[query_file.name])
    if (preflight.keys() != {'articles_sha256', 'queries_sha256', 'documents', 'queries', 'candidate_stops',
                             'certified_stops', 'coalesced_stops', 'input_tokens'}
            or not preflight['documents'] == assay['article_count'] == len(articles)
            or not preflight['queries'] == assay['query_count'] == len(queries)
            or (preflight['articles_sha256'], preflight['queries_sha256']) !=
               (sources[article_file.name], sources[query_file.name])
            or preflight['candidate_stops'] - preflight['certified_stops'] != preflight['coalesced_stops']
            or not 0 < preflight['input_tokens'] <= runtime_budget
            or any(len(article['text']) > 16_000 for article in articles)):
        raise ValueError(f'{stage} preflight differs from the fixed text, questions or compute budget')
    heads = gradio.REPO / 'research/results/qasper-factorial-run/heads.npz'
    if (worker.digest(heads) != HEADS_SHA256
            or worker.digest(gradio.REPO / EVIDENCE_CALIBRATION) != EVIDENCE_CALIBRATION_SHA256):
        raise ValueError('Expected the original QASPER and independently fitted calibration parameters')
    files, v8, run_id, report_sha256, _ = verified_gradio_packet(v8_job, MEDREADME_APP_SOURCE)
    if (v8['checkpoint_sha256'] != checkpoint
            or any(files[name] != (gradio.REPO / name).read_bytes() for name in MEDREADME_APP_SOURCE)):
        raise ValueError('Expected the original candidate serving code')
    files.update({name: (gradio.REPO / name).read_bytes() for name in EXPMRC_SOURCE})
    closure = source_closure(v8_job, files, 'expmrc')
    files.update({worker.EXPMRC_ARTICLES: article_file.read_bytes(), worker.EXPMRC_QUERIES: query_file.read_bytes(),
                  worker.EXPMRC_HEADS: heads.read_bytes()})
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}
    plan = {**preflight, 'candidate_id': v8['checkpoint_sha256'], 'verified_gradio_run_id': run_id,
            'verified_gradio_sha256': report_sha256, 'protocol_sha256': worker.digest(gradio.REPO / protocol),
            'assay_sha256': worker.digest(assay_root / 'assay.json'), 'source_code_sha256': hashes[EXPMRC_EXPORTER],
            'heads_sha256': HEADS_SHA256, 'calibration_sha256': EVIDENCE_CALIBRATION_SHA256}
    tokenizer_audit, _ = audit_inputs(assay_root, snapshot, gradio.REPO / worker.GRADIO_CHECKPOINT, plan,
                                      runtime_budget=runtime_budget)
    plan['runtime_tokens'] = tokenizer_audit['runtime_extraction_tokens']
    plan['runtime_stimuli_sha256'] = tokenizer_audit['reconstructed_stimuli_sha256']
    job = new_job(output)
    archive = job / archive_name
    write_archive(archive, files)
    bundle = {field: v8[field] for field in BACKBONE_FIELDS}
    bundle['requirements'] = [item for item in bundle['requirements'] if not item.startswith('gradio==')]
    bundle.update({key: plan, 'source_closure': {stage: closure}})
    seal_job(job, {archive.name: (archive, stage, hashes)}, bundle, format_version, seconds, **{key: plan})


def prepare_gum_bridge(assay_root, preflight_path, v8_job, output):
    files, v8, run_id, report_sha256, _ = verified_gradio_packet(v8_job, MEDREADME_APP_SOURCE)
    if any(files[name] != (gradio.REPO / name).read_bytes() for name in MEDREADME_APP_SOURCE):
        raise ValueError('Expected the original candidate backbone sources')
    files.update({name: (gradio.REPO / name).read_bytes() for name in GUM_BRIDGE_SOURCE - MEDREADME_APP_SOURCE})
    closure = source_closure(v8_job, files, 'gradio')
    files.update({name: (assay_root / name).read_bytes() for name in GUM_BRIDGE_INPUTS})
    files['preflight.json'] = preflight_path.read_bytes()
    member_hashes = {name: hashlib.sha256(contents).hexdigest() for name, contents in files.items()}
    preflight = json.loads(files['preflight.json'])
    if (preflight['input_sha256'] != {Path(name).name: member_hashes[name] for name in GUM_BRIDGE_INPUTS}
            or preflight['input_documents'] != preflight['encoded_documents']
            or not 0 < preflight['encoded_tokens'] <= 350_000
            or not 0 < preflight['certified_stops'] <= 300_000):
        raise ValueError('Unexpected GUMBridge inputs or compute inventory')
    plan = {'candidate_id': v8['checkpoint_sha256'], 'verified_gradio_run_id': run_id,
            'verified_gradio_sha256': report_sha256, 'protocol_sha256': worker.digest(gradio.REPO / GUM_BRIDGE_PROTOCOL),
            'encoded_documents': preflight['encoded_documents'], 'encoded_tokens': preflight['encoded_tokens'],
            'certified_stops': preflight['certified_stops'],
            'preflight_sha256': member_hashes['preflight.json'], 'source_code_sha256': member_hashes[GUM_BRIDGE_EXPORTER]}
    job = new_job(output)
    archive = job / worker.GUM_BRIDGE_ARCHIVE
    write_archive(archive, files)
    bundle = {field: v8[field] for field in BACKBONE_FIELDS}
    bundle['requirements'] = [item for item in bundle['requirements'] if not item.startswith('gradio==')]
    bundle.update(gum_bridge=plan, source_closure={'gum-bridge': closure})
    seal_job(job, {archive.name: (archive, 'gum-bridge', member_hashes)}, bundle, 20,
             worker.GUM_BRIDGE_SECONDS, gum_bridge=plan)


def prepare_projection(assay_root, protocol, preflight_path, reference_job, output, *, stage='projection'):
    from research.gradio_app.causal_plaba import HIDDEN_WIDTH
    from research.semantic_tracking.beukers_extract import read_stimuli, validate_preflight

    schema, archive_name, seconds = PROJECTION_SCOPES[stage]
    scope = next((scope for name, scope in READER_PROJECTION_ASSAYS.items()
                  if protocol.resolve() == gradio.REPO / name), None)
    stimulus = assay_root / worker.PROJECTION_STIMULI
    assay = assay_root / (scope[0] if scope else 'assay.json')
    stimulus_sha = worker.digest(stimulus)
    if worker.read_json(assay)['file_sha256'][worker.PROJECTION_STIMULI] != stimulus_sha:
        raise ValueError('Grade-free stimuli differ from the sealed local assay')
    preflight = worker.read_json(preflight_path)
    rows = read_stimuli(stimulus)
    validate_preflight(preflight, rows, stimulus_sha)
    if stage == 'projection':
        from research.semantic_tracking.beukers_extract import PROJECTION_WIDTH

        vector_bytes = (preflight['certified_stops'] * PROJECTION_WIDTH * 2
                        + preflight['encoded_documents'] * HIDDEN_WIDTH * 4)
        if any(row['stops'][-1] != len(row['text']) for row in rows) or vector_bytes > worker.archive_budget(stage):
            raise ValueError('Projection inputs need exact last stops and returned arrays within the archive budget')
    if scope:
        _, assay_sha256, stops, inventory, certified_sha256 = scope
        if (stage != 'beukers' or worker.digest(assay) != assay_sha256
                or (certified_sha256 is not None and worker.digest(preflight_path) != certified_sha256)
                or any(preflight[key] != value for key, value in inventory.items())
                or any(len(row['stops']) != stops or row['stops'][-1] != len(row['text']) for row in rows)):
            raise ValueError('The reading assay requires the sealed cue sequences and projection-only route')
        files, original, candidate_id, _, provenance = verified_combined_reader_packet(reference_job)
    else:
        files, original, run_id, report_sha256, _ = verified_gradio_packet(reference_job, MEDREADME_APP_SOURCE)
        if any(files[name] != (gradio.REPO / name).read_bytes() for name in MEDREADME_APP_SOURCE):
            raise ValueError('Expected the original candidate serving source')
        candidate_id = original['checkpoint_sha256']
        provenance = {'verified_gradio_run_id': run_id, 'verified_gradio_sha256': report_sha256}
    files.update({name: (gradio.REPO / name).read_bytes() for name in BEUKERS_SOURCE - MEDREADME_APP_SOURCE})
    closure = source_closure(reference_job, files, 'gradio')
    files.update({worker.PROJECTION_STIMULI: stimulus.read_bytes(), 'preflight.json': preflight_path.read_bytes()})
    hashes = {name: hashlib.sha256(contents).hexdigest() for name, contents in files.items()}
    plan = {'candidate_id': candidate_id, **provenance, 'protocol_sha256': worker.digest(protocol),
            'assay_sha256': worker.digest(assay), 'preflight_sha256': hashes['preflight.json'],
            'stimuli_sha256': hashes[worker.PROJECTION_STIMULI], 'source_code_sha256': hashes[BEUKERS_EXPORTER],
            **{name: preflight[name] for name in ('encoded_documents', 'input_tokens', 'certified_stops')}}
    job = new_job(output)
    archive = job / archive_name
    write_archive(archive, files)
    bundle = {field: original[field] for field in BACKBONE_FIELDS}
    bundle['requirements'] = [item for item in bundle['requirements'] if not item.startswith('gradio==')]
    bundle.update({stage: plan, 'source_closure': {stage: closure}})
    seal_job(job, {archive.name: (archive, stage, hashes)}, bundle, schema, seconds, **{stage: plan})


def prepare_beukers(assay_root, preflight_path, v8_job, output):
    prepare_projection(assay_root, gradio.REPO / BEUKERS_PROTOCOL, preflight_path, v8_job, output, stage='beukers')


def prepare_frick_signals(stimuli, reference_job, output):
    """Reuse the original certified full Frick stories, without standalone controls or grades."""
    archived = gradio.REPO / FRICK_PREVIOUS_PREFLIGHT
    if (worker.digest(stimuli) != FRICK_PREVIOUS_STIMULI_SHA256
            or worker.digest(archived) != FRICK_PREVIOUS_PREFLIGHT_SHA256):
        raise ValueError('Expected the sealed original Frick stimuli and tokenizer certificates')
    source = worker.read_json(archived)
    source_rows = [line for line in stimuli.read_bytes().splitlines(keepends=True) if len(json.loads(line)['stops']) > 1]
    content = b''.join(source_rows)
    documents = [item for item in source['documents'] if len(item['stops']) > 1]
    summary = {**source, 'stimuli_sha256': hashlib.sha256(content).hexdigest(),
               'input_documents': len(documents), 'encoded_documents': len(documents),
               'input_tokens': sum(item['input_tokens'] for item in documents),
               'max_input_tokens': max(item['input_tokens'] for item in documents),
               'requested_stops': sum(len(item['stops']) for item in documents),
               'certified_stops': sum(len(item['stops']) for item in documents), 'documents': documents}
    preflight = (json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    if (summary['stimuli_sha256'] != FRICK_SIGNAL_STIMULI_SHA256
            or hashlib.sha256(preflight).hexdigest() != FRICK_SIGNAL_PREFLIGHT_SHA256):
        raise ValueError('Frick full-story packet differs from the registered study subset')

    files, original, candidate_id, _, provenance = verified_combined_reader_packet(reference_job)
    additional_sources = (BEUKERS_SOURCE - MEDREADME_APP_SOURCE) | {FRICK_SIGNALS_EXPORTER}
    files.update({name: (gradio.REPO / name).read_bytes() for name in additional_sources})
    closure = source_closure(reference_job, files, 'gradio')
    files.update({worker.PROJECTION_STIMULI: content, 'preflight.json': preflight})
    hashes = {name: hashlib.sha256(contents).hexdigest() for name, contents in files.items()}
    plan = {'candidate_id': candidate_id, **provenance, **FRICK_SIGNAL_INVENTORY,
            'protocol_sha256': worker.digest(gradio.REPO / FRICK_SIGNALS_PROTOCOL),
            'assay_sha256': FRICK_PREVIOUS_PREFLIGHT_SHA256, 'preflight_sha256': FRICK_SIGNAL_PREFLIGHT_SHA256,
            'stimuli_sha256': FRICK_SIGNAL_STIMULI_SHA256, 'source_code_sha256': hashes[FRICK_SIGNALS_EXPORTER]}
    job = new_job(output)
    archive = job / worker.FRICK_SIGNALS_ARCHIVE
    write_archive(archive, files)
    bundle = {field: original[field] for field in BACKBONE_FIELDS}
    bundle['requirements'] = [item for item in bundle['requirements'] if not item.startswith('gradio==')]
    bundle.update(frick_signals=plan, source_closure={'frick-signals': closure})
    seal_job(job, {archive.name: (archive, 'frick-signals', hashes)}, bundle, 23, worker.FRICK_SIGNALS_SECONDS,
             frick_signals=plan)


def clear_ease_preparation():
    receipt = worker.read_json(gradio.REPO / CLEAR_EASE_RECEIPT)
    if (receipt['protocol_sha256'] != worker.digest(gradio.REPO / CLEAR_EASE_PROTOCOL)
            or receipt['verification']['model_run'] is not False
            or receipt['verification']['overlapping_source_families'] != 0):
        raise ValueError('Expected the frozen CLEAR material and its independent source split')
    return receipt


def prepare_clear_ease(prepared, reference_job, output):
    from research.stage6.clear_reading_ease_forecast import GEOMETRY, STIMULI, read_rows

    receipt = clear_ease_preparation()
    assay_sha = worker.digest(prepared / 'assay.json')
    if (assay_sha != receipt['verification']['assay_sha256']
            or any(worker.digest(prepared / name) != receipt['file_sha256'][name]
                   for name in (STIMULI, 'materials.jsonl'))):
        raise ValueError('CLEAR input text or coordinates do not match the frozen preparation')
    data = worker.read_json(manifest_path(reference_job))
    verify(reference_job, data)
    reference_receipt, released, route = combined_reader_release(data['run_id'])
    if (route != OVERVIEW_HTTP_ROUTE or data['phase'] != 'finished' or data['release_verified'] is not True
            or any(data[key] != released[key] for key in ('payload_sha256', 'stages', route))):
        raise ValueError('CLEAR requires the released Gradio overview serving checkpoint')
    original = worker.read_json(reference_job / 'bundle.json')
    base_members = original['archives'][worker.ARTICLE_ROUTE_ARCHIVE]['members']
    with tarfile.open(reference_job / worker.ARTICLE_ROUTE_ARCHIVE, 'r:gz') as archive:
        files = {name: archive.extractfile(name).read() for name in base_members}
    files[GRADIO_APP] = (gradio.REPO / GRADIO_APP).read_bytes()
    files[CLEAR_EASE_EXPORTER] = (gradio.REPO / CLEAR_EASE_EXPORTER).read_bytes()
    closure = source_closure(reference_job, files, 'gradio')
    certificates = []
    for stimulus, material in zip(read_rows(prepared / STIMULI), read_rows(prepared / 'materials.jsonl'), strict=True):
        if (set(stimulus) != {'id', 'text', 'stops'} or material['id'] != stimulus['id']
                or stimulus['stops'] != [end for _, end in material['spans']]
                or hashlib.sha256(stimulus['text'].encode()).hexdigest() != stimulus['id']):
            raise ValueError('The CLEAR inference packet must contain only unmodified text and certified coordinates')
        certificates.append({'id': stimulus['id'], 'spans': material['spans'],
                             'tokens': material['sentence_tokens'], 'input_tokens': material['input_tokens']})
    files[STIMULI] = (prepared / STIMULI).read_bytes()
    files[GEOMETRY] = b''.join((json.dumps(row, separators=(',', ':')) + '\n').encode() for row in certificates)
    hashes = {name: hashlib.sha256(contents).hexdigest() for name, contents in files.items()}
    plan = {'candidate_id': receipt['candidate_id'], 'assay_sha256': assay_sha,
            'protocol_sha256': receipt['protocol_sha256'], 'stimuli_sha256': hashes[STIMULI],
            'stops_sha256': hashes[GEOMETRY], 'source_code_sha256': hashes[CLEAR_EASE_EXPORTER],
            'app_sha256': hashes[GRADIO_APP], 'documents': len(certificates),
            'certified_sentences': sum(len(row['spans']) for row in certificates),
            'input_tokens': sum(row['input_tokens'] for row in certificates),
            'combined_route_run_id': data['run_id'], 'combined_route_receipt_sha256': worker.digest(reference_receipt)}
    files['plan.json'] = gradio.json_bytes(plan)
    hashes['plan.json'] = hashlib.sha256(files['plan.json']).hexdigest()
    job = new_job(output)
    path = job / worker.CLEAR_EASE_ARCHIVE
    write_archive(path, files)
    bundle = {field: original[field] for field in BACKBONE_FIELDS}
    bundle.update(clear_ease=plan, source_closure={'clear-ease': closure})
    seal_job(job, {path.name: (path, 'clear-ease', hashes)}, bundle, 24, worker.CLEAR_EASE_SECONDS, clear_ease=plan)


def prepare_first_readback(assay_root, preflight_path, v8_job, output):
    from research.semantic_tracking.first_readback_transfer import STUDIES, load_inputs
    from research.stage6.onestop_rereading_extract import TOKENIZER_SHA256

    assay_path = assay_root / 'assay.json'
    assay = worker.read_json(assay_path)
    source = assay['file_sha256']
    preflight = worker.read_json(preflight_path)
    inputs = {name: assay_root / f'{name}.jsonl' for name in worker.FIRST_READBACK_INPUTS}
    if source.keys() != {path.name for path in inputs.values()} | {'test.jsonl'}:
        raise ValueError('Unexpected source inventory for the first-readback assay')
    hashes = {f'{name}_sha256': source[path.name] for name, path in inputs.items()}
    articles, targets, training = load_inputs(inputs['stimuli'], inputs['targets'], inputs['train'], hashes)
    if (preflight.keys() != {'assay_sha256', 'stimuli_sha256', 'tokenizer_sha256', 'documents', 'stops',
                             'uncertified_stops', 'max_tokens'}
            or preflight['assay_sha256'] != worker.digest(assay_path)
            or preflight['stimuli_sha256'] != hashes['stimuli_sha256']
            or preflight['tokenizer_sha256'] != TOKENIZER_SHA256
            or preflight['documents'] != len(articles)
            or preflight['stops'] != sum(len(row['stops']) for row in articles.values())
            or preflight['uncertified_stops'] != 0 or not 0 < preflight['max_tokens'] <= 6144):
        raise ValueError('First-readback stimuli differ from their independently certified boundaries')
    for study in STUDIES:
        counts = assay['counts'][study]
        observed = [row for row in training if row['study'] == study]
        if (len(observed) != counts['train']['observed_stops']
                or any(sum(row[label] for row in observed) != counts['train'][label]
                       for label in ('negative', 'positive'))
                or any(sum(row['study'] == study and row['split'] == split for row in targets)
                       != counts[split]['designed_stops'] for split in ('train', 'test'))):
            raise ValueError(f'First-readback source counts changed for {study}')
    files, v8, run_id, report_sha256, _ = verified_gradio_packet(v8_job, MEDREADME_APP_SOURCE)
    if any(files[name] != (gradio.REPO / name).read_bytes() for name in MEDREADME_APP_SOURCE):
        raise ValueError('The current serving components differ from the verified checkpoint package')
    files.update({name: (gradio.REPO / name).read_bytes() for name in FIRST_READBACK_SOURCE - MEDREADME_APP_SOURCE})
    closure = source_closure(v8_job, files, 'gradio')
    files.update({member: inputs[name].read_bytes() for name, member in worker.FIRST_READBACK_INPUTS.items()})
    member_hashes = {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}
    plan = {**preflight, **hashes, 'training_stops': len(training),
            'heldout_stops': sum(row['split'] == 'test' for row in targets), 'candidate_id': v8['checkpoint_sha256'],
            'verified_gradio_run_id': run_id, 'verified_gradio_sha256': report_sha256,
            'protocol_sha256': worker.digest(gradio.REPO / FIRST_READBACK_PROTOCOL),
            'source_code_sha256': member_hashes[FIRST_READBACK_EXPORTER]}
    job = new_job(output)
    archive = job / worker.FIRST_READBACK_ARCHIVE
    write_archive(archive, files)
    bundle = {field: v8[field] for field in BACKBONE_FIELDS}
    bundle['requirements'] = [item for item in bundle['requirements'] if not item.startswith('gradio==')]
    bundle.update(first_readback=plan, source_closure={'first-readback': closure})
    seal_job(job, {worker.FIRST_READBACK_ARCHIVE: (archive, 'first-readback', member_hashes)}, bundle, 16,
             worker.FIRST_READBACK_SECONDS, first_readback=plan)


def prepare_context_readback(assay_root, preflight_path, v8_job, output):
    from research.semantic_tracking.context_readback_forecast import CANDIDATE, HEADS_SHA256, read_stimuli
    from research.stage6.onestop_rereading_extract import TOKENIZER_SHA256

    assay_path = assay_root / 'assay.json'
    assay = worker.read_json(assay_path)
    hashes = assay['file_sha256']
    if (hashes.keys() != {'stimuli.jsonl', 'targets.jsonl'}
            or any(worker.digest(assay_root / name) != sha for name, sha in hashes.items())):
        raise ValueError('Independent readback artifacts differ from their outcome-free assay')
    stimulus = assay_root / 'stimuli.jsonl'
    preflight = worker.read_json(preflight_path)
    articles = read_stimuli(stimulus, hashes[stimulus.name])
    if (preflight.keys() != {'assay_sha256', 'stimuli_sha256', 'tokenizer_sha256', 'documents', 'stops',
                             'uncertified_stops', 'max_tokens'}
            or preflight['assay_sha256'] != worker.digest(assay_path)
            or preflight['stimuli_sha256'] != hashes[stimulus.name] or preflight['tokenizer_sha256'] != TOKENIZER_SHA256
            or preflight['documents'] != len(articles) or preflight['stops'] != sum(len(row['stops']) for row in articles)
            or preflight['uncertified_stops'] != 0 or not 0 < preflight['max_tokens'] <= 6144):
        raise ValueError('The independent readback stops differ from their tokenizer preflight')
    expected = {(row['id'], stop) for row in articles for stop in row['stops'][1:]}
    with (assay_root / 'targets.jsonl').open(encoding='utf-8') as stream:
        targets = list(map(json.loads, stream))
    if len(targets) != len(expected) or {(row['id'], row['stop']) for row in targets} != expected:
        raise ValueError('Independent readback targets do not cover the original display regions')
    heads = gradio.REPO / 'research/results/first-readback-transfer-l4/heads.npz'
    if worker.digest(heads) != HEADS_SHA256:
        raise ValueError('The archived pooled first-readback heads changed')
    files, v8, run_id, report_sha256, _ = verified_gradio_packet(v8_job, MEDREADME_APP_SOURCE)
    if (v8['checkpoint_sha256'] != CANDIDATE
            or any(files[name] != (gradio.REPO / name).read_bytes() for name in MEDREADME_APP_SOURCE)):
        raise ValueError('Expected the verified, unmodified Gradio candidate package')
    files.update({name: (gradio.REPO / name).read_bytes() for name in CONTEXT_READBACK_SOURCE - MEDREADME_APP_SOURCE})
    closure = source_closure(v8_job, files, 'gradio')
    files.update({worker.CONTEXT_READBACK_STIMULI: stimulus.read_bytes(), worker.CONTEXT_READBACK_HEADS: heads.read_bytes()})
    member_hashes = {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}
    plan = {**preflight, 'targets_sha256': hashes['targets.jsonl'], 'forecast_stops': len(expected),
            'candidate_id': CANDIDATE, 'heads_sha256': HEADS_SHA256, 'verified_gradio_run_id': run_id,
            'verified_gradio_sha256': report_sha256, 'protocol_sha256': worker.digest(gradio.REPO / CONTEXT_READBACK_PROTOCOL),
            'source_code_sha256': member_hashes[CONTEXT_READBACK_EXPORTER]}
    job = new_job(output)
    archive = job / worker.CONTEXT_READBACK_ARCHIVE
    write_archive(archive, files)
    bundle = {field: v8[field] for field in BACKBONE_FIELDS}
    bundle['requirements'] = [item for item in bundle['requirements'] if not item.startswith('gradio==')]
    bundle.update(context_readback=plan, source_closure={'context-readback': closure})
    seal_job(job, {archive.name: (archive, 'context-readback', member_hashes)}, bundle, 17,
             worker.CONTEXT_READBACK_SECONDS, context_readback=plan)


def prepare(gradio_root, later_root, cwi_root, plaba_root, textde_root, output):
    gradio_path = gradio_root / gradio.PAYLOAD
    gradio_manifest = worker.read_json(gradio_root / gradio.MANIFEST)
    gradio.verify(gradio_root)
    gradio_hashes = {name: value['sha256'] for name, value in gradio_manifest['members'].items()}
    gradio_hashes[gradio.MANIFEST] = worker.digest(gradio_root / gradio.MANIFEST)
    later_model = later_root / 'model-inputs.tar.gz'
    later_manifest = worker.read_json(later_root / 'assay.json')
    later_receipt = worker.read_json(later_root / 'package.json')
    later_hashes = {**later_manifest['inputs_sha256'], 'assay.json': worker.digest(later_root / 'assay.json')}
    if (worker.digest(later_model) != later_receipt['archive_sha256'] or set(later_hashes) != worker.LATER_DATA
            or set(later_receipt['public_members']) != worker.LATER_DATA):
        raise ValueError('Same-paragraph input archive differs from its original seal')
    worker.unpack(later_model, later_hashes)
    cwi_model = cwi_root / 'model-inputs.tar.gz'
    cwi_manifest = worker.read_json(cwi_root / 'assay.json')
    cwi_receipt = worker.read_json(gradio.REPO / 'research/results/cwi-background-transfer-preparation.json')
    cwi_hashes = {**cwi_manifest['inputs_sha256'], 'assay.json': worker.digest(cwi_root / 'assay.json')}
    if (set(cwi_hashes) != worker.CWI_DATA or cwi_hashes != cwi_receipt['archive_members_sha256']
            or worker.digest(cwi_model) != cwi_receipt['model_inputs_archive_sha256']
            or worker.digest(gradio.REPO / CWI_PROTOCOL) != cwi_manifest['protocol_sha256']):
        raise ValueError('CWI transfer inputs or preregistered protocol differ from the original seal')
    worker.unpack(cwi_model, cwi_hashes)
    plaba_source = plaba_root / 'source.tar.gz'
    plaba_model = plaba_root / 'model-inputs.tar.gz'
    plaba_manifest = worker.read_json(plaba_root / 'assay.json')
    plaba_hashes = {**plaba_manifest['inputs_sha256'], 'assay.json': worker.digest(plaba_root / 'assay.json')}
    if (set(plaba_hashes) != worker.PLABA_DATA or worker.digest(plaba_model) != PLABA_DATA_SHA256
            or worker.digest(plaba_source) != PLABA_SOURCE_SHA256):
        raise ValueError('PLABA inference-only source or model-input archive differs from the verified release')
    worker.unpack(plaba_model, plaba_hashes)
    plaba_source_hashes = worker.unpack(plaba_source, PLABA_SOURCE)
    textde_model, textde_manifest, textde_hashes, textde_files, textde_closure = textde_packet(textde_root)
    pin = (gradio.MODEL, gradio.REVISION)
    manifests = (gradio_manifest, later_manifest, cwi_manifest, plaba_manifest, textde_manifest)
    if any((data['model'], data['revision']) != pin for data in manifests):
        raise ValueError('The verification stages need different checkpoints')
    later_source = later_root / 'source.tar.gz'
    gradio_files = archive_code(gradio_path)
    later_files = archive_code(later_source, LATER_SOURCE)
    cwi_files = {name: (gradio.REPO / name).read_bytes() for name in CWI_SOURCE | {CWI_PROTOCOL}}
    plaba_files = archive_code(plaba_source, PLABA_SOURCE)
    closure = {'gradio': source_closure(gradio_path, gradio_files, 'gradio'),
               'later': source_closure(later_source, later_files, 'later'),
               'cwi': source_closure(cwi_model, cwi_files, 'cwi'),
               'plaba': source_closure(plaba_source, plaba_files, 'plaba'),
               'textde': textde_closure}
    compile(worker.SNAPSHOT, '<snapshot-setup>', 'exec')

    job = new_job(output)
    write_archive(job / 'later-source.tar.gz', later_files)
    write_archive(job / 'cwi-source.tar.gz', cwi_files)
    write_archive(job / 'textde-source.tar.gz', textde_files)
    bundles = {'gradio.tar.gz': (gradio_path, 'gradio', gradio_hashes),
               'later-model-inputs.tar.gz': (later_model, 'iirc-later/inputs', later_hashes),
               'later-source.tar.gz': (job / 'later-source.tar.gz', 'iirc-later',
                                       {name: hashlib.sha256(code).hexdigest() for name, code in later_files.items()}),
               'cwi-model-inputs.tar.gz': (cwi_model, 'cwi-transfer/inputs', cwi_hashes),
               'cwi-source.tar.gz': (job / 'cwi-source.tar.gz', 'cwi-transfer',
                                     {name: hashlib.sha256(code).hexdigest() for name, code in cwi_files.items()}),
               'plaba-model-inputs.tar.gz': (plaba_model, 'plaba-span/inputs', plaba_hashes),
               'plaba-source.tar.gz': (plaba_source, 'plaba-span', plaba_source_hashes),
               'textde-model-inputs.tar.gz': (textde_model, 'textcomplexityde/inputs', textde_hashes),
               'textde-source.tar.gz': (job / 'textde-source.tar.gz', 'textcomplexityde',
                                        {name: hashlib.sha256(code).hexdigest() for name, code in textde_files.items()})}
    manifest = {'model': gradio.MODEL, 'revision': gradio.REVISION, 'requirements': gradio_manifest['python_requirements'],
                'source_closure': closure, 'later_data': later_manifest['coverage'],
                'cwi_counts': cwi_manifest['counts'], 'cwi_methods': cwi_manifest['forecast_methods'],
                'plaba_span': {key: plaba_manifest[key] for key in ('documents', 'sentences', 'tokens',
                               'certified_sentences', 'methods', 'top_k')},
                'textde': {key: textde_manifest[key] for key in ('cases', 'article_groups', 'forecast_methods', 'target_order')}}
    with tarfile.open(later_model, 'r:gz') as archive:
        forecasts = {row['case_id']: len(row['stops']) - 1 for row in map(json.loads, archive.extractfile('dev/queries.jsonl'))}
    seal_job(job, bundles, manifest, 5, worker.MAX_SECONDS, later_forecasts=forecasts)


def verify(job, data):
    format_version = data.get('schema_version')
    if format_version not in FORMATS:
        raise ValueError('Unsupported verification batch schema')
    members, minutes, _ = FORMATS[format_version]
    if (not worker.RUN_ID.fullmatch(job.name) or data['run_id'] != job.name or data['minutes_budget'] != minutes
            or worker.digest(job / 'worker.py') != data['worker_sha256']
            or worker.digest(job / 'bundle.json') != data['payload_members_sha256']['bundle.json']
            or worker.digest(job / 'payload.tar.gz') != data['payload_sha256']
            or worker.unpack(job / 'payload.tar.gz', members) != data['payload_members_sha256']):
        raise ValueError('Prepared batch does not match its sealed files')
    if format_version == 20:
        verify_gum_packet(job, data)
    if format_version in (21, 22):
        verify_projection_packet(job, data, 'beukers' if format_version == 21 else 'projection')
    if format_version == 23:
        verify_frick_signals_packet(job, data, worker.read_json(job / 'bundle.json'))
    if format_version == 24:
        verify_clear_ease_packet(job, data, worker.read_json(job / 'bundle.json'))
    if format_version in (7, 8):
        bundle = worker.read_json(job / 'bundle.json')
        inputs = bundle['archives']['gradio.tar.gz']['members']
        if (data['checkpoint_sha256'] != bundle['checkpoint_sha256']
                or data['ordinal_head_sha256'] != inputs[f'{worker.GRADIO_CHECKPOINT}/ordinal/head.npz']):
            raise ValueError('Prepared batch receipt does not identify the packed candidate checkpoint')
    if format_version == 8:
        archive_name = worker.FORECAST_INPUT_ARCHIVE
        forecast = bundle['archives'][archive_name]
        plan = bundle['forecast']
        path = job / archive_name
        if (plan != data['forecast'] or plan.keys() != worker.FORECAST_STIMULI.keys()
                or forecast['directory'] != 'gradio' or forecast['members'].keys() != worker.FORECAST_MEMBERS
                or not worker.digest(path) == forecast['sha256'] == data['payload_members_sha256'][archive_name]):
            raise ValueError('Prepared holdout archive differs from the forecast plan')
        worker.unpack(path, forecast['members'])
        for dataset, spec in plan.items():
            name = worker.FORECAST_STIMULI[dataset]
            preregistered = forecast_preregistration(dataset, data['checkpoint_sha256'])
            if (spec['path'] != name or spec['input_sha256'] != forecast['members'][name]
                    or (spec['input_sha256'], spec['preregistration_sha256']) != preregistered):
                raise ValueError(f'Prepared {dataset} inputs differ from their preregistration')
    if format_version in (9, 11):
        bundle = worker.read_json(job / 'bundle.json')
        student = bundle['student_annotations']
        inputs = bundle['archives'][worker.STUDENT_DATA_ARCHIVE]
        source = bundle['archives'][worker.STUDENT_SOURCE_ARCHIVE]
        receipt_sha256, input_sha256, hashes = student_preparation()
        if (student != data['student_annotations'] or student['preparation_sha256'] != receipt_sha256
                or (bundle['model'], bundle['revision']) != (gradio.MODEL, gradio.REVISION)
                or tuple(bundle['requirements']) != TEXTDE_REQUIREMENTS or tuple(bundle['model_files']) != gradio.MODEL_FILES
                or not inputs['sha256'] == input_sha256 == data['payload_members_sha256'][worker.STUDENT_DATA_ARCHIVE]
                or inputs['members'] != hashes or inputs['directory'] != 'student-annotations/inputs'
                or source['members'] != STUDENT_SOURCE_SHA256 or source['directory'] != 'student-annotations'
                or source['sha256'] != data['payload_members_sha256'][worker.STUDENT_SOURCE_ARCHIVE]):
            raise ValueError('Prepared StudentAnnotations extraction differs from its frozen inputs or source receipt')
    if format_version in (10, 11):
        bundle = worker.read_json(job / 'bundle.json')
        spec, known = ambistory_preparation()
        inputs = bundle['archives'][worker.AMBISTORY_DATA_ARCHIVE]
        source = bundle['archives'][worker.AMBISTORY_SOURCE_ARCHIVE]
        if (not spec == bundle['ambistory'] == data['ambistory']
                or (bundle['model'], bundle['revision']) != (gradio.MODEL, gradio.REVISION)
                or tuple(bundle['requirements']) != TEXTDE_REQUIREMENTS or tuple(bundle['model_files']) != gradio.MODEL_FILES
                or inputs['directory'] != 'ambistory/inputs' or inputs['members'].keys() != ambistory.INPUT_MEMBERS
                or any(inputs['members'][f'{split}/stimuli.jsonl'] != known[f'{split}/model_input/stimuli.jsonl']
                       for split in ambistory.SPLITS)
                or source['directory'] != 'ambistory' or source['members'].keys() != AMBISTORY_SOURCE
                or any(source['members'][name] != expected for name, expected in STUDENT_SOURCE_SHA256.items())
                or any(bundle['archives'][name]['sha256'] != data['payload_members_sha256'][name]
                       for name in (worker.AMBISTORY_DATA_ARCHIVE, worker.AMBISTORY_SOURCE_ARCHIVE))):
            raise ValueError('Prepared AmbiStory extraction differs from its frozen grade-free materials')
    if format_version == 11:
        originals = bundle['source_runs']
        if (originals != data['source_runs'] or originals.keys() != READING_EXTRACTION_INPUTS.keys()
                or len({run['run_id'] for run in originals.values()}) != len(originals)
                or any(not worker.RUN_ID.fullmatch(run['run_id']) or run['run_id'] == job.name
                       or not re.fullmatch(r'[0-9a-f]{64}', run['payload_sha256'])
                       or run['archives_sha256'] != {name: data['payload_members_sha256'][name]
                                                    for name in READING_EXTRACTION_INPUTS[dataset][1]}
                       for dataset, run in originals.items())):
            raise ValueError('Combined extraction packet differs from the sealed standalone source runs')
    if format_version == 12:
        bundle = worker.read_json(job / 'bundle.json')
        registered = bundle['medreadme']
        archive = bundle['archives'][worker.MEDREADME_ARCHIVE]
        inputs = archive['members']
        path = job / worker.MEDREADME_ARCHIVE
        prefix = f'{worker.GRADIO_CHECKPOINT}/'
        if (registered != data['medreadme'] or archive['directory'] != 'medreadme'
                or not worker.digest(path) == archive['sha256'] == data['payload_members_sha256'][worker.MEDREADME_ARCHIVE]
                or inputs[worker.MEDREADME_STIMULI] != registered['stimuli_sha256']
                or inputs[f'{prefix}lexical/heads.npz'] != registered['lexical_weights_sha256']
                or inputs[MEDREADME_EXPORTER] != registered['forecast_code_sha256']
                or registered['protocol_sha256'] not in {
                    worker.digest(gradio.REPO / protocol) for protocol, _ in WORD_HOLDOUT_STUDIES.values()}):
            raise ValueError('Prepared word holdout packet differs from its source and registered experiment')
        worker.unpack(path, inputs)
        with tarfile.open(path, 'r:gz') as package, package.extractfile(f'{prefix}checkpoint.json') as stream:
            checkpoint = json.load(stream)
        expected = (MEDREADME_APP_SOURCE | {MEDREADME_EXPORTER, worker.FORECAST_SOURCE, worker.MEDREADME_STIMULI}
                    | {f'{prefix}{name}' for name in checkpoint['files'].keys() | {'checkpoint.json'}})
        if registered['protocol_sha256'] != worker.digest(gradio.REPO / MEDREADME_PROTOCOL):
            expected |= {REFERENCE_ROUTE_CODE}
            _, released, route = combined_reader_release(registered['combined_route_run_id'],
                                                        registered['combined_route_receipt_sha256'])
            source_hashes = combined_reader_member_hashes(released, route)
            if (registered['candidate_id'] != released[route]['candidate_id']
                    or any(inputs[name] != source_hashes[name] for name in MEDREADME_APP_SOURCE | {REFERENCE_ROUTE_CODE})):
                raise ValueError('Prepared news word holdout differs from the verified combined article route')
        if (expected != inputs.keys() or checkpoint['checkpoint_sha256'] != registered['candidate_id']
                or any(inputs[f'{prefix}{name}'] != value for name, value in checkpoint['files'].items())):
            raise ValueError('The word holdout payload must contain only the verified candidate and grade-free inputs')
    if format_version == 13:
        bundle = worker.read_json(job / 'bundle.json')
        route_keys = {'article_route', 'reference_route', COMBINED_ARTICLE_ROUTE, OVERVIEW_HTTP_ROUTE, WORD_CUE_ROUTE}
        route, = route_keys & bundle.keys()
        plan = bundle[route]
        archive = bundle['archives'][worker.ARTICLE_ROUTE_ARCHIVE]
        inputs = archive['members']
        path = job / worker.ARTICLE_ROUTE_ARCHIVE
        prefix = f'{worker.GRADIO_CHECKPOINT}/'
        if (route_keys & data.keys() != {route} or plan != data[route] or archive['directory'] != 'article-route'
                or not worker.digest(path) == archive['sha256'] == data['payload_members_sha256'][worker.ARTICLE_ROUTE_ARCHIVE]):
            raise ValueError('Prepared article-route packet differs from its registered fixture and serving code')
        worker.unpack(path, inputs)
        with tarfile.open(path, 'r:gz') as package, package.extractfile(f'{prefix}checkpoint.json') as stream:
            checkpoint = json.load(stream)
            if route == 'reference_route':
                closed_route = worker.read_json(gradio.REPO / ARTICLE_ROUTE_RECEIPT)
                if (plan['previous_route_run_id'] != closed_route['run_id']
                        or plan['previous_route_payload_sha256'] != closed_route['payload_sha256']
                        or plan['previous_route_archive_sha256'] != closed_route['payload_members_sha256'][worker.ARTICLE_ROUTE_ARCHIVE]
                        or plan['candidate_id'] != closed_route['article_route']['candidate_id']):
                    raise ValueError('The reference route does not descend from the released V13 candidate')
                with package.extractfile(REFERENCE_ROUTE_VERIFIER) as verifier:
                    if reference_fixture_digest(verifier.read()) != plan['fixture_sha256']:
                        raise ValueError('The packaged reference article differs from its fixture digest')
                artifacts = {}
                for name in REFERENCE_ROUTE_ARTIFACTS:
                    with package.extractfile(name) as artifact:
                        artifacts[name] = json.load(artifact)
                validate_reference_binding(plan, inputs, artifacts)
            elif route in (COMBINED_ARTICLE_ROUTE, OVERVIEW_HTTP_ROUTE, WORD_CUE_ROUTE):
                kind = 'gradio'
                if route == WORD_CUE_ROUTE:
                    validate_word_cue_binding(plan, inputs, package.extractfile(WORD_CUE_VERIFIER).read())
                    if [item for item in bundle['requirements'] if item.startswith('scipy')] != [WORD_CUE_SCIPY]:
                        raise ValueError('The word-cue serving route requires the pinned SciPy runtime')
                    kind = 'word-cues'
                else:
                    binding = validate_overview_http_binding if route == OVERVIEW_HTTP_ROUTE else validate_combined_article_binding
                    binding(plan, inputs)
                code = {name: package.extractfile(name).read() for name in inputs if name.endswith('.py')}
                if gradio.json_bytes(source_closure(path, code, kind)) != gradio.json_bytes(
                        bundle['source_closure']['article-route']):
                    raise ValueError('Combined article route source closure differs from its sealed package')
            elif (plan['source_code_sha256'] != inputs[ARTICLE_ROUTE_VERIFIER]
                  or plan['app_sha256'] != inputs[GRADIO_APP]):
                raise ValueError('Prepared article-route source differs from the registered serving code')
        expected = (ARTICLE_ROUTE_APP_SOURCE | {ARTICLE_ROUTE_VERIFIER}
                    | {f'{prefix}{name}' for name in checkpoint['files'].keys() | {'checkpoint.json'}})
        if route in ('reference_route', COMBINED_ARTICLE_ROUTE, OVERVIEW_HTTP_ROUTE, WORD_CUE_ROUTE):
            expected |= REFERENCE_ROUTE_SOURCE | REFERENCE_ROUTE_ARTIFACTS
        if route in (OVERVIEW_HTTP_ROUTE, WORD_CUE_ROUTE):
            expected |= {OVERVIEW_HTTP_VERIFIER}
        if route == WORD_CUE_ROUTE:
            expected |= WORD_CUE_NEW_SOURCES | {WORD_CUE_RELEASE}
        if (expected != inputs.keys() or checkpoint['checkpoint_sha256'] != plan['candidate_id']
                or any(inputs[f'{prefix}{name}'] != digest for name, digest in checkpoint['files'].items())):
            raise ValueError('The article-route packet must contain the exact verified serving checkpoint')
    if format_version == 14:
        bundle = worker.read_json(job / 'bundle.json')
        plan = bundle['onestop']
        archive = bundle['archives'][worker.ONESTOP_ARCHIVE]
        inputs = archive['members']
        path = job / worker.ONESTOP_ARCHIVE
        prefix = f'{worker.GRADIO_CHECKPOINT}/'
        if (plan != data['onestop'] or archive['directory'] != 'onestop'
                or not worker.digest(path) == archive['sha256'] == data['payload_members_sha256'][worker.ONESTOP_ARCHIVE]
                or plan['source_code_sha256'] != inputs[ONESTOP_EXTRACTOR]
                or plan['stimuli_sha256'] != inputs[worker.ONESTOP_STIMULI]
                or plan['protocol_sha256'] != worker.digest(gradio.REPO / ONESTOP_PROTOCOL)):
            raise ValueError('The prepared OneStop packet differs from the frozen run')
        worker.unpack(path, inputs)
        with tarfile.open(path, 'r:gz') as package, package.extractfile(f'{prefix}checkpoint.json') as stream:
            checkpoint = json.load(stream)
        expected = (MEDREADME_APP_SOURCE | {ONESTOP_EXTRACTOR, worker.ONESTOP_STIMULI}
                    | {f'{prefix}{name}' for name in checkpoint['files'].keys() | {'checkpoint.json'}})
        if (expected != inputs.keys() or checkpoint['checkpoint_sha256'] != plan['candidate_id']
                or any(inputs[f'{prefix}{name}'] != digest for name, digest in checkpoint['files'].items())):
            raise ValueError('The OneStop payload must contain only the verified backbone and outcome-free inputs')
    if format_version in (15, 18, 19):
        bundle = worker.read_json(job / 'bundle.json')
        if format_version == 15 and bundle['expmrc'].get('experiment') in CONTRACTNLI_PROTOCOLS:
            verify_contractnli_packet(job, data, bundle)
            return
        stage, key, archive_name, protocol = (('expmrc', 'expmrc', worker.EXPMRC_ARCHIVE, EXPMRC_PROTOCOL)
                                              if format_version == 15 else EVIDENCE_SCOPES[format_version][:4])
        plan = bundle[key]
        archive = bundle['archives'][archive_name]
        inputs = archive['members']
        path = job / archive_name
        prefix = f'{worker.GRADIO_CHECKPOINT}/'
        bound = {'articles_sha256': worker.EXPMRC_ARTICLES, 'queries_sha256': worker.EXPMRC_QUERIES,
                 'heads_sha256': worker.EXPMRC_HEADS, 'source_code_sha256': EXPMRC_EXPORTER}
        if (plan != data[key] or archive['directory'] != stage
                or not worker.digest(path) == archive['sha256'] == data['payload_members_sha256'][archive_name]
                or any(plan[key] != inputs[member] for key, member in bound.items())
                or plan['protocol_sha256'] != worker.digest(gradio.REPO / protocol)):
            raise ValueError(f'The prepared {stage} packet differs from the frozen transfer run')
        if format_version in EVIDENCE_SCOPES and not (plan['calibration_sha256'] == EVIDENCE_CALIBRATION_SHA256
                                                       == worker.digest(gradio.REPO / EVIDENCE_CALIBRATION)):
            raise ValueError(f'The {stage} calibration must match the frozen ExpMRC parameters')
        worker.unpack(path, inputs)
        with tarfile.open(path, 'r:gz') as package, package.extractfile(f'{prefix}checkpoint.json') as stream:
            checkpoint = json.load(stream)
        expected = (MEDREADME_APP_SOURCE | EXPMRC_SOURCE | set(bound.values())
                    | {f'{prefix}{name}' for name in checkpoint['files'].keys() | {'checkpoint.json'}})
        if (expected != inputs.keys() or checkpoint['checkpoint_sha256'] != plan['candidate_id']
                or any(inputs[f'{prefix}{name}'] != digest for name, digest in checkpoint['files'].items())):
            raise ValueError(f'The {stage} packet must contain only the verified backbone, heads, articles and questions')
    if format_version == 16:
        bundle = worker.read_json(job / 'bundle.json')
        plan = bundle['first_readback']
        archive = bundle['archives'][worker.FIRST_READBACK_ARCHIVE]
        inputs = archive['members']
        path = job / worker.FIRST_READBACK_ARCHIVE
        prefix = f'{worker.GRADIO_CHECKPOINT}/'
        bound = {f'{name}_sha256': member for name, member in worker.FIRST_READBACK_INPUTS.items()}
        bound['source_code_sha256'] = FIRST_READBACK_EXPORTER
        if (plan != data['first_readback'] or archive['directory'] != 'first-readback'
                or not worker.digest(path) == archive['sha256'] == data['payload_members_sha256'][path.name]
                or any(plan[key] != inputs[member] for key, member in bound.items())
                or plan['protocol_sha256'] != worker.digest(gradio.REPO / FIRST_READBACK_PROTOCOL)):
            raise ValueError('The first-readback packet differs from the frozen transfer run')
        worker.unpack(path, inputs)
        with tarfile.open(path, 'r:gz') as package, package.extractfile(f'{prefix}checkpoint.json') as stream:
            checkpoint = json.load(stream)
        expected = (FIRST_READBACK_SOURCE | set(worker.FIRST_READBACK_INPUTS.values())
                    | {f'{prefix}{name}' for name in checkpoint['files'].keys() | {'checkpoint.json'}})
        if (expected != inputs.keys() or checkpoint['checkpoint_sha256'] != plan['candidate_id']
                or any(inputs[f'{prefix}{name}'] != digest for name, digest in checkpoint['files'].items())):
            raise ValueError('The first-readback packet must exclude held-out responses and raw participant records')
    if format_version == 17:
        from research.semantic_tracking.context_readback_forecast import CANDIDATE, HEADS_SHA256

        bundle = worker.read_json(job / 'bundle.json')
        plan = bundle['context_readback']
        archive = bundle['archives'][worker.CONTEXT_READBACK_ARCHIVE]
        inputs = archive['members']
        path = job / worker.CONTEXT_READBACK_ARCHIVE
        prefix = f'{worker.GRADIO_CHECKPOINT}/'
        bound = {'stimuli_sha256': worker.CONTEXT_READBACK_STIMULI, 'heads_sha256': worker.CONTEXT_READBACK_HEADS,
                 'source_code_sha256': CONTEXT_READBACK_EXPORTER}
        if (plan != data['context_readback'] or archive['directory'] != 'context-readback'
                or not worker.digest(path) == archive['sha256'] == data['payload_members_sha256'][path.name]
                or any(plan[key] != inputs[member] for key, member in bound.items())
                or plan['protocol_sha256'] != worker.digest(gradio.REPO / CONTEXT_READBACK_PROTOCOL)
                or (plan['candidate_id'], plan['heads_sha256']) != (CANDIDATE, HEADS_SHA256)):
            raise ValueError('The independent readback packet differs from its locked protocol and pooled heads')
        worker.unpack(path, inputs)
        with tarfile.open(path, 'r:gz') as package, package.extractfile(f'{prefix}checkpoint.json') as stream:
            checkpoint = json.load(stream)
        expected = (CONTEXT_READBACK_SOURCE | set(bound.values())
                    | {f'{prefix}{name}' for name in checkpoint['files'].keys() | {'checkpoint.json'}})
        if (expected != inputs.keys() or checkpoint['checkpoint_sha256'] != plan['candidate_id']
                or any(inputs[f'{prefix}{name}'] != digest for name, digest in checkpoint['files'].items())):
            raise ValueError('The independent packet must contain only the sealed model, heads and text boundaries')


def contractnli_archive_rows(path, experiment):
    with tarfile.open(path, 'r:gz') as archive:
        packet = worker.CONTRACTNLI_PACKET
        stimuli = list(map(json.loads, archive.extractfile(f'{packet}/stimuli.jsonl')))
        maps = list(map(json.loads, archive.extractfile(f'{packet}/articles.jsonl')))
    if experiment == EVIDENCE_INFERENCE:
        return evidence_inference_archive_rows(stimuli, maps)
    document_count, span_count, certified_count = CONTRACTNLI_INVENTORIES[experiment]
    if (len(stimuli) != document_count + 17 or len({row['id'] for row in stimuli}) != len(stimuli)
            or len(maps) != document_count or len({row['id'] for row in maps}) != len(maps)):
        raise ValueError('Unexpected grade-free document and hypothesis inventory')
    if any(row.keys() != {'id', 'text', 'stops'} or type(row['id']) is not str
           or not isinstance(row['text'], str) or not row['text'].strip() or not row['stops']
           or any(type(stop) is not int or not 0 < stop <= len(row['text']) for stop in row['stops'])
           or row['stops'] != sorted(set(row['stops'])) or row['stops'][-1] != len(row['text'])
           for row in stimuli):
        raise ValueError('The contract packet may contain only text and complete certified schedules')
    schedules = {row['id']: row['stops'] for row in stimuli}
    if (any(row.keys() != {'id', 'candidate_stops'} or row['id'] not in schedules
            or any(i is not None and (type(i) is not int or not 0 <= i < len(schedules[row['id']]))
                   for i in row['candidate_stops']) for row in maps)
            or sum(len(row['candidate_stops']) for row in maps) != span_count
            or sum(i is not None for row in maps for i in row['candidate_stops']) != certified_count):
        raise ValueError('The candidate-to-certified-stop maps disagree with the registered inventory')
    articles = {row['id']: schedules[row['id']] for row in maps}
    queries = schedules.keys() - articles.keys()
    if any(len(schedules[key]) != 1 for key in queries):
        raise ValueError('Every hypothesis must have only its final document-independent stop')
    pairs = {(article, query) for article in articles for query in queries}
    return articles, queries, pairs, sum(map(len, schedules.values()))


def evidence_inference_archive_rows(stimuli, maps):
    if not 1 <= len(maps) <= 24 or len({row['id'] for row in stimuli}) != len(stimuli):
        raise ValueError('Invalid Evidence Inference article or stimulus inventory')
    if any(row.keys() != {'id', 'text', 'stops'} or not isinstance(row['id'], str) or not row['id']
           or not isinstance(row['text'], str) or not row['text'].strip() or not row['stops']
           or any(type(stop) is not int or not 0 < stop <= len(row['text']) for stop in row['stops'])
           or row['stops'] != sorted(set(row['stops'])) or row['stops'][-1] != len(row['text'])
           for row in stimuli):
        raise ValueError('Evidence Inference stimuli require complete original text and certified stops')
    sources = {row['id']: row for row in stimuli}
    articles, queries, pairs = {}, set(), set()
    for row in maps:
        identifier, spans, indices, bank = (row[key] for key in ('id', 'spans', 'candidate_stops', 'query_ids'))
        if (row.keys() != {'id', 'spans', 'candidate_stops', 'query_ids'} or identifier not in sources
                or identifier in articles or len(spans) != len(indices) or all(i is None for i in indices)
                or len(bank) < 2 or len(bank) != len(set(bank)) or queries.intersection(bank)
                or any(query not in sources or len(sources[query]['stops']) != 1 for query in bank)):
            raise ValueError('Each Evidence Inference article needs its own unique bank and aligned candidates')
        stops, text = (sources[identifier][key] for key in ('stops', 'text'))
        prior = 0
        for (start, end), index in zip(spans, indices, strict=True):
            if (type(start) is not int or type(end) is not int or not prior <= start < end <= len(text)
                    or not text[start:end].strip()
                    or (index is not None and (type(index) is not int or not 0 <= index < len(stops)
                                               or end != stops[index]))):
                raise ValueError('A candidate must map to its original span and exact causal sentence end')
            prior = end
        articles[identifier] = stops
        queries.update(bank)
        pairs.update((identifier, query) for query in bank)
    if queries.intersection(articles) or articles.keys() | queries != sources.keys():
        raise ValueError('Expected only the audited articles and their associated questions')
    return articles, queries, pairs, sum(len(row['stops']) for row in stimuli)


def verify_contractnli_packet(job, data, bundle):
    from research.semantic_tracking.expmrc_forecast import HEADS_SHA256

    archive_name = worker.EXPMRC_ARCHIVE
    path = job / archive_name
    archive = bundle['archives'][archive_name]
    inputs = archive['members']
    plan = bundle['expmrc']
    experiment = plan['experiment']
    exporter = worker.CONTRACTNLI_MODULES[experiment].replace('.', '/') + '.py'
    reference = worker.read_json(gradio.REPO / CONTRACTNLI_REFERENCE)
    if (plan != data['expmrc'] or archive['directory'] != 'expmrc'
            or not worker.digest(path) == archive['sha256'] == data['payload_members_sha256'][archive_name]
            or plan['reference_run_id'] != reference['run_id']
            or plan['reference_bundle_sha256'] != reference['payload_members_sha256']['bundle.json']
            or plan['protocol_sha256'] != worker.digest(gradio.REPO / CONTRACTNLI_PROTOCOLS[experiment])
            or plan['source_code_sha256'] != inputs[exporter]
            or not plan['heads_sha256'] == inputs[worker.EXPMRC_HEADS] == HEADS_SHA256
            or plan['packet_sha256'] != {Path(name).name: inputs[name] for name in CONTRACTNLI_INPUTS}):
        raise ValueError('ContractNLI needs the registered QASPER heads, source and preflight')
    worker.unpack(path, inputs)
    prefix = f'{worker.GRADIO_CHECKPOINT}/'
    with tarfile.open(path, 'r:gz') as package:
        checkpoint = json.load(package.extractfile(f'{prefix}checkpoint.json'))
        preflight = json.load(package.extractfile(f'{worker.CONTRACTNLI_PACKET}/preflight.json'))
    expected = (MEDREADME_APP_SOURCE | EXPMRC_SOURCE | CONTRACTNLI_INPUTS | {CONTRACTNLI_EXPORTER, exporter, worker.EXPMRC_HEADS}
                | {f'{prefix}{name}' for name in checkpoint['files'].keys() | {'checkpoint.json'}})
    if (inputs.keys() != expected or checkpoint['checkpoint_sha256'] != plan['candidate_id']
            or any(preflight[key] != plan[key] for key in ('candidate_id', 'heads_sha256', 'source_code_sha256'))
            or preflight['model_revision'] != bundle['revision']
            or preflight['file_sha256'] != {name: plan['packet_sha256'][name] for name in ('stimuli.jsonl', 'articles.jsonl')}
            or any(inputs[f'{prefix}{name}'] != sha for name, sha in checkpoint['files'].items())):
        raise ValueError('Only the sealed backbone, readout and grade-free contract stimuli may enter Colab')
    if experiment in {'contractnli_rerank', EVIDENCE_INFERENCE} and preflight['upstream_forecast_code_sha256'] != inputs[CONTRACTNLI_EXPORTER]:
        raise ValueError('The reranking packet must retain the frozen original extraction implementation')
    _, queries, pairs, stop_count = contractnli_archive_rows(path, experiment)
    inventory = preflight['inventory']
    max_tokens = 140_000
    if experiment == EVIDENCE_INFERENCE:
        max_tokens = 145_000
        if preflight['source_audit_sha256'] != plan['assay_sha256']:
            raise ValueError('Evidence Inference provenance changed after certification')
    if (set(preflight['query_ids']) != queries or inventory['certified_stops'] != stop_count
            or inventory['article_hypothesis_pairs'] != len(pairs)
            or not 0 < inventory['max_input_tokens'] <= 8192 or not 0 < inventory['input_tokens'] <= max_tokens):
        raise ValueError('The sealed packet differs from the contract inference budget and preflight')


def verify_contractnli_forecast(job, outputs, bundle):
    from research.semantic_tracking.expmrc_forecast import METHODS

    plan = bundle['expmrc']
    report = worker.read_json(outputs / 'seal.json')
    extraction = worker.read_json(outputs / 'extraction.json')
    causality = worker.read_json(outputs / 'causality.json')['checks']
    articles, queries, pairs, stop_count = contractnli_archive_rows(job / worker.EXPMRC_ARCHIVE, plan['experiment'])
    pins = ('candidate_id', 'assay_sha256', 'protocol_sha256', 'heads_sha256', 'source_code_sha256')
    files = ('predictions', 'extraction', 'causality', 'telemetry')
    if (any(report.get(key) != plan[key] for key in pins)
            or report['model_revision'] != bundle['revision']
            or any(report[f'{name}_sha256'] != worker.digest(outputs / f'{name}.{suffix}')
                   for name, suffix in zip(files, ('jsonl', 'json', 'json', 'jsonl'), strict=True))
            or any(report[f'{name}_sha256'] != plan['packet_sha256'][f'{name}.{suffix}']
                   for name, suffix in (('stimuli', 'jsonl'), ('articles', 'jsonl'), ('preflight', 'json')))
            or report['content_forward_per_document'] != 1 or report['encoded_hypotheses'] != len(queries)
            or set(report['query_ids']) != queries or report['forecast_rows'] != len(pairs)
            or extraction['input_sha256'] != plan['packet_sha256']['stimuli.jsonl']
            or (extraction['model'], extraction['revision'], extraction['gpu'], extraction['dtype'])
            != (bundle['model'], bundle['revision'], 'NVIDIA L4', 'float32')
            or extraction['encoded_texts'] != len(articles) + len(queries) or extraction['excluded_stops']
            or extraction['unique_certified_stops'] != stop_count
            or extraction['tokens'] != report['input_tokens']
            or extraction['projected_tokens']['documents'] != len(articles)
            or len(causality) != 12 or any(not row['passed'] or not row['projected_tokens']['passed'] for row in causality)):
        raise ValueError('ContractNLI predictions lack the exact frozen extraction or causal checks')
    with (outputs / 'predictions.jsonl').open(encoding='utf-8') as stream:
        rows = list(map(json.loads, stream))
    keys = {(row['id'], row['query_id']) for row in rows}
    if (len(rows) != len(keys) or keys != pairs
            or any(row.keys() != {'id', 'query_id', 'stops', 'methods'} or row['stops'] != articles[row['id']]
                   or row['methods'].keys() != set(METHODS)
                   or any(len(values) != len(row['stops']) or any(type(p) not in (int, float) or not 0 < p < 1 for p in values)
                          for values in row['methods'].values()) for row in rows)):
        raise ValueError('Every document and public hypothesis need four complete sealed prediction paths')


def verify_gum_packet(job, data):
    bundle = worker.read_json(job / 'bundle.json')
    packet = bundle['archives'][worker.GUM_BRIDGE_ARCHIVE]
    path = job / worker.GUM_BRIDGE_ARCHIVE
    members = packet['members']
    prefix = f'{worker.GRADIO_CHECKPOINT}/'
    with tarfile.open(path, 'r:gz') as archive:
        candidate = json.load(archive.extractfile(f'{prefix}checkpoint.json'))
    expected = GUM_BRIDGE_SOURCE | GUM_BRIDGE_INPUTS | {'preflight.json', f'{prefix}checkpoint.json'} | {
        f'{prefix}{name}' for name in candidate['files']}
    plan = bundle['gum_bridge']
    if (plan != data['gum_bridge'] or packet['directory'] != 'gum-bridge' or members.keys() != expected
            or plan['candidate_id'] != candidate['checkpoint_sha256']
            or plan['preflight_sha256'] != members['preflight.json']
            or plan['source_code_sha256'] != members[GUM_BRIDGE_EXPORTER]
            or any(members[f'{prefix}{name}'] != sha for name, sha in candidate['files'].items())
            or not worker.digest(path) == packet['sha256'] == data['payload_members_sha256'][path.name]):
        raise ValueError('GUMBridge packet differs from its verified grade-free sources or app checkpoint')
    worker.unpack(path, members)


def verify_gum_projection(outputs, bundle):
    import numpy as np

    plan = bundle['gum_bridge']
    inputs = bundle['archives'][worker.GUM_BRIDGE_ARCHIVE]['members']
    seal = worker.read_json(outputs / 'seal.json')
    artifacts = worker.OUTPUTS['gum-bridge'] - {'seal.json', 'run.log'}
    if (seal['file_sha256'] != {name: worker.digest(outputs / name) for name in artifacts}
            or seal['input_sha256'] != {Path(name).name: inputs[name] for name in GUM_BRIDGE_INPUTS}
            or (seal['model'], seal['revision']) != (bundle['model'], bundle['revision'])
            or seal['candidate_id'] != plan['candidate_id']
            or seal['extractor_sha256'] != plan['source_code_sha256']
            or seal['preflight_sha256'] != plan['preflight_sha256']
            or seal['anchor_heads_sha256'] != inputs[f'{worker.GRADIO_CHECKPOINT}/anchor/heads.npz']
            or any(seal[name] != plan[name] for name in ('encoded_documents', 'encoded_tokens', 'certified_stops'))
            or seal['content_forward_per_document'] != 1 or seal['compute_dtype'] != 'float32'
            or seal['storage_dtype'] != 'float16' or not 0 <= seal['max_storage_relative_l2'] <= 0.001):
        raise ValueError('GUMBridge extraction seal disagrees with the grade-free cause and input inventory')
    coordinates = [json.loads(line) for line in (outputs / 'coordinates.jsonl').read_text(encoding='utf-8').splitlines()]
    if ([(row['id'], len(row['stops'])) for row in coordinates] !=
            [(row['id'], row['certified_stops']) for row in seal['documents'] if row['certified_stops']]
            or sum(len(row['stops']) for row in coordinates) != seal['certified_stops']
            or any(row.keys() != {'id', 'stops'} or row['stops'] != sorted(set(row['stops'])) for row in coordinates)):
        raise ValueError('The returned GUMBridge coordinate order or stop counts differ')
    states = np.load(outputs / 'vectors.npy', mmap_mode='r', allow_pickle=False)
    if states.dtype != np.float16 or list(states.shape) != seal['vectors_shape'] or states.shape != (plan['certified_stops'], 128):
        raise ValueError('Unexpected returned GUMBridge projection shape')
    for start in range(0, len(states), 8192):
        norms = np.linalg.norm(states[start:start + 8192].astype(np.float32), axis=1)
        if not np.all(np.isfinite(norms) & (np.abs(norms - 1) <= 0.001)):
            raise ValueError('Unexpected state norms in the returned GUMBridge projection')
    checks = worker.read_json(outputs / 'causality.json')['checks']
    comparisons = ('substring', 'right_padding', 'suffix_replacement', 'changed_suffix_positive_control')
    pairs = {(row['id'], row['stop']) for row in checks}
    if (len(checks) != seal['independent_causal_checks'] or len(pairs) != 3
            or {(row['id'], row['stop'], row['comparison']) for row in checks} !=
               {(key, stop, method) for key, stop in pairs for method in comparisons}
            or any(not row['passed'] or not row['projected_tokens']['passed'] for row in checks)):
        raise ValueError('GUMBridge independent prefix and suffix controls did not pass')


def verify_projection_packet(job, data, stage):
    bundle = worker.read_json(job / 'bundle.json')
    _, archive_name, _ = PROJECTION_SCOPES[stage]
    packet = bundle['archives'][archive_name]
    path = job / archive_name
    members = packet['members']
    prefix = f'{worker.GRADIO_CHECKPOINT}/'
    with tarfile.open(path, 'r:gz') as archive:
        candidate = json.load(archive.extractfile(f'{prefix}checkpoint.json'))
        preflight = json.load(archive.extractfile('preflight.json'))
    expected = BEUKERS_SOURCE | {worker.PROJECTION_STIMULI, 'preflight.json', f'{prefix}checkpoint.json'} | {
        f'{prefix}{name}' for name in candidate['files']}
    plan = bundle[stage]
    if 'combined_route_run_id' in plan:
        expected |= {REFERENCE_ROUTE_CODE}
        _, released, route = combined_reader_release(plan['combined_route_run_id'], plan['combined_route_receipt_sha256'])
        source_hashes = combined_reader_member_hashes(released, route)
        matching_scope = any(plan['assay_sha256'] == assay_sha256
                             and plan['protocol_sha256'] == worker.digest(gradio.REPO / name)
                             and all(plan[key] == value for key, value in inventory.items())
                             and (certified_sha256 is None or plan['preflight_sha256'] == certified_sha256)
                             and all(len(doc['stops']) == stops for doc in preflight['documents'])
                             for name, (_, assay_sha256, stops, inventory, certified_sha256)
                             in READER_PROJECTION_ASSAYS.items())
        if (stage != 'beukers' or not matching_scope
                or plan['candidate_id'] != released[route]['candidate_id']
                or any(members[name] != source_hashes[name]
                       for name in MEDREADME_APP_SOURCE | {REFERENCE_ROUTE_CODE})):
            raise ValueError('Reading assay inputs diverge from the sealed materials and verified Gradio route')
    if (plan != data[stage] or packet['directory'] != stage or members.keys() != expected
            or plan['candidate_id'] != candidate['checkpoint_sha256']
            or plan['preflight_sha256'] != members['preflight.json']
            or not plan['stimuli_sha256'] == members[worker.PROJECTION_STIMULI] == preflight['stimuli_sha256']
            or plan['source_code_sha256'] != members[BEUKERS_EXPORTER]
            or any(plan[key] != preflight[key] for key in ('encoded_documents', 'input_tokens', 'certified_stops'))
            or any(members[f'{prefix}{name}'] != sha for name, sha in candidate['files'].items())
            or not worker.digest(path) == packet['sha256'] == data['payload_members_sha256'][path.name]):
        raise ValueError(f'{stage} packet differs from the sealed grade-free inputs or verified app checkpoint')
    worker.unpack(path, members)


def verify_frick_signals_packet(job, data, bundle):
    plan = bundle['frick_signals']
    archive = job / worker.FRICK_SIGNALS_ARCHIVE
    packet = bundle['archives'][archive.name]
    members = packet['members']
    prefix = f'{worker.GRADIO_CHECKPOINT}/'
    with tarfile.open(archive, 'r:gz') as stream:
        checkpoint = json.load(stream.extractfile(f'{prefix}checkpoint.json'))
    expected = BEUKERS_SOURCE | {
        FRICK_SIGNALS_EXPORTER, REFERENCE_ROUTE_CODE, worker.PROJECTION_STIMULI,
        'preflight.json', f'{prefix}checkpoint.json', *(f'{prefix}{name}' for name in checkpoint['files']),
    }
    _, released, route = combined_reader_release(plan['combined_route_run_id'], plan['combined_route_receipt_sha256'])
    source_hashes = combined_reader_member_hashes(released, route)
    if (members.keys() != expected or plan != data['frick_signals'] or packet['directory'] != 'frick-signals'
            or not plan['candidate_id'] == checkpoint['checkpoint_sha256'] == released[route]['candidate_id']
            or any(members[key] != source_hashes[key] for key in MEDREADME_APP_SOURCE | {REFERENCE_ROUTE_CODE})
            or any(members[f'{prefix}{name}'] != value for name, value in checkpoint['files'].items())
            or plan['protocol_sha256'] != worker.digest(gradio.REPO / FRICK_SIGNALS_PROTOCOL)
            or plan['assay_sha256'] != FRICK_PREVIOUS_PREFLIGHT_SHA256
            or not plan['preflight_sha256'] == members['preflight.json'] == FRICK_SIGNAL_PREFLIGHT_SHA256
            or not plan['stimuli_sha256'] == members[worker.PROJECTION_STIMULI] == FRICK_SIGNAL_STIMULI_SHA256
            or plan['source_code_sha256'] != members[FRICK_SIGNALS_EXPORTER]
            or any(plan[key] != value for key, value in FRICK_SIGNAL_INVENTORY.items())
            or not worker.digest(archive) == packet['sha256'] == data['payload_members_sha256'][archive.name]):
        raise ValueError('Frick packet changed or includes data beyond the verified model and grade-free story materials')
    worker.unpack(archive, members)


def verify_frick_signals_outputs(job, outputs, bundle):
    import numpy as np

    from research.gradio_app.causal_plaba import HIDDEN_WIDTH

    plan = bundle['frick_signals']
    with tarfile.open(job / worker.FRICK_SIGNALS_ARCHIVE, 'r:gz') as archive:
        preflight = json.load(archive.extractfile('preflight.json'))
        rows = [json.loads(line) for line in archive.extractfile(worker.PROJECTION_STIMULI)]
    seal = worker.read_json(outputs / 'seal.json')
    artifacts = worker.OUTPUTS['frick-signals'] - {'seal.json', 'run.log'}
    if (seal['file_sha256'] != {name: worker.digest(outputs / name) for name in artifacts}
            or any(seal.get(name) != value for name, value in preflight.items())
            or any(seal.get(name) != plan[name] for name in ('candidate_id', 'assay_sha256', 'protocol_sha256',
                                                           'preflight_sha256', 'source_code_sha256'))
            or (seal['model'], seal['revision']) != (bundle['model'], bundle['revision'])
            or seal['gpu'] != 'NVIDIA L4' or seal['compute_dtype'] != 'float32' or seal['storage_dtype'] != 'float32'
            or seal['states_dtype'] != 'float32' or seal['verification_backbone_forward_calls'] != 12
            or seal['attention_layer'] != 23 or seal['attention_heads'] != 8
            or seal['content_forward_per_document'] != 1 or seal['content_backbone_forward_calls'] != len(rows)
            or seal['feature_columns'] != ['surprisal_nats', 'predictive_entropy_nats', 'attention_prior_mass_excess',
                                          'attention_peak_prior_sentence_excess']
            or not 0 < seal['forward_and_readout_s'] <= seal['elapsed_s']
            or not 0 < seal['peak_gpu_allocated_bytes'] <= seal['peak_gpu_reserved_bytes']):
        raise ValueError('Frick token signals do not bind the prepared story input and exact Gradio candidate')

    with (outputs / 'coordinates.jsonl').open(encoding='utf-8') as stream:
        coordinates = [json.loads(line) for line in stream]
    for coordinate, item, row in zip(coordinates, preflight['documents'], rows, strict=True):
        offsets = coordinate['offsets']
        if (coordinate.keys() != {'id', 'stops', 'offsets'}
                or coordinate['id'] != item['id'] or coordinate['stops'] != item['stops']
                or len(offsets) != item['input_tokens']
                or any(not 0 <= a <= b <= len(row['text']) for a, b in offsets)
                or not set(item['stops']) <= {stop for _, stop in offsets}):
            raise ValueError('Returned token offsets do not cover the exact registered source sentence coordinates')
    features = np.load(outputs / 'token_features.npy', mmap_mode='r', allow_pickle=False)
    states = np.load(outputs / 'states.npy', mmap_mode='r', allow_pickle=False)
    if (features.dtype != np.float32 or states.dtype != np.float32
            or features.shape != (plan['input_tokens'], 4) or states.shape != (plan['certified_stops'], HIDDEN_WIDTH)
            or list(features.shape) != seal['token_features_shape'] or list(states.shape) != seal['states_shape']
            or not np.isfinite(features).all() or not np.isfinite(states).all()
            or np.any(features[:, :2] < -1e-6) or np.any(np.abs(features[:, 2:]) > 1.00001)):
        raise ValueError('Malformed vocabulary and attention features or causal sentence-end states')
    starts = np.cumsum([0, *(item['input_tokens'] for item in preflight['documents'][:-1])])
    if not np.all(features[starts] == 0):
        raise ValueError('The first token needs explicit neutral values for unavailable prefix signals')
    checks = worker.read_json(outputs / 'causality.json')['checks']
    comparisons = ('hooked_vs_unhooked', 'prefix_truncation', 'replacement_suffix', 'replacement_suffix_positive')
    if (len(checks) != 12 or seal['independent_causal_checks'] != 12
            or {(row['case'], row['comparison']) for row in checks} !=
               {(case, comparison) for case in range(1, 4) for comparison in comparisons}
            or any(row['passed'] is not True or row['prefix_tokens'] <= 0 for row in checks)
            or any(any(row['hidden_metrics'].values()) for row in checks if row['comparison'] == 'hooked_vs_unhooked')):
        raise ValueError('Frick token signals require all independent and unchanged-forward causal controls')


def verify_clear_ease_packet(job, data, bundle):
    from research.stage6.clear_reading_ease_forecast import GEOMETRY, STIMULI

    receipt = clear_ease_preparation()
    plan = bundle['clear_ease']
    archive = job / worker.CLEAR_EASE_ARCHIVE
    packet = bundle['archives'][archive.name]
    members = packet['members']
    _, released, route = combined_reader_release(plan['combined_route_run_id'], plan['combined_route_receipt_sha256'])
    base = combined_reader_member_hashes(released, route)
    additional = {STIMULI, GEOMETRY, 'plan.json', CLEAR_EASE_EXPORTER}
    if (route != OVERVIEW_HTTP_ROUTE or plan != data['clear_ease'] or packet['directory'] != 'clear-ease'
            or members.keys() != base.keys() | additional
            or any(members[name] != base[name] for name in base.keys() - {GRADIO_APP})
            or not plan['candidate_id'] == receipt['candidate_id'] == released[route]['candidate_id']
            or plan['assay_sha256'] != receipt['verification']['assay_sha256']
            or plan['protocol_sha256'] != receipt['protocol_sha256']
            or not plan['stimuli_sha256'] == receipt['file_sha256'][STIMULI] == members[STIMULI]
            or plan['stops_sha256'] != members[GEOMETRY] or plan['app_sha256'] != members[GRADIO_APP]
            or plan['source_code_sha256'] != members[CLEAR_EASE_EXPORTER]
            or plan['documents'] != sum(split['passages'] for split in receipt['split'].values())
            or plan['certified_sentences'] != sum(split['stops'] for split in receipt['split'].values())
            or plan['input_tokens'] != sum(split['input_tokens'] for split in receipt['split'].values())
            or not worker.digest(archive) == packet['sha256'] == data['payload_members_sha256'][archive.name]):
        raise ValueError('CLEAR payload differs from the fixed serving checkpoint and certified input')
    worker.unpack(archive, members)
    with tarfile.open(archive, 'r:gz') as stream:
        if json.load(stream.extractfile('plan.json')) != plan:
            raise ValueError('CLEAR worker plan differs from its run manifest')
        code = {name: stream.extractfile(name).read() for name in members if name.endswith('.py')}
        if gradio.json_bytes(source_closure(archive, code, 'gradio')) != gradio.json_bytes(
            bundle['source_closure']['clear-ease'],
        ):
            raise ValueError('CLEAR source closure differs from the sealed source package')


def verify_clear_ease_outputs(job, outputs, bundle):
    from research.gradio_app.verify_article_route import SourceMarkup
    from research.stage6.clear_reading_ease_forecast import GEOMETRY, PREFIX_TOLERANCE

    plan = bundle['clear_ease']
    seal = worker.read_json(outputs / 'seal.json')
    artifacts = worker.OUTPUTS['clear-ease'] - {'seal.json', 'run.log'}
    if (any(seal.get(name) != expected for name, expected in plan.items())
            or seal['file_sha256'] != {name: worker.digest(outputs / name) for name in artifacts}
            or (seal['model_revision'], seal['device'], seal['compute_dtype']) != (bundle['revision'], 'NVIDIA L4', 'float32')
            or seal['content_forwards'] != plan['documents'] or seal['causal_control_forwards'] != 3
            or seal['ui_control_forwards'] != 2
            or not 0 < seal['content_forward_and_readout_s'] <= seal['elapsed_after_model_load_s']
            or not 0 < seal['peak_gpu_allocated_bytes'] <= seal['peak_gpu_reserved_bytes']):
        raise ValueError('CLEAR forecasts are not bound to the exact prepared App candidate and forward count')
    with tarfile.open(job / worker.CLEAR_EASE_ARCHIVE, 'r:gz') as archive:
        certificates = [json.loads(line) for line in archive.extractfile(GEOMETRY)]
    with (outputs / 'forecasts.jsonl').open(encoding='utf-8') as stream:
        for row, certificate in zip(map(json.loads, stream), certificates, strict=True):
            if set(row) != {'id', 'sentences'} or row['id'] != certificate['id']:
                raise ValueError('CLEAR result changed its prepared document IDs')
            for score, span, token in zip(row['sentences'], certificate['spans'], certificate['tokens'], strict=True):
                if (set(score) != {'start', 'end', 'token', 'ease'} or [score['start'], score['end']] != span
                        or score['token'] != token or not 1 <= score['ease'] <= 5):
                    raise ValueError('CLEAR result changed sentence or token coordinates, or left the 1–5 scale')
    causal = worker.read_json(outputs / 'causality.json')
    selected = (certificates[i] for i in (0, len(certificates) // 2, len(certificates) - 1))
    if causal['absolute_tolerance'] != PREFIX_TOLERANCE:
        raise ValueError('Unexpected CLEAR causal verification tolerance')
    for control, certificate in zip(causal['controls'], selected, strict=True):
        index = (len(certificate['spans']) - 1) // 2
        if (control['id'] != certificate['id'] or control['end'] != certificate['spans'][index][1]
                or control['token'] != certificate['tokens'][index]
                or not 0 <= control['absolute_ease_difference'] <= PREFIX_TOLERANCE):
            raise ValueError('CLEAR requires the registered causal-prefix controls')
    ui = worker.read_json(outputs / 'ui-verification.json')
    html = (outputs / worker.ARTICLE_OVERVIEW_HTML).read_text(encoding='utf-8')
    parser = SourceMarkup()
    parser.feed(html)
    parser.close()
    layers = {mark.kind for mark in parser.marks}
    checks = ('verified_exact_source_and_coordinates', 'verified_reference_refresh',
              'verified_radio_success_http', 'verified_other_views')
    if (ui['candidate_id'] != plan['candidate_id'] or ui['app_sha256'] != plan['app_sha256']
            or ui['model_forwards_per_article'] != [1, 1] or ui['model_forwards_for_cached_views'] != 0
            or len(ui['cached_http_trace']) != 8 or any(ui[key] is not True for key in checks)
            or ui['fixture_sha256'] != OVERVIEW_HTTP_FIXTURE_SHA256
            or ui['refresh_markup_sha256'] != seal['file_sha256'][worker.ARTICLE_OVERVIEW_HTML]
            or parser.source_count != 1 or parser.in_source or parser.active or layers != {'sentence', 'lexical'}
            or hashlib.sha256(''.join(parser.chunks).encode()).hexdigest() != OVERVIEW_HTTP_FIXTURE_SHA256
            or '模型估計閱讀容易度' not in html or '篇內相對色階' not in html):
        raise ValueError('CLEAR run must verify the current Gradio App and both full-text heatmap layers')


def verify_projection_outputs(job, outputs, bundle, stage):
    import numpy as np

    from research.semantic_tracking.beukers_extract import FP16_TOLERANCE, PROJECTION_WIDTH

    plan = bundle[stage]
    _, archive_name, _ = PROJECTION_SCOPES[stage]
    members = bundle['archives'][archive_name]['members']
    seal = worker.read_json(outputs / 'seal.json')
    artifacts = worker.OUTPUTS[stage] - {'seal.json', 'run.log'}
    with tarfile.open(job / archive_name, 'r:gz') as archive:
        preflight = json.load(archive.extractfile('preflight.json'))
    if (seal['file_sha256'] != {name: worker.digest(outputs / name) for name in artifacts}
            or any(seal.get(name) != value for name, value in preflight.items())
            or (seal['model'], seal['revision']) != (bundle['model'], bundle['revision'])
            or any(seal[name] != plan[name] for name in ('candidate_id', 'assay_sha256', 'protocol_sha256',
                                                        'source_code_sha256', 'stimuli_sha256', 'preflight_sha256'))
            or seal['anchor_heads_sha256'] != members[f'{worker.GRADIO_CHECKPOINT}/anchor/heads.npz']
            or seal['projection_width'] != PROJECTION_WIDTH or seal['content_forward_per_document'] != 1
            or seal['gpu'] != 'NVIDIA L4' or seal['compute_dtype'] != 'float32' or seal['storage_dtype'] != 'float16'
            or not 0 <= seal['max_storage_relative_l2'] <= FP16_TOLERANCE
            or not 0 < seal['forward_and_readout_s'] <= seal['elapsed_s'] or seal['peak_gpu_reserved_bytes'] <= 0):
        raise ValueError(f'{stage} projection is not bound to its preflight, app checkpoint and independent inputs')
    with (outputs / 'coordinates.jsonl').open(encoding='utf-8') as stream:
        coords = [json.loads(line) for line in stream]
    if coords != [{key: row[key] for key in ('id', 'stops')} for row in preflight['documents']]:
        raise ValueError('The returned projection coordinates differ from the preflight stop identities or order')
    states = np.load(outputs / 'vectors.npy', mmap_mode='r', allow_pickle=False)
    if states.dtype != np.float16 or list(states.shape) != seal['vectors_shape'] or states.shape != (plan['certified_stops'], PROJECTION_WIDTH):
        raise ValueError('Unexpected returned projection shape or precision')
    for start in range(0, len(states), 8192):
        norms = np.linalg.norm(states[start:start + 8192].astype(np.float32), axis=1)
        if not np.all(np.isfinite(norms) & (np.abs(norms - 1) <= FP16_TOLERANCE)):
            raise ValueError('Unexpected vector norms in the returned projection')
    if stage == 'projection':
        from research.gradio_app.causal_plaba import HIDDEN_WIDTH

        terminal = np.load(outputs / 'terminal_states.npy', mmap_mode='r', allow_pickle=False)
        if (terminal.dtype != np.float32 or seal['terminal_state_dtype'] != 'float32'
                or list(terminal.shape) != seal['terminal_states_shape']
                or terminal.shape != (len(coords), HIDDEN_WIDTH)):
            raise ValueError('Unexpected returned terminal state shape or precision')
        for start in range(0, len(terminal), 512):
            block = terminal[start:start + 512]
            squared_norm = np.einsum('ij,ij->i', block, block, dtype=np.float64)
            if not np.all(np.isfinite(squared_norm) & (squared_norm > 0)):
                raise ValueError('Expected finite nonzero FP32 terminal hidden states')
    checks = worker.read_json(outputs / 'causality.json')['checks']
    comparisons = ('substring', 'right_padding', 'suffix_replacement', 'changed_suffix_positive_control')
    pairs = {(row['id'], row['stop']) for row in checks}
    if (len(checks) != 12 or len(checks) != seal['independent_causal_checks'] or len(pairs) != 3
            or not {key for key, _ in pairs} <= {row['id'] for row in coords}
            or {(row['id'], row['stop'], row['comparison']) for row in checks} !=
               {(key, stop, method) for key, stop in pairs for method in comparisons}
            or any(row['passed'] is not True or row['projected_tokens']['passed'] is not True for row in checks)):
        raise ValueError('The independent FP32 prefix and suffix controls did not pass')


def validate_stage(job, stage, receipt):
    archive = job / f'{stage}.tar.gz'
    outputs = job / 'outputs' / stage
    names = receipt['members'].keys()
    bundle = worker.read_json(job / 'bundle.json')
    expected = worker.output_members(stage,
                                     combined_article_route=bool({COMBINED_ARTICLE_ROUTE, OVERVIEW_HTTP_ROUTE} & bundle.keys()),
                                     word_cue_route=WORD_CUE_ROUTE in bundle)
    complete = receipt['state'] == 'complete'
    extra = names - expected
    if (not 0 < receipt['archive_bytes'] == archive.stat().st_size <= worker.archive_budget(stage)
            or worker.digest(archive) != receipt['archive_sha256'] or 'run.log' not in names
            or complete and not expected <= names
            or extra and (stage != 'iirc-later' or any(not worker.TELEMETRY.fullmatch(name) for name in extra))):
        raise ValueError(f'Stage archive disagrees with its remote receipt: {stage}')
    worker.unpack(archive, receipt['members'], outputs)
    if not complete:
        return False

    if stage == 'setup':
        runtime = worker.read_json(outputs / 'runtime.json')
        if (runtime['device'] != 'NVIDIA L4' or Path(runtime['snapshot']).name != bundle['revision']
                or any(runtime['packages'][item] != version for item, version in
                       (requirement.split('==') for requirement in bundle['requirements']))):
            raise ValueError('CUDA setup differs from the pinned model or environment')
    elif stage == 'gradio':
        verify_gradio(outputs, bundle, candidate=worker.read_json(manifest_path(job))['schema_version'] in (7, 8))
    elif stage == 'forecast':
        verify_forecast(job, outputs, bundle)
    elif stage == 'medreadme':
        verify_medreadme(job, outputs, bundle)
    elif stage == 'article-route':
        verify_article_route(outputs, bundle)
    elif stage == 'onestop':
        verify_onestop(outputs, bundle)
    elif stage == 'expmrc':
        if bundle['expmrc'].get('experiment') in CONTRACTNLI_PROTOCOLS:
            verify_contractnli_forecast(job, outputs, bundle)
        else:
            verify_evidence_forecast(outputs, bundle, bundle['expmrc'], bundle['expmrc']['input_tokens'])
    elif stage in ('nq-evidence', 'qasper-tail'):
        plan = bundle[stage.replace('-', '_')]
        if worker.read_json(outputs / 'extraction.json')['input_sha256'] != plan['runtime_stimuli_sha256']:
            raise ValueError(f'The {stage} extraction differs from its preflight runtime stimuli')
        verify_evidence_forecast(outputs, bundle, plan, plan['runtime_tokens'])
    elif stage == 'gum-bridge':
        verify_gum_projection(outputs, bundle)
    elif stage in PROJECTION_SCOPES:
        verify_projection_outputs(job, outputs, bundle, stage)
    elif stage == 'frick-signals':
        verify_frick_signals_outputs(job, outputs, bundle)
    elif stage == 'clear-ease':
        verify_clear_ease_outputs(job, outputs, bundle)
    elif stage == 'first-readback':
        verify_first_readback(job, outputs, bundle)
    elif stage == 'context-readback':
        verify_context_readback(job, outputs, bundle)
    elif stage == 'student-annotations':
        verify_student_annotations(job, outputs, bundle)
    elif stage == 'ambistory':
        verify_ambistory(job, outputs, bundle)
    elif stage == 'iirc-later':
        verify_later(job, outputs, bundle)
    elif stage == 'cwi-transfer':
        verify_cwi(job, outputs, bundle)
    elif stage == 'plaba-span':
        verify_plaba(job, outputs, bundle)
    else:
        verify_textde(job, outputs, bundle)
    return True


def verify_gradio(outputs, bundle, *, candidate=False):
    report = worker.read_json(outputs / 'verification.json')
    seal = worker.read_json(outputs / 'serving-seal.json')
    replay = worker.read_json(outputs / 'replay-report.json')
    inputs = bundle['archives']['gradio.tar.gz']['members']
    if (report['model'] != bundle['revision'] or report['device'] != 'NVIDIA L4 FP32'
            or report['known_unicode_token_count'] != 277 or report['model_forwards_per_combined_gradio_request'] != 1
            or report['iirc_head_sha256'] != inputs['research/results/iirc-anchor-run/heads.npz']
            or report['plaba_head_sha256'] != inputs['research/results/causal-plaba-checkpoint/head.npz']
            or report['lexical_head_sha256'] != inputs['research/results/stage5-complex/fit/heads.npz']
            or seal['runtime'] != report['device'] or seal['model'] != bundle['model']
            or seal['revision'] != bundle['revision'] or seal['heads_sha256'] != report['iirc_head_sha256']
            or seal['evaluation_sha256'] != inputs['fixtures/iirc-dev/stimuli.jsonl']
            or seal['l4_form_control_source_sha256'] != inputs['research/results/iirc-anchor-run/predictions.jsonl']
            or replay != report['iirc_serving_replay'] or replay['paragraphs'] != 430
            or replay['seal_sha256'] != worker.digest(outputs / 'serving-seal.json')
            or replay['predictions_sha256'] != seal['predictions_sha256']
            or seal['predictions_sha256'] != worker.digest(outputs / 'predictions.jsonl')):
        raise ValueError('Gradio verification or IIRC serving replay differs from its receipt')
    if candidate:
        ordinal_error = report['ordinal_l4_max_probability_difference']
        if (report['checkpoint_sha256'] != bundle['checkpoint_sha256']
                or report['ordinal_head_sha256'] != inputs[f'{worker.GRADIO_CHECKPOINT}/ordinal/head.npz']
                or not 0 <= ordinal_error <= GRADIO_ORDINAL_TOLERANCE
                or report['gradio_cached_views_verified'] is not True
                or report['model_forwards_for_three_cached_interactions'] != 0
                or replay['identical_ordered_spans'] != replay['paragraphs']):
            raise ValueError('Candidate checkpoint, ordinal replay, cached views, or anchor ordering failed verification')
    with (outputs / 'predictions.jsonl').open(encoding='utf-8') as stream:
        if len({row['paragraph_id'] for row in map(json.loads, stream)}) != 430:
            raise ValueError('Gradio replay predictions do not cover 430 paragraphs')


def verify_forecast(job, outputs, bundle):
    with TemporaryDirectory() as directory:
        root = Path(directory)
        archive_name = worker.FORECAST_INPUT_ARCHIVE
        worker.unpack(job / archive_name, bundle['archives'][archive_name]['members'], root)
        for dataset, spec in bundle['forecast'].items():
            source = read_stimuli(root / spec['path'], spec['input_sha256'])
            predictions = outputs / dataset / 'predictions.jsonl'
            seal = worker.read_json(outputs / dataset / 'seal.json')
            if (seal['candidate_id'] != bundle['checkpoint_sha256'] or seal['input_sha256'] != source.sha256
                    or seal['predictions_sha256'] != worker.digest(predictions)
                    or not seal['documents'] == seal['forwards'] == spec['documents'] == len(source.documents)
                    or type(seal['unavailable_sentences']) is not int or seal['unavailable_sentences'] < 0):
                raise ValueError(f'Invalid {dataset} forecast seal')
            certified = 0
            with predictions.open(encoding='utf-8') as stream:
                for (identifier, text), row in zip(source.documents, map(json.loads, stream), strict=True):
                    if set(row) != {'id', 'sentences'} or row['id'] != identifier or not isinstance(row['sentences'], list):
                        raise ValueError(f'Unexpected {dataset} forecast row: {identifier}')
                    previous_end, previous_token = 0, -1
                    for sentence in row['sentences']:
                        start, end, token = (sentence[key] for key in ('start', 'end', 'token'))
                        if (any(type(value) is not int for value in (start, end, token))
                                or not previous_end <= start < end <= len(text) or token <= previous_token):
                            raise ValueError(f'Unaligned {dataset} forecast span: {identifier}')
                        for target in ('ease', 'understanding'):
                            p = sentence[f'probability_{target}']
                            if (not isinstance(p, list) or len(p) != 5
                                    or any(type(value) not in (int, float) or not 0 <= value <= 1 for value in p)
                                    or not math.isclose(sum(p), 1, rel_tol=0, abs_tol=1e-12)
                                    or not math.isclose(sentence[target], sum(k * v for k, v in enumerate(p, start=1)),
                                                        rel_tol=0, abs_tol=1e-12)):
                                raise ValueError(f'Invalid {dataset} forecast distribution: {identifier}')
                        previous_end, previous_token = end, token
                        certified += 1
            if not 0 < certified == seal['certified_sentences']:
                raise ValueError(f'{dataset} forecast does not cover the certified sentence inventory')


def verify_article_route(outputs, bundle):
    report = worker.read_json(outputs / 'verification.json')
    if WORD_CUE_ROUTE in bundle:
        verify_word_cue_outputs(outputs, bundle, report)
        return
    if OVERVIEW_HTTP_ROUTE in bundle:
        verify_overview_http_outputs(outputs, bundle, report)
        return
    if 'reference_route' in bundle:
        plan = bundle['reference_route']
        identity = (*REFERENCE_ROUTE_IDENTITY, 'candidate_id', 'fixture_sha256')
        counts = ('model_forwards', 'model_forwards_for_cached_reference', 'reference_queries')
        if (any(report[key] != plan[key] for key in identity)
                or (report['model_revision'], report['device']) != (bundle['revision'], 'NVIDIA L4 FP32')
                or report['verified_exact_source_and_coordinates'] is not True
                or any(type(report[key]) is not int for key in counts)
                or tuple(report[key] for key in counts) != (1, 0, 2)):
            raise ValueError('Gradio did not verify the sealed reference and its two cached queries on L4')
        return
    combined_overview = COMBINED_ARTICLE_ROUTE in bundle
    plan = bundle[COMBINED_ARTICLE_ROUTE if combined_overview else 'article_route']
    identity = ('candidate_id', 'fixture_sha256', 'characters', 'paragraphs', 'source_code_sha256', 'app_sha256')
    if (any(report[key] != plan[key] for key in identity)
            or (report['model_revision'], report['device']) != (bundle['revision'], 'NVIDIA L4 FP32')
            or report['default_article_only'] is not True or report['verified_exact_source_and_coordinates'] is not True
            or (report['model_forwards'], report['model_forwards_for_cached_view']) != (1, 0)
            or not plan['paragraphs'] <= report['tokens'] <= plan['characters']):
        raise ValueError('Gradio did not verify the sealed article and checkpoint in one forward on L4')
    for kind in ('sentence', 'lexical'):
        counts = report[f'{kind}_marks_by_paragraph']
        if (len(counts) != plan['paragraphs'] or any(type(count) is not int or count <= 0 for count in counts)
                or sum(counts) != report[f'{kind}_marks']
                or not re.fullmatch(r'[0-9a-f]{64}', report[f'{kind}_markup_sha256'])):
            raise ValueError(f'Incomplete {kind} heatmap for the sealed article')
    if combined_overview:
        layers = {'combined_ease_lexical_marks_by_paragraph': 'lexical_marks_by_paragraph',
                  'combined_understanding_sentence_marks_by_paragraph': 'sentence_marks_by_paragraph',
                  'combined_understanding_lexical_marks_by_paragraph': 'lexical_marks_by_paragraph'}
        nested, gaps = (report.get(key) for key in ('combined_lexical_marks_nested', 'combined_lexical_marks_in_gaps'))
        if (report.get('verified_combined_overview') is not True
                or type(report.get('cached_view_switches')) is not int or report['cached_view_switches'] != 3
                or any(not isinstance(report.get(key), list) or any(type(count) is not int for count in report[key])
                       or report[key] != report[baseline] for key, baseline in layers.items())
                or type(nested) is not int or nested <= 0 or type(gaps) is not int or gaps < 0
                or nested + gaps != report['lexical_marks']
                or not re.fullmatch(r'[0-9a-f]{64}', report.get('understanding_markup_sha256', ''))
                or worker.digest(outputs / worker.ARTICLE_OVERVIEW_HTML) != report['sentence_markup_sha256']):
            raise ValueError('Incomplete verified dual-layer overview or cached-view replay artifact')


def verify_word_cue_outputs(outputs, bundle, report):
    plan = bundle[WORD_CUE_ROUTE]
    members = bundle['archives'][worker.ARTICLE_ROUTE_ARCHIVE]['members']
    fixtures = report['fixtures']
    expected = plan['fixtures']
    prefix = report['prefix_invariance']
    markup = outputs / worker.WORD_CUE_HTML
    preview = outputs / worker.WORD_CUE_PREVIEW
    if (any(report[key] != members[name] for key, name in WORD_CUE_IDENTITY.items())
            or report['candidate_id'] != plan['candidate_id'] or report['cue_head_id'] != plan['cue_head_id']
            or (report['model_revision'], report['device']) != (bundle['revision'], 'NVIDIA L4')
            or len(fixtures) != len(expected) or {row['id']: row['text_sha256'] for row in fixtures} != expected
            or any(row['offsets_verified'] is not True or type(row['content_forwards']) is not int
                   or row['content_forwards'] != 1 or type(row['cached_forwards']) is not int
                   or row['cached_forwards'] != 0 or type(row['cue_marks']) is not int or row['cue_marks'] < 1
                   for row in fixtures)
            or type(report['content_forwards']) is not int or report['content_forwards'] != len(fixtures)
            or type(report['cached_forwards']) is not int or report['cached_forwards'] != 0
            or prefix['prefix_sha256'] != expected['plain'] or prefix['absolute_tolerance'] != WORD_CUE_PREFIX_TOLERANCE
            or type(prefix['prefix_tokens']) is not int or type(prefix['compared_cues']) is not int
            or not 0 < prefix['compared_cues'] < prefix['prefix_tokens']
            or not 0 <= prefix['maximum_absolute_score_difference'] <= WORD_CUE_PREFIX_TOLERANCE
            or report['markup_sha256'] != worker.digest(markup) or report['preview_sha256'] != worker.digest(preview)
            or preview.read_bytes().count(markup.read_bytes()) != 1):
        raise ValueError('Gradio must verify both exact word-cue fixtures with one L4 forward and cached view changes')


def verify_overview_http_outputs(outputs, bundle, report):
    from research.gradio_app.verify_article_route import SourceMarkup

    plan = bundle[OVERVIEW_HTTP_ROUTE]
    members = bundle['archives'][worker.ARTICLE_ROUTE_ARCHIVE]['members']
    fixture_digest = plan['fixture_sha256']
    spans = [[0, 50], [54, 110]]
    tokens = [13, 29]
    checks = ('verified_exact_source_and_coordinates', 'verified_reference_refresh',
              'verified_radio_success_http', 'verified_other_views')
    preview = outputs / worker.ARTICLE_OVERVIEW_HTML
    if (any(report[key] != members[name] for key, name in OVERVIEW_HTTP_IDENTITY.items())
            or report['candidate_id'] != plan['candidate_id'] or report['fixture_sha256'] != fixture_digest
            or (report['model_revision'], report['device']) != (bundle['revision'], 'NVIDIA L4 FP32')
            or type(report['model_forwards']) is not int or report['model_forwards'] != 2
            or report['model_forwards_per_article'] != [1, 1]
            or type(report['model_forwards_for_cached_views']) is not int or report['model_forwards_for_cached_views'] != 0
            or report['paragraph_sentence_spans'] != spans or report['paragraph_sentence_tokens'] != tokens
            or report['paragraph_sentence_marks'] != 2 or any(report[key] is not True for key in checks)
            or report['refresh_markup_sha256'] != worker.digest(preview)):
        raise ValueError('Gradio did not verify the bound overview, reference refresh, and causal paragraph stops on L4')
    markup = SourceMarkup()
    try:
        markup.feed(preview.read_text(encoding='utf-8'))
        markup.close()
    except AssertionError as error:
        raise ValueError('Invalid refreshed overview markup') from error
    marks = [mark for mark in markup.marks if mark.kind == 'sentence']
    expected = [(*span, str(token)) for span, token in zip(spans, tokens, strict=True)]
    observed = [(mark.start, mark.end, mark.attrs.get('data-token')) for mark in marks]
    source_sha256 = hashlib.sha256(''.join(markup.chunks).encode('utf-8')).hexdigest()
    if (markup.source_count != 1 or markup.in_source or markup.active or source_sha256 != fixture_digest
            or observed != expected or not any(mark.kind == 'lexical' for mark in markup.marks)):
        raise ValueError('The returned overview must preserve source text and both heatmap layers')


def verify_onestop(outputs, bundle):
    from research.stage6.onestop_rereading_extract import CHECKS, PREFIX_TOLERANCE, WIDTH

    report = worker.read_json(outputs / 'seal.json')
    plan = bundle['onestop']
    identity = ('candidate_id', 'stimuli_sha256', 'source_code_sha256', 'positions_sha256',
                'documents', 'stops', 'input_tokens')
    controls = report['causality_controls']
    if (any(report[key] != plan[key] for key in identity)
            or (report['model_revision'], report['device'], report['width']) != (bundle['revision'], 'NVIDIA L4 FP32', WIDTH)
            or report['model_forwards'] != plan['documents'] or report['causality_control_forwards'] != 2 * CHECKS
            or report['causality_tolerance'] != PREFIX_TOLERANCE or len(controls) != CHECKS
            or any(not re.fullmatch(r'[0-9a-f]{64}', key)
                   or checks.keys() != {'standalone_prefix', 'replaced_suffix'}
                   or any(not 0 <= value <= PREFIX_TOLERANCE for value in checks.values())
                   for key, checks in controls.items())
            or any(report[key] != worker.digest(outputs / filename) for key, filename in (
                ('features_sha256', 'features.npz'), ('telemetry_sha256', 'telemetry.jsonl')))):
        raise ValueError('The returned OneStop features do not match the serving backbone and certified stops')


def verify_evidence_forecast(outputs, bundle, plan, extraction_tokens):
    from research.semantic_tracking.expmrc_forecast import METHODS

    report = worker.read_json(outputs / 'seal.json')
    keys = ('candidate_id', 'articles_sha256', 'queries_sha256', 'assay_sha256', 'protocol_sha256', 'heads_sha256',
            'source_code_sha256', 'documents', 'queries', 'candidate_stops', 'certified_stops', 'coalesced_stops',
            'input_tokens')
    predictions = outputs / 'predictions.jsonl'
    extraction = worker.read_json(outputs / 'extraction.json')
    causality = worker.read_json(outputs / 'causality.json')['checks']
    if (any(report[key] != plan[key] for key in keys) or report['model_revision'] != bundle['revision']
            or report['predictions_sha256'] != worker.digest(predictions)
            or report['extraction_sha256'] != worker.digest(outputs / 'extraction.json')
            or (extraction['model'], extraction['revision'], extraction['gpu'], extraction['dtype'])
            != (bundle['model'], bundle['revision'], 'NVIDIA L4', 'float32')
            or extraction['encoded_texts'] != plan['documents'] + plan['queries'] or extraction['excluded_stops']
            or extraction['tokens'] != extraction_tokens or extraction['unique_certified_stops'] != plan['certified_stops']
            or extraction['projected_tokens']['documents'] != plan['documents']
            or not causality or any(not item['passed'] or not item['projected_tokens']['passed'] for item in causality)):
        raise ValueError('The L4 evidence forecasts differ from the sealed candidate and stop inventory')
    with predictions.open(encoding='utf-8') as stream:
        rows = list(map(json.loads, stream))
    if (len({row['id'] for row in rows}) != len(rows) or len(rows) != plan['queries']
            or len({row['article_id'] for row in rows}) != plan['documents']
            or any(row['methods'].keys() != set(METHODS) for row in rows)):
        raise ValueError('Evidence forecasts do not cover all frozen questions and readouts')


def verify_first_readback(job, outputs, bundle):
    from research.semantic_tracking.first_readback_transfer import METHODS, MODES, TARGET_FIELDS, key, read_rows

    plan = bundle['first_readback']
    report = worker.read_json(outputs / 'seal.json')
    keys = ('candidate_id', 'stimuli_sha256', 'targets_sha256', 'train_sha256', 'assay_sha256', 'protocol_sha256',
            'source_code_sha256', 'training_stops', 'heldout_stops')
    extraction = worker.read_json(outputs / 'extraction.json')
    causality = worker.read_json(outputs / 'causality.json')['checks']
    files = ('predictions.jsonl', 'heads.npz', 'extraction.json')
    if (any(report[field] != plan[field] for field in keys) or report['model_revision'] != bundle['revision']
            or any(report[f'{name.split(".")[0]}_sha256'] != worker.digest(outputs / name) for name in files)
            or (extraction['model'], extraction['revision'], extraction['gpu'], extraction['dtype'])
            != (bundle['model'], bundle['revision'], 'NVIDIA L4', 'float32')
            or not extraction['encoded_texts'] == extraction['input_texts'] == plan['documents']
            or extraction['input_sha256'] != plan['stimuli_sha256'] or extraction['excluded_stops']
            or extraction['unique_certified_stops'] != plan['stops']
            or not causality or any(not check['passed'] for check in causality)):
        raise ValueError('The returned first-readback forecasts differ from the fixed Qwen candidate or input stops')
    archive = bundle['archives'][worker.FIRST_READBACK_ARCHIVE]
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        worker.unpack(job / worker.FIRST_READBACK_ARCHIVE, archive['members'], root)
        targets = read_rows(root / worker.FIRST_READBACK_INPUTS['targets'], plan['targets_sha256'], TARGET_FIELDS)
    expected = {key(row): row for row in targets if row['split'] == 'test'}
    methods = {f'{mode}_{method}' for mode in MODES for method in METHODS}
    rows = read_rows(outputs / 'predictions.jsonl', report['predictions_sha256'], TARGET_FIELDS | {'methods'})
    if (not len(rows) == len(expected) == plan['heldout_stops'] or {key(row) for row in rows} != expected.keys()
            or any(row['methods'].keys() != methods or {field: row[field] for field in TARGET_FIELDS} != expected[key(row)]
                   or any(type(p) is not float or not 0 < p < 1 for p in row['methods'].values()) for row in rows)):
        raise ValueError('The first-readback forecasts fail held-out stop alignment')


def verify_context_readback(job, outputs, bundle):
    from research.semantic_tracking.context_readback_forecast import METHODS, PREDICTION_FIELDS, read_rows, read_stimuli

    plan = bundle['context_readback']
    report = worker.read_json(outputs / 'seal.json')
    required = ('candidate_id', 'stimuli_sha256', 'targets_sha256', 'assay_sha256', 'protocol_sha256', 'heads_sha256',
                'source_code_sha256', 'documents', 'forecast_stops')
    extraction = worker.read_json(outputs / 'extraction.json')
    checks = worker.read_json(outputs / 'causality.json')['checks']
    if (any(report[field] != plan[field] for field in required) or report['model_revision'] != bundle['revision']
            or any(report[f'{field}_sha256'] != worker.digest(outputs / f'{field}{suffix}')
                   for field, suffix in (('predictions', '.jsonl'), ('extraction', '.json'), ('causality', '.json')))
            or (extraction['model'], extraction['revision'], extraction['gpu'], extraction['dtype'])
            != (bundle['model'], bundle['revision'], 'NVIDIA L4', 'float32')
            or not extraction['encoded_texts'] == extraction['input_texts'] == plan['documents']
            or extraction['input_sha256'] != plan['stimuli_sha256'] or extraction['excluded_stops']
            or extraction['unique_certified_stops'] != plan['stops']
            or len(checks) != 12 or any(not check['passed'] for check in checks)):
        raise ValueError('Independent readback forecasts differ from the frozen candidate and boundary preflight')
    archive = bundle['archives'][worker.CONTEXT_READBACK_ARCHIVE]
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        worker.unpack(job / worker.CONTEXT_READBACK_ARCHIVE, archive['members'], root)
        stimuli = read_stimuli(root / worker.CONTEXT_READBACK_STIMULI, plan['stimuli_sha256'])
    expected = {(row['id'], stop) for row in stimuli for stop in row['stops'][1:]}
    rows = read_rows(outputs / 'predictions.jsonl', report['predictions_sha256'], PREDICTION_FIELDS)
    if (not len(rows) == len(expected) == plan['forecast_stops']
            or {(row['id'], row['stop']) for row in rows} != expected
            or any(row['methods'].keys() != set(METHODS)
                   or any(type(p) is not float or not 0 < p < 1 for p in row['methods'].values()) for row in rows)):
        raise ValueError('Independent readback predictions do not exactly cover the outcome-free display regions')


def verify_medreadme(job, outputs, bundle):
    from research.semantic_tracking.iirc_anchor_fit import WORD

    plan = bundle['medreadme']
    archive = bundle['archives'][worker.MEDREADME_ARCHIVE]
    predictions = outputs / 'predictions.jsonl'
    seal = worker.read_json(outputs / 'seal.json')
    if (any(seal[key] != plan[key] for key in ('candidate_id', 'lexical_weights_sha256', 'stimuli_sha256',
                                              'forecast_code_sha256'))
            or seal['model_forwards'] != plan['retained_sentences']
            or seal['forecasts_sha256'] != worker.digest(predictions)):
        raise ValueError('MedReadMe served lexical outputs differ from the exact single-forward receipt')
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        worker.unpack(job / worker.MEDREADME_ARCHIVE, archive['members'], root)
        source = read_stimuli(root / worker.MEDREADME_STIMULI, plan['stimuli_sha256'])
    if len(source.documents) != plan['retained_sentences']:
        raise ValueError('MedReadMe forecast sentence inventory differs from the preregistration')
    with predictions.open(encoding='utf-8') as stream:
        for (identifier, text), row in zip(source.documents, map(json.loads, stream), strict=True):
            if row.keys() != {'id', 'words'} or row['id'] != identifier or not isinstance(row['words'], list):
                raise ValueError(f'Unexpected MedReadMe forecast row: {identifier}')
            expected, previous_end = {word.span() for word in WORD.finditer(text)}, 0
            for start, end, score in row['words']:
                if (type(start) is not int or type(end) is not int or (start, end) not in expected
                        or start < previous_end or type(score) not in (int, float) or not 0 <= score <= 1
                        or round(score, 2) != score):
                    raise ValueError(f'Unexpected MedReadMe source word or displayed lexical score: {identifier}')
                previous_end = end


def verify_later(job, outputs, bundle):
    extraction = worker.read_json(outputs / 'features/extraction.json')
    causality = worker.read_json(outputs / 'features/causality.json')
    seal = worker.read_json(outputs / 'fit/fit-manifest.json')
    progress = worker.read_json(outputs / 'features/progress.json')
    materials = bundle['later_data']
    tokens = sum(split['tokens'] for split in materials.values())
    inputs = bundle['archives']['later-model-inputs.tar.gz']['members']
    if (any((item['model'], item['revision'], item['assay_sha256']) !=
            (bundle['model'], bundle['revision'], inputs['assay.json']) for item in (extraction, seal))
            or extraction['gpu'] != 'NVIDIA L4' or extraction['dtype'] != 'float32' or extraction['tokens'] != tokens
            or extraction['input_paths'] != sum(split['paragraphs'] for split in materials.values())
            or extraction['certified_character_stops'] != sum(split['stops'] for split in materials.values())
            or extraction['inputs_sha256'] != {'assay': inputs['assay.json'],
               **{name: inputs[name] for name in ('train/stimuli.jsonl', 'dev/stimuli.jsonl')}}
            or extraction['causality_sha256'] != worker.digest(outputs / 'features/causality.json')
            or extraction['plan_sha256'] != worker.digest(outputs / 'features/plan.json')
            or causality['plan_sha256'] != extraction['plan_sha256'] or len(causality['checks']) != 12
            or not all(row['passed'] and row['projected_tokens']['passed'] for row in causality['checks'])
            or progress['plan_sha256'] != extraction['plan_sha256'] or not progress['complete']
            or progress['tokens_completed'] != tokens or seal['features_sha256'] != extraction['features_sha256']
            or seal['extraction_sha256'] != worker.digest(outputs / 'features/extraction.json')
            or seal['heads_sha256'] != worker.digest(outputs / 'fit/heads.npz')
            or seal['predictions_sha256'] != worker.digest(outputs / 'fit/predictions.jsonl')
            or seal['train_cases'] != materials['train']['questions'] or seal['dev_cases'] != materials['dev']['questions']):
        raise ValueError('Same-paragraph extraction or fit receipts disagree with the sealed assay')
    expected = worker.read_json(manifest_path(job))['later_forecasts']
    with (outputs / 'fit/predictions.jsonl').open(encoding='utf-8') as stream:
        for row in map(json.loads, stream):
            count = expected.pop(row['case_id'])
            if (set(row['methods']) != {'geometry', 'path_text', 'anchor_conditioned'} or any(
                    len(points) != count or any(not math.isfinite(h) or not 0 < h < 1 for h in points)
                    for points in row['methods'].values())):
                raise ValueError('Later-support prediction has invalid hazard outputs')
    if expected:
        raise ValueError('Later-support predictions do not cover all sealed questions')


def verify_cwi(job, outputs, bundle):
    extraction = worker.read_json(outputs / 'extraction/extraction.json')
    causality = worker.read_json(outputs / 'extraction/causality.json')
    seal = worker.read_json(outputs / 'fit/fit-manifest.json')
    cwi_inputs = bundle['archives']['cwi-model-inputs.tar.gz']['members']
    protocol = bundle['archives']['cwi-source.tar.gz']['members'][CWI_PROTOCOL]
    if (any((item['model'], item['revision']) != (bundle['model'], bundle['revision']) for item in (extraction, seal))
            or seal['assay_sha256'] != cwi_inputs['assay.json'] or seal['protocol_sha256'] != protocol
            or seal['inputs_sha256'] != {name: value for name, value in cwi_inputs.items() if name != 'assay.json'}
            or extraction['input_sha256'] != cwi_inputs['stimuli.jsonl'] or extraction['gpu'] != 'NVIDIA L4'
            or extraction['segment_means'] is not False or extraction['dtype'] != 'float32'
            or extraction['input_texts'] != sum(split['documents'] for split in bundle['cwi_counts'].values())
            or seal['extraction_sha256'] != worker.digest(outputs / 'extraction/extraction.json')
            or seal['heads_sha256'] != worker.digest(outputs / 'fit/heads.npz')
            or seal['lexicon_sha256'] != worker.digest(outputs / 'fit/lexicon.json')
            or seal['predictions_sha256'] != worker.digest(outputs / 'fit/cwi-predictions.jsonl')
            or seal['annotator_order'] != ['native', 'nonnative']
            or len(causality['checks']) != 12 or not all(row['passed'] for row in causality['checks'])
            or set(seal['methods']) != set(bundle['cwi_methods'])):
        raise ValueError('CWI transfer forecasts or receipts disagree with the sealed protocol')

    excluded = {(row['id'], row['stop']) for row in extraction['excluded_stops']}
    with tarfile.open(job / 'payload.tar.gz', 'r:gz') as payload:
        with payload.extractfile('cwi-model-inputs.tar.gz') as archive:
            with tarfile.open(fileobj=BytesIO(archive.read()), mode='r:gz') as inputs:
                with inputs.extractfile('cwi-eval.jsonl') as eval_file:
                    eligible = {(row['id'], row['start'], row['stop']): row['split'] for row in map(json.loads, eval_file)
                                if (row['id'], row['stop']) not in excluded}
    counts = {'dev': 0, 'test': 0}
    methods = set(bundle['cwi_methods'])
    with (outputs / 'fit/cwi-predictions.jsonl').open(encoding='utf-8') as stream:
        for row in map(json.loads, stream):
            split = eligible.pop((row['id'], row['start'], row['stop']))
            if any(len(row[name]) != 2 or any(not math.isfinite(p) or not 0 < p < 1 for p in row[name]) for name in methods):
                raise ValueError('CWI forecast must contain two finite probabilities per method')
            counts[split] += 1
    if eligible or counts != seal['forecast_candidates']:
        raise ValueError('CWI forecasts differ from the certified evaluation coordinates or counts')


def verify_plaba(job, outputs, bundle):
    extraction = worker.read_json(outputs / 'extraction/extraction.json')
    causality = worker.read_json(outputs / 'extraction/causality.json')
    seal = worker.read_json(outputs / 'forecast-manifest.json')
    inputs = bundle['archives']['plaba-model-inputs.tar.gz']['members']
    inventory = bundle['plaba_span']
    if (any((item['model'], item['revision']) != (bundle['model'], bundle['revision']) for item in (extraction, seal))
            or seal['assay_sha256'] != inputs['assay.json'] or seal['heads_sha256'] != inputs['heads.npz']
            or extraction['input_sha256'] != inputs['stimuli.jsonl'] or extraction['gpu'] != 'NVIDIA L4'
            or extraction['dtype'] != 'float32' or extraction['tokens'] != inventory['tokens']
            or extraction['input_texts'] != inventory['documents']
            or extraction['unique_certified_stops'] != inventory['certified_sentences']
            or extraction['projected_tokens']['components'] != 128 or extraction['excluded_stops']
            or seal['extraction_sha256'] != worker.digest(outputs / 'extraction/extraction.json')
            or seal['causality_sha256'] != worker.digest(outputs / 'extraction/causality.json')
            or len(causality['checks']) != 12
            or not all(row['passed'] and row['projected_tokens']['passed'] for row in causality['checks'])
            or seal['predictions_sha256'] != worker.digest(outputs / 'predictions.jsonl')
            or seal['sentences'] != inventory['sentences']):
        raise ValueError('PLABA span predictions or extraction differ from the sealed inputs')

    with tarfile.open(job / 'payload.tar.gz', 'r:gz') as payload:
        with payload.extractfile('plaba-model-inputs.tar.gz') as archive:
            with tarfile.open(fileobj=BytesIO(archive.read()), mode='r:gz') as materials:
                with materials.extractfile('sentences.jsonl') as coordinates:
                    sentences = {row['sentence_id']: row for row in map(json.loads, coordinates)}
    with (outputs / 'predictions.jsonl').open(encoding='utf-8') as forecasts:
        for row in map(json.loads, forecasts):
            sentence = sentences.pop(row['sentence_id'])
            if (set(row['methods']) != set(inventory['methods'])
                    or any(len(spans) > inventory['top_k'] or (spans and sentence['token_count'] is None)
                           or any(not sentence['start'] <= span['start'] < span['end'] <= sentence['end']
                                  or not math.isfinite(span['score']) or not 0 <= span['score'] <= 1 for span in spans)
                           for spans in row['methods'].values())):
                raise ValueError('PLABA span predictions exceed the author sentence or model output bounds')
    if sentences:
        raise ValueError('PLABA span predictions do not cover the sealed sentence IDs')


def verify_textde(job, outputs, bundle):
    extraction = worker.read_json(outputs / 'extraction/extraction.json')
    causality = worker.read_json(outputs / 'extraction/causality.json')
    seal = worker.read_json(outputs / 'forecast/seal.json')
    inputs = bundle['archives']['textde-model-inputs.tar.gz']['members']
    code = bundle['archives']['textde-source.tar.gz']['members']['research/stage6/textcomplexityde_eval.py']
    inventory = bundle['textde']
    transformer = next(requirement.split('==')[1] for requirement in bundle['requirements']
                       if requirement.startswith('transformers=='))
    if (any((item['model'], item['revision']) != (bundle['model'], bundle['revision']) for item in (extraction, seal))
            or extraction['gpu'] != 'NVIDIA L4' or extraction['dtype'] != 'float32' or extraction['segment_means'] is not False
            or extraction['transformers'] != transformer or extraction['input_texts'] != inventory['cases']
            or extraction['input_sha256'] != inputs['stimuli.jsonl'] or seal['assay_sha256'] != inputs['assay.json']
            or seal['heads_sha256'] != inputs['heads.npz'] or seal['evaluation_code_sha256'] != code
            or seal['extraction_sha256'] != worker.digest(outputs / 'extraction/extraction.json')
            or seal['forecasts_sha256'] != worker.digest(outputs / 'forecast/predictions.jsonl')
            or seal['methods'] != inventory['forecast_methods'] or seal['target_order'] != inventory['target_order']
            or seal['forecasts'] != extraction['encoded_texts'] or seal['forecasts'] != extraction['unique_certified_stops']
            or len(causality['checks']) != 12 or not all(row['passed'] for row in causality['checks'])):
        raise ValueError('TextComplexityDE predictions or extraction differ from the sealed readout')

    with tarfile.open(job / 'payload.tar.gz', 'r:gz') as payload:
        with payload.extractfile('textde-model-inputs.tar.gz') as archive:
            with tarfile.open(fileobj=BytesIO(archive.read()), mode='r:gz') as materials:
                with materials.extractfile('stimuli.jsonl') as stops:
                    expected = {row['id']: row['stops'][0] for row in map(json.loads, stops)}
    if 'textde_control_stops' in bundle:
        controls = {(row['id'], row['stop']) for row in causality['checks']}
        if (causality.get('control_stops') != bundle['textde_control_stops'] or len(controls) != 3
                or any(not 0 < stop < expected[case] for case, stop in controls)):
            raise ValueError('TextComplexityDE requires separately certified interior controls')
    for row in extraction['excluded_stops']:
        if expected.pop(row['id']) != row['stop']:
            raise ValueError('TextComplexityDE exclusion does not match a sealed sentence end')
    if not expected or seal['forecasts'] != len(expected):
        raise ValueError('TextComplexityDE encoded or excluded stops differ from the expected sentence inventory')
    with (outputs / 'forecast/predictions.jsonl').open(encoding='utf-8') as forecasts:
        for row in map(json.loads, forecasts):
            if (row['stop'] != expected.pop(row['id']) or set(row['methods']) != set(inventory['forecast_methods'])
                    or any(len(scores) != len(inventory['target_order']) or
                           any(not math.isfinite(score) or not 1 <= score <= 5 for score in scores)
                           for scores in row['methods'].values())):
                raise ValueError('TextComplexityDE forecast exceeds the certified coordinates or rating range')
    if expected:
        raise ValueError('TextComplexityDE forecasts omit certified sentences')


def verify_interior_controls(causality, sentence_ends, dataset, *, projected=False):
    stable = {'substring', 'right_padding', 'suffix_replacement'}
    positive = 'changed_suffix_positive_control'
    comparisons = stable | {positive}
    controls = {}
    for row in causality['checks']:
        case, stop, comparison = row['id'], row['stop'], row['comparison']
        group = controls.setdefault((case, stop), set())
        if not 0 < stop < sentence_ends[case] or comparison not in comparisons or comparison in group:
            raise ValueError(f'{dataset} independently certified interior token controls are misaligned')
        metrics = (row, row['projected_tokens']) if projected else (row,)
        for check in metrics:
            error, maximum = check['max_token_relative_rms'], check['max_absolute']
            if (check['passed'] is not True or not math.isfinite(error) or not math.isfinite(maximum) or maximum < 0
                    or not (error > 0.02 if comparison == positive else 0 <= error <= 0.002)):
                raise ValueError(f'{dataset} independently certified interior token control failed')
        group.add(comparison)
    if (causality['control_stops'] != TEXTDE_INTERIOR_CONTROLS or len(controls) != 3
            or len({case for case, _ in controls}) != 3 or any(group != comparisons for group in controls.values())):
        raise ValueError(f'{dataset} requires four causal comparisons for three distinct input texts')


def verify_student_annotations(job, outputs, bundle):
    import numpy as np

    output = outputs / 'extraction'
    extraction = worker.read_json(output / 'extraction.json')
    causality = worker.read_json(output / 'causality.json')
    seal = worker.read_json(output / 'seal.json')
    spec = bundle['student_annotations']
    hashes = bundle['archives'][worker.STUDENT_DATA_ARCHIVE]['members']
    source = bundle['archives'][worker.STUDENT_SOURCE_ARCHIVE]['members']
    with tarfile.open(job / 'payload.tar.gz', 'r:gz') as payload:
        with payload.extractfile(worker.STUDENT_DATA_ARCHIVE) as archive:
            with tarfile.open(fileobj=BytesIO(archive.read()), mode='r:gz') as materials:
                plan, expected = student_inputs(materials, hashes)
    if (any((item['model'], item['revision']) != (bundle['model'], bundle['revision']) for item in (extraction, seal))
            or extraction['gpu'] != plan['planned_accelerator'] or extraction['dtype'] != plan['dtype']
            or extraction['transformers'] != plan['transformers'] or extraction['segment_means'] is not False
            or extraction['tokens'] != plan['total_input_tokens'] or extraction['excluded_stops'] != []
            or not extraction['input_texts'] == extraction['encoded_texts'] == plan['documents']
            or extraction['unique_certified_stops'] != plan['expected_certified_stops']
            or extraction['input_sha256'] != hashes['stimuli.jsonl']
            or not seal['tokenizer_sha256'] == spec['tokenizer_sha256'] == plan['tokenizer_sha256']
            or seal['preparation_sha256'] != spec['preparation_sha256'] or seal['inputs_sha256'] != hashes
            or seal['source_sha256'] != source or 'projected_tokens' in extraction
            or any(seal[field] != worker.digest(output / name) for field, name in worker.STUDENT_ARTIFACTS.items())):
        raise ValueError('StudentAnnotations state extraction differs from its sealed source, input, or model')

    verify_interior_controls(causality, expected, 'StudentAnnotations')
    count = len(expected)
    with np.load(output / 'features.npz', allow_pickle=False) as archive:
        if len(archive.files) != 3 or set(archive.files) != {'ids', 'stops', 'states'}:
            raise ValueError('StudentAnnotations state archive contains an unexpected array')
        ids, stops, states = archive['ids'], archive['stops'], archive['states']
        if (ids.shape != (count,) or ids.dtype.kind != 'U' or stops.shape != (count,) or stops.dtype.kind != 'i'
                or dict(zip(ids.tolist(), stops.tolist(), strict=True)) != expected
                or states.shape != (count, spec['state_width']) or states.dtype != np.float32
                or not bool(np.isfinite(states).all()) or not bool(np.any(states, axis=1).all())):
            raise ValueError('StudentAnnotations state rows do not match the certified sentence-end inventory')


def read_projected_states(path, coordinate, expected):
    import numpy as np

    count = len(expected)
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != 3 or set(archive.files) != {'ids', coordinate, 'states'}:
            raise ValueError(f'{path.name} contains an unexpected state array')
        ids, stops, states = archive['ids'], archive[coordinate], archive['states']
    keys = list(zip(ids.tolist(), stops.tolist(), strict=True))
    coordinate_dtype = 'i' if coordinate == 'stops' else 'U'
    if (ids.shape != (count,) or ids.dtype.kind != 'U' or stops.shape != (count,) or stops.dtype.kind != coordinate_dtype
            or set(keys) != expected or states.shape != (count, ambistory.PROJECTION_WIDTH) or states.dtype != np.float32
            or not bool(np.isfinite(states).all()) or not np.allclose(np.linalg.norm(states, axis=1), 1, atol=2e-6, rtol=0)):
        raise ValueError(f'{path.name} does not match the supplied AmbiStory coordinate inventory or float32 unit norms')
    return keys, states


def verify_ambistory(job, outputs, bundle):
    output = outputs / 'extraction'
    extraction = worker.read_json(output / 'extraction.json')
    causality = worker.read_json(output / 'causality.json')
    seal = worker.read_json(output / 'seal.json')
    spec = bundle['ambistory']
    inputs = bundle['archives'][worker.AMBISTORY_DATA_ARCHIVE]['members']
    source = bundle['archives'][worker.AMBISTORY_SOURCE_ARCHIVE]['members']
    inventory = spec['splits']
    if (any((item['model'], item['revision']) != (bundle['model'], bundle['revision']) for item in (extraction, seal))
            or extraction['gpu'] != 'NVIDIA L4' or extraction['dtype'] != 'float32'
            or f"transformers=={extraction['transformers']}" not in bundle['requirements']
            or extraction['projection_width'] != ambistory.PROJECTION_WIDTH
            or not extraction['projection_sha256'] == seal['projection_sha256'] == spec['projection_sha256']
            or extraction['story_splits'] != {split: row['stories'] for split, row in inventory.items()}
            or extraction['stories'] != sum(row['stories'] for row in inventory.values())
            or extraction['story_stops'] != sum(row['certified_stops'] for row in inventory.values())
            or extraction['story_tokens'] != sum(row['tokens'] for row in inventory.values())
            or not extraction['sense_definitions'] == extraction['sense_examples'] == spec['families'] * 2
            or extraction['sense_tokens'] < extraction['sense_definitions'] + extraction['sense_examples']
            or seal['preparation_sha256'] != spec['preparation_sha256'] or seal['inputs_sha256'] != inputs
            or seal['source_sha256'] != source
            or any(seal[field] != worker.digest(output / name) for field, name in worker.AMBISTORY_ARTIFACTS.items())):
        raise ValueError('AmbiStory sparse states or receipts differ from the sealed setup')
    with TemporaryDirectory() as temporary, tarfile.open(job / 'payload.tar.gz', 'r:gz') as payload:
        root = Path(temporary)
        path = root / worker.AMBISTORY_DATA_ARCHIVE
        with payload.extractfile(worker.AMBISTORY_DATA_ARCHIVE) as stream, path.open('wb') as file:
            shutil.copyfileobj(stream, file)
        worker.unpack(path, inputs, root / 'inputs')
        stories, senses, prefixes, _ = ambistory.read_inputs(root / 'inputs')

    verify_interior_controls(causality, {case: len(row['text']) for case, row in stories.items()}, 'AmbiStory', projected=True)
    keys, states = read_projected_states(output / 'story-states.npz', 'stops',
                                         {(case, stop) for case, story in stories.items() for stop in story['stops']})
    read_projected_states(output / 'sense-states.npz', 'kinds', set(senses))
    common_prefix = ambistory.prefix_report(keys, states, prefixes)
    reported = extraction['common_prefix']
    if (reported.keys() != common_prefix.keys()
            or any(reported[key] != common_prefix[key] for key in ('families', 'comparisons'))
            or any(not math.isclose(reported[key], common_prefix[key], rel_tol=1e-9, abs_tol=1e-12)
                   for key in ('max_relative_rms', 'max_absolute'))):
        raise ValueError('AmbiStory shared-prefix report differs from archived states')


def stages_for(data):
    return FORMATS[data['schema_version']][2]


def collect(machine, deadline):
    returned = machine.data.setdefault('stages', {})
    stages = stages_for(machine.data)
    attempts = dict.fromkeys(stages, 0)
    previous = None
    launch_deadline = monotonic() + 50
    while monotonic() + 25 < deadline:
        if machine.download('/status.json', 'remote-status.json'):
            status = worker.read_json(machine.job / 'remote-status.json')
            if (any(status[key] != machine.data[key] for key in ('run_id', 'worker_sha256', 'payload_sha256'))
                    or set(status['stages']) != set(stages)):
                raise ValueError('Remote batch status differs from its owner and payload')
            if status['phase'] != previous:
                previous = status['phase']
                machine.update(f'remote-{previous}')
            outstanding = False
            for stage, receipt in status['stages'].items():
                if stage in returned or receipt['state'] not in {'complete', 'failed'}:
                    continue
                attempts[stage] += 1
                remaining = deadline - monotonic() - 20
                if remaining <= 0:
                    raise TimeoutError('No download time remains before runtime cleanup')
                transfer_seconds = {'student-annotations': 40, 'ambistory': 40, 'gum-bridge': 110}.get(stage, 19)
                if not machine.download(f'/return/{stage}.tar.gz', f'{stage}.tar.gz', timeout=min(transfer_seconds, remaining)):
                    if attempts[stage] < 2:
                        outstanding = True
                        continue
                    returned[stage] = {'verified': False, 'error': 'DownloadFailed', **receipt}
                else:
                    try:
                        verified = validate_stage(machine.job, stage, receipt)
                        returned[stage] = {'verified': verified, **receipt}
                    except (KeyError, ValueError, TypeError, OSError, tarfile.TarError) as error:
                        returned[stage] = {'verified': False, 'error': str(error), **receipt}
                machine.update(f'retrieved-{stage}')
            if status['finished'] and not outstanding:
                if status['phase'] == 'worker-failed':
                    machine.download('.launch.log', 'remote-launch.log')
                return
        elif previous is None and monotonic() >= launch_deadline:
            machine.download('.launch.log', 'remote-launch.log')
            raise RuntimeError('No initial batch status received from the worker')
        sleep(min(10, max(0, deadline - monotonic() - 25)))
    raise TimeoutError('Batch deadline reached; verified earlier stages are retained')


def release(machine):
    machine.update('releasing')
    machine.usage('before-stop', timeout=12)
    machine.data['endpoint_absent_after_stop'] = machine.release()
    machine.data['release_usage_recorded'] = machine.usage('after-stop')
    machine.data['release_verified'] = (machine.data['endpoint_absent_after_stop'] and machine.data['release_usage_recorded'])


def run(job, data):
    verify(job, data)
    with run_lock(job, 'x'):
        return run_owned(Colab(job, data))


def run_owned(machine):
    data, job = machine.data, machine.job
    attempted = False

    def interrupt(signum, _):
        raise KeyboardInterrupt(f'Batch interrupted by signal {signum}')

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, interrupt)
    try:
        if re.search(rf'^Version: {re.escape(CLI_VERSION)}$', machine.cli('version', timeout=20).stdout, re.M) is None:
            raise RuntimeError(f'Expected shared Colab CLI {CLI_VERSION}')
        before = machine.sessions()
        if machine.endpoint(before) is not None:
            raise RuntimeError('The prepared batch name is already active')
        data['endpoints_before_allocation'] = sorted(before)
        machine.usage('before-allocation', required=True)
        machine.update('allocating')
        deadline = monotonic() + data['minutes_budget'] * 60 - RELEASE_RESERVE_S
        attempted = True
        machine.cli('new', '-s', job.name, '--gpu', 'L4', timeout=75)
        endpoint = machine.endpoint(machine.sessions())
        if not endpoint or endpoint in before:
            raise RuntimeError('Could not identify the created L4 endpoint')
        data['endpoint'] = endpoint
        machine.update('allocated')
        machine.usage('after-allocation')
        for filename, suffix in (('payload.tar.gz', '.tar.gz'), ('worker.py', '.worker.py')):
            machine.cli('upload', '-s', job.name, job / filename, f'{machine.remote}{suffix}', timeout=65)
        machine.launch(deadline)
        machine.update('worker-running')
        collect(machine, deadline)
    except (RuntimeError, ValueError, KeyError, OSError, subprocess.SubprocessError, KeyboardInterrupt) as error:
        data['failure'] = str(error) or type(error).__name__
    finally:
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, signal.SIG_IGN)
        if attempted:
            release(machine)
        ok = ('failure' not in data and data.get('release_verified') and
              all(data.get('stages', {}).get(stage, {}).get('verified') for stage in stages_for(data)))
        machine.update('finished' if ok else 'incomplete')
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    pack = actions.add_parser('prepare')
    for option in ('gradio-root', 'later-root', 'cwi-root', 'plaba-root', 'textde-root', 'output'):
        pack.add_argument(f'--{option}', type=Path, required=True)
    textde_pack = actions.add_parser('prepare-textde')
    textde_pack.add_argument('--textde-root', type=Path, required=True)
    textde_pack.add_argument('--output', type=Path, required=True)
    student_pack = actions.add_parser('prepare-student-annotations')
    student_pack.add_argument('--student-annotations-root', type=Path,
                              default=gradio.CACHE / 'studentannotations/guide-v1')
    student_pack.add_argument('--output', type=Path, required=True)
    ambistory_pack = actions.add_parser('prepare-ambistory')
    ambistory_pack.add_argument('--ambistory-root', type=Path, default=gradio.CACHE / 'ambistory/causal-v3')
    ambistory_pack.add_argument('--output', type=Path, required=True)
    reading_pack = actions.add_parser('prepare-reading-extractions')
    reading_pack.add_argument('--student-job', type=Path, required=True)
    reading_pack.add_argument('--ambistory-job', type=Path, required=True)
    reading_pack.add_argument('--output', type=Path, required=True)
    gradio_pack = actions.add_parser('prepare-gradio')
    gradio_pack.add_argument('--gradio-root', type=Path, required=True)
    gradio_pack.add_argument('--output', type=Path, required=True)
    gradio_pack.add_argument('--with-forecasts', action='store_true', help='Add the preregistered Blott and Kerwer forecasts')
    medreadme_pack = actions.add_parser('prepare-medreadme')
    medreadme_pack.add_argument('--medreadme-root', type=Path, default=gradio.CACHE / 'medreadme-audit/lexical-holdout-v3')
    medreadme_pack.add_argument('--gradio-job', type=Path, required=True)
    medreadme_pack.add_argument('--output', type=Path, required=True)
    cwi_news_pack = actions.add_parser('prepare-cwi-news')
    for field in ('cwi-news-root', 'gradio-job', 'output'):
        cwi_news_pack.add_argument(f'--{field}', type=Path, required=True)
    cwi_wikinews_pack = actions.add_parser('prepare-cwi-wikinews')
    for field in ('cwi-wikinews-root', 'gradio-job', 'output'):
        cwi_wikinews_pack.add_argument(f'--{field}', type=Path, required=True)
    article_pack = actions.add_parser('prepare-article-route')
    article_pack.add_argument('--gradio-job', type=Path, required=True)
    article_pack.add_argument('--reference-job', type=Path, required=True)
    article_pack.add_argument('--output', type=Path, required=True)
    overview_http_pack = actions.add_parser('prepare-overview-http')
    overview_http_pack.add_argument('--base-job', type=Path, required=True)
    overview_http_pack.add_argument('--output', type=Path, required=True)
    word_cue_pack = actions.add_parser('prepare-word-cue-route')
    word_cue_pack.add_argument('--overview-job', type=Path, required=True)
    word_cue_pack.add_argument('--output', type=Path, required=True)
    reference_pack = actions.add_parser('prepare-reference-route')
    reference_pack.add_argument('--article-job', type=Path, required=True)
    reference_pack.add_argument('--output', type=Path, required=True)
    onestop_pack = actions.add_parser('prepare-onestop')
    onestop_pack.add_argument('--assay-root', type=Path, required=True)
    onestop_pack.add_argument('--preflight', type=Path, required=True)
    onestop_pack.add_argument('--gradio-job', type=Path, required=True)
    onestop_pack.add_argument('--output', type=Path, required=True)
    expmrc_pack = actions.add_parser('prepare-expmrc')
    expmrc_pack.add_argument('--assay-root', type=Path, required=True)
    expmrc_pack.add_argument('--preflight', type=Path, required=True)
    expmrc_pack.add_argument('--gradio-job', type=Path, required=True)
    expmrc_pack.add_argument('--output', type=Path, required=True)
    contractnli_pack = actions.add_parser('prepare-contractnli')
    contractnli_pack.add_argument('--experiment', choices=tuple(CONTRACTNLI_PROTOCOLS), default='contractnli')
    for field in ('packet', 'assay', 'reference-job', 'output'):
        contractnli_pack.add_argument(f'--{field}', type=Path, required=True)
    for action in ('prepare-nq-evidence', 'prepare-qasper-tail'):
        pack = actions.add_parser(action)
        for field in ('assay-root', 'preflight', 'snapshot', 'gradio-job', 'output'):
            pack.add_argument(f'--{field}', type=Path, required=True)
    gum_pack = actions.add_parser('prepare-gum-bridge')
    for field in ('assay-root', 'preflight', 'gradio-job', 'output'):
        gum_pack.add_argument(f'--{field}', type=Path, required=True)
    beukers_pack = actions.add_parser('prepare-beukers')
    for field in ('assay-root', 'preflight', 'gradio-job', 'output'):
        beukers_pack.add_argument(f'--{field}', type=Path, required=True)
    projection_pack = actions.add_parser('prepare-projection')
    for field in ('assay-root', 'protocol', 'preflight', 'gradio-job', 'output'):
        projection_pack.add_argument(f'--{field}', type=Path, required=True)
    frick_signals_pack = actions.add_parser('prepare-frick-signals')
    for field in ('stimuli', 'gradio-job', 'output'):
        frick_signals_pack.add_argument(f'--{field}', type=Path, required=True)
    clear_pack = actions.add_parser('prepare-clear-ease')
    for field in ('prepared', 'gradio-job', 'output'):
        clear_pack.add_argument(f'--{field}', type=Path, required=True)
    reader_projection_actions = {'prepare-core-vocabulary': CORE_VOCABULARY_PROTOCOL,
                                 'prepare-core-vocabulary-disjoint': CORE_VOCABULARY_DISJOINT_PROTOCOL,
                                 'prepare-act': ACT_PROTOCOL}
    for action in reader_projection_actions:
        readout_pack = actions.add_parser(action)
        for field in ('assay-root', 'preflight', 'gradio-job', 'output'):
            readout_pack.add_argument(f'--{field}', type=Path, required=True)
    readback_pack = actions.add_parser('prepare-first-readback')
    readback_pack.add_argument('--assay-root', type=Path, required=True)
    readback_pack.add_argument('--preflight', type=Path, required=True)
    readback_pack.add_argument('--gradio-job', type=Path, required=True)
    readback_pack.add_argument('--output', type=Path, required=True)
    context_pack = actions.add_parser('prepare-context-readback')
    context_pack.add_argument('--assay-root', type=Path, required=True)
    context_pack.add_argument('--preflight', type=Path, required=True)
    context_pack.add_argument('--gradio-job', type=Path, required=True)
    context_pack.add_argument('--output', type=Path, required=True)
    for action in ('verify', 'run', 'status', 'release'):
        actions.add_parser(action).add_argument('job', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare-gum-bridge':
        prepare_gum_bridge(args.assay_root, args.preflight, args.gradio_job, args.output)
        return 0
    if args.action == 'prepare-beukers':
        prepare_beukers(args.assay_root, args.preflight, args.gradio_job, args.output)
        return 0
    if args.action == 'prepare-projection':
        prepare_projection(args.assay_root, args.protocol, args.preflight, args.gradio_job, args.output)
        return 0
    if args.action == 'prepare-frick-signals':
        prepare_frick_signals(args.stimuli, args.gradio_job, args.output)
        return 0
    if args.action == 'prepare-clear-ease':
        prepare_clear_ease(args.prepared, args.gradio_job, args.output)
        return 0
    if args.action in reader_projection_actions:
        prepare_projection(args.assay_root, gradio.REPO / reader_projection_actions[args.action], args.preflight,
                           args.gradio_job, args.output, stage='beukers')
        return 0
    if args.action == 'prepare':
        prepare(args.gradio_root, args.later_root, args.cwi_root, args.plaba_root, args.textde_root, args.output)
        return 0
    if args.action == 'prepare-textde':
        prepare_textde(args.textde_root, args.output)
        return 0
    if args.action == 'prepare-student-annotations':
        prepare_student_annotations(args.student_annotations_root, args.output)
        return 0
    if args.action == 'prepare-ambistory':
        prepare_ambistory(args.ambistory_root, args.output)
        return 0
    if args.action == 'prepare-reading-extractions':
        prepare_reading_extractions(args.student_job, args.ambistory_job, args.output)
        return 0
    if args.action == 'prepare-gradio':
        prepare_gradio(args.gradio_root, args.output, with_forecasts=args.with_forecasts)
        return 0
    if args.action == 'prepare-medreadme':
        prepare_word_holdout(args.medreadme_root, args.gradio_job, args.output, 'medreadme')
        return 0
    if args.action == 'prepare-cwi-news':
        prepare_word_holdout(args.cwi_news_root, args.gradio_job, args.output, 'cwi_news')
        return 0
    if args.action == 'prepare-cwi-wikinews':
        prepare_word_holdout(args.cwi_wikinews_root, args.gradio_job, args.output, 'cwi_wikinews')
        return 0
    if args.action == 'prepare-article-route':
        prepare_article_route(args.gradio_job, args.reference_job, args.output)
        return 0
    if args.action == 'prepare-overview-http':
        prepare_overview_http(args.base_job, args.output)
        return 0
    if args.action == 'prepare-word-cue-route':
        prepare_word_cue_route(args.overview_job, args.output)
        return 0
    if args.action == 'prepare-reference-route':
        prepare_reference_route(args.article_job, args.output)
        return 0
    if args.action == 'prepare-onestop':
        prepare_onestop(args.assay_root, args.preflight, args.gradio_job, args.output)
        return 0
    if args.action == 'prepare-expmrc':
        prepare_expmrc(args.assay_root, args.preflight, args.gradio_job, args.output)
        return 0
    if args.action == 'prepare-contractnli':
        prepare_contractnli(args.packet, args.assay, args.reference_job, args.output, args.experiment)
        return 0
    if args.action == 'prepare-nq-evidence':
        prepare_nq_evidence(args.assay_root, args.preflight, args.snapshot, args.gradio_job, args.output)
        return 0
    if args.action == 'prepare-qasper-tail':
        prepare_qasper_tail(args.assay_root, args.preflight, args.snapshot, args.gradio_job, args.output)
        return 0
    if args.action == 'prepare-first-readback':
        prepare_first_readback(args.assay_root, args.preflight, args.gradio_job, args.output)
        return 0
    if args.action == 'prepare-context-readback':
        prepare_context_readback(args.assay_root, args.preflight, args.gradio_job, args.output)
        return 0
    job = args.job.resolve()
    data = worker.read_json(manifest_path(job))
    if args.action == 'run':
        return run(job, data)
    if args.action == 'release':
        with run_lock(job, 'r+'):
            data = worker.read_json(manifest_path(job))
            if (not worker.RUN_ID.fullmatch(job.name) or data['run_id'] != job.name or
                    'allocating' not in data['timeline']):
                raise ValueError('No owned allocation to recover')
            machine = Colab(job, data)
            release(machine)
            machine.update('release-verified' if data['release_verified'] else 'release-unverified')
            return int(not data['release_verified'])
    if args.action == 'verify':
        verify(job, data)
    else:
        print(json.dumps(data, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
