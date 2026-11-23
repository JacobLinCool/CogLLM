"""Seal the readout artifacts and source-boundary rules used by the combined app."""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from nltk.tabdata import PunktDecoder
from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer

from research.stage3.run import MODEL, TRANSFORMERS_VERSION

from .causal_plaba import QWEN_FILES, digest
from .plaba import BASE_REVISION
from .tokenizer import TOKENIZER_CONFIG_SHA256


RESULTS = Path(__file__).resolve().parents[1] / 'results'
DEFAULT_CHECKPOINT = RESULTS / 'gradio-candidate-checkpoint'
MANIFEST = 'checkpoint.json'
RESULT_FILES = {
    'sentence/head.npz': 'causal-plaba-checkpoint/head.npz',
    'sentence/manifest.json': 'causal-plaba-checkpoint/manifest.json',
    'sentence/metrics.json': 'causal-plaba-checkpoint/metrics.json',
    'ordinal/head.npz': 'causal-plaba-ordinal/head.npz',
    'ordinal/manifest.json': 'causal-plaba-ordinal/manifest.json',
    'lexical/heads.npz': 'stage5-complex/fit/heads.npz',
    'lexical/fitting.json': 'stage5-complex/fit/fitting.json',
    'lexical/run-manifest.json': 'stage5-complex/manifest.json',
    'lexical/replay.json': 'stage5-complex/lexical-runtime-replay.json',
    'anchor/heads.npz': 'iirc-anchor-run/heads.npz',
    'anchor/fit-manifest.json': 'iirc-anchor-run/fit-manifest.json',
    'anchor/scores.json': 'iirc-anchor-run/scores.json',
    'evidence/l4-route.json': 'gradio-combined-l4/verification.json',
    'evidence/l4-serving-seal.json': 'gradio-combined-l4/serving-seal.json',
    'evidence/l4-iirc-scores.json': 'gradio-combined-l4/iirc-locked-scores.json',
}
PUNKT_FILES = ('README', 'english/abbrev_types.txt', 'english/collocations.tab',
               'english/ortho_context.tab', 'english/sent_starters.txt')
MEMBERS = set(RESULT_FILES) | {f'punkt/{name}' for name in PUNKT_FILES}
BACKBONE = {'model': MODEL, 'revision': BASE_REVISION, 'compute_dtype': 'float32',
            'transformers': TRANSFORMERS_VERSION,
            'snapshot_sha256': {**QWEN_FILES, 'tokenizer_config.json': TOKENIZER_CONFIG_SHA256}}
CONTRACT = {
    'input': 'article',
    'offsets': 'half-open Unicode code-point indices',
    'sentence_boundaries': 'bundled English Punkt parameters',
    'sentence_mean': 'PLABA qwen_plus_length',
    'sentence_distribution': 'PLABA effect_calibrated_weights, experimental',
    'lexical': 'CompLex target_end, 100% training',
    'anchor': 'IIRC qwen, at paragraph end',
    'reader_background': 'withheld',
    'question_onset': 'withheld',
    'question_resolution': 'withheld',
}


@dataclass(frozen=True)
class Checkpoint:
    directory: Path
    identifier: str

    def segmenter(self) -> PunktSentenceTokenizer:
        root = self.directory / 'punkt/english'
        decoder = PunktDecoder()
        parameters = PunktParameters()
        with (root / 'collocations.tab').open(encoding='utf-8') as source:
            parameters.collocations = set(decoder.tab2tups(source))
        with (root / 'sent_starters.txt').open(encoding='utf-8') as source:
            parameters.sent_starters = decoder.txt2set(source)
        with (root / 'abbrev_types.txt').open(encoding='utf-8') as source:
            parameters.abbrev_types = decoder.txt2set(source)
        with (root / 'ortho_context.tab').open(encoding='utf-8') as source:
            parameters.ortho_context = decoder.tab2intdict(source)
        return PunktSentenceTokenizer(parameters)


def read(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def fingerprint(manifest: dict) -> str:
    canonical = json.dumps(manifest, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def verify_evidence(directory: Path, hashes: dict[str, str]) -> None:
    sentence = read(directory / 'sentence/manifest.json')
    ordinal = read(directory / 'ordinal/manifest.json')
    lexical = read(directory / 'lexical/fitting.json')
    lexical_run = read(directory / 'lexical/run-manifest.json')
    lexical_replay = read(directory / 'lexical/replay.json')
    anchor = read(directory / 'anchor/fit-manifest.json')
    anchor_scores = read(directory / 'anchor/scores.json')
    route = read(directory / 'evidence/l4-route.json')
    seal = read(directory / 'evidence/l4-serving-seal.json')
    scored = read(directory / 'evidence/l4-iirc-scores.json')
    replay = route['iirc_serving_replay']

    expected = {
        'sentence/head.npz': (sentence['head_sha256'], route['plaba_head_sha256']),
        'sentence/metrics.json': (sentence['metrics_sha256'],),
        'ordinal/head.npz': (ordinal['head_sha256'],),
        'lexical/heads.npz': (lexical['artifacts']['heads.npz']['sha256'], lexical_replay['head_sha256'],
                              route['lexical_head_sha256']),
        'anchor/heads.npz': (anchor['heads_sha256'], anchor_scores['input_sha256']['heads'],
                             route['iirc_head_sha256'], seal['heads_sha256'], scored['input_sha256']['heads']),
        'anchor/fit-manifest.json': (anchor_scores['input_sha256']['seal'],),
        'evidence/l4-serving-seal.json': (replay['seal_sha256'], scored['input_sha256']['seal']),
    }
    if any(hashes[name] != reference for name, references in expected.items() for reference in references):
        raise ValueError('Checkpoint files do not match the fit and verification receipts')
    model = (MODEL, BASE_REVISION)
    if (any((report['model'], report['revision']) != model for report in (sentence, ordinal, anchor, seal))
            or (lexical_run['model'], lexical_run['model_revision']) != model
            or route['model'] != BASE_REVISION or route['device'] != 'NVIDIA L4 FP32'
            or route['model_forwards_per_combined_gradio_request'] != 1
            or replay['identical_ordered_spans'] != replay['paragraphs']
            or replay['predictions_sha256'] != seal['predictions_sha256']
            or seal['predictions_sha256'] != scored['input_sha256']['predictions']
            or not anchor_scores['acceptance']['all_conditions_passed']
            or not scored['acceptance']['all_conditions_passed']
            or not ordinal['all_data_head_not_scored_out_of_fold']):
        raise ValueError('Checkpoint verification receipts refer to incompatible methods or evaluation states')


def load_checkpoint(directory: Path) -> Checkpoint:
    manifest = read(directory / MANIFEST)
    identifier = manifest.pop('checkpoint_sha256', None)
    hashes = manifest.get('files', {})
    if (manifest.get('schema') != 1 or manifest.get('status') != 'candidate'
            or manifest.get('backbone') != BACKBONE or manifest.get('contract') != CONTRACT
            or hashes.keys() != MEMBERS or identifier != fingerprint(manifest)):
        raise ValueError('The CogLLM candidate checkpoint manifest is invalid')
    paths = {path.relative_to(directory).as_posix() for path in directory.rglob('*') if path.is_file()}
    if paths != MEMBERS | {MANIFEST}:
        raise ValueError('Checkpoint directory differs from its declared file inventory')
    for name, expected in hashes.items():
        path = directory / name
        if path.is_symlink() or digest(path) != expected:
            raise ValueError(f'Checkpoint artifact differs from its digest: {name}')
    verify_evidence(directory, hashes)
    return Checkpoint(directory, identifier)


def pack(results: Path, punkt: Path, destination: Path) -> Checkpoint:
    if destination.exists():
        raise FileExistsError(destination)
    sources = {name: results / source for name, source in RESULT_FILES.items()}
    sources.update({f'punkt/{name}': punkt / name for name in PUNKT_FILES})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.checkpoint-', dir=destination.parent) as temporary:
        staged = Path(temporary) / 'bundle'
        hashes = {}
        for name, source in sorted(sources.items()):
            target = staged / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            hashes[name] = digest(target)
        manifest = {'schema': 1, 'status': 'candidate', 'backbone': BACKBONE,
                    'contract': CONTRACT, 'files': hashes}
        manifest['checkpoint_sha256'] = fingerprint(manifest)
        (staged / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        checkpoint = load_checkpoint(staged)
        staged.rename(destination)
    return Checkpoint(destination, checkpoint.identifier)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    package = actions.add_parser('pack')
    package.add_argument('--results', type=Path, default=RESULTS)
    package.add_argument('--punkt', type=Path, default=Path.home() / 'nltk_data/tokenizers/punkt_tab')
    package.add_argument('--output', type=Path, default=DEFAULT_CHECKPOINT)
    verify = actions.add_parser('verify')
    verify.add_argument('checkpoint', type=Path, nargs='?', default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    checkpoint = (pack(args.results, args.punkt, args.output) if args.action == 'pack'
                  else load_checkpoint(args.checkpoint))
    print(checkpoint.identifier)


if __name__ == '__main__':
    main()
