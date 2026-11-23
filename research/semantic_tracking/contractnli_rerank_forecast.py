"""Export frozen QASPER hazards for the independent ContractNLI EDGAR dev assay."""

import argparse
from collections import Counter
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory

from research.semantic_tracking import contractnli_forecast as frozen


SPLIT = 'dev'
ARTICLES = 23
SOURCE_SPANS = 1739
NONEMPTY_SPANS = 1685
SOURCE_FILES = ('stimuli.jsonl', 'layout.jsonl', 'queries.jsonl')
RETURN_FILES = ('predictions.jsonl', 'extraction.json', 'causality.json', 'telemetry.jsonl')


def identity():
    return {'schema': 1, 'split': SPLIT, 'candidate_id': frozen.CANDIDATE, 'model_revision': frozen.REVISION,
            'heads_sha256': frozen.HEADS_SHA256, 'tokenizer_sha256': frozen.TOKENIZER_SHA256,
            'source_code_sha256': frozen.digest(Path(__file__)),
            'upstream_forecast_code_sha256': frozen.digest(Path(frozen.__file__))}


def source_material(assay_root):
    audit_path = assay_root / 'source-audit.json'
    audit = frozen.read_json(audit_path)
    sources = {name: assay_root / name for name in SOURCE_FILES}
    hashes = {name: frozen.digest(path) for name, path in sources.items()}
    if (audit['retained_split'] != SPLIT or set(audit['grade_free_files']) != hashes.keys()
            or any(audit['output_sha256'][name] != value for name, value in hashes.items())):
        raise ValueError('Expected exactly the frozen official EDGAR dev article, candidate and hypothesis bytes')
    articles = frozen.read_rows(sources['stimuli.jsonl'], frozen.STIMULUS_FIELDS)
    layout = frozen.read_rows(sources['layout.jsonl'], frozen.LAYOUT_FIELDS)
    queries = frozen.read_rows(sources['queries.jsonl'], {'id', 'text'})
    if (len(articles) != ARTICLES or len(queries) != frozen.HYPOTHESES or articles.keys() != layout.keys()
            or articles.keys() & queries.keys() or len({row['text'] for row in queries.values()}) != len(queries)):
        raise ValueError('Expected 23 official EDGAR dev articles and 17 distinct hypotheses')
    for key, row in articles.items():
        frozen.checked_stimulus(row, terminal=False)
        if row['stops'] != sorted({end for _, end in layout[key]['spans']}):
            raise ValueError(f'Original candidate endpoints and layout disagree for {key}')
    if any(type(row['text']) is not str or not row['text'].strip() for row in queries.values()):
        raise ValueError('An official ContractNLI hypothesis is missing')
    return articles, layout, queries, {'source_sha256': hashes, 'source_audit_sha256': frozen.digest(audit_path)}


def candidate_positions(stops, candidates):
    index = {stop: i for i, stop in enumerate(stops)}
    return [index.get(end) for end in candidates]


def certify_material(articles, layout, queries, tokenizer):
    from research.semantic_tracking.gum_bridge_extract import certify

    documents = articles | queries
    encodings = {key: tokenizer.encode(row['text'], add_special_tokens=False) for key, row in documents.items()}
    too_long = {key: len(item.ids) for key, item in encodings.items() if len(item.ids) > frozen.MAX_TOKENS}
    if too_long:
        raise ValueError(f'Whole texts exceed the frozen {frozen.MAX_TOKENS} token cap: {too_long}')

    def prefixes(texts):
        return [row.ids for row in tokenizer.encode_batch(texts, add_special_tokens=False)]

    stimuli, maps = [], []
    counts, exclusions = Counter(), Counter()
    for key, row in documents.items():
        full = encodings[key]
        text = row['text']
        candidates = frozen.trimmed_stops(text, layout[key]['spans']) if key in articles else []
        requested = sorted({end for end in candidates if end is not None} | {len(text)})
        certified = certify({'id': key, 'text': text, 'stops': requested}, lambda _: (full.ids, full.offsets), prefixes)
        stops = [end for end, _ in certified.certified]
        if (not stops or stops[-1] != len(text) or len(stops) + sum(certified.rejected.values()) != len(requested)):
            raise ValueError(f'Complete text or candidate accounting failed exact prefix certification for {key}')
        stimuli.append({'id': key, 'text': text, 'stops': stops})
        counts.update(input_tokens=len(full.ids), certified_stops=len(stops))
        counts['max_input_tokens'] = max(counts['max_input_tokens'], len(full.ids))
        if key in articles:
            positions = candidate_positions(stops, candidates)
            maps.append({'id': key, 'candidate_stops': positions})
            counts.update(source_candidate_spans=len(candidates),
                          nonempty_candidate_spans=sum(end is not None for end in candidates),
                          certified_candidate_spans=sum(position is not None for position in positions),
                          article_input_tokens=len(full.ids), article_certified_stops=len(stops))
            exclusions.update(certified.rejected)
    exclusions['whitespace_only'] = counts['source_candidate_spans'] - counts['nonempty_candidate_spans']
    inventory = {'articles': len(articles), 'hypotheses': len(queries), 'documents': len(documents),
                 'article_hypothesis_pairs': len(articles) * len(queries), **counts,
                 'candidate_exclusions': dict(exclusions)}
    return stimuli, maps, inventory


def preflight(assay_root, tokenizer_path, output):
    from tokenizers import Tokenizer

    if output.exists():
        raise FileExistsError(output)
    if frozen.digest(tokenizer_path) != frozen.TOKENIZER_SHA256:
        raise ValueError('Expected the pinned Qwen Rust tokenizer')
    articles, layout, queries, provenance = source_material(assay_root)
    stimuli, maps, inventory = certify_material(articles, layout, queries, Tokenizer.from_file(str(tokenizer_path)))
    if (inventory['source_candidate_spans'] != SOURCE_SPANS or inventory['nonempty_candidate_spans'] != NONEMPTY_SPANS
            or inventory['source_candidate_spans'] != inventory['certified_candidate_spans']
            + sum(inventory['candidate_exclusions'].values())):
        raise ValueError('Dev candidate stop certification does not reconcile to the frozen source spans')
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.contractnli-rerank-preflight-') as temp:
        directory = Path(temp) / 'packet'
        directory.mkdir()
        frozen.write_rows(directory / 'stimuli.jsonl', stimuli)
        frozen.write_rows(directory / 'articles.jsonl', maps)
        frozen.write_json(directory / 'preflight.json', {
            **identity(), **provenance, 'query_ids': list(queries), 'inventory': inventory,
            'file_sha256': {name: frozen.digest(directory / name) for name in frozen.PACKET_FILES},
        })
        directory.rename(output)


def load_packet(packet):
    receipt = frozen.read_json(packet / 'preflight.json')
    if (any(receipt[key] != value for key, value in identity().items())
            or receipt['file_sha256'] != {name: frozen.digest(packet / name) for name in frozen.PACKET_FILES}):
        raise ValueError('Dev input packet does not match the sealed preflight and exporter source')
    stimuli = frozen.read_rows(packet / 'stimuli.jsonl', frozen.STIMULUS_FIELDS)
    mapped = frozen.read_rows(packet / 'articles.jsonl', frozen.MAP_FIELDS)
    queries = tuple(receipt['query_ids'])
    if (len(mapped) != ARTICLES or len(queries) != frozen.HYPOTHESES or len(set(queries)) != len(queries)
            or mapped.keys() & set(queries) or stimuli.keys() != mapped.keys() | set(queries)):
        raise ValueError('Unexpected article or hypothesis inventory in sealed dev input')
    for row in stimuli.values():
        frozen.checked_stimulus(row)
    if any(len(stimuli[key]['stops']) != 1 for key in queries):
        raise ValueError('Hypothesis input must contain exactly one terminal stop')
    verified, total = 0, 0
    for key, row in mapped.items():
        positions = row['candidate_stops']
        count = len(stimuli[key]['stops'])
        if type(positions) is not list or not positions:
            raise ValueError(f'Missing candidate stop indices for {key}')
        available = [index for index in positions if index is not None]
        if (any(type(index) is not int or not 0 <= index < count for index in available)
                or available != sorted(set(available)) or not set(range(count - 1)) <= set(available)):
            raise ValueError(f'Unordered or incomplete candidate stop indices for {key}')
        verified += len(available)
        total += len(positions)
    observed = {'articles': len(mapped), 'hypotheses': len(queries), 'documents': len(stimuli),
                'article_hypothesis_pairs': len(mapped) * len(queries), 'source_candidate_spans': total,
                'nonempty_candidate_spans': NONEMPTY_SPANS, 'certified_candidate_spans': verified,
                'article_certified_stops': sum(len(stimuli[key]['stops']) for key in mapped),
                'certified_stops': sum(len(row['stops']) for row in stimuli.values())}
    if total != SOURCE_SPANS or any(receipt['inventory'][key] != value for key, value in observed.items()):
        raise ValueError('Exact dev stimulus coverage differs from the frozen preflight inventory')
    return stimuli, tuple(mapped), queries, receipt


def verify_extraction(info, receipt, document_count, article_count):
    inventory = receipt['inventory']
    expected = {'input_sha256': receipt['file_sha256']['stimuli.jsonl'], 'model': frozen.MODEL,
                'revision': frozen.REVISION, 'max_batch_tokens': frozen.MAX_TOKENS, 'excluded_stops': [],
                'input_texts': document_count, 'encoded_texts': document_count, 'tokens': inventory['input_tokens'],
                'unique_certified_stops': inventory['certified_stops'], 'segment_means': True}
    if (any(info[key] != value for key, value in expected.items())
            or info['projected_tokens']['documents'] != article_count
            or info['projected_tokens']['tokens'] != inventory['article_input_tokens']):
        raise ValueError('Frozen CUDA extraction disagrees with the dev Rust tokenizer preflight')


def forecast(packet, model, heads_path, assay_sha256, protocol_sha256, output):
    import numpy as np

    from research.core_features.extract import run
    from research.gradio_app.causal_plaba import QWEN_FILES
    from research.semantic_tracking.qasper_fit import projection

    if output.exists():
        raise FileExistsError(output)
    if not all(re.fullmatch(r'[0-9a-f]{64}', value) for value in (assay_sha256, protocol_sha256)):
        raise ValueError('Expected exact parent assay and protocol SHA-256 values')
    documents, articles, queries, receipt = load_packet(packet)
    if (model.name != frozen.REVISION or any(frozen.digest(model / name) != value for name, value in QWEN_FILES.items())
            or frozen.digest(heads_path) != frozen.HEADS_SHA256):
        raise ValueError('Expected the frozen shared backbone and QASPER heads')
    matrix = projection()
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.contractnli-rerank-forecast-') as temp:
        scratch = Path(temp)
        extraction, result = scratch / 'extraction', scratch / 'return'
        result.mkdir()
        run(packet / 'stimuli.jsonl', extraction, segment_means=True, max_batch_tokens=frozen.MAX_TOKENS,
            projected_tokens=(set(articles), matrix))
        verify_extraction(frozen.read_json(extraction / 'extraction.json'), receipt, len(documents), len(articles))
        with np.load(heads_path, allow_pickle=False) as heads:
            if not np.array_equal(heads['projection'], matrix):
                raise ValueError('Frozen QASPER heads do not match the extractor projection')
            frozen.write_rows(result / 'predictions.jsonl',
                              frozen.predictions(documents, articles, queries, extraction, scratch, matrix, heads))
        for name in RETURN_FILES[1:]:
            shutil.copyfile(extraction / name, result / name)
        frozen.write_json(result / 'seal.json', {
            **receipt['inventory'], **identity(), 'assay_sha256': assay_sha256, 'protocol_sha256': protocol_sha256,
            'source_audit_sha256': receipt['source_audit_sha256'], 'content_forward_per_document': 1,
            'encoded_hypotheses': len(queries), 'forecast_rows': len(articles) * len(queries), 'query_ids': list(queries),
            'stimuli_sha256': receipt['file_sha256']['stimuli.jsonl'], 'articles_sha256': receipt['file_sha256']['articles.jsonl'],
            'preflight_sha256': frozen.digest(packet / 'preflight.json'),
            **{f'{Path(name).stem}_sha256': frozen.digest(result / name) for name in RETURN_FILES},
        })
        result.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest='command', required=True)
    prepare = subcommands.add_parser('preflight')
    prepare.add_argument('--assay-root', required=True, type=Path)
    prepare.add_argument('--tokenizer', dest='tokenizer_path', required=True, type=Path)
    export = subcommands.add_parser('forecast')
    export.add_argument('--packet', required=True, type=Path)
    export.add_argument('--model', required=True, type=Path)
    export.add_argument('--heads', dest='heads_path', required=True, type=Path)
    export.add_argument('--assay-sha256', required=True)
    export.add_argument('--protocol-sha256', required=True)
    for command in (prepare, export):
        command.add_argument('--output', required=True, type=Path)
    options = vars(parser.parse_args())
    action = options.pop('command')
    (preflight if action == 'preflight' else forecast)(**options)


if __name__ == '__main__':
    main()
