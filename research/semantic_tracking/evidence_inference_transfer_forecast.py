"""Certify Evidence Inference sentences and export the frozen QASPER hazards."""

import argparse
from collections import Counter
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory

from research.semantic_tracking import contractnli_forecast as frozen


MAX_INPUT_TOKENS = 145_000
ARTICLE_FIELDS = {'id', 'text'}
QUERY_FIELDS = {'id', 'article_id', 'text'}
LAYOUT_FIELDS = {'id', 'spans', 'candidate_stops', 'query_ids'}
SOURCES = ('articles.jsonl', 'queries.jsonl')
RETURN_FILES = ('predictions.jsonl', 'extraction.json', 'causality.json', 'telemetry.jsonl')


def identity():
    return {'schema': 1, 'candidate_id': frozen.CANDIDATE, 'model_revision': frozen.REVISION,
            'heads_sha256': frozen.HEADS_SHA256, 'tokenizer_sha256': frozen.TOKENIZER_SHA256,
            'source_code_sha256': frozen.digest(Path(__file__)),
            'upstream_forecast_code_sha256': frozen.digest(Path(frozen.__file__))}


def trimmed_sentences(text, segmenter):
    """Return the candidate's Punkt sentence coordinates without the UI's character cap."""
    spans = []
    previous = 0
    for start, end in segmenter.span_tokenize(text):
        sentence = text[start:end]
        start += len(sentence) - len(sentence.lstrip())
        end -= len(sentence) - len(sentence.rstrip())
        if start >= end:
            continue
        if start < previous or text[previous:start].strip():
            raise ValueError('Frozen Punkt did not cover all source text in order')
        spans.append([start, end])
        previous = end
    if not spans or text[previous:].strip():
        raise ValueError('Frozen Punkt did not cover all source text in order')
    return spans


def question_banks(articles, queries):
    banks = {key: [] for key in articles}
    if articles.keys() & queries.keys():
        raise ValueError('Article and query identifiers must be disjoint')
    for key, row in queries.items():
        if row['article_id'] not in banks or type(row['text']) is not str or not row['text'].strip():
            raise ValueError(f'Invalid source article or query text for {key}')
        banks[row['article_id']].append(key)
    if any(not bank for bank in banks.values()):
        raise ValueError('Each approved article needs a nonempty bank of its own queries')
    return banks


def read_material(assay_root):
    audit_path = assay_root / 'source-audit.json'
    audit = frozen.read_json(audit_path)
    paths = {name: assay_root / name for name in SOURCES}
    hashes = {name: frozen.digest(path) for name, path in paths.items()}
    if any(audit['output_sha256'][name] != value for name, value in hashes.items()):
        raise ValueError('Evidence Inference sources do not match the locked grade-free intake hashes')
    articles = frozen.read_rows(paths['articles.jsonl'], ARTICLE_FIELDS)
    queries = frozen.read_rows(paths['queries.jsonl'], QUERY_FIELDS)
    if any(type(row['text']) is not str or not row['text'].strip() for row in articles.values()):
        raise ValueError('An approved Evidence Inference article has no text')
    return articles, queries, {'source_sha256': hashes, 'source_audit_sha256': frozen.digest(audit_path)}


def certify_material(articles, queries, tokenizer, segmenter):
    from research.semantic_tracking.gum_bridge_extract import certify

    banks = question_banks(articles, queries)
    documents = articles | queries
    encodings = {key: tokenizer.encode(row['text'], add_special_tokens=False) for key, row in documents.items()}
    lengths = {key: len(item.ids) for key, item in encodings.items()}
    too_long = {key: size for key, size in lengths.items() if not 0 < size <= frozen.MAX_TOKENS}
    if too_long or sum(lengths.values()) > MAX_INPUT_TOKENS:
        raise ValueError(f'Exact token budget exceeded for {too_long}; total {sum(lengths.values())}/{MAX_INPUT_TOKENS}')

    def prefixes(texts):
        return [row.ids for row in tokenizer.encode_batch(texts, add_special_tokens=False)]

    stimuli, layout = [], []
    counts, exclusions = Counter(), Counter()
    for key, row in documents.items():
        text, complete = row['text'], encodings[key]
        spans = trimmed_sentences(text, segmenter) if key in articles else []
        requested = sorted({end for _, end in spans} | {len(text)})
        result = certify({'id': key, 'text': text, 'stops': requested}, lambda _: (complete.ids, complete.offsets), prefixes)
        stops = [end for end, _ in result.certified]
        if (not stops or stops[-1] != len(text) or len(stops) + sum(result.rejected.values()) != len(requested)):
            raise ValueError(f'Complete input or candidate accounting failed tokenizer certification for {key}')
        stimuli.append({'id': key, 'text': text, 'stops': stops})
        counts.update(certified_stops=len(stops))
        if key in articles:
            stop_index = {end: i for i, end in enumerate(stops)}
            candidate = [stop_index.get(end) for _, end in spans]
            counts.update(source_candidate_spans=len(spans), certified_candidate_spans=sum(x is not None for x in candidate),
                          article_certified_stops=len(stops))
            exclusions.update(result.rejected)
            layout.append({'id': key, 'spans': spans, 'candidate_stops': candidate, 'query_ids': banks[key]})
    inventory = {'articles': len(articles), 'hypotheses': len(queries), 'documents': len(documents),
                 'article_hypothesis_pairs': len(queries), 'input_tokens': sum(lengths.values()),
                 'article_input_tokens': sum(lengths[key] for key in articles), 'max_input_tokens': max(lengths.values()),
                 **counts, 'candidate_exclusions': dict(exclusions)}
    if counts['source_candidate_spans'] != counts['certified_candidate_spans'] + sum(exclusions.values()):
        raise ValueError('Sentence availability must reconcile to every registered source sentence')
    return stimuli, layout, inventory


def preflight(assay_root, tokenizer_path, checkpoint, output):
    from tokenizers import Tokenizer

    from research.gradio_app.checkpoint import load_checkpoint

    if output.exists():
        raise FileExistsError(output)
    candidate = load_checkpoint(checkpoint)
    if candidate.identifier != frozen.CANDIDATE or frozen.digest(tokenizer_path) != frozen.TOKENIZER_SHA256:
        raise ValueError('The tokenizer and Punkt segmenter must belong to the fixed CogLLM backbone')
    articles, queries, provenance = read_material(assay_root)
    stimuli, layout, inventory = certify_material(articles, queries, Tokenizer.from_file(str(tokenizer_path)),
                                                   candidate.segmenter())
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.evidence-inference-preflight-') as temp:
        directory = Path(temp) / 'packet'
        directory.mkdir()
        frozen.write_rows(directory / 'stimuli.jsonl', stimuli)
        frozen.write_rows(directory / 'articles.jsonl', layout)
        frozen.write_json(directory / 'preflight.json', {
            **identity(), **provenance, 'checkpoint_manifest_sha256': frozen.digest(checkpoint / 'checkpoint.json'),
            'query_ids': [query for row in layout for query in row['query_ids']], 'inventory': inventory,
            'file_sha256': {name: frozen.digest(directory / name) for name in frozen.PACKET_FILES},
        })
        directory.rename(output)


def validated_layout(documents, layout):
    queries, banks = set(), {}
    source, certified = 0, 0
    for row in documents.values():
        frozen.checked_stimulus(row)
    for key, row in layout.items():
        if key not in documents:
            raise ValueError(f'Unknown article in the sealed candidate map: {key}')
        text, stops = documents[key]['text'], documents[key]['stops']
        spans, candidate, bank = row['spans'], row['candidate_stops'], row['query_ids']
        if (type(spans) is not list or not spans or type(candidate) is not list or len(candidate) != len(spans)
                or type(bank) is not list or not bank or any(type(query) is not str or query not in documents
                                                            or query in layout for query in bank)
                or len(set(bank)) != len(bank) or queries.intersection(bank)):
            raise ValueError(f'Invalid or repeated grade-free article sentence or query map for {key}')
        index = {stop: i for i, stop in enumerate(stops)}
        previous, registered = 0, set()
        for bounds, position in zip(spans, candidate, strict=True):
            if (type(bounds) is not list or len(bounds) != 2 or any(type(value) is not int for value in bounds)
                    or not previous <= bounds[0] < bounds[1] <= len(text)
                    or text[previous:bounds[0]].strip() or text[bounds[0]:bounds[1]] != text[bounds[0]:bounds[1]].strip()
                    or (position is not None and type(position) is not int) or position != index.get(bounds[1])):
                raise ValueError(f'Invalid or uncertified sentence endpoint for {key}')
            if position is not None:
                registered.add(position)
            previous = bounds[1]
        if text[previous:].strip() or not set(range(len(stops) - 1)) <= registered:
            raise ValueError(f'Incomplete coverage of visible source and registered stops for {key}')
        source += len(spans)
        certified += len(registered)
        banks[key] = tuple(bank)
        queries.update(bank)
    if documents.keys() != layout.keys() | queries or any(len(documents[key]['stops']) != 1 for key in queries):
        raise ValueError('The packet must contain each article and each independent query exactly once')
    inventory = {'articles': len(layout), 'hypotheses': len(queries), 'documents': len(documents),
                 'article_hypothesis_pairs': len(queries), 'source_candidate_spans': source, 'certified_candidate_spans': certified,
                 'article_certified_stops': sum(len(documents[key]['stops']) for key in layout),
                 'certified_stops': sum(len(row['stops']) for row in documents.values())}
    return banks, inventory


def load_packet(packet):
    receipt = frozen.read_json(packet / 'preflight.json')
    if (any(receipt[key] != value for key, value in identity().items())
            or receipt['file_sha256'] != {name: frozen.digest(packet / name) for name in frozen.PACKET_FILES}):
        raise ValueError('The grade-free packet does not match preflight and candidate bytes')
    documents = frozen.read_rows(packet / 'stimuli.jsonl', frozen.STIMULUS_FIELDS)
    layout = frozen.read_rows(packet / 'articles.jsonl', LAYOUT_FIELDS)
    banks, observed = validated_layout(documents, layout)
    inventory = receipt['inventory']
    if (receipt['query_ids'] != [query for bank in banks.values() for query in bank]
            or any(inventory[key] != value for key, value in observed.items())
            or not 0 < inventory['article_input_tokens'] < inventory['input_tokens'] <= MAX_INPUT_TOKENS
            or not 0 < inventory['max_input_tokens'] <= frozen.MAX_TOKENS
            or sum(inventory['candidate_exclusions'].values()) != observed['source_candidate_spans']
            - observed['certified_candidate_spans']):
        raise ValueError('Declared token or sentence inventory differs from certified packet')
    return documents, banks, receipt


def predictions(documents, banks, extraction, scratch, matrix, heads, device):
    import numpy as np
    import torch

    from research.semantic_tracking.qasper_factorial_fit import question_features
    from research.semantic_tracking.qasper_fit import unit

    features = frozen.article_states(extraction, scratch, documents, tuple(banks), matrix)
    full, segment, dense, offsets, counts, lookup, slices = features
    with torch.inference_mode():
        for key, query_ids in banks.items():
            stops = documents[key]['stops']
            indices = [lookup[key, stop] for stop in stops]
            qkeys = [(query, documents[query]['stops'][0]) for query in query_ids]
            qmeans = dict(zip(qkeys, unit(segment[[lookup[coordinate] for coordinate in qkeys]]), strict=True))
            rows = [{'paper_id': key, 'question_id': query} for query in query_ids]
            questions = question_features(documents, qmeans, matrix, rows)
            states = {(key, stop): vector for stop, vector in zip(stops, unit(full[indices]), strict=True)}
            means = qmeans | {(key, stop): vector for stop, vector in zip(stops, unit(segment[indices]), strict=True)}
            token_stop = counts[indices].astype(np.int64)
            block = slices[key]
            shapes = offsets[block]
            token_states = np.array(dense[block], copy=True)
            lengths = np.linalg.norm(token_states, axis=-1)
            if (token_stop[0] <= 0 or np.any(np.diff(token_stop) <= 0) or token_stop[-1] != len(shapes)
                    or np.any(shapes[:, 0] < 0) or np.any(shapes[:, 0] > shapes[:, 1])
                    or np.any(shapes[:, 1] > len(documents[key]['text']))
                    or np.any(np.maximum.accumulate(shapes[:, 1])[token_stop - 1] > stops)
                    or not np.all(np.isfinite(lengths) & (np.abs(lengths - 1) < 2e-5))):
                raise ValueError(f'Dense token geometry is not certified for {key}')
            hazards = frozen.article_forecasts(rows, documents, states, means, matrix, questions, token_states,
                                                token_stop, heads, device)
            for i, query in enumerate(query_ids):
                yield {'id': key, 'query_id': query, 'stops': stops,
                       'methods': {method: hazards[j, i].tolist() for j, method in enumerate(frozen.METHODS)}}


def verify_extraction(info, receipt):
    inventory = receipt['inventory']
    expected = {'input_sha256': receipt['file_sha256']['stimuli.jsonl'], 'model': frozen.MODEL, 'revision': frozen.REVISION,
                'max_batch_tokens': frozen.MAX_TOKENS, 'excluded_stops': [], 'segment_means': True,
                'input_texts': inventory['documents'], 'encoded_texts': inventory['documents'], 'tokens': inventory['input_tokens'],
                'unique_certified_stops': inventory['certified_stops']}
    if (any(info[key] != value for key, value in expected.items()) or info['projected_tokens']['documents'] != inventory['articles']
            or info['projected_tokens']['tokens'] != inventory['article_input_tokens']):
        raise ValueError('CUDA extraction differs from the audited Evidence Inference prefix layout')


def forecast(packet, model, heads_path, assay_sha256, protocol_sha256, output):
    import numpy as np

    from research.core_features.extract import run
    from research.gradio_app.causal_plaba import QWEN_FILES
    from research.semantic_tracking.qasper_fit import projection

    if output.exists():
        raise FileExistsError(output)
    if not all(re.fullmatch(r'[0-9a-f]{64}', value) for value in (assay_sha256, protocol_sha256)):
        raise ValueError('Expected exact parent assay and protocol SHA-256 values')
    documents, banks, receipt = load_packet(packet)
    if (model.name != frozen.REVISION or any(frozen.digest(model / name) != value for name, value in QWEN_FILES.items())
            or frozen.digest(heads_path) != frozen.HEADS_SHA256):
        raise ValueError('Expected the verified Qwen backbone and frozen factorial QASPER heads')
    matrix = projection()
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.evidence-inference-forecast-') as temp:
        scratch = Path(temp)
        extraction, result = scratch / 'extraction', scratch / 'return'
        result.mkdir()
        run(packet / 'stimuli.jsonl', extraction, segment_means=True, max_batch_tokens=frozen.MAX_TOKENS,
            projected_tokens=(set(banks), matrix))
        verify_extraction(frozen.read_json(extraction / 'extraction.json'), receipt)
        with np.load(heads_path, allow_pickle=False) as heads:
            if not np.array_equal(heads['projection'], matrix):
                raise ValueError('The extractor and readout projections differ')
            frozen.write_rows(result / 'predictions.jsonl', predictions(documents, banks, extraction, scratch, matrix, heads, 'cuda'))
        for name in RETURN_FILES[1:]:
            shutil.copyfile(extraction / name, result / name)
        frozen.write_json(result / 'seal.json', {
            **receipt['inventory'], **identity(), 'assay_sha256': assay_sha256, 'protocol_sha256': protocol_sha256,
            'source_audit_sha256': receipt['source_audit_sha256'], 'content_forward_per_document': 1,
            'encoded_hypotheses': len(receipt['query_ids']), 'forecast_rows': len(receipt['query_ids']),
            'query_ids': receipt['query_ids'], 'checkpoint_manifest_sha256': receipt['checkpoint_manifest_sha256'],
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
    prepare.add_argument('--checkpoint', required=True, type=Path)
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
