"""Certify ContractNLI endpoints and export frozen QASPER hazards on one L4."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from research.semantic_tracking.expmrc_forecast import HEADS_SHA256, METHODS, digest


CANDIDATE = '74c26ff02569b2b3bc28ed291bf15fd2f85db5b4c93a3ad7bfc34f6a5d333ea0'
MODEL = 'Qwen/Qwen3.5-2B'
REVISION = '15852e8c16360a2fea060d615a32b45270f8a8fc'
TOKENIZER_SHA256 = '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42'
ARTICLES = 47
HYPOTHESES = 17
SOURCE_SPANS = 4318
NONEMPTY_SPANS = 4216
CERTIFIED_SPANS = 4207
MAX_TOKENS = 8192
HIDDEN = 2048
WIDTH = 128
PACKET_FILES = ('stimuli.jsonl', 'articles.jsonl')
STIMULUS_FIELDS = {'id', 'text', 'stops'}
LAYOUT_FIELDS = {'id', 'official_id', 'group_id', 'document_type', 'spans'}
MAP_FIELDS = {'id', 'candidate_stops'}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_rows(path, fields):
    with path.open(encoding='utf-8') as handle:
        records = [json.loads(line) for line in handle]
    if not records or any(type(row) is not dict or row.keys() != fields or type(row['id']) is not str or not row['id']
                          for row in records):
        raise ValueError(f'Unexpected grade-free rows in {path.name}')
    indexed = {row['id']: row for row in records}
    if len(indexed) != len(records):
        raise ValueError(f'Duplicate identifiers in {path.name}')
    return indexed


def write_rows(path, rows):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')


def write_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def checked_stimulus(row, *, terminal=True):
    body, stops = row['text'], row['stops']
    if (type(body) is not str or not body.strip() or type(stops) is not list or not stops
            or any(type(stop) is not int or not 0 < stop <= len(body) for stop in stops)
            or stops != sorted(set(stops)) or (terminal and stops[-1] != len(body))):
        raise ValueError(f'Invalid complete source layout for {row["id"]}')


def trimmed_stops(body, spans):
    stops = []
    before = 0
    for bounds in spans:
        if (type(bounds) is not list or len(bounds) != 2 or any(type(value) is not int for value in bounds)
                or not before <= bounds[0] <= bounds[1] <= len(body)):
            raise ValueError('ContractNLI candidate spans must be ordered and disjoint original coordinates')
        start, end = bounds
        trimmed = body[start:end].rstrip()
        stops.append(start + len(trimmed) if trimmed else None)
        before = end
    return stops


def preflight(assay_root, tokenizer_path, output):
    from tokenizers import Tokenizer
    from research.semantic_tracking.gum_bridge_extract import certify

    if output.exists():
        raise FileExistsError(output)
    if digest(tokenizer_path) != TOKENIZER_SHA256:
        raise ValueError('Expected the frozen Qwen Rust tokenizer for exact prefix certification')
    source_paths = {name: assay_root / name for name in ('stimuli.jsonl', 'layout.jsonl', 'queries.jsonl')}
    source_sha = {name: digest(path) for name, path in source_paths.items()}
    source_audit = assay_root / 'source-audit.json'
    intake = read_json(source_audit)
    if (set(intake['grade_free_files']) != source_sha.keys()
            or any(intake['output_sha256'][name] != value for name, value in source_sha.items())):
        raise ValueError('ContractNLI stimuli, layout or hypotheses differ from the frozen intake audit')
    articles = read_rows(source_paths['stimuli.jsonl'], STIMULUS_FIELDS)
    layout = read_rows(source_paths['layout.jsonl'], LAYOUT_FIELDS)
    queries = read_rows(source_paths['queries.jsonl'], {'id', 'text'})
    if (len(articles) != ARTICLES or len(queries) != HYPOTHESES or articles.keys() != layout.keys()
            or articles.keys() & queries.keys() or len({row['text'] for row in queries.values()}) != HYPOTHESES):
        raise ValueError('Expected exactly the fixed EDGAR article inventory and 17 independent hypotheses')

    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    def encode(text):
        full = tokenizer.encode(text, add_special_tokens=False)
        return full.ids, full.offsets

    def prefixes(texts):
        return [row.ids for row in tokenizer.encode_batch(texts, add_special_tokens=False)]

    stimuli, mapped = [], []
    counts, exclusions = Counter(), Counter()
    for key, row in (*articles.items(), *queries.items()):
        body = row['text']
        source = key in layout
        if source:
            checked_stimulus(row, terminal=False)
            span_stops = trimmed_stops(body, layout[key]['spans'])
            if row['stops'] != sorted({end for _, end in layout[key]['spans']}):
                raise ValueError(f'Source candidate boundaries and layout do not agree for {key}')
        else:
            if type(body) is not str or not body.strip():
                raise ValueError(f'Empty hypothesis {key}')
            span_stops = []
        requested = sorted({stop for stop in span_stops if stop is not None} | {len(body)})
        doc = certify({'id': key, 'text': body, 'stops': requested}, encode, prefixes)
        stops = [end for end, _ in doc.certified]
        if not stops or stops[-1] != len(body) or len(doc.tokens) > MAX_TOKENS:
            raise ValueError(f'Complete text must be certified and at most 8192 tokens for {key}')
        stimuli.append({'id': key, 'text': body, 'stops': stops})
        counts.update(input_tokens=len(doc.tokens), certified_stops=len(stops))
        counts['max_input_tokens'] = max(counts['max_input_tokens'], len(doc.tokens))
        if source:
            stop_indices = {stop: i for i, stop in enumerate(stops)}
            candidate_stops = [stop_indices.get(stop) for stop in span_stops]
            mapped.append({'id': key, 'candidate_stops': candidate_stops})
            counts.update(source_candidate_spans=len(span_stops),
                          nonempty_candidate_spans=sum(stop is not None for stop in span_stops),
                          certified_candidate_spans=sum(stop is not None for stop in candidate_stops),
                          article_certified_stops=len(stops), article_input_tokens=len(doc.tokens))
            exclusions.update(doc.rejected)
    if tuple(counts[key] for key in ('source_candidate_spans', 'nonempty_candidate_spans', 'certified_candidate_spans')) != (
            SOURCE_SPANS, NONEMPTY_SPANS, CERTIFIED_SPANS):
        raise ValueError('Candidate span layout differs from the fixed ContractNLI grade-free inventory')
    inventory = {'articles': len(articles), 'hypotheses': len(queries), 'documents': len(stimuli),
                 'article_hypothesis_pairs': len(articles) * len(queries), **counts,
                 'candidate_exclusions': {'whitespace_only': SOURCE_SPANS - NONEMPTY_SPANS, **exclusions}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.contractnli-preflight-') as temp:
        directory = Path(temp) / 'packet'
        directory.mkdir()
        write_rows(directory / 'stimuli.jsonl', stimuli)
        write_rows(directory / 'articles.jsonl', mapped)
        write_json(directory / 'preflight.json', {
            'schema': 1, 'candidate_id': CANDIDATE, 'model_revision': REVISION, 'heads_sha256': HEADS_SHA256,
            'tokenizer_sha256': TOKENIZER_SHA256, 'query_ids': list(queries), 'inventory': inventory,
            'source_sha256': source_sha, 'source_audit_sha256': digest(source_audit),
            'source_code_sha256': digest(Path(__file__)),
            'file_sha256': {name: digest(directory / name) for name in PACKET_FILES},
        })
        directory.rename(output)


def load_packet(packet):
    receipt = read_json(packet / 'preflight.json')
    inventory = receipt['inventory']
    fingerprints = {'schema': 1, 'candidate_id': CANDIDATE, 'model_revision': REVISION,
                    'heads_sha256': HEADS_SHA256, 'tokenizer_sha256': TOKENIZER_SHA256,
                    'source_code_sha256': digest(Path(__file__))}
    if (any(receipt[key] != value for key, value in fingerprints.items())
            or receipt['file_sha256'] != {name: digest(packet / name) for name in PACKET_FILES}):
        raise ValueError('Unexpected ContractNLI preflight or grade-free packet bytes')
    stimuli = read_rows(packet / 'stimuli.jsonl', STIMULUS_FIELDS)
    mapped = read_rows(packet / 'articles.jsonl', MAP_FIELDS)
    query_ids = receipt['query_ids']
    if (len(mapped) != ARTICLES or len(query_ids) != HYPOTHESES or len(set(query_ids)) != HYPOTHESES
            or mapped.keys() & set(query_ids) or stimuli.keys() != mapped.keys() | set(query_ids)):
        raise ValueError('Exact article and shared hypothesis coverage is required')
    for row in stimuli.values():
        checked_stimulus(row)
    if any(len(stimuli[key]['stops']) != 1 for key in query_ids):
        raise ValueError('A hypothesis must have exactly its full-text endpoint')
    for key, row in mapped.items():
        size = len(stimuli[key]['stops'])
        sequence = row['candidate_stops']
        if type(sequence) is not list or not sequence or any(type(stop) is not int or not 0 <= stop < size
                                                              for stop in sequence if stop is not None):
            raise ValueError(f'Invalid candidate-to-stop map for {key}')
        indices = [stop for stop in sequence if stop is not None]
        if indices != sorted(set(indices)) or not set(range(size - 1)) <= set(indices):
            raise ValueError(f'Candidate-to-stop order or coverage does not match article {key}')
    total_map = sum(len(row['candidate_stops']) for row in mapped.values())
    certified = sum(stop is not None for row in mapped.values() for stop in row['candidate_stops'])
    if (total_map != SOURCE_SPANS or certified != CERTIFIED_SPANS
            or inventory['source_candidate_spans'] != total_map or inventory['certified_candidate_spans'] != certified
            or inventory['certified_stops'] != sum(len(row['stops']) for row in stimuli.values())):
        raise ValueError('ContractNLI grade-free inventory differs from exact preflight')
    return stimuli, tuple(mapped), tuple(query_ids), receipt


def disk_array(path, field, scratch):
    import numpy as np

    target = scratch / f'{path.stem}-{field}.npy'
    with ZipFile(path) as archive, archive.open(f'{field}.npy') as source, target.open('xb') as destination:
        shutil.copyfileobj(source, destination, length=1024 * 1024)
    return np.load(target, mmap_mode='r', allow_pickle=False)


def article_states(extraction, scratch, documents, article_ids, projection):
    """Keep large token and paragraph states disk-backed until one article is read."""
    import numpy as np

    features, dense = extraction / 'features.npz', extraction / 'token-features.npz'
    expected = {(identifier, end) for identifier, row in documents.items() for end in row['stops']}
    with np.load(features, allow_pickle=False) as paragraph, np.load(dense, allow_pickle=False) as tokens:
        keys = list(zip(paragraph['ids'].tolist(), paragraph['stops'].tolist(), strict=True))
        if (set(paragraph.files) != {'ids', 'stops', 'states', 'segment_means'} or len(keys) != len(expected)
                or set(keys) != expected
                or set(tokens.files) != {'ids', 'indptr', 'offsets', 'stop_counts', 'states', 'projection'}
                or not np.array_equal(tokens['projection'], projection)):
            raise ValueError('Extraction differs from the certified QASPER feature inventory')
        token_ids, offsets, split = tokens['ids'].tolist(), tokens['offsets'], tokens['indptr']
        token_counts = tokens['stop_counts']
    full = disk_array(features, 'states', scratch)
    means = disk_array(features, 'segment_means', scratch)
    projected = disk_array(dense, 'states', scratch)
    if (len(token_ids) != len(article_ids) or set(token_ids) != set(article_ids)
            or split.shape != (len(article_ids) + 1,) or split.dtype.kind not in 'iu' or split[0] != 0
            or np.any(split[1:] <= split[:-1]) or split[-1] != len(projected)
            or offsets.shape != (len(projected), 2) or offsets.dtype.kind not in 'iu'
            or token_counts.shape != (len(keys),) or token_counts.dtype.kind not in 'iu'
            or full.shape != (len(keys), HIDDEN) or means.shape != full.shape
            or projected.shape != (len(offsets), WIDTH)
            or any(values.dtype != np.float32 for values in (full, means, projected))):
        raise ValueError('Feature shapes do not agree with the sealed stop coordinates')
    lookup = {key: i for i, key in enumerate(keys)}
    slices = {key: slice(left, right) for key, left, right in zip(token_ids, split[:-1], split[1:], strict=True)}
    return full, means, projected, offsets, token_counts, lookup, slices


def local_logmeanexp(states, stop_counts, questions, weights):
    """Equivalent to the frozen TokenDesign, without materializing question×token×width interactions."""
    import torch

    from research.semantic_tracking.qasper_factorial_fit import SQRT_WIDTH

    split = torch.cat((stop_counts.new_zeros(1), stop_counts)).diff()
    factors = questions * SQRT_WIDTH * weights
    values = states @ factors.T
    maxima = torch.segment_reduce(values, 'max', lengths=split, unsafe=True)
    groups = torch.repeat_interleave(torch.arange(len(split), device=values.device), split, output_size=len(states))
    masses = torch.segment_reduce(torch.exp(values - maxima[groups]), 'sum', lengths=split, unsafe=True)
    return (maxima + torch.log(masses / split[:, None])).T


def article_forecasts(rows, documents, states, means, matrix, questions, dense, stop_counts, heads, device):
    import torch

    from research.semantic_tracking.qasper_factorial_fit import LOCAL_END, LOCAL_START, question_only_features
    from research.semantic_tracking.qasper_fit import assemble, pack

    center, scale = heads['position_mean'], heads['position_scale']
    geometries, body = assemble(rows, documents, states, means, matrix)
    body = torch.as_tensor(pack(geometries, body, center, scale), device=device)
    query_design = torch.as_tensor(question_only_features(geometries, rows, questions, center, scale), device=device)
    position, question, pooled, token = (torch.as_tensor(heads[method], device=device) for method in METHODS)
    common = body[..., :LOCAL_START] @ token[:LOCAL_START] + body[..., LOCAL_END:] @ token[LOCAL_END:]
    ordered_queries = torch.stack([torch.as_tensor(questions[row['question_id']], device=device) for row in rows])
    local = local_logmeanexp(torch.as_tensor(dense, device=device), torch.as_tensor(stop_counts, device=device),
                             ordered_queries, token[LOCAL_START:LOCAL_END])
    logits = (body[..., :4] @ position, query_design @ question, body @ pooled, common + local)
    hazards = torch.stack([torch.sigmoid(values.double()) for values in logits])
    if not (torch.isfinite(hazards) & (hazards > 0) & (hazards < 1)).all().item():
        raise ValueError('Frozen QASPER output must contain finite strictly interior hazards')
    return hazards.cpu().numpy()


def predictions(documents, articles, query_ids, extraction, scratch, matrix, heads):
    import numpy as np
    import torch

    from research.semantic_tracking.qasper_factorial_fit import question_features
    from research.semantic_tracking.qasper_fit import unit

    full, segment, dense, offsets, counts, lookup, slices = article_states(extraction, scratch, documents, articles, matrix)
    query_means = {(key, documents[key]['stops'][0]): unit(segment[lookup[key, documents[key]['stops'][0]]])
                   for key in query_ids}
    questions = question_features(documents, query_means, matrix, [{'question_id': key} for key in query_ids])
    device = torch.device('cuda')
    with torch.inference_mode():
        for key in articles:
            stops = documents[key]['stops']
            indices = [lookup[key, stop] for stop in stops]
            states = {(key, stop): value for stop, value in zip(stops, unit(full[indices]), strict=True)}
            means = query_means | {(key, stop): value for stop, value in zip(stops, unit(segment[indices]), strict=True)}
            block = slices[key]
            token_states = np.array(dense[block], copy=True)
            spans = offsets[block]
            token_stop = counts[indices].astype(np.int64)
            lengths = np.linalg.norm(token_states, axis=-1)
            if (token_stop[0] <= 0 or np.any(np.diff(token_stop) <= 0) or token_stop[-1] != len(spans)
                    or np.any(spans[:, 0] < 0) or np.any(spans[:, 0] > spans[:, 1])
                    or np.any(spans[:, 1] > len(documents[key]['text']))
                    or np.any(np.maximum.accumulate(spans[:, 1])[token_stop - 1] > stops)
                    or not np.all(np.isfinite(lengths) & (np.abs(lengths - 1) < 2e-5))):
                raise ValueError(f'Uncertified dense ContractNLI geometry for {key}')
            rows = [{'paper_id': key, 'question_id': query} for query in query_ids]
            hazards = article_forecasts(rows, documents, states, means, matrix, questions, token_states, token_stop, heads, device)
            for i, query_id in enumerate(query_ids):
                yield {'id': key, 'query_id': query_id, 'stops': stops,
                       'methods': {method: hazards[j, i].tolist() for j, method in enumerate(METHODS)}}


def forecast(packet, model, heads_path, assay_sha256, protocol_sha256, output):
    import numpy as np

    from research.core_features.extract import run
    from research.gradio_app.causal_plaba import QWEN_FILES
    from research.semantic_tracking.qasper_fit import projection

    if output.exists():
        raise FileExistsError(output)
    if not all(re.fullmatch(r'[0-9a-f]{64}', value) for value in (assay_sha256, protocol_sha256)):
        raise ValueError('Assay and protocol digests must be exact SHA-256 values')
    documents, article_ids, query_ids, receipt = load_packet(packet)
    if (model.name != REVISION or any(digest(model / name) != value for name, value in QWEN_FILES.items())
            or digest(heads_path) != HEADS_SHA256):
        raise ValueError('Expected the exact frozen Qwen backbone and QASPER factorial heads')
    matrix = projection()
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.contractnli-forecast-') as temp:
        scratch = Path(temp)
        extraction, result = scratch / 'extraction', scratch / 'return'
        result.mkdir()
        run(packet / 'stimuli.jsonl', extraction, segment_means=True, max_batch_tokens=MAX_TOKENS,
            projected_tokens=(set(article_ids), matrix))
        info = read_json(extraction / 'extraction.json')
        inventory = receipt['inventory']
        if (info['input_sha256'] != receipt['file_sha256']['stimuli.jsonl'] or info['model'] != MODEL
                or info['revision'] != REVISION or info['max_batch_tokens'] != MAX_TOKENS or info['excluded_stops']
                or info['encoded_texts'] != len(documents) or info['tokens'] != inventory['input_tokens']
                or info['unique_certified_stops'] != inventory['certified_stops']
                or info['projected_tokens']['documents'] != len(article_ids)
                or info['projected_tokens']['tokens'] != inventory['article_input_tokens']):
            raise ValueError('Pinned CUDA extraction differs from independent Rust endpoint certification')
        forecasts_path = result / 'predictions.jsonl'
        with np.load(heads_path, allow_pickle=False) as heads:
            if not np.array_equal(heads['projection'], matrix):
                raise ValueError('QASPER local and pooled heads must use the exact same frozen projection')
            write_rows(forecasts_path, predictions(documents, article_ids, query_ids, extraction, scratch, matrix, heads))
        for name in ('extraction.json', 'causality.json', 'telemetry.jsonl'):
            shutil.copyfile(extraction / name, result / name)
        seal = {**inventory, 'assay_sha256': assay_sha256, 'protocol_sha256': protocol_sha256, 'candidate_id': CANDIDATE,
                'model_revision': REVISION, 'heads_sha256': HEADS_SHA256, 'content_forward_per_document': 1,
                'encoded_hypotheses': len(query_ids), 'forecast_rows': len(article_ids) * len(query_ids),
                'query_ids': list(query_ids), 'stimuli_sha256': receipt['file_sha256']['stimuli.jsonl'],
                'articles_sha256': receipt['file_sha256']['articles.jsonl'], 'preflight_sha256': digest(packet / 'preflight.json'),
                'source_audit_sha256': receipt['source_audit_sha256'],
                'source_code_sha256': digest(Path(__file__)), **{f'{name.split(".")[0]}_sha256': digest(result / name)
                                                               for name in ('predictions.jsonl', 'extraction.json',
                                                                            'causality.json', 'telemetry.jsonl')}}
        write_json(result / 'seal.json', seal)
        result.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest='command', required=True)
    prepare = subcommands.add_parser('preflight')
    prepare.add_argument('--assay-root', type=Path, required=True)
    prepare.add_argument('--tokenizer', dest='tokenizer_path', type=Path, required=True)
    export = subcommands.add_parser('forecast')
    export.add_argument('--packet', type=Path, required=True)
    export.add_argument('--model', type=Path, required=True)
    export.add_argument('--heads', dest='heads_path', type=Path, required=True)
    export.add_argument('--assay-sha256', required=True)
    export.add_argument('--protocol-sha256', required=True)
    for subcommand in (prepare, export):
        subcommand.add_argument('--output', type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop('command')
    (preflight if command == 'preflight' else forecast)(**args)


if __name__ == '__main__':
    main()
