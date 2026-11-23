"""Audit whether the plain-text Gradio route recovers held-out NDA evidence geometry."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path
from types import SimpleNamespace

from tokenizers import Tokenizer

from research.gradio_app.causal_plaba import CausalPlabaReader, MAX_TOKENS
from research.gradio_app.checkpoint import DEFAULT_CHECKPOINT, load_checkpoint
from research.gradio_app.combined import paragraphs
from research.gradio_app.plaba import MAX_CHARACTERS, TOKENIZER_SHA256, sentence_spans
from research.semantic_tracking import contractnli_forecast as base
from research.semantic_tracking import contractnli_rerank as rerank
from research.semantic_tracking.contractnli_rerank_forecast import load_packet


def app_candidates(article, reader, tokenizer):
    encoded = tokenizer.encode(article, add_special_tokens=False)
    tokens = len(encoded.ids)
    if len(article) > MAX_CHARACTERS or tokens > MAX_TOKENS:
        return None
    record = SimpleNamespace(text=article, tokens=tuple(encoded.ids), offsets=tuple(encoded.offsets))
    spans = [(paragraph_start + start, paragraph_start + end)
             for paragraph_start, paragraph_end, _ in paragraphs(article)
             for start, end in sentence_spans(article[paragraph_start:paragraph_end], reader.segmenter)]
    positions = [reader.position(record, start, end) for start, end in spans]
    return [span for span, position in zip(spans, positions, strict=True) if position is not None]


def coverage(body, article_stops, source_spans, candidate_stops, app_spans):
    exact_positions = set(app_spans)
    possible_ends = {end for _, end in app_spans}
    app_starts = [start for start, _ in app_spans]
    classes = {}
    official = base.trimmed_stops(body, source_spans)
    for i, ((left, _), stop_index) in enumerate(zip(source_spans, candidate_stops, strict=True)):
        if stop_index is None:
            continue
        right = article_stops[stop_index]
        if right != official[i]:
            raise ValueError('Source and tokenizer no longer agree on the official candidate endpoint')
        left += len(body[left:right]) - len(body[left:right].lstrip())
        end_match = right in possible_ends
        exact = (left, right) in exact_positions
        containing = bisect_right(app_starts, left) - 1
        enclosed = containing >= 0 and app_spans[containing][1] >= right
        classes[i] = {'exact_span': exact, 'same_causal_end': end_match, 'enclosed_in_one_sentence': enclosed}
    return classes


def geometry_counts(classifications):
    result = Counter(total=len(classifications))
    for row in classifications.values():
        result.update({key: int(value) for key, value in row.items()})
    return result


def audit(prepared, packet, tokenizer_path):
    source = rerank.audit_sources(prepared, rerank.EVAL_AUDIT, ('layout.jsonl', 'gold.jsonl', 'stimuli.jsonl'))
    stimuli, articles, hypotheses, certificate = load_packet(packet)
    checkpoint = load_checkpoint(DEFAULT_CHECKPOINT)
    tokenizer_sha256 = base.digest(tokenizer_path)
    if (checkpoint.identifier != certificate['candidate_id'] or tokenizer_sha256 != TOKENIZER_SHA256
            or any(source['output_sha256'][name] != expected for name, expected in certificate['source_sha256'].items())):
        raise ValueError('Expected the exact held-out evidence sources and Gradio candidate')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    reader = CausalPlabaReader(checkpoint.directory, backbone=SimpleNamespace(tokenizer=tokenizer),
                              checkpoint=checkpoint.directory / 'sentence', segmenter=checkpoint.segmenter())
    layout = base.read_rows(prepared / 'layout.jsonl', rerank.LAYOUT_FIELDS)
    mapping = base.read_rows(packet / 'articles.jsonl', base.MAP_FIELDS)
    gold = rerank.hypothesis_rows(prepared / 'gold.jsonl', rerank.GOLD_FIELDS)
    if layout.keys() != set(articles) or {(row['id'], row['query_id']) for row in gold} != {
        (identifier, query) for identifier in articles for query in hypotheses
    }:
        raise ValueError('Unexpected coverage of held-out documents and annotations')
    supported = set()
    candidates = Counter()
    candidates_within_input_limits = Counter()
    by_article = {}
    for article_id in articles:
        coordinate = mapping[article_id]
        article = stimuli[article_id]
        app = app_candidates(article['text'], reader, tokenizer)
        if app is not None:
            supported.add(article_id)
        classifications = coverage(article['text'], article['stops'], layout[article_id]['spans'],
                                   coordinate['candidate_stops'], app or [])
        by_article[article_id] = classifications
        candidates.update(geometry_counts(classifications))
        if article_id in supported:
            candidates_within_input_limits.update(geometry_counts(classifications))
    if candidates['total'] != certificate['inventory']['certified_candidate_spans']:
        raise ValueError('Certified source inventory changed')

    evidence = Counter()
    unique = defaultdict(set)
    cases = Counter()
    for row in gold:
        classifications = by_article[row['id']]
        labels = row['evidence_indices']
        if not labels:
            continue
        valid = [index for index in labels if index in classifications]
        evidence['uncertified_source_assignments'] += len(labels) - len(valid)
        unique[row['id']].update(valid)
        cases['evidence_bearing_claims'] += 1
        cases['claims_with_all_source_spans_certified'] += len(valid) == len(labels)
        cases['claims_with_every_evidence_span_exact'] += (
            len(valid) == len(labels) and all(classifications[i]['exact_span'] for i in valid)
        )
        evidence.update(geometry_counts({i: classifications[i] for i in valid}))
    unique_counts = Counter()
    unique_within_input_limits = Counter()
    for article_id, indices in unique.items():
        counts = geometry_counts({i: by_article[article_id][i] for i in indices})
        unique_counts.update(counts)
        if article_id in supported:
            unique_within_input_limits.update(counts)
    return {
        'candidate_id': checkpoint.identifier, 'source_audit_sha256': rerank.EVAL_AUDIT,
        'packet_sha256': base.digest(packet / 'preflight.json'), 'tokenizer_sha256': tokenizer_sha256,
        'held_out_documents': len(articles), 'within_app_article_limits': len(supported),
        'documents_with_all_certified_candidates_exact': sum(
            bool(rows) and all(row['exact_span'] for row in rows.values()) for rows in by_article.values()
        ),
        'all_certified_candidates': dict(candidates),
        'unique_annotation_spans': dict(unique_counts),
        'within_input_limits': {'all_certified_candidates': dict(candidates_within_input_limits),
                                'unique_annotation_spans': dict(unique_within_input_limits)},
        'evidence_assignments_across_claims': dict(evidence),
        'evidence_claim_counts': dict(cases),
        'scope': ('Retrospective geometry comparison on already-scored EDGAR material. '
                  'No model forecasts recomputed. Exact source comparisons trim surrounding whitespace.'),
    }


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared', 'packet', 'tokenizer', 'output'):
        cli.add_argument(f'--{name}', type=Path, required=True)
    args = cli.parse_args()
    result = audit(args.prepared, args.packet, args.tokenizer)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False, sort_keys=True)
        stream.write('\n')
