"""Bound the recorded full-answer endpoint under the TED-Q retirement rule."""

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path

from research.stage5.tedq_audit import CSV_SHA256, REVISION, THRESHOLD, observation, validate_probes


FULL = 5


def summarize(counts):
    total = sum(counts.values())
    observed = counts['full_at_first'] + counts['full_at_second']
    missing = counts['partial_at_first_no_second_rating']
    return {
        'questions': total,
        'counts': dict(sorted(counts.items())),
        'observed_full': observed,
        'unobserved_eligible_second_ratings': missing,
        'full_count_lower': observed,
        'full_count_upper': observed + missing,
        'full_fraction_lower': observed / total,
        'full_fraction_upper': (observed + missing) / total,
    }


def partition(first, second, opportunities):
    if first.score == FULL:
        return 'full_at_first'
    if first.score >= THRESHOLD:
        return ('partial_at_first_no_second_rating' if opportunities == 2
                else 'partial_at_first_end_of_excerpt')
    if second is None:
        return 'below_partial_at_first_end_of_excerpt'
    return 'full_at_second' if second.score == FULL else 'below_full_in_both_observed_ratings'


def audit(path):
    data = path.read_bytes()
    if sha256(data).hexdigest() != CSV_SHA256:
        raise ValueError('Expected the pinned TED-Q author elicitation file')
    probes = defaultdict(list)
    observed = {}
    with StringIO(data.decode('utf-8-sig'), newline='') as stream:
        for record in csv.DictReader(stream):
            row = observation(record)
            if row.id in observed:
                raise ValueError('Duplicate TED-Q annotation ID')
            probes[row.excerpt, row.chunk].append(row)
            observed[row.id] = row
    question_count = sum(row.kind == 'question' for row in observed.values())
    last, _ = validate_probes(probes)
    question_ratings, paired = defaultdict(dict), defaultdict(dict)
    for rating in observed.values():
        if rating.question_id is None:
            continue
        original = observed[rating.question_id]
        lag = rating.chunk - original.chunk
        if (original.kind != 'question' or rating.excerpt != original.excerpt or lag not in (1, 2)
                or lag in question_ratings[original.id] or lag in paired[rating.excerpt, rating.chunk]):
            raise ValueError('Follow-ups must identify a unique prior question for the same reader and excerpt')
        question_ratings[original.id][lag] = rating
        paired[rating.excerpt, rating.chunk][lag] = rating
    if len(question_ratings) != question_count:
        raise ValueError('Every question must have a first observed follow-up')
    cohorts = {1: Counter(), 2: Counter()}
    talks = defaultdict(Counter)
    for key, ratings in question_ratings.items():
        first, second = ratings[1], ratings.get(2)
        question = observed[key]
        opportunities = min(2, last[question.excerpt] - question.chunk)
        if (opportunities not in (1, 2) or question.chunk_span[1] != first.chunk_span[0]
                or (second is not None) != (opportunities == 2 and first.score < THRESHOLD)
                or (second is not None and first.chunk_span[1] != second.chunk_span[0])):
            raise ValueError('TED-Q actual follow-ups must match the published stop rule')
        key = partition(first, second, opportunities)
        cohorts[opportunities][key] += 1
        talks[question.excerpt[1]][key] += 1
    totals = cohorts[1] + cohorts[2]
    if sum(totals.values()) != question_count:
        raise ValueError('Every unique question must belong to exactly one final rating category')
    concurrent = [ratings for ratings in paired.values() if len(ratings) == 2]
    conservative_counterexamples = sum(
        ratings[2].score == FULL and ratings[1].score < THRESHOLD for ratings in concurrent
    )
    by_talk = [summarize(counts) for counts in talks.values()]
    return {
        'source_revision': REVISION,
        'source_sha256': CSV_SHA256,
        'readers': len({excerpt[0] for excerpt in last}),
        'talks': len(talks),
        'retirement_rating_threshold': THRESHOLD,
        'full_answer_rating': FULL,
        'estimand': 'At least one rating of 5 by the last available probe, capped at two probes after the question',
        'bound_assumption': ('Keep observed ratings fixed and complete only the protocol-missing second ratings '
                             'on already displayed chunks. Excerpt-end absences are outside the window.'),
        'all_available_windows': summarize(totals),
        'by_available_followup_count': {str(n): summarize(counts) for n, counts in cohorts.items()},
        'talk_range_of_bound_width_pp': [
            100 * metric(row['full_fraction_upper'] - row['full_fraction_lower'] for row in by_talk)
            for metric in (min, max)
        ],
        'simultaneously_rated_older_and_newer_questions': {
            'probes': len(concurrent),
            'older_score_5_newer_below_3': conservative_counterexamples,
        },
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--elicitation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.elicitation)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False, sort_keys=True)
        stream.write('\n')
