"""Reconstruct Study 1 responses from Anglin et al. (2025)'s fixed public release."""

import argparse
from hashlib import file_digest
from itertools import product
import json
from pathlib import Path

import numpy as np
import pyreadstat
from scipy.stats import f, t


SOURCES = {
    'study1.sav': ('https://osf.io/download/v6nkd/', '5e637de7a479315ea46286400c0003cb213d3aa2a419a1b1467c4ba632dea4aa'),
    'study1.sps': ('https://osf.io/download/zwfqc/', '208452a0038f256d1438fc1181201311d89e6fbd1875015813d6c6a0f3cb75b5'),
    'stimulus.pdf': ('https://osf.io/download/juam5/', 'ce8bf04b616ae86b0bf471b7b66162d01e2405b1b0319f62cf9dfb011b0ca850'),
    'followup.pdf': ('https://osf.io/download/nyrgb/', 'de28284153116f1a383ed3a9ec58f67f73374d9ed5e4a615f6246ba1a77b0e3a'),
}
ARMS = ('both_ineffective', 'both_effective', 'ineffective_first', 'effective_first')
CONTRASTS = ((0, 1), (0, 2), (0, 3), (1, 2), (2, 3))
BELIEF = [f'DP_belief_T{time}' for time in range(1, 5)]
CERTAINTY = [f'DP_certainty_T{time}' for time in range(1, 5)]
DESIGN = ('ID', 'StudyA', 'StudyB', 'Study_order', 'results_combined')
ATTENTION = ('MCA_pass', 'MCB_pass', 'MCcombined_pass')

# Table 1, rows T1–T4, using the author's displayed rounded values.
PUBLISHED_COUNTS = (68, 93, 72, 79)
PUBLISHED_MEANS = (
    (4.10, 4.12, 3.82, 3.65), (3.84, 4.85, 4.88, 5.05),
    (4.03, 3.74, 4.26, 4.10), (4.42, 5.11, 4.65, 4.52),
)
PUBLISHED_SDS = (
    (2.16, 2.03, 2.03, 2.05), (2.20, 2.06, 2.12, 2.05),
    (2.08, 1.91, 1.93, 1.89), (2.01, 1.85, 1.96, 2.01),
)


def check_sources(directory):
    result = {}
    for name, (url, expected) in SOURCES.items():
        with (directory / name).open('rb') as source:
            actual = file_digest(source, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError(f'Source hash changed for {name}: {actual}')
        result[name] = {'url': url, 'sha256': actual}
    return result


def get_design(frame):
    a, b, order = (np.asarray(frame[name], dtype=np.float64) for name in DESIGN[1:4])
    first = np.where(order == 1, a, np.where(order == 2, b, np.nan))
    both_valid = np.isin(a, (1, 2)) & np.isin(b, (1, 2))
    # The author's syntax groups like outcomes even if order is missing.
    derived = np.where(both_valid, np.where(a == b, a, first + 2), np.nan)
    stored = np.asarray(frame['results_combined'], dtype=np.float64)
    known = np.isfinite(stored)
    if not np.array_equal(derived[known], stored[known]):
        raise ValueError('Stored condition disagrees with the author’s SPSS condition mapping')
    return a, b, order, derived, known


def verify_cohort(frame, mask, a, b, order, certainty):
    ids = frame.loc[mask, 'ID']
    if ids.isna().any() or not ids.is_unique or not np.isfinite(certainty[mask]).all():
        raise ValueError('Analytic rows must have unique respondent keys and all included certainty ratings')
    if not np.all(np.isin(np.column_stack((a, b, order))[mask], (1, 2))):
        raise ValueError('Analytic rows must map to eight fully specified presentation sequences')


def summarize(values, include_contrasts):
    profiles = {}
    for index, column in enumerate(values.T, 1):
        profiles[f'T{index}'] = {
            'mean': float(column.mean()),
            'sample_sd': float(column.std(ddof=1)),
            'counts_1_to_8': np.bincount(column.astype(int), minlength=9)[1:].tolist(),
        }
    summary = {'timepoints': profiles}
    if include_contrasts:
        comparisons = {}
        df = len(values) - 1
        for before, after in CONTRASTS:
            if after >= values.shape[1]:
                continue
            delta = values[:, after] - values[:, before]
            estimate = float(delta.mean())
            sd = float(delta.std(ddof=1))
            se = sd / np.sqrt(len(values))
            statistic = estimate / se
            radius = float(t.ppf(0.995, df) * se)
            comparisons[f'T{after + 1}-T{before + 1}'] = {
                'mean_change_scale_points': estimate,
                'paired_t': float(statistic), 'df': df,
                'two_sided_p_unadjusted': float(2 * t.sf(abs(statistic), df)),
                'ci_99_scale_points': [estimate - radius, estimate + radius],
            }
        summary['paired_contrasts'] = comparisons
    return summary


def interaction(values, arms):
    centered = values - values.mean(axis=1, keepdims=True)
    indices = arms.astype(int) - 1
    means = np.stack([centered[indices == group].mean(axis=0) for group in range(len(ARMS))])
    counts = np.bincount(indices)
    ss_effect = float(np.sum(counts[:, None] * (means - centered.mean(axis=0)) ** 2))
    ss_error = float(np.sum((centered - means[indices]) ** 2))
    time_df = values.shape[1] - 1
    df1, df2 = (len(ARMS) - 1) * time_df, (len(values) - len(ARMS)) * time_df
    statistic = (ss_effect / df1) / (ss_error / df2)
    return {'F': statistic, 'df_num': df1, 'df_denom': df2, 'p': float(f.sf(statistic, df1, df2))}


def run(directory, destination):
    sources = check_sources(directory)
    used = [*DESIGN, *ATTENTION, *BELIEF, *CERTAINTY]
    frame, _ = pyreadstat.read_sav(directory / 'study1.sav', usecols=used)
    a, b, order, arms, known = get_design(frame)
    belief, certainty = (frame[columns].to_numpy() for columns in (BELIEF, CERTAINTY))
    for values in (belief, certainty):
        if not np.isin(values[np.isfinite(values)], np.arange(1, 9)).all():
            raise ValueError('Unexpected belief or certainty scale value')

    complete = known & np.isfinite(belief).all(axis=1)
    immediate = known & np.isfinite(belief[:, :3]).all(axis=1) & np.isfinite(certainty[:, :3]).all(axis=1)
    verify_cohort(frame, complete, a, b, order, certainty)
    verify_cohort(frame, immediate, a, b, order, certainty[:, :3])

    complete_arms, immediate_arms, sd_discrepancies = {}, {}, []
    for code, arm in enumerate(ARMS, 1):
        full = complete & (arms == code)
        early = immediate & (arms == code)
        if (int(full.sum()) != PUBLISHED_COUNTS[code - 1]
                or not np.allclose(belief[full].mean(axis=0), PUBLISHED_MEANS[code - 1], atol=0.005, rtol=0)):
            raise ValueError(f'Participant count or belief means do not match Table 1 for {arm}')
        sd_pairs = zip(belief[full].std(axis=0, ddof=1), PUBLISHED_SDS[code - 1], strict=True)
        for time, (reconstructed, published) in enumerate(sd_pairs, 1):
            if abs(reconstructed - published) > 0.005:
                sd_discrepancies.append({'arm': arm, 'time': f'T{time}', 'published': published,
                                         'reconstructed': float(reconstructed)})
        complete_arms[arm] = {
            'count': int(full.sum()),
            'attention_failures_retained': {name: int(frame.loc[full, name].eq(0).sum()) for name in ATTENTION},
            'belief': summarize(belief[full], True),
            'certainty_exploratory': summarize(certainty[full], True),
        }
        immediate_arms[arm] = {
            'count': int(early.sum()),
            'belief': summarize(belief[early, :3], True),
            'certainty_exploratory': summarize(certainty[early, :3], True),
        }

    sequences = []
    for value_a, value_b, value_order in product((1, 2), repeat=3):
        exact = (a == value_a) & (b == value_b) & (order == value_order)
        full, early = complete & exact, immediate & exact
        sequences.append({
            'longitudinal_outcome_code': value_a, 'cross_sectional_outcome_code': value_b,
            'first_study': 'longitudinal' if value_order == 1 else 'cross_sectional',
            'published_cohort_count': int(full.sum()), 'immediate_cohort_count': int(early.sum()),
            'immediate_belief': summarize(belief[early, :3], False),
            'immediate_certainty_exploratory': summarize(certainty[early, :3], False),
        })

    condition_by_time = interaction(belief[complete], arms[complete])
    if not np.isclose(condition_by_time['F'], 11.25, atol=0.005, rtol=0):
        raise ValueError('Belief condition-by-time interaction does not match Study 1 Results')
    result = {
        'article': 'https://doi.org/10.1525/collabra.133914',
        'project': 'https://osf.io/gfp48/',
        'project_license': 'https://creativecommons.org/licenses/by/4.0/',
        'license_record': 'https://api.osf.io/v2/licenses/563c1cf88c5e4a3877f9e96a/',
        'sources': sources,
        'raw_counts': {'saved_rows': len(frame), 'with_saved_condition': int(known.sum()),
                       'with_t4_belief': int(np.isfinite(belief[:, 3]).sum())},
        'published_belief_cohort': {
            'count': int(complete.sum()), 'selection': 'Saved condition and all four belief scores. Attention checks retained.',
            'n_with_all_four_certainty_scores': int(np.isfinite(certainty[complete]).all(axis=1).sum()),
            'table_1': {'matched_arm_counts': 4, 'means_within_rounding': 16,
                        'sds_within_rounding': 16 - len(sd_discrepancies), 'sd_discrepancies': sd_discrepancies},
            'belief_condition_by_time': condition_by_time,
            'certainty_condition_by_time_exploratory': interaction(certainty[complete], arms[complete]),
            'arms': complete_arms,
        },
        'immediate_observed_cohort': {
            'count': int(immediate.sum()), 'selection': 'Saved condition and all paired T1–T3 belief and certainty scores.',
            'arms': immediate_arms, 'presentation_sequences': sequences,
        },
        'stimulus_visuals': {
            'code_1': 'Evidence opposes deterrence.', 'code_2': 'Evidence supports deterrence.',
            'longitudinal_code_1': {'pdf_page': 7, 'raster_object_id': 63, 'numeric_table_rows': 7},
            'longitudinal_code_2': {'pdf_page': 5, 'raster_object_id': 58, 'numeric_table_rows': 7},
            'cross_sectional_code_1': {'pdf_page': 12, 'raster_object_id': 74, 'numeric_table_rows': 10},
            'cross_sectional_code_2': {'pdf_page': 13, 'raster_object_id': 77, 'numeric_table_rows': 10},
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    run(options.source_dir, options.output)
