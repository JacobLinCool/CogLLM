"""Fit and compare question-specific first-hit curves on fixed Qwen states."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from research.backbone_transfer.extract import digest, stream_rows, write_json, write_rows
from research.semantic_tracking.load_curve_fit import cluster_interval, projection


PROTOCOL = Path(__file__).with_name('same-question-protocol.md')
METHODS = ('steps', 'local', 'lexical', 'question')
WORDS = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")
STOP_WORDS = frozenset('a an and are as at be been by did do does for from had has have in is it its of on or the to was were what when where which who why will with would'.split())
L2 = 0.08
SEPARATION = 0.05


@dataclass(frozen=True)
class PathWindow:
    group: str
    doc: str
    identity: str
    split: str
    query: tuple
    risk: int
    left: int
    end: int
    hit: bool
    focus: bool


def end_token(doc, stop):
    index = doc.exact_stop(stop)
    if index is None:
        raise ValueError(f'Expected an exact original-token endpoint at character {stop}')
    return index + 1


def left_token(doc, start):
    return int(np.searchsorted(doc.ends, start, side='right'))


def authored_windows(rows, docs):
    windows = []
    alignment = []
    for row in rows:
        a, b = row['variants']
        if {a['answer_identity'], b['answer_identity']} != {'A', 'B'} or set(row['anchors']) != {'A', 'B'}:
            raise ValueError('Every authored family must cross two answer identities')
        da, db = docs[a['id']], docs[b['id']]
        risk_a = end_token(da, row['shared_prefix_end'])
        risk_b = end_token(db, row['shared_prefix_end'])
        if (da.text[:row['shared_prefix_end']] != db.text[:row['shared_prefix_end']]
                or risk_a != risk_b or da.ids[:risk_a] != db.ids[:risk_b]
                or not np.array_equal(da.offsets[:risk_a], db.offsets[:risk_b])):
            raise ValueError('The designed shared prefix was not preserved in extraction')
        hidden_gap = float(np.max(np.abs(da.states[:risk_a] - db.states[:risk_b])))
        if hidden_gap > 1e-4:
            raise ValueError('Shared prefix projection depends on the suffix')
        cue_a = (left_token(da, a['cue'][0]), end_token(da, a['cue'][1]))
        cue_b = (left_token(db, b['cue'][0]), end_token(db, b['cue'][1]))
        if not (cue_a == cue_b and risk_a <= cue_a[0] < cue_a[1] and len(da.ids) == len(db.ids)):
            raise ValueError('Crossed continuations must have identical step and cue-window geometry')
        alignment.append({'family_id': row['family_id'], 'split': row['split'], 'prefix_tokens': risk_a,
                          'cue_start_token': cue_a[0], 'cue_end_exclusive': cue_a[1],
                          'text_tokens_per_branch': len(da.ids), 'maximum_prefix_feature_difference': hidden_gap})
        for variant in (a, b):
            for identity, span in row['anchors'].items():
                if not 0 <= span[0] < span[1] <= row['shared_prefix_end']:
                    raise ValueError('Question anchor is not contained in the common prefix')
                windows.append(PathWindow(row['family_id'], variant['id'], identity, row['split'], tuple(span),
                                          risk_a, cue_a[0], cue_a[1],
                                          identity == variant['answer_identity'], True))
    return windows, alignment


def evidence_windows(rows, docs):
    windows = []
    for row in rows:
        doc = docs[row['id']]
        lookup = {query['id']: query for query in row['queries']}
        prior = row['article_start']
        prefix = len(doc.text[:prior].rstrip())
        risk = end_token(doc, prefix)
        early, late = row['earlier_window'], row['later_window']
        interval = lambda window: (left_token(doc, window['start']), end_token(doc, window['end']))
        first, end = interval(early)
        next_first, next_end = interval(late)
        if not risk <= first < end <= next_first < next_end:
            raise ValueError('ExpMRC support windows must be ordered and disjoint')
        questions = (row['earlier_query'], row['later_query'])
        for i, question in enumerate(questions):
            span = tuple(lookup[question]['span'])
            if not 0 <= span[0] < span[1] <= prior:
                raise ValueError('ExpMRC question must be visible before original text')
            windows.append(PathWindow(row['group_id'], row['id'], question, row['split'], span, risk,
                                      first, end, i == 0, True))
            if i == 1:
                windows.append(PathWindow(row['group_id'], row['id'], question, row['split'], span, risk,
                                          next_first, next_end, True, False))
    return windows


def words(text):
    return frozenset(word for word in WORDS.findall(text.casefold()) if word not in STOP_WORDS)


def make_features(doc, query, risk, end):
    q = doc.states[end_token(doc, query[1]) - 1].astype(np.float64)
    steps = np.arange(risk, end)
    vectors = doc.states[risk:end].astype(np.float64)
    previous = doc.states[risk-1:end-1].astype(np.float64)
    delta = vectors - previous
    elapsed = steps - risk + 1
    age = np.column_stack((np.log1p(elapsed), np.minimum(elapsed / 64, 3)))
    terms = words(doc.text[query[0]:query[1]])
    recent = [words(doc.text[max(doc.offsets[max(t - 11, risk), 0], 0):doc.ends[t]]) for t in steps]
    overlap = np.asarray([[len(terms & seen), len(terms & seen) / max(len(terms | seen), 1)] for seen in recent])
    local = np.column_stack((age, vectors))
    lexical = np.column_stack((age, overlap))
    scale = np.linalg.norm(q) * np.maximum(np.linalg.norm(vectors, axis=-1), 1e-10)
    cosine = (vectors @ q) / max(np.linalg.norm(q), 1e-10) / np.maximum(np.linalg.norm(vectors, axis=-1), 1e-10)
    update = (delta @ q) / np.maximum(scale, 1e-10)
    interaction = np.column_stack((local, overlap, cosine, update, (vectors * q) / np.sqrt(len(q)),
                                   (delta * q) / np.sqrt(len(q))))
    return dict(zip(METHODS, (age, local, lexical, interaction), strict=True))


@dataclass(frozen=True)
class Features:
    paths: tuple
    x: dict


def feature_sets(paths, docs):
    data = tuple(make_features(docs[p.doc], p.query, p.risk, p.end) for p in paths)
    return Features(tuple(paths), {method: np.concatenate([a[method] for a in data]) for method in METHODS})


class FirstHitLoss:
    def __init__(self, traces):
        family_size = Counter(trace.group for trace in traces)
        self.weight = np.array([1 / (len(family_size) * family_size[trace.group]) for trace in traces])
        self.hits = np.array([trace.hit for trace in traces])
        length = np.array([trace.end - trace.risk for trace in traces])
        if np.any(length <= 0):
            raise ValueError('Every event or censored trajectory needs at least one token')
        self.starts = np.r_[0, np.cumsum(length)[:-1]]
        self.trace_id = np.repeat(np.arange(len(length)), length)
        offsets = np.arange(length.sum()) - self.starts[self.trace_id]
        window_left = np.array([trace.left - trace.risk for trace in traces])
        self.inside = self.hits[self.trace_id] & (offsets >= window_left[self.trace_id])

    def values(self, logits):
        average_step = np.logaddexp(0, logits)
        survival_loss = np.add.reduceat(np.where(self.inside, 0, average_step), self.starts)
        event_mass = np.add.reduceat(np.where(self.inside, average_step, 0), self.starts)
        hit_loss = -np.log(np.maximum(-np.expm1(-event_mass[self.hits]), np.finfo(float).tiny))
        survival_loss[self.hits] += hit_loss
        gradient = expit(logits)
        idx = self.trace_id[self.inside]
        gradient[self.inside] *= np.exp(-event_mass[idx]) / np.minimum(np.expm1(-event_mass[idx]), -1e-300)
        return survival_loss, gradient * self.weight[self.trace_id]


def train(features, method, block_widths=None):
    raw = features.x[method]
    center = raw.mean(axis=0)
    scale = np.maximum(raw.std(axis=0), .1)
    if block_widths is not None:
        if sum(block_widths) != raw.shape[1] or min(block_widths) < 1:
            raise ValueError('Feature blocks must match the fixed readout geometry')
        scale *= np.repeat(np.sqrt(np.asarray(block_widths) / raw.shape[1]), block_widths)
    design = (raw - center) / scale
    design /= np.sqrt(design.shape[1])
    likelihood = FirstHitLoss(features.paths)

    def objective(parameters):
        weights = parameters[1:]
        neglog, score_gradient = likelihood.values(design @ weights + parameters[0])
        loss = neglog @ likelihood.weight + L2 / 2 * (weights @ weights)
        return loss, np.r_[score_gradient.sum(), design.T @ score_gradient + L2 * weights]

    initial = np.zeros(design.shape[1] + 1)
    initial[0] = -np.log(np.median([p.end - p.risk for p in features.paths]))
    solution = minimize(objective, initial, method='L-BFGS-B', jac=True,
                        options={'maxiter': 220, 'gtol': 1e-8, 'ftol': 1e-12})
    if not np.isfinite(solution.fun) or (not solution.success and solution.status != 1):
        raise RuntimeError(f'First-hit fit failed for {method}: {solution.message}')
    return {'center': center, 'scale': scale, 'w': solution.x[1:], 'b': float(solution.x[0]),
            'fit': {'iterations': int(solution.nit), 'converged': bool(solution.success),
                    'nll_regularized': float(solution.fun)}}


def apply_readout(x, fit):
    return ((x - fit['center']) / fit['scale']) @ fit['w'] / np.sqrt(x.shape[1]) + fit['b']


def evaluate(features, method, model):
    logits = apply_readout(features.x[method], model)
    likelihood = FirstHitLoss(features.paths)
    losses, _ = likelihood.values(logits)
    stop = 0
    scores = []
    for p, loss in zip(features.paths, losses, strict=True):
        start = stop
        stop += p.end - p.risk
        cumulative = np.r_[0., np.cumsum(-np.logaddexp(0, logits[start:stop]))]
        lo = p.left - p.risk
        drop = np.exp(cumulative[lo]) * -np.expm1(cumulative[-1] - cumulative[lo])
        scores.append({'group': p.group, 'doc': p.doc, 'identity': p.identity,
                       'hit': p.hit, 'focus': p.focus, 'loss': float(loss), 'drop': float(drop),
                       'before_cue_drop': float(-np.expm1(cumulative[lo])),
                       'initial_token': p.risk, 'cue_left_token': p.left, 'cue_end_exclusive': p.end,
                       'survival': np.exp(cumulative[1:]).tolist()})
    return scores


def family_scores(scores):
    grouped = defaultdict(list)
    for row in scores:
        grouped[row['group']].append(row)
    summary = {}
    for key, rows in grouped.items():
        outcomes = defaultdict(dict)
        for row in rows:
            if row['focus']:
                flags = outcomes[row['doc']]
                if row['hit'] in flags:
                    raise ValueError('Exactly one focus question can occupy each condition')
                flags[row['hit']] = row
        if not all(choices.keys() == {True, False} for choices in outcomes.values()):
            raise ValueError('A paired cue needs two labels for the same visible token window')
        differences = np.array([choice[True]['drop'] - choice[False]['drop'] for choice in outcomes.values()])
        separated = []
        for choice in outcomes.values():
            matched, unmatched = choice[True], choice[False]
            diff = np.asarray(unmatched['survival']) - matched['survival']
            where = np.flatnonzero(diff >= SEPARATION)
            first = int(where[0]) + matched['initial_token'] if len(where) else None
            separated.append({'doc': matched['doc'], 'token': first,
                              'in_cue': first is not None and first >= matched['cue_left_token']})
        summary[key] = {'nll': float(np.mean([r['loss'] for r in rows])),
                        'matched_minus_unmatched_drop': float(differences.mean()),
                        'matched_order_score': float(((differences > 1e-12) + .5 * (abs(differences) <= 1e-12)).mean()),
                        'matched_before_cue_drop': float(np.mean([r['before_cue_drop'] for r in rows if r['focus'] and r['hit']])),
                        'first_5pp_separation': separated}
    return summary


def scores_by_endpoint(trained, held, output, name):
    scores = {mode: evaluate(held, mode, fit) for mode, fit in trained.items()}
    summaries = {method: family_scores(rows) for method, rows in scores.items()}
    group = sorted(summaries['steps'])
    if len(group) < 2 or any(set(s) != set(group) for s in summaries.values()):
        raise ValueError('Held-out methods do not have identical independent material groups')
    report = {}
    for method, summary in summaries.items():
        report[method] = {}
        for measure in ('nll', 'matched_minus_unmatched_drop', 'matched_order_score', 'matched_before_cue_drop'):
            values = np.array([summary[g][measure] for g in group])
            report[method][measure] = cluster_interval(values, group, np.ones(len(group)))
            if method != 'steps':
                delta = values - [summaries['steps'][g][measure] for g in group]
                report[method][measure + '_minus_steps'] = cluster_interval(delta, group, np.ones(len(group)))
            if method == 'question':
                delta = values - [summaries['lexical'][g][measure] for g in group]
                report[method][measure + '_minus_lexical'] = cluster_interval(delta, group, np.ones(len(group)))
        separations = [row for family in summary.values() for row in family['first_5pp_separation']]
        report[method]['5pp_separation_locations'] = {
            'before_cue': sum(item['token'] is not None and not item['in_cue'] for item in separations),
            'in_cue': sum(item['in_cue'] for item in separations),
            'not_observed_by_cue_end': sum(item['token'] is None for item in separations),
        }
    write_rows(output / f'{name}-paths.jsonl', (
        {**row, 'method': method} for method in METHODS for row in scores[method]
    ))
    return {'test_traces': len(held.paths), 'test_groups': len(group), 'methods': report,
            'family_scores': summaries}


def equal_window_scores(rows, docs, trained):
    differences = defaultdict(dict)
    for row in rows:
        sham = row['earlier_sham']
        if row['split'] != 'test' or sham is None:
            continue
        doc = docs[row['id']]
        target = next(q for q in row['queries'] if q['id'] == row['earlier_query'])
        risk = end_token(doc, len(doc.text[:row['article_start']].rstrip()))
        event = row['earlier_window']
        left, right = left_token(doc, event['start']), end_token(doc, event['end'])
        sham_left, sham_right = left_token(doc, sham['start']), end_token(doc, sham['end'])
        if not risk <= sham_left < sham_right <= left < right or sham_right - sham_left != right - left:
            raise ValueError('Equal-width comparison must precede known support and be native-token matched')
        features = make_features(doc, target['span'], risk, right)
        for method, fitted in trained.items():
            hazard = np.logaddexp(0, apply_readout(features[method], fitted))
            true_drop = -np.expm1(-hazard[left-risk:right-risk].sum())
            sham_drop = -np.expm1(-hazard[sham_left-risk:sham_right-risk].sum())
            differences[method][row['group_id']] = float(true_drop - sham_drop)
    groups = sorted(differences['steps'])
    if len(groups) < 2 or any(set(group) != set(groups) for group in differences.values()):
        raise ValueError('Paired equal windows require at least two fixed article groups')
    baseline = np.array([differences['steps'][key] for key in groups])
    weights = np.ones(len(groups))
    return {method: {
        'support_minus_earlier_sham': cluster_interval([values[key] for key in groups], groups, weights),
        'minus_steps': cluster_interval(np.array([values[key] for key in groups]) - baseline, groups, weights),
    } for method, values in differences.items()}


def run(job, authored, evidence, output):
    if output.exists():
        raise FileExistsError(output)
    docs, seal = projection(job, protocol=PROTOCOL)
    authored_rows, evidence_rows = list(stream_rows(authored)), list(stream_rows(evidence))
    expected = {v['id'] for row in authored_rows for v in row['variants']} | {row['id'] for row in evidence_rows}
    if docs.keys() != expected:
        raise ValueError('A sealed input does not have one crossed-control or paired-evidence role')
    constructed, alignment = authored_windows(authored_rows, docs)
    recorded = evidence_windows(evidence_rows, docs)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.same-question-') as directory:
        stage = Path(directory) / 'assembled'
        stage.mkdir()
        metrics, weights = {}, {}
        for name, windows in (('authored', constructed), ('evidence', recorded)):
            training = [p for p in windows if p.split == 'train']
            test = [p for p in windows if p.split == 'test']
            if len(training) + len(test) != len(windows) or {p.group for p in training} & {p.group for p in test}:
                raise ValueError('Training and held-out families must be separate')
            features = feature_sets(training, docs)
            fitted = {method: train(features, method) for method in METHODS}
            for method, model in fitted.items():
                for key in ('center', 'scale', 'w', 'b'):
                    weights[f'{name}/{method}/{key}'] = model[key]
            held = feature_sets(test, docs)
            stats = scores_by_endpoint(fitted, held, stage, name)
            stats['training_traces'] = len(training)
            stats['training_groups'] = len({p.group for p in training})
            stats['fit'] = {method: model['fit'] for method, model in fitted.items()}
            if name == 'evidence':
                stats['equal_width_earlier_window'] = equal_window_scores(evidence_rows, docs, fitted)
            metrics[name] = stats
            print(name, stats['training_groups'], stats['test_groups'], flush=True)
        np.savez_compressed(stage / 'weights.npz', **weights)
        write_json(stage / 'metrics.json', metrics)
        write_json(stage / 'control-alignment.json', alignment)
        write_json(stage / 'receipt.json', {
            'protocol_sha256': digest(PROTOCOL), 'authored_pair_sha256': digest(authored),
            'evidence_pair_sha256': digest(evidence), 'projection': seal,
            'scorer_sha256': digest(Path(__file__)), 'l2': L2,
            'file_sha256': {file.name: digest(file) for file in stage.iterdir()},
        })
        stage.rename(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('job', 'authored', 'evidence', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    run(**vars(p.parse_args()))


if __name__ == '__main__':
    main()
