"""Build a position-aligned development packet without accessing formal outcomes."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unicodedata

from research.backbone_transfer.extract import (
    MAX_INPUT_TOKENS, certify, digest, read_json, stream_rows, verified_model, write_json, write_rows,
)

HERE = Path(__file__).resolve().parent


def sha(value):
    return sha256(value.encode('utf-8')).hexdigest()


def normal_words(text):
    return re.findall(r'[^\W_]+', unicodedata.normalize('NFKC', text).casefold())


class Union:
    """Collect response sufficient statistics and keep observed coordinates explicit."""

    def __init__(self, cache, holdout):
        self.cache = cache
        self.reserved = {
            name.casefold() for row in holdout['families'] for name in (row['id'], *row['aliases'])
        }
        self.documents = {}
        self.families = {}
        self.parents = {}
        self.tasks = {}
        self.effects = defaultdict(list)
        self.sources = {}
        self.filters = Counter()

    def input(self, source, relative):
        if source.casefold() in self.reserved:
            raise ValueError(f'Formal validation family requested by training: {source}')
        path = self.cache / relative
        record = self.sources.setdefault(source, {})
        record[relative] = digest(path)
        return path

    def rows(self, source, relative):
        return list(stream_rows(self.input(source, relative)))

    def root(self, family):
        while self.parents[family] != family:
            self.parents[family] = self.parents[self.parents[family]]
            family = self.parents[family]
        return family

    def unite(self, left, right):
        lroot, rroot = sorted((self.root(left), self.root(right)))
        self.parents[rroot] = lroot

    def document(self, source, family, text):
        if source.casefold() in self.reserved:
            raise ValueError(f'Formal validation family requested by training: {source}')
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f'Empty {source} stimulus')
        group = f'{source}:{family}'
        self.parents.setdefault(group, group)
        key = sha(text)
        if key in self.documents:
            self.unite(group, self.families[key])
        else:
            self.documents[key] = text
            self.families[key] = group
        return key

    def statistic(self, name, source, key, stop, kind, labels, statistics):
        if not 0 < stop <= len(self.documents[key]) or any(not math.isfinite(v) for v in statistics):
            raise ValueError(f'Invalid {name} readout or observations')
        task = self.tasks.setdefault(name, (source, kind, tuple(labels), {}))
        if task[:3] != (source, kind, tuple(labels)):
            raise ValueError(f'An endpoint changes its observation contract: {name}')
        row = task[3].setdefault((key, stop), [0.0] * len(statistics))
        for i, value in enumerate(statistics):
            row[i] += value
        return key, stop

    def counts(self, name, source, key, stop, labels, counts):
        if len(counts) != len(labels) or not all(int(v) == v and v >= 0 for v in counts) or not sum(counts):
            raise ValueError(f'Invalid category count for {name}')
        return self.statistic(name, source, key, stop, 'categorical', labels, counts)

    def category(self, name, source, key, stop, labels, value):
        if value not in labels:
            raise ValueError(f'Unregistered category for {name}: {value!r}')
        return self.counts(name, source, key, stop, labels, [int(i == value) for i in labels])

    def scalar(self, name, source, key, stop, value, count=1, variance=0.0):
        if count < 1 or variance < 0:
            raise ValueError(f'Invalid Gaussian sufficient statistics for {name}')
        return self.statistic(name, source, key, stop, 'gaussian', (name,),
                              (count, value * count, (variance + value * value) * count))

    def contrast(self, name, label, left, right, scale, metric='mean', category=None):
        """Accumulate matched alternatives, with equal weight for each material pair."""
        if not left or not right or scale <= 0 or not set(left).isdisjoint(right):
            raise ValueError(f'Invalid pair for {name}/{label}')
        self.effects[name, label, scale, metric, category].append((list(left), list(right)))
        keys = [self.families[key] for key, _ in left + right]
        for group in keys[1:]:
            self.unite(keys[0], group)

    def grouped_roles(self):
        # Exact repeated documents and registered matched versions never cross roles.
        by_source = defaultdict(set)
        for source, _, _, rows in self.tasks.values():
            by_source[source].update(self.root(self.families[doc]) for doc, _ in rows)
        dev = set()
        fit_only = {next(iter(groups)) for groups in by_source.values() if len(groups) == 1}
        for source, groups in sorted(by_source.items()):
            if len(groups) == 1:
                continue
            n_dev = max(1, len(groups) // 5)
            dev.update(sorted(groups - fit_only, key=lambda group: sha('union-dev-v1|' + group))[:n_dev])
        for source, groups in by_source.items():
            if not groups - dev or (len(groups) > 1 and groups - fit_only and not groups & dev):
                raise ValueError(f'Shared materials caused a degenerate development split for {source}')
        return {key: int(self.root(group) in dev) for key, group in self.families.items()}


def sealed_texts(root, holdout):
    hashes = set()
    for entry in holdout['families']:
        for member in entry.get('source_pins', []):
            if 'path' not in member:
                continue
            relative = Path(member['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Formal verification file must be a project-relative path')
            path = root / relative
            if not member.get('sha256') or digest(path) != member['sha256']:
                raise ValueError(f'Formal verification fingerprint changed: {relative}')
            if member.get('kind') == 'normalized_text_hashes':
                hashes.update(row['sha256'] for row in stream_rows(path))
    return hashes


def expected(payload, kind, metric, category):
    if kind == 'gaussian':
        return payload[1]
    if metric == 'category':
        return payload[category] / sum(payload)
    return sum(i * count for i, count in enumerate(payload)) / sum(payload)


def compile_effects(union, name, rows, role):
    metadata = union.tasks[name]
    index = {(row['doc'], row['stop']): i for i, row in enumerate(rows)}
    results = {'train': [], 'development': []}
    for (endpoint, label, scale, metric, category), pairs in sorted(union.effects.items()):
        if endpoint != name:
            continue
        for side, destination in enumerate(('train', 'development')):
            coefficients = Counter()
            matched = 0
            for left, right in pairs:
                if not all(key in index and role[key[0]] == side for key in left + right):
                    continue
                matched += 1
                for group, sign in ((left, -1), (right, 1)):
                    for key in group:
                        coefficients[index[key]] += sign / len(group)
            if matched:
                indices = sorted(i for i, coefficient in coefficients.items() if coefficient)
                values = [coefficients[i] / matched for i in indices]
                actual = sum(value * expected(rows[i]['outcome'], metadata[1], metric, category)
                             for i, value in zip(indices, values, strict=True))
                if indices:
                    record = {'indices': indices, 'coefficients': values, 'target': actual, 'scale': scale,
                              'weight': 1.0, 'metric': metric}
                    if category is not None:
                        record['category'] = category
                    if side:
                        record.update(label=label, matched_pairs=matched)
                    results[destination].append(record)
    return results


def prepare(cache, tokenizer_path, holdout_path, output):
    from tokenizers import Tokenizer
    from research.union_training.sources import add_sources

    if output.exists():
        raise FileExistsError(output)
    _, registered = verified_model(tokenizer_path.parent, 'qwen', tokenizer_only=True)
    if tokenizer_path != tokenizer_path.parent / 'tokenizer.json' or digest(tokenizer_path) != registered['sha256']['tokenizer.json']:
        raise ValueError('Development preparation requires the pinned Qwen tokenizer')
    holdout = read_json(holdout_path)
    if holdout['schema_version'] != 1 or not any(family['status'] == 'sealed' for family in holdout['families']):
        raise ValueError('At least one independent formal verification family must be sealed first')
    blocked = sealed_texts(HERE.parents[1], holdout)
    union = Union(cache, holdout)
    add_sources(union)

    requested = defaultdict(set)
    for _, _, _, rows in union.tasks.values():
        for key, stop in rows:
            requested[key].add(stop)
    for key in requested:
        text = union.documents[key]
        if sha(' '.join(normal_words(text))) in blocked:
            raise ValueError(f'Development contains a formal held-out stimulus: {key}')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    certified = {}
    input_tokens = 0
    for key, stops in sorted(requested.items()):
        text = union.documents[key]
        encoding = tokenizer.encode(text, add_special_tokens=False)
        if len(encoding.ids) > MAX_INPUT_TOKENS:
            union.filters['document_exceeds_extraction_window'] += len(stops)
            continue
        doc = certify(key, text, sorted(stops), encoding, tokenizer)
        for reason in doc.missing.values():
            union.filters[reason] += 1
        if doc.certified:
            certified[key] = set(doc.certified)
            input_tokens += len(doc.ids)

    roles = union.grouped_roles()
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.union-prepared-') as temporary:
        stage = Path(temporary) / 'packet'
        stage.mkdir()
        write_rows(stage / 'stimuli.jsonl', [
            {'id': key, 'text': union.documents[key], 'stops': sorted(stops)}
            for key, stops in certified.items()
        ])
        index, inventory = [], {}
        for name, (source, kind, labels, records) in sorted(union.tasks.items()):
            rows = []
            lost = 0
            for (key, stop), stats in sorted(records.items()):
                if stop not in certified.get(key, ()):
                    lost += int(sum(stats) if kind == 'categorical' else stats[0])
                    continue
                if kind == 'gaussian':
                    count, total, total_sq = stats
                    mean = total / count
                    outcome = [count, mean, max(0.0, total_sq / count - mean * mean)]
                else:
                    outcome = stats
                rows.append({'id': f'{key}:{stop}', 'doc': key, 'stop': stop, 'split': roles[key],
                             'family': union.root(union.families[key]), 'outcome': outcome})
            if not any(row['split'] == 0 for row in rows):
                raise ValueError(f'No certifiable fitting observations for {name}')
            effects = compile_effects(union, name, rows, roles)
            relative = f'{name}.jsonl'
            write_rows(stage / relative, rows)
            index.append({'name': name, 'source': source, 'kind': kind, 'file': relative, 'labels': labels,
                          'effects': effects['train'], 'dev_effects': effects['development']})
            inventory[name] = {
                'source': source, 'lost_observations_to_token_geometry': lost,
                'fit_supervision': int(sum((row['outcome'][0] if kind == 'gaussian' else sum(row['outcome']))
                                          for row in rows if row['split'] == 0)),
                'dev_supervision': int(sum((row['outcome'][0] if kind == 'gaussian' else sum(row['outcome']))
                                          for row in rows if row['split'] == 1)),
                'fit_rows': sum(row['split'] == 0 for row in rows),
                'dev_rows': sum(row['split'] == 1 for row in rows),
                'fit_families': len({row['family'] for row in rows if row['split'] == 0}),
                'dev_families': len({row['family'] for row in rows if row['split'] == 1}),
                'fit_effects': len(effects['train']), 'dev_effects': len(effects['development']),
            }
        write_json(stage / 'index.json', {'tasks': index})
        write_json(stage / 'receipt.json', {
            'formal_holdout_sha256': digest(holdout_path), 'tokenizer_sha256': digest(tokenizer_path),
            'source_files_sha256': union.sources, 'source_filters': dict(union.filters),
            'normalized_formal_text_hashes_checked': len(blocked),
            'development_documents': len(certified), 'input_tokens': input_tokens, 'endpoints': inventory,
            'packet_sha256': {path.name: digest(path) for path in sorted(stage.iterdir())},
        })
        stage.rename(output)
    print(f'{len(inventory)} endpoints, {len(certified)} documents, {input_tokens} Qwen input tokens', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=Path('.cache/cogllm'))
    parser.add_argument('--tokenizer-path', type=Path, required=True)
    parser.add_argument('--holdout-path', type=Path, default=HERE / 'holdout.json')
    parser.add_argument('--output', type=Path, required=True)
    prepare(**vars(parser.parse_args()))
