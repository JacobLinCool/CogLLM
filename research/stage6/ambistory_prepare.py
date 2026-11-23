"""Prepare the original AmbiStory homonym split and separate text from judgments."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from statistics import mean, stdev
from tempfile import TemporaryDirectory

from research.semantic_tracking.iirc_prepare import digest
from research.stage2.records import write_jsonl


REVISION = '705f1a3784b62f1aa42435b9e107dba901547441'
SOURCES = {
    'train': ('7b3207fa81513027d39193a9a601fba6f886323a893661eb3a94c14f432bbf60', 2280),
    'dev': ('a3110b61fd20a37caac2ed56fe0a7b0d17ce29d34b156ce65cc2396d5bda705a', 588),
    'test': ('37c0264c7ba0a5bcdce262d25cb6dba9b66bc06161f337e8f1a4742b54c30968', 930),
}


def identifier(*values):
    return sha256(json.dumps(values, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def collect_split(path, expected):
    checksum, samples = expected
    contents = path.read_bytes()
    rows = json.loads(contents)
    if sha256(contents).hexdigest() != checksum or len(rows) != samples or len({r['sample_id'] for r in rows.values()}) != samples:
        raise ValueError(f'Unexpected author JSON or duplicate sample at {path}')
    setups = defaultdict(list)
    for row in rows.values():
        setups[row['homonym'], row['precontext'], row['sentence']].append(row)
    return setups


def records(setups, heldout_homonyms):
    texts, designs, votes, exclusions = [], [], [], []
    for (homonym, precontext, sentence), samples in sorted(setups.items()):
        family = identifier(homonym, precontext, sentence)
        if homonym.casefold() in heldout_homonyms:
            exclusions.append({'id': family, 'reason': 'homonym_in_later_author_split'})
            continue
        pattern = rf'(?<!\w){re.escape(homonym)}(?!\w)'
        targets = list(re.finditer(pattern, sentence, flags=re.IGNORECASE))
        if len(targets) != 1:
            exclusions.append({'id': family, 'reason': 'target_surface_missing_or_nonunique'})
            continue
        offset = len(precontext.strip()) + 1
        excerpt = precontext.strip() + ' ' + sentence
        target = [offset + targets[0].start(), offset + targets[0].end()]
        if excerpt[slice(*target)].casefold() != homonym.casefold():
            raise ValueError(f'Unaligned target: {family}')
        target_pre_stop = len(excerpt[:target[0]].rstrip())
        definitions = sorted({(r['judged_meaning'], r['example_sentence']) for r in samples})
        endings = sorted({r['ending'] for r in samples})
        if len(samples) != 6 or len(definitions) != 2 or len(endings) != 3 or endings[0] != '':
            raise ValueError(f'Expected 2 senses by 3 story variants: {family}')
        cells = {(r['ending'], r['judged_meaning']): r for r in samples}
        for ending in endings:
            text = f'{excerpt} {ending}' if ending else excerpt
            case = identifier(family, ending)
            stops = sorted({target_pre_stop, target[1], len(excerpt), len(text)})
            texts.append({'id': case, 'text': text, 'stops': stops})
            definitions_for_case = []
            for definition, example in definitions:
                row = cells[ending, definition]
                labels, nonsensical = row['choices'], row['nonsensical']
                if (len(labels) < 5 or any(type(x) is not int or not 1 <= x <= 5 for x in labels)
                        or len(labels) != len(nonsensical) or any(type(x) is not bool for x in nonsensical)
                        or not math.isclose(mean(labels), row['average'], abs_tol=0.001)
                        or not math.isclose(stdev(labels), row['stdev'], abs_tol=0.001)):
                    raise ValueError(f'Unexpected individual ratings: {row["sample_id"]}')
                sense = identifier(family, definition)
                definitions_for_case.append({'id': sense, 'definition': definition, 'example': example})
                votes.append({'id': case, 'sense_id': sense, 'author_sample': row['sample_id'],
                              'ratings': labels, 'nonsensical': nonsensical})
            designs.append({'id': case, 'family': family, 'homonym': homonym.casefold(), 'target': target,
                            'target_pre_stop': target_pre_stop, 'setup_stop': len(excerpt), 'post_stop': len(text),
                            'has_ending': bool(ending),
                            'senses': definitions_for_case})
    return texts, designs, votes, exclusions


def prepare(source, output):
    if output.exists():
        raise FileExistsError(output)
    splits = {split: collect_split(source / f'{split}.json', expected) for split, expected in SOURCES.items()}
    homonyms = {split: {key[0].casefold() for key in setups} for split, setups in splits.items()}
    later_homonyms = {'train': homonyms['dev'] | homonyms['test'], 'dev': homonyms['test'], 'test': set()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.ambistory-', dir=output.parent) as temporary:
        staged = Path(temporary) / 'data'
        staged.mkdir(mode=0o700)
        statistics, hashes = {}, {}
        for split, setups in splits.items():
            texts, designs, votes, exclusions = records(setups, later_homonyms[split])
            files = ((f'{split}/model_input/stimuli.jsonl', texts), (f'{split}/design/stops.jsonl', designs),
                     (f'{split}/scoring/ordinal-ratings.jsonl', votes))
            for name, items in files:
                path = staged / name
                path.parent.mkdir(parents=True, mode=0o700)
                write_jsonl(path, items)
                hashes[name] = digest(path)
            statistics[split] = {'story_families': len(setups) - len(exclusions), 'stories': len(texts),
                                 'sense_samples': len(votes), 'votes': sum(len(row['ratings']) for row in votes),
                                 'nonsensical_votes': sum(sum(row['nonsensical']) for row in votes),
                                 'excluded_families': exclusions}
        receipt = {'author_revision': REVISION, 'source_sha256': {k: v[0] for k, v in SOURCES.items()},
                   'files_sha256': hashes, 'preparation_sha256': digest(Path(__file__)), 'splits': statistics}
        (staged / 'manifest.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        staged.rename(output)
    return receipt


def certify(prepared, model, output):
    """Verify tokenizer boundaries using the actual Qwen tokenizer and no model weights."""
    from transformers import AutoTokenizer, __version__

    from research.core_features.extract import plan
    from research.gradio_app.plaba import TOKENIZER_SHA256
    from research.gradio_app.tokenizer import TOKENIZER_CONFIG_SHA256
    from research.stage3.run import REVISION as QWEN_REVISION, TRANSFORMERS_VERSION

    tokenizer_checks = {'tokenizer.json': TOKENIZER_SHA256, 'tokenizer_config.json': TOKENIZER_CONFIG_SHA256}
    if (__version__ != TRANSFORMERS_VERSION or model.name != QWEN_REVISION
            or any(digest(model / name) != expected for name, expected in tokenizer_checks.items())):
        raise ValueError('Expected the pinned Qwen tokenizer files and Transformers release')
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    report = json.loads((prepared / 'manifest.json').read_bytes())
    if report['preparation_sha256'] != digest(Path(__file__)):
        raise ValueError('Prepared inputs use different material-selection code')
    for split in SOURCES:
        def read(relative):
            member = f'{split}/{relative}'
            data = (prepared / member).read_bytes()
            if sha256(data).hexdigest() != report['files_sha256'][member]:
                raise ValueError(f'Unexpected prepared bytes: {member}')
            return list(map(json.loads, data.splitlines()))

        texts = read('model_input/stimuli.jsonl')
        design = {row['id']: row for row in read('design/stops.jsonl')}
        prefixes = defaultdict(set)
        counts = defaultdict(int)
        for item in texts:
            row = design[item['id']]
            prefixes[row['family']].add(item['text'][:row['setup_stop']])
            counts[row['family']] += 1
        documents, rejected = plan(tokenizer, texts, batch_prefixes=True)
        if (len(documents) != len(design) or len(documents) != len(texts) or rejected
                or any(len(items) != 1 for items in prefixes.values()) or set(counts.values()) != {3}):
            raise ValueError(f'Expected all stops and exactly one common prefix per story family in {split}')
        report['splits'][split]['tokenizer_preflight'] = {
            'stories': len(documents), 'tokens': sum(len(row[1]) for row in documents),
            'certified_stops': sum(len(row[2]) for row in documents), 'rejected_stops': len(rejected),
        }
    report['tokenizer'] = {'revision': QWEN_REVISION, 'transformers': __version__, 'files_sha256': tokenizer_checks}
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    build, verify = actions.add_parser('prepare'), actions.add_parser('certify')
    for action, names in ((build, ('source', 'output')), (verify, ('prepared', 'model', 'output'))):
        for name in names:
            action.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        print(json.dumps(prepare(args.source, args.output)['splits'], indent=2))
    else:
        certify(args.prepared, args.model, args.output)


if __name__ == '__main__':
    main()
