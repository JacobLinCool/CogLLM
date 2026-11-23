"""Add audited IIRC prose anchor→later evidence to training only."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

from tokenizers import Tokenizer

from research.backbone_transfer.extract import digest, read_json, stream_rows, write_json, write_rows
from research.token_readout.data import TOKENIZER, twelve_word_windows
from research.token_readout.transfer_data import MODELS, TOKENIZER_ROOT, EncodedText


ROOT = Path(__file__).parents[1]
SEED_AUDIT = ROOT / 'results' / 'iirc-seed-evidence-audit.json'
QUESTION_AUDIT = ROOT / 'results' / 'iirc-preparation-audit.json'
DEFAULT_SOURCE = Path('.cache/cogllm/iirc')
SELECTION_SALT = 'CogLLM portable readout same-paragraph IIRC v1'
MAX_DOCUMENT_TOKENS = 640


def verified_source(root):
    seed = read_json(SEED_AUDIT)
    prepared = read_json(QUESTION_AUDIT)
    filenames = ('stimuli.jsonl', 'locked-evidence.jsonl', 'locked-contrast-pairs.jsonl')
    files = {name: root / 'seed-evidence' / 'train' / name for name in filenames}
    questions = root / 'prepared' / 'train' / 'questions.jsonl'
    if (any(digest(path) != seed['files_sha256'][f'train/{name}'] for name, path in files.items())
            or digest(questions) != prepared['files_sha256']['train/questions.jsonl']):
        raise ValueError('IIRC official-train derivatives differ from the archived source audit')
    support = list(stream_rows(files['locked-evidence.jsonl']))
    paragraph_names = {label['paragraph_id'] for label in support if label['direction'] == 'later'}
    text = {row['id']: row['text'] for row in stream_rows(files['stimuli.jsonl']) if row['id'] in paragraph_names}
    unique = Counter(row['anchor_id'] for row in stream_rows(questions))
    pairs = {tuple(sorted(pair['question_ids'])) for pair in stream_rows(files['locked-contrast-pairs.jsonl'])
             if pair['direction'] == 'later'}
    source_sha = {name: digest(path) for name, path in {**files, 'questions.jsonl': questions}.items()}
    source_sha.update({'seed_audit': digest(SEED_AUDIT), 'official_preparation_audit': digest(QUESTION_AUDIT)})
    return text, [row for row in support if row['direction'] == 'later'], unique, pairs, source_sha


def verified_base(base):
    receipt = read_json(base / 'receipt.json')
    if receipt['schema'] != 1 or 'linked_supplement' in receipt:
        raise ValueError('A linked packet must be derived from the original portable bundle')
    paths = {name: base / name for name in receipt['file_sha256']}
    if any(digest(path) != receipt['file_sha256'][name] for name, path in paths.items()):
        raise ValueError('Original portable bundle changed after certification')
    stimuli = list(stream_rows(paths['stimuli.jsonl']))
    if any(set(row) != {'id', 'text'} for row in stimuli) or len(stimuli) != receipt['documents']:
        raise ValueError('Invalid sealed original stimuli')
    return receipt, stimuli


def geometry(entry, labels):
    anchors, windows = {}, {}
    for label in labels:
        name = label['question_id']
        anchor = label['anchor']
        coords = ((anchor['start'], anchor['end']), (label['support_start'], label['support_end']))
        if not 0 <= coords[0][0] < coords[0][1] <= coords[1][0] < coords[1][1] <= len(entry.text):
            raise ValueError('IIRC did not supply a disjoint later support')
        converted = []
        for start, end in coords:
            span = entry.overlap_window([start, end])
            if entry.text[entry.ends[span[1] - 1]:end].strip():
                raise ValueError('Native tokenizer would omit a non-space label suffix')
            converted.append(span)
        anchors[name], windows[name] = converted
    start = max(span[1] for span in anchors.values())
    ordered = sorted(windows, key=lambda name: windows[name][0])
    if (any(span[0] < start for span in windows.values())
            or (len(ordered) == 2 and windows[ordered[0]][1] > windows[ordered[1]][0])):
        raise ValueError('Native tokenizer has no disjoint anchor/support ordering')
    return {'risk_start': start, 'anchor_tokens': anchors}, {
        'window_tokens': windows, 'earlier_query': ordered[0],
        'later_query': ordered[1] if len(ordered) == 2 else None, 'sham_tokens': None,
    }


def shortlist(texts, labels, unique, pairs, base_windows, tokenizers):
    by_paragraph = defaultdict(list)
    skipped = Counter()
    for label in labels:
        if unique[label['anchor']['anchor_id']] == 1:
            by_paragraph[label['paragraph_id']].append(label)
        else:
            skipped['ambiguous_question_at_anchor'] += 1
    eligible = []
    for key, paragraph_labels in by_paragraph.items():
        text = texts[key]
        windows = set(twelve_word_windows(text))
        if not windows or windows & base_windows:
            skipped['overlap_or_too_short'] += 1
            continue
        encoded = {name: EncodedText.encode(text, tokenizer) for name, tokenizer in tokenizers.items()}
        if max(len(entry.ids) for entry in encoded.values()) > MAX_DOCUMENT_TOKENS:
            skipped['over_max_native_tokens'] += 1
            continue
        label_index = {row['question_id']: row for row in paragraph_labels}
        matching_pairs = sorted(pair for pair in pairs if set(pair) <= label_index.keys())
        alternatives = [tuple(label_index[name] for name in pair) for pair in matching_pairs]
        alternatives.extend((label_index[name],) for name in sorted(label_index))
        for group in alternatives:
            try:
                projected = {name: geometry(entry, group) for name, entry in encoded.items()}
            except ValueError:
                skipped['nonaligning_anchor_or_evidence_option'] += 1
                continue
            family = group[0]['article_family']
            if any(label['article_family'] != family for label in group):
                raise ValueError('Source article family differs between labels in the same paragraph')
            stable = sha256(f'{SELECTION_SALT}:{family}:{key}'.encode()).hexdigest()
            eligible.append((len(group) == 1, stable, family, key, group, windows, encoded, projected))
            break
    return sorted(eligible, key=lambda row: row[:3]), dict(skipped)


def build(base, source, output, candidate, qwen_tokenizer, tokenizer_root, families):
    if not 120 <= families <= 180:
        raise ValueError('Expected a bounded 120–180 family training supplement')
    receipt, stimuli = verified_base(base)
    files = {'qwen': qwen_tokenizer, **{name: tokenizer_root / name / 'tokenizer.json' for name in MODELS[1:]}}
    if {name: digest(path) for name, path in files.items()} != receipt['tokenizer_sha256']:
        raise ValueError('Supplement tokenizers differ from the base portable benchmark')
    tokenizers = {name: Tokenizer.from_file(str(path)) for name, path in files.items()}
    texts, labels, unique, pairs, source_hashes = verified_source(source)
    previous_windows = {window for row in stimuli for window in twelve_word_windows(row['text'])}
    eligible, skipped = shortlist(texts, labels, unique, pairs, previous_windows, tokenizers)
    selected_families, used_windows = set(), set()
    layout = read_json(base / f'{candidate}.json')
    learning = read_json(base / f'{candidate}-learn.json')
    if 'linked' in learning['train'] or 'linked' in learning['dev']:
        raise ValueError('Original training packet already includes linked IIRC labels')
    learning['train']['linked'], learning['dev']['linked'] = [], []
    original_test = {key: value for key, value in layout['rows'].items() if value['split'] == 'test'}
    selection = []
    for singleton, _, family, key, group, windows, encoded, projected in eligible:
        if family in selected_families or windows & used_windows:
            continue
        selected_families.add(family)
        used_windows.update(windows)
        text = texts[key]
        identity = sha256(text.encode()).hexdigest()
        if identity in layout['rows']:
            raise ValueError('Duplicate original document identity after text isolation')
        coordinates, targets = projected[candidate]
        stimuli.append({'id': identity, 'text': text})
        layout['rows'][identity] = {'kind': 'linked', 'split': 'train', 'offsets': encoded[candidate].offsets, **coordinates}
        learning['train']['linked'].append({'id': identity, 'group_id': f'iirc:{family}', **coordinates, **targets})
        selection.append({'id': identity, 'source_article_family': family, 'source_paragraph': key,
                          'strict_pair': not singleton, 'source_questions': sorted(targets['window_tokens']),
                          'native_token_counts': {name: len(entry.ids) for name, entry in encoded.items()}})
        if len(selection) == families:
            break
    if len(selection) != families or used_windows & previous_windows:
        raise ValueError(f'Only {len(selection)} isolated IIRC families for the required {families}')
    final_test = {key: value for key, value in layout['rows'].items() if value['split'] == 'test'}
    if final_test != original_test or len(stimuli) != len(layout['rows']):
        raise ValueError('IIRC training supplement changed a sealed evaluation row')
    report = {'candidate': candidate, 'source': 'IIRC official train, same seed paragraph',
              'source_files_sha256': source_hashes, 'selection_salt': SELECTION_SALT,
              'base_receipt_sha256': digest(base / 'receipt.json'), 'families': families,
              'official_train_later_labels': len(labels), 'eligible_before_family_selection': len(eligible),
              'exclusions': skipped, 'strict_pair_families': sum(row['strict_pair'] for row in selection),
              'labeled_anchors': sum(len(row['source_questions']) for row in selection),
              'exact_twelve_word_windows_overlapping_base': 0, 'unannotated_windows_used_as_sham': 0,
              'meaning': 'A human-selected hyperlink span with subsequent verbatim human-selected main evidence. '
                         'Supported subset only, no observed participant resolution timing or unlabeled negatives.',
              'selection': selection}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.linked-supplement-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'bundle'
        stage.mkdir()
        copyfile(base / 'design.json', stage / 'design.json')
        write_rows(stage / 'stimuli.jsonl', stimuli)
        write_json(stage / f'{candidate}.json', layout)
        write_json(stage / f'{candidate}-learn.json', learning)
        write_json(stage / 'linked-manifest.json', report)
        receipt['documents'] = len(stimuli)
        receipt['split_counts']['linked'] = {'train': families}
        receipt['linked_supplement'] = {'candidate': candidate, 'families': families,
                                        'base_receipt_sha256': report['base_receipt_sha256']}
        receipt['file_sha256'] = {path.name: digest(path) for path in stage.iterdir()}
        write_json(stage / 'receipt.json', receipt)
        if output.exists():
            expected = {path.name: digest(path) for path in stage.iterdir()}
            if expected != {path.name: digest(path) for path in output.iterdir()}:
                raise ValueError('Existing output differs from the rebuilt supplement')
        else:
            stage.rename(output)
    return report


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate', choices=MODELS, required=True)
    parser.add_argument('--qwen-tokenizer', type=Path, default=TOKENIZER)
    parser.add_argument('--tokenizer-root', type=Path, default=TOKENIZER_ROOT)
    parser.add_argument('--families', type=int, default=160)
    result = build(**vars(parser.parse_args()))
    print(f'{result["families"]} isolated IIRC article families, {result["labeled_anchors"]} labeled anchors, '
          f'{result["strict_pair_families"]} strict paired families')
