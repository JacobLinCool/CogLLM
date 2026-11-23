"""Prepare, lock and score the formal Traditional Chinese MECO passage estimand."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unicodedata
from xml.etree import ElementTree
from zipfile import ZipFile

from research.backbone_transfer.extract import (
    MODELS, checked_digest, digest, read_json, stream_rows, write_json, write_rows,
)
from research.union_training.verify_observations import require, text_digest, verified_member


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / 'formal-meco-plan.md'
HOLDOUT = HERE / 'holdout.json'
FAMILY = 'traditional-chinese-meco'
HEAD = 'onestop_rereading_first_read_correct'
OUTCOME = 'ch_t_bi_all_acc.rda'
PASSAGES = range(1, 13)
PAGES = (1, 2)
GLYPHS = {
    # Character, horizontal passage/page, row and column within the printed grid.
    '，': (1, 1, 0, 10), '。': (1, 1, 3, 2), '、': (1, 1, 7, 14),
    '「': (1, 1, 0, 17), '」': (1, 1, 0, 20), '：': (7, 1, 8, 3),
    '；': (12, 2, 1, 3), '（': (12, 1, 2, 9), '）': (12, 1, 2, 21),
}
MAIN = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
READER_COUNT = 60
ITEMS_PER_PASSAGE = 4
HORIZONTAL_PASSAGES = 6
TRIAL_COUNT = READER_COUNT * ITEMS_PER_PASSAGE * HORIZONTAL_PASSAGES
BOOTSTRAP_DRAWS = 6000
BOOTSTRAP_SEED = 1729
EXTRACT_FILES = {'coordinates.jsonl', 'preflight.json', 'causality.json', 'states.npy'}
MECO_SESSIONS = re.compile(r'HK(0[1-9]|[1-5][0-9]|60)([HV])')


def passage_id(passage):
    return f'{FAMILY}:{passage:02d}'


def canonical(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text).casefold() if c.isalnum())


def source_pins():
    family = next(item for item in read_json(HOLDOUT)['families'] if item['id'] == FAMILY)
    require(family['status'] == 'sealed', 'MECO must be a registered reserved family')
    return {Path(item['path']).name: item['sha256'] for item in family['source_pins']}


def workbook_words(path):
    with ZipFile(path) as workbook:
        sheet = ElementTree.fromstring(workbook.read('xl/worksheets/sheet1.xml'))
        strings = [
            ''.join(node.text or '' for node in item.findall('.//m:t', MAIN))
            for item in ElementTree.fromstring(workbook.read('xl/sharedStrings.xml'))
        ]
    rows = sheet.findall('./m:sheetData/m:row', MAIN)
    header = [strings[int(cell.find('m:v', MAIN).text)] for cell in list(rows[0])[:5]]
    require(header == ['Passage', 'Page', 'WordNr', 'SenID', 'Word'], 'MECO WordUnit header changed')
    pages = defaultdict(list)
    word_numbers = Counter()
    for number, row in enumerate(rows[1:2096], 2):
        cells = list(row)[:5]
        require([cell.get('r') for cell in cells] == [f'{column}{number}' for column in 'ABCDE'],
                'MECO stimulus workbook coordinates changed')
        passage, page, word, _, value = [int(cell.find('m:v', MAIN).text) for cell in cells]
        text = strings[value]
        word_numbers[passage] += 1
        require(passage in PASSAGES and page in PAGES and word == word_numbers[passage]
                and cells[-1].get('t') == 's' and bool(text) and all(c.isalnum() for c in text),
                f'Unexpected MECO workbook word in passage {passage}, cell E{number}')
        pages[passage, page].append(text)
    require(set(pages) == {(i, j) for i in PASSAGES for j in PAGES} and sum(word_numbers.values()) == 2095,
            'Incomplete MECO WordUnit stimulus material')
    return pages


def page_glyphs(path):
    from PIL import Image, ImageOps

    with Image.open(path) as original:
        require(original.size == (1920, 1080), 'MECO horizontal picture dimensions changed')
        image = ImageOps.invert(original.convert('L'))
    glyphs = {}
    for row in range(11):
        for column in range(24):
            x, y = 500 + 43 * column, 79 + 80 * row
            tile = image.crop((x, y, x + 43, y + 65))
            if bounds := tile.getbbox():
                glyph = tile.crop(bounds)
                glyphs[row, column] = glyph.size, glyph.tobytes()
    return glyphs


def prepare(source, output):
    pins = source_pins()
    images = {Path(item['path']).name: item['sha256'] for item in
              read_json(verified_member(source, 'pin-receipt.json', pins['pin-receipt.json']))}
    workbook = 'meco-hk-wordinfo.xlsx'
    word_pages = workbook_words(verified_member(source, workbook, pins[workbook]))
    canonical_manifest = 'normalized-stimulus-sha256.json'
    originals = read_json(verified_member(source, canonical_manifest, pins[canonical_manifest]))
    require(originals['stimulus_workbook_sha256'] == pins[workbook], 'Unexpected MECO stimulus manifest basis')
    picture_pages = {}
    image_hashes = {}
    for passage in PASSAGES:
        for page in PAGES:
            name = f'H_{passage:03d}_p{page}.png'
            image_hashes[name] = images[name]
            picture_pages[passage, page] = page_glyphs(verified_member(source, name, images[name]))

    punctuation = {picture_pages[passage, page][row, column]: c
                   for c, (passage, page, row, column) in GLYPHS.items()}
    require(len(punctuation) == len(GLYPHS), 'Distinct punctuation shapes must remain distinguishable')
    glyph_for_char, char_for_glyph = {}, {}
    stimuli, diagnostics = [], []
    for passage, original in zip(PASSAGES, originals['passages'], strict=True):
        pages = []
        for page in PAGES:
            words = ''.join(word_pages[passage, page])
            glyphs = picture_pages[passage, page].values()
            require(sum(glyph not in punctuation for glyph in glyphs) == len(words),
                    f'MECO image and workbook counts differ at passage {passage}, page {page}')
            letters = iter(words)
            result = []
            for glyph in glyphs:
                if glyph in punctuation:
                    result.append(punctuation[glyph])
                    continue
                char = next(letters)
                require(glyph_for_char.setdefault(char, glyph) == glyph
                        and char_for_glyph.setdefault(glyph, char) == char,
                        f'MECO character alignment differs at passage {passage}, page {page}')
                result.append(char)
            pages.append(''.join(result))
        text = ''.join(pages)
        clean = canonical(text)
        require(original == {'passage': passage, 'characters': len(clean), 'sha256': text_digest(clean),
                             'words': sum(len(word_pages[passage, page]) for page in PAGES)},
                f'MECO canonical text differs from the sealed material: passage {passage}')
        stimuli.append({'id': passage_id(passage), 'text': text, 'stops': [len(text)]})
        diagnostics.append({'passage': passage, 'page_character_counts': list(map(len, pages)),
                            'canonical_sha256': original['sha256'], 'input_text_sha256': text_digest(text)})

    output.mkdir(parents=True, exist_ok=False)
    write_rows(output / 'stimuli.jsonl', stimuli)
    write_json(output / 'stimulus-seal.json', {
        'family': FAMILY, 'protocol_sha256': digest(PROTOCOL),
        'formal_holdout_sha256': digest(HOLDOUT), 'stimuli_sha256': digest(output / 'stimuli.jsonl'),
        'source_workbook_sha256': pins[workbook], 'source_image_sha256': image_hashes,
        'reserved_outcome_sha256': pins[OUTCOME], 'passages': diagnostics,
    })


def baseline(packet, packet_receipt, stimuli):
    files = packet_receipt['packet_sha256']
    index = read_json(verified_member(packet, 'index.json', files['index.json']))
    task = next(task for task in index['tasks'] if task['name'] == HEAD)
    require((task['source'], task['likelihood'], task['coordinates'], task['labels'], task['states'])
            == ('onestop-rereading', 'categorical', 'point', [0, 1], ['interpretation']),
            'OneStop training response contract changed')
    outcome_file = task['file']
    target_hashes = {text_digest(canonical(row['text'])) for row in stimuli}
    development_rows = stream_rows(verified_member(packet, 'stimuli.jsonl', files['stimuli.jsonl']))
    require(all(text_digest(canonical(row['text'])) not in target_hashes for row in development_rows),
            'A complete MECO passage overlaps a development or training stimulus')
    wrong = correct = 0
    for row in stream_rows(verified_member(packet, outcome_file, files[outcome_file])):
        require(row['role'] in {0, 1}, 'Unexpected registered OneStop split')
        if row['role'] == 0:
            no, yes = row['outcome']
            require(no == int(no) and yes == int(yes) and min(no, yes) >= 0 and no + yes > 0,
                    'Unexpected OneStop training counts')
            wrong += int(no)
            correct += int(yes)
    require(wrong > 0 and correct > 0, 'Both training response categories must have support')
    return {'study': 'onestop-rereading', 'readout': HEAD, 'correct': correct, 'trials': correct + wrong,
            'probability': correct / (correct + wrong)}


def checked_projection(prepared, projection, stimuli):
    spec = read_json(MODELS)['models']['qwen']
    seal = read_json(projection / 'seal.json')
    require(seal['candidate'] == 'qwen' and seal['revision'] == spec['revision']
            and seal['model_files_sha256'] == spec['sha256'] and seal['gpu'] == 'NVIDIA L4'
            and seal['content_forward_per_document'] == 1
            and seal['states_dtype'] == seal['compute_dtype'] == 'float32'
            and seal['protocol_sha256'] == digest(PROTOCOL)
            and seal['stimuli_sha256'] == digest(prepared / 'stimuli.jsonl')
            and seal['documents'] == seal['certified_stops'] == len(PASSAGES)
            and seal['states_shape'] == [len(PASSAGES), spec['hidden_width']]
            and seal['targets_sha256'] is None and seal['independent_causal_checks'] >= 6
            and set(seal['file_sha256']) == EXTRACT_FILES, 'Projection does not meet the registered one-pass contract')
    for name, fingerprint in seal['file_sha256'].items():
        verified_member(projection, name, fingerprint)
    checks = read_json(projection / 'causality.json')['checks']
    require(len(checks) == seal['independent_causal_checks'] and all(row['passed'] is True for row in checks),
            'Projection failed the registered prefix checks')
    coordinates = [{'id': row['id'], 'stops': row['stops']} for row in stimuli]
    require(list(stream_rows(projection / 'coordinates.jsonl')) == coordinates,
            'Certified model passage ends and grade-free coordinates differ')
    return spec


def checked_candidate(candidate, candidate_sha256, packet, spec):
    seal = read_json(verified_member(candidate, 'seal.json', candidate_sha256))
    receipt_sha = digest(packet / 'receipt.json')
    require(seal['packet_receipt_sha256'] == receipt_sha and seal['formal_holdout_sha256'] == digest(HOLDOUT)
            and seal['encoder_revision'] == spec['revision'] and seal['encoder_gpu'] == 'NVIDIA L4'
            and seal['formal_data_in_worker'] is False and seal['one_forward_per_stimulus'] is True
            and seal['backbone_updates'] == 0, 'MECO requires the verified frozen full-data candidate')
    files = seal['files_sha256']
    fit = read_json(verified_member(candidate, 'fitting.json', files['fitting.json']))
    require(fit.get('stop_reason') == 'development_plateau' and fit['checkpoint_at_budget_edge'] is False
            and fit['history'][-1]['plateau_action'] == 'plateau'
            and fit['continuation_policy_sha256'] == seal['continuation_policy_sha256']
            and fit['device'] == 'cuda' and HEAD in fit['development'],
            'Wait for the predeclared joint development plateau before locking MECO forecasts')
    return verified_member(candidate, 'heads.npz', files['heads.npz']), fit['selected_epoch']


def readout(projection, heads, stimuli):
    import numpy as np
    import torch

    from research.union_training.fit_observations import STATE_WIDTH, load_projection

    with torch.inference_mode():
        cache, lookup = load_projection(projection, torch.device('cpu'))
        require(len(lookup) == len(PASSAGES), 'Unexpected projected facet count')
        with np.load(heads, allow_pickle=False) as archive:
            shapes = {
                'shared.interpretation.weight': (STATE_WIDTH, cache.shape[1]),
                'shared.interpretation.bias': (STATE_WIDTH,),
                f'measurements.{HEAD}.linear.weight': (2, STATE_WIDTH),
                f'measurements.{HEAD}.linear.bias': (2,),
            }
            parameters = []
            for name, shape in shapes.items():
                value = archive[name]
                require(value.shape == shape and value.dtype == np.float32 and np.isfinite(value).all(),
                        f'Invalid parameter {name}')
                parameters.append(torch.from_numpy(value))
        state_weight, state_bias, head_weight, head_bias = parameters
        rows = torch.as_tensor([lookup[row['id'], row['stops'][0]] for row in stimuli])
        interpretation = torch.nn.functional.linear(cache[rows], state_weight, state_bias).tanh()
        logits = torch.nn.functional.linear(interpretation, head_weight, head_bias)
        log_probs = torch.nn.functional.log_softmax(logits.to(torch.float64), dim=-1)
    return [{'id': row['id'], 'log_probabilities': values}
            for row, values in zip(stimuli, log_probs.tolist(), strict=True)]


def lock(prepared, projection, candidate, candidate_sha256, packet, output):
    if output.exists():
        raise FileExistsError(output)
    preparation = read_json(prepared / 'stimulus-seal.json')
    pins = source_pins()
    require(preparation['family'] == FAMILY and preparation['protocol_sha256'] == digest(PROTOCOL)
            and preparation['formal_holdout_sha256'] == digest(HOLDOUT)
            and preparation['reserved_outcome_sha256'] == pins[OUTCOME],
            'Prepared data no longer match the registered MECO contract')
    stimuli_file = verified_member(prepared, 'stimuli.jsonl', preparation['stimuli_sha256'])
    stimuli = list(stream_rows(stimuli_file))
    require([row['id'] for row in stimuli] == [passage_id(passage) for passage in PASSAGES]
            and all(set(row) == {'id', 'text', 'stops'} and row['stops'] == [len(row['text'])]
                    for row in stimuli), 'Required endpoint is the only stop for each of the 12 passages')
    spec = checked_projection(prepared, projection, stimuli)
    heads, epoch = checked_candidate(candidate, candidate_sha256, packet, spec)
    packet_receipt = read_json(packet / 'receipt.json')
    require(packet_receipt['formal_holdout_sha256'] == digest(HOLDOUT)
            and packet_receipt['tokenizer_sha256'] == spec['sha256']['tokenizer.json'],
            'Joint development packet is unrelated to the registered holdout or tokenizer')
    forecasts = readout(projection, heads, stimuli)
    reference = baseline(packet, packet_receipt, stimuli)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.meco-forecast-') as temporary:
        stage = Path(temporary) / 'forecast'
        stage.mkdir()
        write_rows(stage / 'forecasts.jsonl', forecasts)
        write_json(stage / 'forecast-seal.json', {
            'family': FAMILY, 'readout': HEAD, 'epoch': epoch, 'baseline': reference,
            'protocol_sha256': digest(PROTOCOL), 'formal_holdout_sha256': digest(HOLDOUT),
            'analysis_code_sha256': digest(Path(__file__)), 'data_sha256': pins[OUTCOME],
            'candidate_seal_sha256': candidate_sha256, 'candidate_heads_sha256': digest(heads),
            'training_packet_receipt_sha256': digest(packet / 'receipt.json'),
            'projection_seal_sha256': digest(projection / 'seal.json'),
            'stimulus_seal_sha256': digest(prepared / 'stimulus-seal.json'),
            'forecasts_sha256': digest(stage / 'forecasts.jsonl'),
        })
        stage.rename(output)


def outcome_cells(path):
    """Read only horizontal binary grades, keeping crossed clusters in memory."""
    import numpy as np
    import rdata
    from rdata.conversion import DEFAULT_CLASS_MAP

    constructors = {**DEFAULT_CLASS_MAP, 'tbl_df': DEFAULT_CLASS_MAP['data.frame']}
    records = rdata.read_rda(path, constructor_dict=constructors)
    require(set(records) == {'ch_t_bi_all_acc'}, 'Unexpected MECO source objects')
    raw = records['ch_t_bi_all_acc']
    fields = ['Session_Name_', 'cond', 'paragraph', 'number', 'subject', 'QUESTIONNUM', 'ACCURACY']
    require(len(raw) == 2 * TRIAL_COUNT and all(name in raw for name in fields),
            'MECO raw response denominator or schema changed')
    readers, responses = defaultdict(set), defaultdict(dict)
    for session, condition, filename, number, reader, item, result in raw[fields].itertuples(index=False, name=None):
        key = MECO_SESSIONS.fullmatch(session) if isinstance(session, str) else None
        require(key is not None and number in PASSAGES and isinstance(reader, str) and bool(reader),
                'Invalid MECO trial key')
        # The session identifies a reader and orientation. The displayed file confirms the passage number.
        passage, orientation = int(number), key[2]
        require(filename == f'{orientation}_{passage:03d}_p1.png'
                and condition == {'H': 1, 'V': 2}[orientation],
                'MECO assignment condition or displayed passage changed')
        if orientation == 'V':
            continue
        require(item in range(1, ITEMS_PER_PASSAGE + 1) and result in {0, 1},
                'Horizontal MECO trial must have an item key and a binary recorded outcome')
        require(item not in responses[reader, passage], 'Duplicate MECO reader-passage-item trial')
        responses[reader, passage][item] = int(result)
        readers[reader].add(passage)

    require(len(readers) == READER_COUNT and all(len(p) == HORIZONTAL_PASSAGES for p in readers.values())
            and len(responses) == READER_COUNT * HORIZONTAL_PASSAGES
            and all(len(items) == ITEMS_PER_PASSAGE for items in responses.values())
            and {p for _, p in responses} == set(PASSAGES), 'MECO horizontal design or denominator differs')
    cells = np.zeros((READER_COUNT, len(PASSAGES), 2), dtype=np.int64)
    for i, reader in enumerate(sorted(readers)):
        for passage in readers[reader]:
            answers = responses[reader, passage]
            correct = sum(answers.values())
            cells[i, passage - 1] = len(answers) - correct, correct
    require(cells.sum() == TRIAL_COUNT, 'Unexpected number of scored horizontal MECO outcomes')
    return cells


def crossed_interval(delta, trials):
    import numpy as np

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    readers, passages = trials.shape
    by_reader = rng.multinomial(readers, np.full(readers, 1 / readers), size=BOOTSTRAP_DRAWS)
    by_passage = rng.multinomial(passages, np.full(passages, 1 / passages), size=BOOTSTRAP_DRAWS)
    denominator = np.einsum('br,rp,bp->b', by_reader, trials, by_passage, optimize=True)
    require(np.all(denominator > 0), 'Crossed bootstrap produced an empty resample')
    numerator = np.einsum('br,rp,bp->b', by_reader, delta, by_passage, optimize=True)
    return np.quantile(numerator / denominator, [.025, .975], method='linear').tolist()


def metrics(counts, log_probs, baseline_prob):
    import numpy as np

    trials = counts.sum(axis=2)
    marginals = counts.sum(axis=0)
    n = trials.sum()
    reference = np.log([1 - baseline_prob, baseline_prob])
    delta = (counts * (reference - log_probs)).sum(axis=2)
    change = float(delta.sum() / n)
    interval = crossed_interval(delta, trials)
    if interval[1] < 0:
        decision = 'supports_improvement'
    elif interval[0] > 0:
        decision = 'supports_worse_performance'
    else:
        decision = 'difference_inconclusive'
    predicted = np.exp(log_probs[:, 1])
    measured = marginals[:, 1].sum() / n
    return {
        'counts': {'readers': len(counts), 'passages': len(PASSAGES), 'trials': int(n),
                   'correct': int(marginals[:, 1].sum())},
        'estimand': 'Horizontal passage-marginal post-reading correctness',
        'primary': {'metric': 'mean_bernoulli_nll_nats', 'model': float(-(marginals * log_probs).sum() / n),
                    'source_baseline': float(-(marginals * reference).sum() / n),
                    'model_minus_source_baseline': change, 'crossed_bootstrap_95_interval': interval,
                    'bootstrap_draws': BOOTSTRAP_DRAWS, 'bootstrap_seed': BOOTSTRAP_SEED, 'decision': decision},
        'descriptive': {
            'model_brier': float((marginals * np.stack((predicted ** 2, (1 - predicted) ** 2), axis=1)).sum() / n),
            'source_baseline_brier': float(marginals.dot([baseline_prob ** 2, (1 - baseline_prob) ** 2]).sum() / n),
            'model_mean_probability': float(marginals.sum(axis=1).dot(predicted) / n),
            'source_baseline_probability': baseline_prob,
            'human_accuracy': float(measured),
            'model_minus_human_accuracy': float(marginals.sum(axis=1).dot(predicted) / n - measured),
            'source_baseline_minus_human_accuracy': float(baseline_prob - measured),
        },
        'passages': [{'id': passage_id(passage), 'correct': int(yes), 'trials': int(no + yes),
                      'model_probability': float(prob)}
                     for passage, (no, yes), prob in zip(PASSAGES, marginals, predicted, strict=True)],
    }


def score(source, forecast, forecast_sha256, output):
    import numpy as np

    if output.exists():
        raise FileExistsError(output)
    pins = source_pins()
    seal = read_json(verified_member(forecast, 'forecast-seal.json', forecast_sha256))
    require(seal['family'] == FAMILY and seal['readout'] == HEAD
            and seal['protocol_sha256'] == digest(PROTOCOL)
            and seal['formal_holdout_sha256'] == digest(HOLDOUT)
            and seal['analysis_code_sha256'] == digest(Path(__file__))
            and seal['data_sha256'] == pins[OUTCOME], 'Forecast seal does not match this MECO assessment')
    predictions = list(stream_rows(verified_member(forecast, 'forecasts.jsonl', seal['forecasts_sha256'])))
    require([row['id'] for row in predictions] == [passage_id(passage) for passage in PASSAGES]
            and all(set(row) == {'id', 'log_probabilities'} for row in predictions),
            'Frozen forecasts do not cover all registered passage IDs')
    log_probs = np.asarray([row['log_probabilities'] for row in predictions], dtype=np.float64)
    require(log_probs.shape == (len(PASSAGES), 2) and np.isfinite(log_probs).all() and np.all(log_probs < 0)
            and np.allclose(np.exp(log_probs).sum(axis=1), 1, atol=1e-12, rtol=0),
            'Frozen forecasts do not contain valid Bernoulli probabilities')
    reference = seal['baseline']
    base = reference['probability']
    require(reference['readout'] == HEAD and reference['study'] == 'onestop-rereading'
            and 0 < base < 1 and base == reference['correct'] / reference['trials'],
            'Invalid frozen source-only baseline')
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = outcome_cells(verified_member(source, OUTCOME, seal['data_sha256']))
    evaluation = metrics(counts, log_probs, base)
    write_json(output, {**evaluation, 'forecast_seal_sha256': forecast_sha256,
                        'protocol_sha256': seal['protocol_sha256'], 'source_sha256': seal['data_sha256']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='phase', required=True)
    preparation = commands.add_parser('prepare')
    freezing = commands.add_parser('lock')
    scoring = commands.add_parser('score')
    for command, paths in ((preparation, ('source', 'output')),
                           (freezing, ('prepared', 'projection', 'candidate', 'packet', 'output')),
                           (scoring, ('source', 'forecast', 'output'))):
        for name in paths:
            command.add_argument(f'--{name}', type=Path, required=True)
    freezing.add_argument('--candidate-sha256', type=checked_digest, required=True)
    scoring.add_argument('--forecast-sha256', type=checked_digest, required=True)
    for command, action in [(preparation, prepare), (freezing, lock), (scoring, score)]:
        command.set_defaults(action=action)
    args = vars(parser.parse_args())
    action = args.pop('action')
    args.pop('phase')
    action(**args)


if __name__ == '__main__':
    main()
