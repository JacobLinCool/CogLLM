"""Compare frozen causal bridge readouts at the original annotated query stops."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import scipy
from scipy.special import logsumexp

from research.semantic_tracking import bridge_interaction_fit as ridge
from research.semantic_tracking import gum_bridge_rank as gum
from research.stage6.external_projection_prepare import GENTLE_AUDIT, GUM_FIT, read


PROTOCOL = Path(__file__).with_name('bridge-rerank-protocol.md')
EXTRA = ('function_word', 'has_digit', 'following_punctuation', 'preceding_preposition', 'same_paragraph')
PREPOSITIONS = set('about above across after against along among around at before behind below beneath beside between '
                   'beyond by despite down during except for from in into near of off on onto out over past since '
                   'through to toward under until up upon with within without'.split())
FUNCTION_WORDS = PREPOSITIONS | set(
    'a an the and or but nor if because although as than that which who whom whose what when where why how some any '
    'each every all both either neither no not it its i me my he him his she her hers we us our they them their you '
    'your this these those is am are was were be been being has have had do does did can could should would will '
    'might may must shall very also only then there here so too'.split()
)
POS = ('ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN',
       'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X', 'missing')
NOMINAL = ('NOUN', 'PROPN', 'PRON', 'NUM')
SURFACE = gum.FORM + EXTRA
COSINE = SURFACE + ('qwen_cosine',)
INTERACTION = COSINE + tuple(f'qwen_pair_{index}' for index in range(128))
FEATURES = INTERACTION + tuple(f'upos_{tag}' for tag in POS)
WIDTHS = {'surface': len(SURFACE), 'cosine': len(COSINE), 'interaction': len(INTERACTION), 'pos': len(FEATURES)}
METRICS = ('nll_nats', 'top1', 'top5', 'mrr')
COMPARISONS = (('cosine', 'surface'), ('interaction', 'cosine'), ('pos', 'interaction'),
               ('interaction', 'original_cosine'))


def checked_projection(directory, reference):
    seal = directory / 'seal.json'
    expected = read(reference)
    if (gum.digest(seal) != gum.digest(reference) or expected['candidate_id'] != gum.CANDIDATE
            or any(gum.digest(directory / name) != expected['file_sha256'][name]
                   for name in ('coordinates.jsonl', 'vectors.npy'))):
        raise ValueError(f'Unexpected projection files in {directory}')
    return expected, gum.read_projection(directory / 'vectors.npy', directory / 'coordinates.jsonl')


def sidecar(path, article_path, audit):
    if gum.digest(path) != audit['feature_sha256'] or gum.digest(article_path) != audit['articles_sha256']:
        raise ValueError(f'Word annotations do not match the original articles in {article_path}')
    with path.open(encoding='utf-8') as stream:
        return {row['id']: {word['stop']: (word['start'], POS.index(word['upos'])) for word in row['words']}
                for row in map(json.loads, stream)}


def augment(article, target, coordinate, vectors, tags, original):
    body, paragraphs = article['text'], article['paragraphs']
    starts = [left for left, _ in paragraphs]
    onset, stop = target['trigger_span']
    origin = gum.paragraph_for(paragraphs, starts, (onset, stop))
    words = list(gum.WORD.finditer(body, 0, onset))
    eligible = []
    for index, match in enumerate(words):
        paragraph = gum.paragraph_for(paragraphs, starts, match.span())
        if paragraph is not None and match.end() in coordinate:
            eligible.append((index, match, paragraph))
    if len(eligible) != len(original.gold):
        raise ValueError('Augmented readout changed the original candidate inventory')

    design = np.zeros((len(eligible), len(FEATURES)), dtype=np.float32)
    design[:, :len(gum.FORM)] = original.design[:, :len(gum.FORM)]
    design[:, len(SURFACE)] = original.design[:, -1]
    candidates = []
    for row, (index, match, paragraph) in enumerate(eligible):
        word = gum.surface(match.group())
        prior = gum.surface(words[index - 1].group()) if index else ''
        next_char = min(match.end(), onset)
        follows_punctuation = next_char < onset and body[next_char] in ',.;:!?)]}'
        design[row, len(gum.FORM):len(SURFACE)] = (
            word in FUNCTION_WORDS, any(char.isdigit() for char in word), follows_punctuation,
            prior in PREPOSITIONS, paragraph == origin,
        )
        start, pos = tags.get(match.end(), (match.start(), POS.index('missing')))
        if start != match.start():
            raise ValueError(f'Word annotation extends past its exact source span in {article["id"]}')
        design[row, len(INTERACTION) + pos] = 1
        candidates.append(coordinate[match.end()])

    cosine, products = ridge.pair_features(vectors[coordinate[stop]], vectors[candidates])
    if not np.allclose(cosine, original.design[:, -1], atol=2e-6, rtol=0):
        raise ValueError('Augmented readout changed the original causal vector order')
    design[:, len(COSINE):len(INTERACTION)] = products
    return gum.Example(original.document, design, original.gold)


def cases(inputs, labels, word_path, coordinates, vectors, audit, outcomes):
    articles = {row['id']: row for row in gum.read_rows(inputs)}
    word_tags = sidecar(word_path, inputs, audit)
    if (articles.keys() != word_tags.keys() or not articles.keys() <= coordinates.keys()
            or any(not coordinates[key].keys() <= set(row['stops']) for key, row in articles.items())):
        raise ValueError(f'Invalid reading coordinate inventory in {inputs}')
    for target in gum.read_rows(labels):
        key = target['id']
        reason, original = gum.example(articles[key], target, coordinates[key], vectors)
        outcomes[reason] += 1
        if original is not None:
            yield original, augment(articles[key], target, coordinates[key], vectors, word_tags[key], original)


def top_probabilities(scores, gold):
    best = max(scores[gold])
    ahead = np.count_nonzero(scores > best)
    tied = scores == best
    hits = np.count_nonzero(tied & gold)
    tied_count = np.count_nonzero(tied)
    survival, top1, top5, reciprocal = 1.0, 0.0, 0.0, 0.0
    for skipped in range(tied_count - hits + 1):
        chance = survival * hits / (tied_count - skipped)
        rank = ahead + skipped + 1
        if rank == 1:
            top1 += chance
        if rank <= 5:
            top5 += chance
        reciprocal += chance / rank
        survival -= chance
    return top1, top5, reciprocal


def measure(scores, gold):
    return (logsumexp(scores) - logsumexp(scores[gold]), *top_probabilities(scores, gold))


def score(examples, original_params, fitted, prior):
    names = ('original_surface', 'original_cosine', *fitted)
    center, scale = (np.asarray(original_params[key]) for key in ('center', 'scale'))
    old_betas = [np.asarray(original_params['weights'][key]) for key in ('form', 'qwen_form')]
    by_document = defaultdict(list)
    diagnostic = defaultdict(list)
    by_paragraph = defaultdict(list)
    nominal = [len(INTERACTION) + POS.index(tag) for tag in NOMINAL]
    filters = Counter()
    for original, row in examples:
        scaled = (original.design - center) / scale
        original_scores = [scaled[:, :len(beta)] @ beta for beta in old_betas]
        scores = [*original_scores, *(ridge.predict(row.design, head) for head in fitted.values())]
        by_document[row.document].append([measure(values, row.gold) for values in scores])

        earlier = row.design[:, SURFACE.index('same_paragraph')] == 0
        if np.any(row.gold & ~earlier):
            raise ValueError('Every positive must be in a strictly earlier paragraph')
        paragraph_nll = []
        for values in scores:
            earlier_log_mass = logsumexp(values[earlier])
            paragraph_nll.append((logsumexp(values) - earlier_log_mass,
                                  earlier_log_mass - logsumexp(values[row.gold])))
        by_paragraph[row.document].append(paragraph_nll)

        allowed = np.any(row.design[:, nominal], axis=1)
        positive_remaining = np.any(row.gold & allowed)
        hit1, hit5, _ = top_probabilities(scores[1][allowed], row.gold[allowed]) if positive_remaining else (0, 0, 0)
        old_first = scores[1] == scores[1].max()
        diagnostic[row.document].append((allowed.mean(), positive_remaining, allowed[old_first].mean(), hit1, hit5))
        filters.update(total=len(allowed), retained=int(allowed.sum()), gold=int(row.gold.sum()),
                       gold_retained=int((row.gold & allowed).sum()), queries_any_gold=int(positive_remaining))

    matrix = np.array([np.mean(by_document[key], axis=0) for key in sorted(by_document)])
    aggregate = dict(zip(names, (dict(zip(METRICS, totals.tolist(), strict=True)) for totals in matrix.mean(axis=0)),
                         strict=True))
    for name, old_name in zip(names[:2], ('form', 'qwen_form'), strict=True):
        for metric, value in prior['metrics'][old_name].items():
            if not np.isclose(aggregate[name][metric], value, atol=5e-12, rtol=0):
                raise ValueError(f'Failed exact original score replay for {name}/{metric}')

    article_draws = np.random.default_rng(735972).integers(0, len(matrix), size=(10_000, len(matrix)))
    contrasts = {}
    for challenge, reference in COMPARISONS:
        diff = matrix[:, names.index(challenge)] - matrix[:, names.index(reference)]
        intervals = np.quantile(diff[article_draws].mean(axis=1), [0.025, 0.975], axis=0)
        contrasts[f'{challenge}_minus_{reference}'] = {
            metric: {'difference': float(diff[:, index].mean()), 'ci95': intervals[:, index].tolist()}
            for index, metric in enumerate(METRICS)
        }
    averages = np.mean([np.mean(rows, axis=0) for rows in diagnostic.values()], axis=0)
    paragraph_averages = np.mean([np.mean(rows, axis=0) for rows in by_paragraph.values()], axis=0)
    if not np.allclose(paragraph_averages.sum(axis=1), matrix.mean(axis=0)[:, 0], atol=5e-12, rtol=0):
        raise ValueError('The probability decomposition did not recover the full candidate NLL')
    return {'articles': len(matrix), 'queries': sum(map(len, by_document.values())), 'metrics': aggregate,
            'contrasts': contrasts, 'paragraph_nll_diagnostic': {
                name: dict(zip(('earlier_paragraph_mass_nll', 'within_earlier_paragraphs_nll'),
                               values.tolist(), strict=True))
                for name, values in zip(names, paragraph_averages, strict=True)
            }, 'nominal_only_diagnostic': {
                'counts': dict(filters), 'document_weighted': dict(zip(
                    ('candidate_retention', 'queries_with_gold_retained', 'original_top_choice_nominal', 'top1', 'top5'),
                    averages.tolist(), strict=True))}}


def save(path, value):
    with path.open('x', encoding='utf-8') as output:
        output.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def fit_and_score(args, output):
    gum_seal, gum_states = checked_projection(args.gum_extraction, GUM_FIT.with_name('extraction.json'))
    gentle_result = GUM_FIT.parent.parent / 'external-projection-l4'
    gentle_seal, gentle_states = checked_projection(args.gentle_extraction, gentle_result / 'extraction.json')
    packet_path = args.gentle_packet / 'assay.json'
    packet = read(packet_path)
    if (gum_seal['projection_sha256'] != gentle_seal['projection_sha256']
            or gum.digest(packet_path) != gentle_seal['assay_sha256']
            or gum.digest(GENTLE_AUDIT) != packet['gentle_audit_sha256']
            or gum.digest(GUM_FIT) != packet['gum_fit_sha256']):
        raise ValueError('Original studies used different models, projection or material')
    old_fit = read(GUM_FIT)
    old_scores = read(GUM_FIT.with_name('scores.json'))['splits']
    old_scores['gentle'] = read(gentle_result / 'gentle.json')
    word_audit = read(args.word_features / 'audit.json')
    inventories = {}
    provenance = {'protocol': gum.digest(PROTOCOL), 'scorer': gum.digest(Path(__file__)),
                  'fitter': gum.digest(Path(ridge.__file__)), 'original_fit': gum.digest(GUM_FIT),
                  'candidate_id': gum_seal['candidate_id'], 'projection': gum_seal['projection_sha256'],
                  'gum_extraction': gum.digest(args.gum_extraction / 'seal.json'),
                  'gentle_extraction': gum.digest(args.gentle_extraction / 'seal.json'),
                  'word_audit': gum.digest(args.word_features / 'audit.json'),
                  'numpy': np.__version__, 'scipy': scipy.__version__}

    def split_cases(split):
        if split == 'gentle':
            inputs, labels = (args.gentle / f'{name}.jsonl' for name in ('articles', 'scoring'))
            expected = packet['gentle_material_sha256']
            input_sha, label_sha = expected[inputs.name], expected[labels.name]
            states = gentle_states
        else:
            inputs = args.gum_assay / 'model_input' / f'{split}.jsonl'
            labels = args.gum_assay / 'scoring' / f'{split}.jsonl'
            input_sha = gum_seal['input_sha256'][inputs.name]
            label_sha = old_fit['provenance']['train_labels'] if split == 'train' else old_scores[split]['labels_sha256']
            states = gum_states
        if (gum.digest(inputs), gum.digest(labels)) != (input_sha, label_sha):
            raise ValueError(f'Data changed since the original {split} experiment')
        outcomes = Counter()
        inventories[split] = {'inputs_sha256': input_sha, 'labels_sha256': label_sha, 'annotations': outcomes}
        return cases(inputs, labels, args.word_features / f'{split}.jsonl', *states,
                     word_audit['paths'][split], outcomes)

    training = [row for _, row in split_cases('train')]
    fitted = {name: ridge.fit(training, slice(width)) for name, width in WIDTHS.items()}
    del training
    fit_path = output / 'fit.json'
    save(fit_path, {'methods': fitted, 'features': FEATURES, 'provenance': provenance,
                    'train_inventory': inventories['train']})
    frozen = read(fit_path)['methods']
    report = {}
    for split in ('dev', 'test', 'gentle'):
        result = score(split_cases(split), old_fit['params'], frozen, old_scores[split])
        report[split] = {**result, **inventories[split]}
    main_deltas = [report[split]['contrasts']['interaction_minus_cosine'] for split in ('test', 'gentle')]
    passes = all(delta['nll_nats']['ci95'][1] < 0 and delta['top5']['ci95'][0] > 0 for delta in main_deltas)
    save(output / 'scores.json', {'fit_sha256': gum.digest(fit_path), 'passes_retrospective_comparison': passes,
                                  'evaluations': report})


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    with TemporaryDirectory(dir=args.output.parent, prefix='.bridge-rerank-') as temporary:
        destination = Path(temporary) / 'results'
        destination.mkdir()
        fit_and_score(args, destination)
        destination.rename(args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('gum-assay', 'gum-extraction', 'gentle', 'gentle-packet', 'gentle-extraction',
                   'word-features', 'output'):
        parser.add_argument(f'--{option}', required=True, type=Path)
    run(parser.parse_args())
