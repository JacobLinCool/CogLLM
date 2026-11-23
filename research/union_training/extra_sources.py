"""Register more paired and position-aligned observations from the research cache."""

from ast import literal_eval
from collections import defaultdict

from research.backbone_transfer.extract import read_json
from research.stage6.yadav_audit import CONDITIONS, RELEASES, build_stimulus_lookup


BINARY = (0, 1)
FIVE = tuple(range(1, 6))


def index(rows, field='id'):
    lookup = {row[field]: row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError(f'Duplicate keys for {field}')
    return lookup


def student_annotations(union):
    source = 'studentannotations'
    base = 'studentannotations/guide-v1'
    receipt = read_json(union.input(source, f'{base}/preparation-receipt.json'))
    material = index(union.rows(source, f'{base}/model_input/stimuli.jsonl'))
    design = index(union.rows(source, f'{base}/local/trial-design.jsonl'), 'trial')
    outcomes = index(union.rows(source, f'{base}/local/outcomes.jsonl'), 'trial')
    groups = index(union.rows(source, f'{base}/local/sentences.jsonl'))
    if material.keys() != groups.keys() or design.keys() != outcomes.keys():
        raise ValueError('StudentAnnotations stimuli, trial keys or outcomes fail alignment')
    for name in ('model_input/stimuli.jsonl', 'local/trial-design.jsonl', 'local/outcomes.jsonl', 'local/sentences.jsonl'):
        if union.sources[source][f'{base}/{name}'] != receipt['sha256'][name]:
            raise ValueError(f'StudentAnnotations release changed: {name}')
    for trial, context in design.items():
        stimulus = material[context['id']]
        text = stimulus['text']
        if stimulus['stops'] != [len(text)]:
            raise ValueError('StudentAnnotations ratings must align to the shown sentence')
        doc = union.document(source, groups[stimulus['id']]['group'], text)
        union.category('studentannotations_comprehension_difficulty', source, doc, len(text), FIVE,
                       outcomes[trial]['grade'])


def qiao(union):
    source = 'qiao'
    base = 'semantic-tracking/qiao-evidence/prepared-v2'
    receipt = read_json(union.input(source, f'{base}/source-audit.json'))
    material = index(union.rows(source, f'{base}/stimuli.jsonl'))
    layout = index(union.rows(source, f'{base}/layout.jsonl'))
    outcomes = union.rows(source, f'{base}/outcomes.jsonl')
    if material.keys() != layout.keys():
        raise ValueError('Qiao case presentations differ from the registered layout')
    for name in ('stimuli.jsonl', 'layout.jsonl', 'outcomes.jsonl'):
        if union.sources[source][f'{base}/{name}'] != receipt['output_sha256'][name]:
            raise ValueError(f'Qiao release changed: {name}')

    phases = {
        (key, row['stage']): row['stop'] for key, item in layout.items() for row in item['observed_stages']
    }
    observed_fields = ('guilt_rating', 'confidence', 'evidence_likelihood_if_not_guilty',
                       'evidence_likelihood_if_guilty')
    accepted = {'verified_published_evidence_tag', 'inferred_from_published_design'}
    for row in outcomes:
        if row['alignment'] not in accepted:
            union.filters['qiao_conflicting_display_sequence'] += 1
            continue
        key, stop = row['id'], row['stop']
        if phases[key, row['stage']] != stop or stop not in material[key]['stops']:
            raise ValueError('Qiao response is outside its identified evidence exposure')
        doc = union.document(source, layout[key]['case_id'], material[key]['text'])
        for field in observed_fields:
            response = row[field]
            if not 0 <= response <= 100:
                raise ValueError(f'Qiao score outside the published scale: {field}')
            union.scalar(f'qiao_{field}', source, doc, stop, response)
        if row['final_stage']:
            union.category('qiao_final_verdict', source, doc, stop, ('not_guilty', 'guilty'), row['verdict'])


def onestop_rereading(union):
    from research.union_training.prepare import normal_words

    source = 'onestop-rereading'
    base = 'onestop-rereading/assay-v1'
    receipt = read_json(union.input(source, f'{base}/assay.json'))
    material = index(union.rows(source, f'{base}/stimuli.jsonl'))
    outcomes = []
    for split in ('development', 'holdout'):
        outcomes.extend(union.rows(source, f'{base}/{split}.jsonl'))
    if set(row['id'] for row in outcomes) != material.keys():
        raise ValueError('OneStop rereading observations do not cover the prepared material')
    for name in ('stimuli.jsonl', 'development.jsonl', 'holdout.jsonl'):
        if union.sources[source][f'{base}/{name}'] != receipt['file_sha256'][name]:
            raise ValueError(f'OneStop rereading release changed: {name}')

    qa_by_text = defaultdict(set)
    qa_stimuli = union.rows('onestop', 'stage2/onestop/stimuli.jsonl')
    qa_articles = {row['metadata']['source_article_number'] for row in qa_stimuli}
    for row in qa_stimuli:
        qa_by_text[tuple(normal_words(row['text']))].add(row['metadata']['source_article_number'])
    linked_articles = set()
    family_by_doc = {}
    for row in outcomes:
        article = row['article']
        identity = row['id']
        if family_by_doc.setdefault(identity, article) != article:
            raise ValueError('OneStop reread text shares inconsistent source identifiers')
        if row['stop'] not in material[identity]['stops'] or row['repeat'] not in BINARY or row['correct'] not in BINARY:
            raise ValueError('Invalid OneStop rereading observation')
    for identity, article in family_by_doc.items():
        text = material[identity]['text']
        family = f'article-{article}'
        material[identity]['key'] = union.document(source, family, text)
        # Compare actual body paragraphs and source article numbers before
        # linking the whole-article and QA-with-options versions.
        matched = {
            number for paragraph in text.split('\n\n')[1:]
            for number in qa_by_text.get(tuple(normal_words(paragraph)), ())
        }
        if matched and matched != {article}:
            raise ValueError(f'Conflicting OneStop same-material match: article {article}')
        if matched:
            linked_articles.add(article)
        if article in qa_articles:
            union.unite(f'onestop:onestop-article-{article}', f'{source}:{family}')
    if linked_articles != qa_articles:
        raise ValueError('OneStop QA/rereading family linkage lacks text verification')

    for row in outcomes:
        condition = 'reread' if row['repeat'] else 'first_read'
        union.category(f'onestop_rereading_{condition}_correct', source, material[row['id']]['key'], row['stop'],
                       BINARY, row['correct'])


def yadav(union):
    import rdata

    source = 'yadav'
    base = 'yadav/raw'
    for experiment, trials, stimuli, rsvp in (
        ('full_sentence', 'experiment1-slider.rda', 'experiment1-stimuli.txt', False),
        ('paced_words', 'experiment2-all-slider.rda', 'experiment2-stimuli.txt', True),
    ):
        data_path = union.input(source, f'{base}/{trials}')
        material_path = union.input(source, f'{base}/{stimuli}')
        for name in (trials, stimuli):
            if union.sources[source][f'{base}/{name}'] != RELEASES[name][1]:
                raise ValueError(f'Yadav release changed: {name}')
        table = rdata.read_rda(data_path)['dat.slider']
        lookup = build_stimulus_lookup(material_path)
        observations = table.loc[table['condition'] != 'F']
        items = defaultdict(dict)
        for row in observations.itertuples(index=False):
            language, item = row.Language, int(row.item)
            if CONDITIONS[row.condition] != (row.Level, row.Paraphrase) or not 1 <= row.response <= 9:
                raise ValueError('Yadav timing condition or response outside its original design')
            original = lookup[language, item, row.Level, row.Paraphrase]
            shown = ' '.join(literal_eval(row.stimulus)) if rsvp else row.stimulus
            if shown.rstrip() != original.rstrip():
                raise ValueError('Yadav stimulus and observation differ')
            text = original.replace('_', ' ').rstrip() if rsvp else original.rstrip()
            doc = union.document(source, f'item-{item}', text)
            task = f'yadav_{experiment}_difficulty'
            point = union.scalar(task, source, doc, len(text), (float(row.response) - 1) * 12.5)
            items[language, item, row.Level][row.Paraphrase] = point
        for (language, _, level), versions in items.items():
            if versions.keys() == {'No', 'Yes'} and versions['No'] != versions['Yes']:
                union.contrast(task, f'{language.lower()}_{level.lower().replace(" ", "_")}_embedded_minus_paraphrase',
                               [versions['Yes']], [versions['No']], 100.0)


def turkish_preferences(union):
    from research.stage6 import turkish_sentence_prepare as release

    source = 'turkish'
    base = 'turkish-v5/assay-v1'
    receipt = read_json(union.input(source, f'{base}/assay.json'))
    material = index(union.rows(source, f'{base}/model_input/stimuli.jsonl'))
    metadata = index(union.rows(source, f'{base}/sentence-metadata.jsonl'))
    if material.keys() != metadata.keys():
        raise ValueError('Turkish comparison material and components differ')
    for basename in ('model_input/stimuli.jsonl', 'sentence-metadata.jsonl'):
        if union.sources[source][f'{base}/{basename}'] != receipt['file_sha256'][basename]:
            raise ValueError(f'Turkish material release changed: {basename}')
    for part in ('fit', 'test'):
        comparisons = index(union.rows(source, f'{base}/{part}.jsonl'))
        outcomes = index(union.rows(source, f'{base}/{part}-truth.jsonl'))
        if comparisons.keys() != outcomes.keys():
            raise ValueError('Turkish same-length pair outcomes cannot be joined')
        for basename in (f'{part}.jsonl', f'{part}-truth.jsonl'):
            if union.sources[source][f'{base}/{basename}'] != receipt['file_sha256'][basename]:
                raise ValueError(f'Turkish comparison release changed: {basename}')
        for trial, row in comparisons.items():
            earlier, later = row['earlier_id'], row['later_id']
            first, second = metadata[earlier], metadata[later]
            if first['component_id'] != second['component_id'] or first['word_count'] != second['word_count']:
                raise ValueError('Turkish same-length pair crosses connected sentence components')
            if not outcomes[trial]['quality_retained']:
                union.filters['turkish_author_quality_filter'] += 1
                continue
            text = material[earlier]['text'] + '\n\n' + material[later]['text']
            doc = union.document(source, first['component_id'], text)
            union.category('turkish_earlier_sentence_easier', source, doc, len(text), BINARY,
                           outcomes[trial]['earlier_easier'])

    # The release records the actual selected answer, not every option shown.
    # Preserve the authored question and learn the recorded correctness endpoint.
    raw = 'turkish-v5/source'
    workbook = union.input(source, f'{raw}/{release.WORKBOOK}')
    comprehension = union.input(source, f'{raw}/{release.QUESTION_CSV}')
    for basename in (release.WORKBOOK, release.QUESTION_CSV):
        if union.sources[source][f'{raw}/{basename}'] != release.FILES[basename][1]:
            raise ValueError(f'Turkish comprehension source changed: {basename}')
    sentences = release.read_workbook(workbook)
    trials = set()
    used_stimuli = set()
    for row in release.read_csv(comprehension, release.QUESTION_FIELDS):
        sid, shown = row['Cumle_ID'], row['Cumle_Metni']
        question, response = row['Soru_Metni'], row['Dogru_Sik_Mi']
        if (shown != sentences[sid].text or not question.strip() or not row['Secilen_Sik'].strip()
                or response not in {'Doğru', 'Yanlış'}):
            raise ValueError('Turkish comprehension response lacks an author stimulus, question, or graded choice')
        trial = (*release.trial_id(row), sid, row['Soru_Tipi'], question)
        if trial in trials:
            raise ValueError('Duplicate original Turkish comprehension response')
        trials.add(trial)
        used_stimuli.add(sid)
        stimulus_id = release.opaque('sentence', sid, shown)
        if stimulus_id in metadata:
            if material[stimulus_id]['text'] != shown:
                raise ValueError('Turkish pair and comprehension editions differ')
            group = metadata[stimulus_id]['component_id']
        else:
            group = f'comprehension-only-{stimulus_id}'
        text = shown + '\n\n' + question
        doc = union.document(source, group, text)
        union.category('turkish_comprehension_correct', source, doc, len(text), BINARY, response == 'Doğru')
    if (len(trials), len(used_stimuli)) != (22490, 787):
        raise ValueError('Turkish comprehension counts differ from the release inventory')


def add_extra_sources(union):
    for reader in (student_annotations, qiao, onestop_rereading, yadav, turkish_preferences):
        reader(union)
