"""Audit verbatim named difficulties in public PLABA-EVAL Round 1 feedback."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re

from research.semantic_tracking.iirc_prepare import digest
from research.stage3.run import save_json


# Reviewed expressions, keyed by 1-based data row in the pinned annotations CSV.
# A response must explicitly question, report unfamiliarity with, or ask to define
# or clarify the expression. Generic editing comments, factual questions, bare
# terminology without an explicit issue, and whole unlocalized clauses are excluded.
# Candidate expressions can still fail verbatim matching below.
REVIEWED = {
    1: ("physiological state",),
    2: ("metabolomics",),
    4: ("Endogenous", "exogenous"),
    7: ("prospective tests",),
    9: ("Metabolomics", "metabolic profiles"),
    11: ("Endogenous", "exogenous", "gas-phase compounds"),
    14: ("Prospective tests",),
    16: ("metabolomics",),
    18: ("endogenous", "exogenous"),
    22: ("phlebology",),
    24: ("compression garment",),
    25: ("high stiffness", "low stiffness"),
    30: ("long stretch bandages", "compression garments"),
    46: ("Cosmesis",),
    69: ("Moist wound",),
    70: ("cosmesis",),
    76: ("ureteric",),
    77: ("Large distal ureteric stones",),
    78: ("mainstay",),
    81: ("Dissemination",),
    89: ("oropharynx", "nasopharynx", "otorhinolaryngology"),
    126: ("VMAT2 inhibitors",),
    150: ("mmil",),
    151: ("plasma glucose", "mol"),
    153: ("phenotype", "predicted value"),
    186: ("cosmesis",),
    208: ("atopic",),
    217: ("HbA1c", "FPG", "OGTT"),
    224: ("TAWH", "hollow viscera perforation"),
    234: ("pole cortex",),
    246: ("orofacial muscles",),
    258: ("schizoaffective",),
    270: ("remarkable",),
    281: ("inguinal",),
    282: ("bimodal",),
    283: ("inguinal region",),
    286: ("open anterior approach",),
    288: ("epidemiology", "etiology"),
    300: ("orofacial muscles",),
    316: ("status epilepticus",),
    319: ("norepinephrine",),
    321: ("epileptiform activity",),
    366: ("hydatid cyst",),
    367: ("septations", "lower pole cortex"),
    368: ("pericystectomy",),
    387: ("hydatid",),
    388: ("septations",),
    389: ("Laparoscopic",),
    391: ("Schizoaffective disorder", "schizophrenia"),
    393: ("confound the clinical picture",),
    395: ("bilateral", "lower extremity", "edema"),
    397: ("3+ pitting edema",),
    398: ("lower extremity swelling",),
    401: ("diabetic neuropathy",),
    403: ("hollow viscera", "sepsis SOURCE control"),
    405: ("intraoperatively",),
    409: ("hemoglobin A(1c)",),
    410: ("glucose tolerance test",),
    417: ("aneuploidy",),
    424: ("turner syndrome phenotypes",),
    427: ("inguinal region",),
    429: ("spinal—anasthesia",),
    432: ("etiology",),
    434: ("tachycardia",),
    435: ("bupropion",),
    436: ("extubated",),
    437: ("intubated",),
    438: ("status epilepticus", "titrated"),
    439: ("receiving hospital",),
    440: ("QRS", "mEq"),
    442: ("dysrhythmias",),
    444: ("serum lamotrigine", "mcg/mL"),
    447: ("metabolomics",),
    448: ("metabolomics",),
    450: ("metabolomics application",),
    455: ("biomarkers",),
    470: ("metabolomics",),
    471: ("metabolomics", "metabolites"),
    481: ("mol",),
    482: ("mol",),
    484: ("predictive value",),
    520: ("medical expulsive therapy (MET)",),
    521: ("shockwave lithotripsy", "ureteroscopy"),
    537: ("metbolomics",),
    566: ("gene-environment interactions",),
    591: ("intravenous pressure",),
    593: ("ambulatory venous hypertension",),
    595: ("short stretch system",),
    600: ("higher optimal pressure",),
    684: ("dopamine",),
    687: ("FDA", "vesicular monoamine transporter type 2 (VMAT2)"),
    736: ("chromosomal",),
    738: ("aneuploidy", "phenotypes"),
    749: ("systemic (full-body) response to inflammation (infection-fighting)",),
    838: ("hydrocodone",),
    857: ("phenotype",),
    875: ("out-of-hours specialist cardiology",),
    878: ("schizoaffective",),
    902: ("inguinal",),
}

LEXICAL_RUN = re.compile(r"[^\W_]+(?:[’'\-][^\W_]+)*")
SOURCE_FILES = ('annotations.csv', 'documents.csv', 'sentences.csv')
PREPARED_FILES = ('stimuli.jsonl', 'trials.jsonl')


def matches(text, phrase):
    """Exact casefold match at whole-word boundaries, in original Unicode coordinates."""
    folded, starts, ends = [], {}, {}
    position = 0
    for index, character in enumerate(text):
        part = character.casefold()
        starts[position] = index
        folded.append(part)
        position += len(part)
        ends[position] = index + 1
    pattern = r'(?<!\w)' + re.escape(phrase.casefold()) + r'(?!\w)'
    return [(starts[item.start()], ends[item.end()]) for item in re.finditer(pattern, ''.join(folded))
            if item.start() in starts and item.end() in ends]


def reader_groups(trials, documents):
    by_reader = defaultdict(set)
    for trial in trials:
        if trial['participant_id'] is not None:
            by_reader[trial['participant_id']].add(documents[trial['stimulus_id']]['family_id'])
    parents = {document['family_id']: document['family_id'] for document in documents.values()}

    def root(family):
        while family != parents[family]:
            family = parents[family]
        return family

    for families in by_reader.values():
        first, *rest = sorted(families)
        for family in rest:
            parents[root(family)] = root(first)
    return {family: root(family) for family in parents}


def run(prepared, raw, stage2_audit, output):
    config = json.loads(stage2_audit.read_text(encoding='utf-8'))['plaba']
    sources = {name: digest(raw / name) for name in SOURCE_FILES}
    artifacts = {name: digest(prepared / name) for name in PREPARED_FILES}
    if any(checksum != config['source']['files'][name]['sha256'] for name, checksum in sources.items()):
        raise ValueError('Unexpected PLABA release files')
    if any(checksum != config['artifacts'][name]['sha256'] for name, checksum in artifacts.items()):
        raise ValueError('Prepared PLABA files do not match the Stage 2 audit')
    with (raw / 'annotations.csv').open(encoding='utf-8', newline='') as source:
        raw_rows = {idx: row for idx, row in enumerate(csv.DictReader(source), 1)
                    if row['round'] == 'round1' and row['included_in_paper_sentence_analysis'] == 'True'}
    with (prepared / 'stimuli.jsonl').open(encoding='utf-8') as source:
        documents = {row['id']: row for line in source if (row := json.loads(line))}
    with (prepared / 'trials.jsonl').open(encoding='utf-8') as source:
        trials = [json.loads(line) for line in source]
    round_one = [row for row in trials if row['metadata'].get('round') == 'round1'
                 and row['source_ref']['file'] == 'annotations.csv']
    if (len(round_one) != len(raw_rows)
            or {row['source_ref']['row'] for row in round_one} != raw_rows.keys()):
        raise ValueError('Round 1 sentence-rating inventory differs from the author release')

    components = reader_groups(round_one, documents)
    all_round_components = reader_groups(trials, documents)
    observed, accepted, rejected = {}, [], []
    readers, seen = set(), set()
    for trial in round_one:
        idx = trial['source_ref']['row']
        original = raw_rows[idx]
        doc = documents[trial['stimulus_id']]
        segment, = (item for item in doc['segments'] if item['id'] == trial['target_segment_id'])
        sentence = doc['text'][segment['start']:segment['end']]
        feedback = trial['observations']['feedback_text']['value'] or ''
        if (trial['id'] != original['annotation_id'] or feedback != original['feedback_text']
                or sentence != original['annotation_text'] or segment['id'] != original['sentence_uid']
                or doc['id'] != original['doc_id'] or doc['family_id'] != original['base_text_id']
                or trial['metadata']['feedback_schema'] != 'free_text'
                or trial['exposure'] != {'mode': 'full_text', 'visible_end': len(doc['text']), 'task': 'sentence_rating'}):
            raise ValueError(f'Round 1 response or observed sentence changed at source row {idx}')
        if feedback.strip():
            observed[idx] = feedback
        if idx not in REVIEWED:
            continue
        seen.add(idx)
        spans = []
        for phrase in REVIEWED[idx]:
            if not 1 <= len(LEXICAL_RUN.findall(phrase)) <= 8 or not matches(feedback, phrase):
                raise ValueError(f'Adjudicated expression is absent from its feedback at source row {idx}: {phrase}')
            found = matches(sentence, phrase)
            if len(found) != 1:
                rejected.append({'source_row': idx, 'candidate': phrase, 'in_sentence_matches': len(found)})
                continue
            (start, end), = found
            spans.append({'start': segment['start'] + start, 'end': segment['start'] + end,
                          'expression': sentence[start:end]})
        if spans:
            readers.add(trial['participant_id'])
            accepted.append({'source_row': idx, 'document_id': doc['id'], 'sentence_id': segment['id'],
                             'family_id': doc['family_id'], 'variant': doc['metadata']['variant'],
                             'round1_reader_connected_group': components[doc['family_id']], 'spans': spans})
    if seen != REVIEWED.keys():
        raise ValueError('Adjudication does not refer to the pinned Round 1 source')

    unique_spans = {(row['document_id'], span['start'], span['end']) for row in accepted for span in row['spans']}
    families = {row['family_id'] for row in accepted}
    stats = {variant: {'feedback_rows': len(rows),
                       'unique_sentences': len({row['sentence_id'] for row in rows}),
                       'span_mentions': sum(len(row['spans']) for row in rows)}
             for variant in ('orig', 'simp') if (rows := [row for row in accepted if row['variant'] == variant])}
    save_json(output, {
        'source': {'revision': config['source']['revision'], 'release': config['source']['source'],
                   'license': config['source']['license'], 'raw_sha256': sources, 'prepared_sha256': artifacts},
        'scope': 'Explicitly named difficulty in full-document, sentence-linked Round 1 feedback. The named span is '
                 'retrospective; the data do not record when the reader first encountered the difficulty or resolved it.',
        'development_reuse': 'PLABA documents and readers served in project development. This free-text endpoint can '
                             'supply a positive-only diagnostic for a frozen head from a different corpus.',
        'round1_included_sentence_ratings': len(round_one), 'round1_nonempty_feedback_rows': len(observed),
        'round1_question_mark_feedback_rows': sum('?' in feedback for feedback in observed.values()),
        'reviewed_candidate_feedback_rows': len(seen),
        'retained_feedback_rows': len(accepted), 'retained_span_mentions': sum(len(row['spans']) for row in accepted),
        'unique_exact_document_spans': len(unique_spans),
        'unique_sentences': len({row['sentence_id'] for row in accepted}),
        'document_variants': len({row['document_id'] for row in accepted}), 'unique_readers': len(readers - {None}),
        'article_families': len(families), 'round1_reader_connected_groups': len({components[key] for key in families}),
        'all_round_reader_connected_groups': len({all_round_components[key] for key in families}),
        'by_variant': stats, 'feedback_rows_per_family': dict(sorted(Counter(row['family_id'] for row in accepted).items())),
        'rejected_verbatim_candidates': rejected, 'positives': accepted,
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared', 'raw', 'stage2-audit', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    run(**vars(parser.parse_args()))
