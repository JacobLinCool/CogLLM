"""Verify the released Abel text-order manipulation and its observation level."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np
import pyreadstat
from scipy.stats import t


FILES = {'ratings.sav': 'e64eace00774c951579b31c612da3f365b115c472a514edaf6033ad0cf358adb',
         'blocked.docx': 'd1bbf84bf650f29ebee8ae918b55d1b31caf46049c35a67faedd2f4a94c2a5e7',
         'interleaved.docx': '4ce8baf3ce70d8770b21933802a7f7ba64b4303d132d6d4c166602bbc297021d'}
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
BODY = {'blocked': (6, (8, 11, 14, 17, 20, 23)), 'interleaved': (3, (5, 8, 11, 14, 17, 20))}
INFERENCE = 'processes_relational_inferences_'
FACTORS = ('sequence', 'prompting', 'interaction')


def words(text):
    return re.findall(r'[^\W_]+', text.casefold())


def materials(path, stem):
    with ZipFile(path) as document:
        root = ElementTree.fromstring(document.read('word/document.xml'))
    text = [''.join(run.text or '' for run in item.findall('.//w:t', NS)) for item in root.findall('./w:body/w:p', NS)]
    intro, heads = BODY[stem]
    note = text[1]
    if not note.casefold().startswith('translated from german') or len(heads) != 6:
        raise ValueError('Expected the authors’ translated and sectioned stimulus')
    blocks = [text[index + 1] for index in heads]
    if any(not text[index].strip() or not block.strip() for index, block in zip(heads, blocks, strict=True)):
        raise ValueError('Section position mapping disagrees with the released document')
    sections = [text[intro], *blocks]
    return {
        'translated_edition': True, 'nonempty_document_paragraphs': sum(bool(x.strip()) for x in text),
        'substantive_section_counts': [len(words(item)) for item in sections],
        'total_body_words_excluding_headings': sum(len(words(item)) for item in sections),
        'section_structure': 'one shared introduction and six headed body sections',
        'same_intro_sha256_for_comparison': sha256(text[intro].encode()).hexdigest(),
    }


def mean_stats(values):
    return {'count': len(values), 'mean': float(np.mean(values)), 'sd': float(np.std(values, ddof=1))}


def summarize(frame):
    ordering, prompts = (frame[key].to_numpy() for key in ('Sequence', 'SelfQuestioning'))
    first, second, total = (frame[INFERENCE + key].to_numpy() for key in ('reading_cycle_1', 'reading_cycle_2', 'sum'))
    if (set(ordering) != {1, 2} or set(prompts) != {1, 2} or len(total) != 114 or not np.array_equal(total, first + second)
            or not np.all(np.isfinite(total) & (total >= 0))):
        raise ValueError('Released subject totals do not match the study’s 2×2 design and reading cycles')
    cells = {}
    for sequence, sequence_name in ((1, 'blocked'), (2, 'interleaved')):
        for prompt, prompt_name in ((1, 'spontaneous_thought_reports'), (2, 'prompted_self_questioning')):
            mask = (ordering == sequence) & (prompts == prompt)
            cells[f'{sequence_name}/{prompt_name}'] = {key: mean_stats(var[mask])
                for key, var in (('first_cycle', first), ('second_cycle', second), ('both_cycles_total', total),
                                 ('count_per_cycle', total / 2))}
    if [cell['both_cycles_total']['count'] for cell in cells.values()] != [28, 28, 28, 30]:
        raise ValueError('Unexpected allocation of retained readers')

    x, z = ordering - 1.5, prompts - 1.5
    design = np.column_stack((np.ones(len(total)), x, z, x * z))
    response = total / 2
    coefs = np.linalg.lstsq(design, response, rcond=None)[0]
    residual = response - design @ coefs
    df = len(total) - design.shape[1]
    covariance = np.linalg.inv(design.T @ design) * (residual @ residual / df)
    critical = t.ppf(0.975, df)
    tests = {}
    for name, contrast in zip((*FACTORS, 'interleaving_in_spontaneous', 'interleaving_in_prompted'),
                              np.array([[0,1,0,0], [0,0,1,0], [0,0,0,1], [0,1,0,-.5], [0,1,0,.5]]), strict=True):
        difference = contrast @ coefs
        se = (contrast @ covariance @ contrast) ** 0.5
        tests[name] = {'difference_per_reading_cycle': float(difference), 'ci95': (difference + np.array([-1, 1]) * critical * se).tolist(),
                       'F': float((difference / se) ** 2), 'df': [1, df]}
    return cells, tests


def run(raw, output):
    for name, expected in FILES.items():
        if sha256((raw / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Source differs from the released edition: {name}')
    frame, metadata = pyreadstat.read_sav(str(raw / 'ratings.sav'))
    columns = metadata.column_names
    if len(columns) != 29:
        raise ValueError('Expected the 29-column original reader-level release')
    text = {stem: materials(raw / f'{stem}.docx', stem) for stem in ('blocked', 'interleaved')}
    if text['blocked']['same_intro_sha256_for_comparison'] != text['interleaved']['same_intro_sha256_for_comparison']:
        raise ValueError('Translated editions disagree on their common introduction')
    cells, tests = summarize(frame)
    report = {'source_sha256': FILES, 'reader_rows': len(frame), 'columns': columns, 'cells': cells,
              'observations': {'reader_by_cycle_inference_counts': True, 'subject_and_paragraph_keys': False,
                               'verbatim_thought_reports': False, 'same_question_updates': False},
              'inductive_inference_2x2': tests, 'published_english_materials': text}
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('raw', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    run(**vars(parser.parse_args()))
