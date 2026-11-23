"""Score the one publicly available Freeling abstract family with a frozen readout."""

import argparse
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np

from research.gradio_app.causal_plaba import CHECKPOINT, CausalPlabaReader, digest
from research.stage3.run import save_json


ARCHIVE = 'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8430246/supplementaryFiles'
ARCHIVE_SHA256 = 'f3a4e5fce8a3c4344eff4e7bfc8c4d987240fa70cb1a6854b5fadcacdc5c8ef1'
WORD_XML = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
CONDITIONS = ('Hardest', 'Harder', 'Easier', 'Easiest')


def read_texts(archive):
    if digest(archive) != ARCHIVE_SHA256:
        raise ValueError('Expected the verified supplementary archive from Europe PMC')
    with ZipFile(archive) as supplement:
        with ZipFile(BytesIO(supplement.read('Data_Sheet_1.docx'))) as document:
            root = ET.fromstring(document.read('word/document.xml'))
    versions = {}
    for row in root.findall('./w:body/w:tbl[1]/w:tr', WORD_XML):
        cells = row.findall('w:tc', WORD_XML)
        label = ''.join(node.text or '' for node in cells[0].findall('.//w:t', WORD_XML))
        if label in CONDITIONS:
            if label in versions or len(cells) != 2:
                raise ValueError('Unexpected condition rows in Table S1')
            paragraphs = [''.join(node.text or '' for node in paragraph.findall('.//w:t', WORD_XML))
                          for paragraph in cells[1].findall('./w:p', WORD_XML)]
            versions[label] = '\n\n'.join(text for text in paragraphs if text.strip())
    if set(versions) != set(CONDITIONS) or any(not text.strip() for text in versions.values()):
        raise ValueError('Expected the four released versions of the same abstract')
    return versions


def run(model, archive, output):
    texts = read_texts(archive)
    reader = CausalPlabaReader(model)
    estimates = {}
    for condition in CONDITIONS:
        analysis = reader.analyze(texts[condition])
        weights = [len(analysis.text[row.start:row.end].split()) for row in analysis.sentences]
        estimates[condition] = {
            'scorable_sentences': len(weights), 'unavailable_sentences': len(analysis.unavailable_sentences),
            'scored_words': sum(weights), 'article_words': len(analysis.text.split()),
            'word_weighted_mean_processing_ease': float(np.average([row.ease for row in analysis.sentences], weights=weights)),
        }
    values = [estimates[condition]['word_weighted_mean_processing_ease'] for condition in CONDITIONS]
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, {
        'scope': 'Directional frozen-checkpoint diagnostic on one new source family (populist politics). '
                 'The readers rated the whole abstract. The model averages causal sentence-end scores by sentence word count.',
        'publication': 'https://doi.org/10.3389/fpsyg.2021.714321', 'material': 'Supplementary Table S1',
        'human_reference': 'Supplementary Table S2 interaction model: readability contrast per writing-style unit '
                           'for populism is positive (2.73 + 0.20 percentage points by the published coefficients). '
                           'A combined confidence interval cannot be recovered from the marginal intervals.',
        'archive': ARCHIVE, 'archive_sha256': digest(archive), 'checkpoint_sha256': digest(CHECKPOINT / 'head.npz'),
        'models_scored': 1, 'source_families': 1, 'per_version': estimates,
        'easiest_minus_hardest_points': values[-1] - values[0],
        'monotonic_in_manipulated_order': bool(np.all(np.diff(values) >= 0)),
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'archive', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    run(**vars(parser.parse_args()))
