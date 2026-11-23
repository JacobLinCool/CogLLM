"""Run the exact Gradio sentence readout once per CLEAR excerpt on Colab L4."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

from research.gradio_app.causal_plaba import digest
from research.gradio_app.combined import CombinedReader, paragraphs
from research.gradio_app.plaba import BASE_REVISION, sentence_spans
from research.gradio_app.verify_overview_http import verify_route


STIMULI = 'stimuli.jsonl'
GEOMETRY = 'stops.jsonl'
PREFIX_TOLERANCE = 0.00002


def read_rows(path):
    with path.open(encoding='utf-8') as stream:
        yield from map(json.loads, stream)


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write('\n')


def content_forecast(reader, article, certificate):
    identifier = article['id']
    text = article['text']
    spans = [(start + left, start + right) for start, end, _ in paragraphs(text)
             for left, right in sentence_spans(text[start:end], reader.segmenter)]
    if ([list(span) for span in spans] != certificate['spans']
            or [end for _, end in spans] != article['stops']):
        raise ValueError(f'CLEAR paragraph or sentence alignment changed: {identifier}')
    encoded = reader.backbone.encode(text)
    ratings = reader.rate(encoded, spans)
    observed = [(row.start, row.end) for row in ratings.sentences]
    if (ratings.text != text or observed != spans or ratings.unavailable_sentences
            or len(encoded.tokens) != certificate['input_tokens']
            or list(ratings.sentence_tokens) != certificate['tokens']):
        raise ValueError(f'CLEAR causal sentence stops changed: {identifier}')
    return {'id': identifier, 'sentences': [{'start': row.start, 'end': row.end, 'token': token, 'ease': row.ease}
            for row, token in zip(ratings.sentences, ratings.sentence_tokens, strict=True)]}


def prefix_control(reader, article, full):
    row = full['sentences'][(len(full['sentences']) - 1) // 2]
    prefix = article['text'][:row['end']]
    if len(prefix) >= len(article['text']):
        raise ValueError('The prefix control must remove later text')
    encoded = reader.backbone.encode(prefix)
    analysis = reader.rate(encoded, [(row['start'], row['end'])])
    difference = abs(analysis.sentences[0].ease - row['ease'])
    if analysis.sentence_tokens != (row['token'],) or difference > PREFIX_TOLERANCE:
        raise ValueError('The CLEAR sentence estimate changed when dropping its unseen continuation')
    return {'id': article['id'], 'end': row['end'], 'token': row['token'], 'absolute_ease_difference': difference}


def run(model, checkpoint, plan_path, output):
    import torch

    plan = json.loads(plan_path.read_bytes())
    directory = plan_path.parent
    articles_path, stops_path = directory / STIMULI, directory / GEOMETRY
    if (digest(articles_path) != plan['stimuli_sha256'] or digest(stops_path) != plan['stops_sha256']
            or digest(Path(__file__)) != plan['source_code_sha256']):
        raise ValueError('CLEAR input files or exporter differ from the sealed plan')
    articles, stops = list(read_rows(articles_path)), list(read_rows(stops_path))
    if len(articles) != plan['documents'] or len(stops) != len(articles) or len(articles) < 3:
        raise ValueError('Unexpected CLEAR document count')
    reader = CombinedReader(model, checkpoint=checkpoint, cuda=True, with_reference=True)
    if reader.checkpoint.identifier != plan['candidate_id']:
        raise ValueError('Unexpected Gradio serving candidate')
    if output.exists():
        raise FileExistsError(output)
    started = monotonic()
    with TemporaryDirectory(prefix='.clear-reading-ease-', dir=output.parent) as tmp:
        stage = Path(tmp) / 'outputs'
        stage.mkdir()
        preview = stage / 'reading-overview.html'
        ui_report = verify_route(reader, 7861, save_markup=preview)
        markup = preview.read_text(encoding='utf-8')
        if '模型估計閱讀容易度' not in markup or '篇內相對色階' not in markup:
            raise ValueError('Gradio did not display the current difficulty legend')
        write_json(stage / 'ui-verification.json', {
            **ui_report, 'candidate_id': reader.checkpoint.identifier, 'app_sha256': plan['app_sha256'],
        })
        controls = {}
        checkpoints = {0, len(articles) // 2, len(articles) - 1}
        count = 0

        def record_forward(*_):
            nonlocal count
            count += 1

        hook = reader.backbone.encoder.register_forward_hook(record_forward)
        forward_started = monotonic()
        completed, sentences = 0, 0
        try:
            with (stage / 'forecasts.jsonl').open('x', encoding='utf-8') as stream:
                for index, (article, certificate) in enumerate(zip(articles, stops, strict=True)):
                    if set(article) != {'id', 'text', 'stops'} or article['id'] != certificate['id']:
                        raise ValueError('Expected only text and preflight coordinates in CLEAR inference')
                    row = content_forecast(reader, article, certificate)
                    if count != index + 1:
                        raise ValueError('Expected one forward per CLEAR excerpt')
                    stream.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')
                    sentences += len(row['sentences'])
                    completed += 1
                    if index in checkpoints:
                        controls[index] = row
                    if completed % 100 == 0:
                        print(f'CLEAR scored {completed}/{len(articles)} excerpts', flush=True)
            runtime = monotonic() - forward_started
            if sentences != plan['certified_sentences'] or completed != len(articles):
                raise ValueError('Unexpected number of certified CLEAR forecasts')
            causality = [prefix_control(reader, articles[index], row) for index, row in controls.items()]
            if count != completed + len(causality):
                raise ValueError('Each causal control must run once')
        finally:
            hook.remove()
        write_json(stage / 'causality.json', {'controls': causality, 'absolute_tolerance': PREFIX_TOLERANCE})
        artifacts = ('forecasts.jsonl', 'causality.json', 'ui-verification.json', 'reading-overview.html')
        seal = {
            **plan, 'model_revision': BASE_REVISION, 'device': torch.cuda.get_device_name(),
            'compute_dtype': 'float32', 'content_forwards': completed, 'causal_control_forwards': len(causality),
            'ui_control_forwards': ui_report['model_forwards'], 'content_forward_and_readout_s': runtime,
            'elapsed_after_model_load_s': monotonic() - started,
            'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved(),
            'file_sha256': {name: digest(stage / name) for name in artifacts},
        }
        write_json(stage / 'seal.json', seal)
        stage.rename(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'checkpoint', 'plan-path', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    run(**vars(parser.parse_args()))
