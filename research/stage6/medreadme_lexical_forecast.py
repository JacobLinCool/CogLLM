"""Export the exact displayed Gradio word scores for outcome-free holdout sentences."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from research.gradio_app.forecast import read_stimuli
from research.semantic_tracking.iirc_anchor_fit import WORD
from research.semantic_tracking.iirc_prepare import digest
from research.stage2.records import write_jsonl
from research.stage3.run import save_json


def forecast(reader, stimuli, output):
    if output.exists():
        raise FileExistsError(output)
    count = 0

    def recorded_forward(*_):
        nonlocal count
        count += 1

    def records():
        for index, (case, text) in enumerate(stimuli.documents, 1):
            prediction = reader.analyze(text)
            available = [(row.start, row.end) for row in prediction.lexical]
            known_words = [match.span() for match in WORD.finditer(text)]
            if (prediction.text != text or count != index
                    or sorted([*available, *prediction.unavailable_words]) != known_words):
                raise ValueError('Expected one forward and a full partition of source word offsets')
            yield {'id': case, 'words': [[row.start, row.end, round(row.difficulty, 2)] for row in prediction.lexical]}

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.medreadme-forecasts-') as temporary:
        stage = Path(temporary) / 'forecasts'
        stage.mkdir()
        path = stage / 'predictions.jsonl'
        hook = reader.backbone.encoder.register_forward_hook(recorded_forward)
        try:
            write_jsonl(path, records())
        finally:
            hook.remove()
        save_json(stage / 'seal.json', {'candidate_id': reader.checkpoint.identifier,
                  'lexical_weights_sha256': reader.lexical_head.head_sha256, 'model_forwards': count,
                  'stimuli_sha256': stimuli.sha256, 'forecasts_sha256': digest(path),
                  'forecast_code_sha256': digest(Path(__file__))})
        stage.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for value in ('model', 'checkpoint', 'stimuli', 'output'):
        parser.add_argument(f'--{value}', type=Path, required=True)
    parser.add_argument('--stimuli-sha256', required=True)
    arguments = parser.parse_args()
    stimuli = read_stimuli(arguments.stimuli, arguments.stimuli_sha256)
    from research.gradio_app.combined import CombinedReader

    forecast(CombinedReader(arguments.model, checkpoint=arguments.checkpoint, cuda=True), stimuli, arguments.output)


if __name__ == '__main__':
    main()
