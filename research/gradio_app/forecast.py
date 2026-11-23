"""Export the served ordinal readout for sealed, outcome-free holdout text."""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .combined import CombinedReader


@dataclass(frozen=True)
class Stimuli:
    sha256: str
    documents: tuple[tuple[str, str], ...]


def read_stimuli(path: Path, expected_sha256: str) -> Stimuli:
    contents = path.read_bytes()
    checksum = hashlib.sha256(contents).hexdigest()
    if checksum != expected_sha256:
        raise ValueError(f'Unexpected holdout input checksum: {path}')
    documents = {}
    for line in contents.decode('utf-8').split('\n'):
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) not in ({'id', 'text'}, {'id', 'text', 'body_spans'}):
            raise ValueError('Holdout inputs may only contain ID, source text and body boundaries')
        case, text = row['id'], row['text']
        if not isinstance(case, str) or not case.strip() or case in documents:
            raise ValueError('Holdout inputs must have unique nonempty IDs')
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f'Empty or invalid holdout text: {case}')
        if 'body_spans' in row:
            previous = 0
            spans = row['body_spans']
            if not isinstance(spans, list) or not spans:
                raise ValueError(f'Expected nonempty body boundaries: {case}')
            for span in spans:
                if (not isinstance(span, list) or len(span) != 2 or any(type(x) is not int for x in span)
                        or not previous <= span[0] < span[1] <= len(text) or not text[span[0]:span[1]].strip()):
                    raise ValueError(f'Invalid or overlapping body boundary: {case}')
                previous = span[1]
        documents[case] = text
    if not documents:
        raise ValueError('Expected at least one holdout document')
    return Stimuli(checksum, tuple(documents.items()))


def export_forecasts(reader: 'CombinedReader', stimuli: Stimuli, output: Path) -> dict:
    """Analyze each original string once, using the same boundaries and readout as Gradio."""
    from .causal_plaba import digest
    from .ordinal import LEVELS

    if output.exists():
        raise FileExistsError(output)
    count = 0
    scored = 0
    unavailable = 0

    def record_forward(module, args, result):
        nonlocal count
        count += 1

    hook = reader.backbone.encoder.register_forward_hook(record_forward)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix='.forecast-', dir=output.parent) as temporary:
            staged = Path(temporary) / 'forecast'
            staged.mkdir()
            forecasts = staged / 'predictions.jsonl'
            with forecasts.open('w', encoding='utf-8') as destination:
                for case, text in stimuli.documents:
                    prior = count
                    analysis = reader.analyze(text)
                    if analysis.text != text or count != prior + 1:
                        raise RuntimeError(f'Expected one model forward against the unmodified source: {case}')
                    rows = []
                    for span, token, p in zip(analysis.sentences, analysis.sentence_tokens,
                                              analysis.distributions, strict=True):
                        rows.append({'start': span.start, 'end': span.end, 'token': token,
                                     'ease': sum(k * v for k, v in zip(LEVELS, p.ease, strict=True)),
                                     'understanding': sum(k * v for k, v in zip(LEVELS, p.understanding, strict=True)),
                                     'probability_ease': p.ease, 'probability_understanding': p.understanding})
                    scored += len(rows)
                    unavailable += len(analysis.unavailable_sentences)
                    destination.write(json.dumps({'id': case, 'sentences': rows}, allow_nan=False) + '\n')
            seal = {'candidate_id': reader.checkpoint.identifier, 'input_sha256': stimuli.sha256,
                    'predictions_sha256': digest(forecasts), 'documents': len(stimuli.documents), 'forwards': count,
                    'certified_sentences': scored, 'unavailable_sentences': unavailable}
            (staged / 'seal.json').write_text(json.dumps(seal, indent=2, sort_keys=True) + '\n', encoding='utf-8')
            staged.rename(output)
    finally:
        hook.remove()
    return seal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'checkpoint', 'stimuli', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--input-sha256', required=True)
    args = parser.parse_args()
    stimuli = read_stimuli(args.stimuli, args.input_sha256)

    from .combined import CombinedReader

    reader = CombinedReader(args.model, checkpoint=args.checkpoint, cuda=True)
    print(json.dumps(export_forecasts(reader, stimuli, args.output), sort_keys=True))


if __name__ == '__main__':
    main()
