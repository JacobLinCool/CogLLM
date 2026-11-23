"""Write pre-score answer-value and sufficiency stops for the sealed test branches."""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from research.semantic_tracking.same_question_controls import TOKENIZER_SHA256
from research.stage2.records import write_jsonl


INPUTS_SHA256 = '08497809787124c1ead4297ba7db343ff36bab9e9be1171ec07865a696d3006a'
PAIRS_SHA256 = 'ac2ca9ef7178ffa01f4ebdee3605aefd66d7f23dc87e8db6b5d0b59e77587c84'


@dataclass(frozen=True)
class Markup:
    value: str
    sufficient: str | None
    earliest_contextual: str | None
    reasoning: str


MARKS = {
    'sqc-0dc64cb4535f204f': Markup(
        'His aunt Clara', 'those sketches', 'had drawn',
        'Clara is named first. Drawing suggests the intended action and those sketches links her to the sketchbook.'),
    'sqc-355566088b61f292': Markup(
        'Dublin', 'Dublin', None,
        'The letters and their stated origin jointly specify the place they came from.'),
    'sqc-78c03ef564fb9bde': Markup(
        'The lower switch', 'the bell', 'activated',
        'The answer candidate appears in the subject. Activation suggests the bell question and the object confirms it.'),
    'sqc-bbc595486cd36b24': Markup(
        'steady', 'current', 'steady',
        'Steady matches one offered option once the lamp is named. Current completes the property phrase.'),
    'sqc-e8cf541263b49718': Markup(
        'six', 'glasses', 'boxes',
        'Six gives the number. Boxes makes the requested quantity plausible and glasses identifies the shipment.'),
    'sqc-793ccd8f51de3e55': Markup(
        'the freezer', 'the freezer', None,
        'The crew and repair action are established before the freezer names the machine.'),
    'sqc-c0073351755002f3': Markup(
        'by the pier', 'replacing', 'by the pier',
        'The location identifies a candidate stake. Replacing explicitly links that stake to the requested action.'),
    'sqc-dc420062c5ac20c1': Markup(
        'the eastern shoreline', 'shoreline', 'eastern',
        'Eastern suggests the relevant direction after the omission is established. Shoreline completes the omitted entity.'),
    'sqc-2b6c6f87cd26ddcf': Markup(
        'Milan', 'Milan', None,
        'The crate and its packing are established before Milan supplies the location.'),
    'sqc-04413c1bbd288d65': Markup(
        'oak', 'oak', None,
        'The frame and its material are established before oak supplies the material value.'),
    'sqc-29624c6392aba633': Markup(
        'the upstairs tenant', None, 'the upstairs tenant',
        'The cue explicitly names a recipient by floor. A particular apartment is inferable only if the reader assumes a unique upstairs tenant.'),
    'sqc-3ba4c09eb06186b3': Markup(
        'Mr Lewis', 'Mr Lewis', None,
        'The envelope and its recipient role are established before the surname completes the recipient name.'),
    'sqc-0344e46b91fb60aa': Markup(
        'Maya Bell', 'the narration', 'record',
        'The name precedes the action. Record narrows the role and narration identifies the requested responsibility.'),
    'sqc-df8d76a61caf003a': Markup(
        'beside the empty stone fountain', 'fountain', None,
        'The last shot is the only filming location at issue in the prefix. Fountain completes the location described for filming.'),
}


def _read_pinned_rows(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f'Unexpected sealed packet digest for {path.name}')
    return [json.loads(line) for line in data.decode('utf-8').splitlines()]


def _locate(cue_text, fragment, offset):
    start = cue_text.index(fragment)
    if cue_text.find(fragment, start + 1) != -1:
        raise ValueError(f'A decisive fragment occurs more than once: {fragment!r}')
    return [offset + start, offset + start + len(fragment)]


def decisive_rows(packet, tokenizer):
    texts = {row['id']: row['text'] for row in _read_pinned_rows(packet / 'inputs.jsonl', INPUTS_SHA256)}
    pairs = _read_pinned_rows(packet / 'pairs.jsonl', PAIRS_SHA256)
    test_variants = [variant for family in pairs if family['split'] == 'test' for variant in family['variants']]
    if len(test_variants) != len(MARKS) or {variant['id'] for variant in test_variants} != MARKS.keys():
        raise ValueError('Decisive annotation must match the complete sealed test inventory')

    rows = []
    for variant in sorted(test_variants, key=lambda row: row['id']):
        identifier = variant['id']
        text = texts[identifier]
        start, end = variant['cue']
        cue_text = text[start:end]
        mark = MARKS[identifier]
        value = _locate(cue_text, mark.value, start)
        sufficient = _locate(cue_text, mark.sufficient, start)[1] if mark.sufficient else None
        contextual = _locate(cue_text, mark.earliest_contextual, start)[1] if mark.earliest_contextual else None
        if sufficient is not None and (sufficient < value[1] or contextual is not None and contextual > sufficient):
            raise ValueError(f'Sufficiency precedes its value or contextual cue for {identifier}')

        full_ids = tokenizer.encode(text, add_special_tokens=False).ids
        for boundary in (value[1], sufficient, contextual):
            if boundary is not None:
                stop_ids = tokenizer.encode(text[:boundary], add_special_tokens=False).ids
                if not stop_ids or full_ids[:len(stop_ids)] != stop_ids:
                    raise ValueError(f'Annotation does not align to a causal token endpoint for {identifier}')

        rows.append({
            'id': identifier,
            'value_span': value,
            'first_sufficient_end': sufficient,
            'contextual_end': contextual,
            'sufficiency_window': [contextual, sufficient] if contextual is not None and sufficient is not None else None,
            'reasoning': mark.reasoning,
        })
    return rows


def main():
    from tokenizers import Tokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if hashlib.sha256(args.tokenizer.read_bytes()).hexdigest() != TOKENIZER_SHA256:
        raise ValueError('Expected the pinned native Qwen3.5-2B tokenizer.json')
    records = decisive_rows(args.packet, Tokenizer.from_file(str(args.tokenizer)))
    write_jsonl(args.output, records)
    print(json.dumps({
        'test_variants': len(records),
        'sufficient_stops': sum(row['first_sufficient_end'] is not None for row in records),
        'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }, indent=2))


if __name__ == '__main__':
    main()
