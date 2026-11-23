"""Mark earliest sufficient answer stops for the sealed authored training branches."""

import argparse
import hashlib
import json
from pathlib import Path

from research.semantic_tracking.same_question_controls import TOKENIZER_SHA256
from research.semantic_tracking.same_question_decisive import (
    INPUTS_SHA256, PAIRS_SHA256, Markup, _locate, _read_pinned_rows,
)
from research.stage2.records import write_jsonl


MARKS = {
    'sqc-5b6081aaa1b78878': Markup('King Street', 'King Street', None,
                                  'The street name completes the waiting location for Saturday passengers.'),
    'sqc-2a268eba6d097fed': Markup('six tonight', 'six tonight', 'six',
                                  'Six gives the hour. Tonight identifies the relevant part of the day.'),
    'sqc-f37763af879e1e09': Markup('the loading gate', 'the loading gate', None,
                                  'The gate completes the entrance that the courier can use.'),
    'sqc-ed9221da0d30b017': Markup('before lunch', 'before lunch', None,
                                  'Lunch completes the departure deadline for the return shipment.'),
    'sqc-fd8beacd23db77a0': Markup('Leah', 'the extra seat', 'was taking',
                                  'Leah appears first. The seat phrase confirms which role she is taking.'),
    'sqc-6ece424d1f7b1c1f': Markup('pumpkin soup', 'pumpkin soup', 'pumpkin',
                                  'Pumpkin suggests the soup type. Soup confirms what the second pot contains.'),
    'sqc-23557a02d566ad00': Markup('the rooftop', 'the rooftop', None,
                                  'The yellow sample and origin relation precede the location value.'),
    'sqc-e3dc19d9d638f2a1': Markup('twelve minutes', 'twelve minutes', None,
                                  'Minutes completes the duration for the stated settling event.'),
    'sqc-5cb88d4a23729fb4': Markup('first thing Wednesday morning', 'Wednesday morning', 'first thing',
                                  'First thing suggests an early start and Wednesday morning fixes the relevant day and period.'),
    'sqc-ebfaa540915d4a3d': Markup('The building caretaker', 'the elevator inspection', 'should log',
                                  'The caretaker is named before the clause links that person to the elevator inspection.'),
    'sqc-9e8a3f8eaed49796': Markup('blocked the path', 'blocked the path', None,
                                  'Obstructing the path explains why the already mentioned cans had to move.'),
    'sqc-7e538c4ce7d74a59': Markup('Friday morning', 'Friday morning', 'Friday',
                                  'Friday gives the day and morning completes the stated arrival window.'),
    'sqc-23a5f99eda065576': Markup('Birch Bend', 'Birch Bend', None,
                                  'The full place name specifies where the new sign will stand.'),
    'sqc-ebbe2814a8bf2042': Markup('next Monday', 'next Monday', None,
                                  'The weekday completes the schedule for the fence repair.'),
    'sqc-b30510855be135d4': Markup('The clay tray', 'to drain', 'took longer',
                                  'The clay tray is named first. To drain identifies the process that took longer.'),
    'sqc-bcaf153fb1bf88c3': Markup('a ceramic plate', 'a ceramic plate', None,
                                  'The plate completes the stated surface for the mass check.'),
    'sqc-2bc5fcf38c2eebd7': Markup('clear glass', 'clear glass', None,
                                  'Glass specifies the material from which the prism was made.'),
    'sqc-ce77093b2e237cd7': Markup('pale blue', 'pale blue', None,
                                  'Blue completes the stated color of the replacement screen.'),
    'sqc-d06811fab2e959ec': Markup('The wall shelves', 'reserved books', 'will hold',
                                  'The answer starts with the shelves. Reserved books confirms the purpose asked about.'),
    'sqc-6d960ba5ac01cf76': Markup('their reader card', 'their reader card', None,
                                  'The shared prefix puts a single visitor credential requirement under discussion. Card completes the credential.'),
    'sqc-64cada1723fccdfe': Markup('The library basement', 'the maps', 'will hold',
                                  'The place is mentioned first and the maps links the storage statement to the target object.'),
    'sqc-420c47418f75922d': Markup('The duty clerk', 'records inquiries', 'will answer',
                                  'The role is stated up front. Records inquiries confirms what the clerk will answer.'),
    'sqc-37ec95090755ecd4': Markup('four that afternoon', 'that afternoon', 'four',
                                  'Four supplies the hour. That afternoon establishes which occurrence of four is meant.'),
    'sqc-d88812498579e3a0': Markup('the small studio', 'the small studio', None,
                                  'The studio completes the room where the group would practice.'),
    'sqc-0653a32ae75aa017': Markup('Foam pads', 'the chamber', 'absorbed sound',
                                  'Absorbed sound supplies the role. The chamber confirms the location asked about.'),
    'sqc-c4f32e6ccbcd71b8': Markup('Two wooden clips', 'the small instrument', 'held',
                                  'The clips are named first. Instrument completes the object being held.'),
    'sqc-996210a76d46eb19': Markup('the west cupboard', 'the west cupboard', None,
                                  'The cupboard completes the storage location for the spare mats.'),
    'sqc-b6be6f69e6db776b': Markup('at seven on Tuesday', 'on Tuesday', 'at seven',
                                  'The cue gives the nominal hour and weekday. It leaves the meridiem unspecified.'),
    'sqc-bc14cd52c89d2df4': Markup('lost its synchronization', 'lost its synchronization', None,
                                  'Loss of synchronization provides the causal explanation for the time-stamp fault.'),
    'sqc-5ac3629b9295f43b': Markup('A loose connector', 'the power supply', 'interrupted',
                                  'A connector is named before the affected supply is identified as the power supply.'),
    'sqc-8ae193fe98dc8149': Markup('The brass quartet', 'the concert', 'would open',
                                  'The group is named first. The concert specifies the event it will open.'),
    'sqc-9ea181a47a597181': Markup('The school volunteers', 'the food vouchers', 'would collect',
                                  'The group is named first. Food vouchers specifies the items they will collect.'),
    'sqc-da0a5d0ec4ba6c00': Markup('A glass coating', 'its gloss', 'gave the tile',
                                  'The coating is linked to gloss at the end. The first tile is the only tile whose gloss is at issue.'),
    'sqc-05af50b28b47f71a': Markup('Water loss', 'shrink', 'the second tile',
                                  'The cause and affected sample appear early. Shrink completes the link to the requested effect.'),
}


def decisive_rows(packet, tokenizer):
    texts = {row['id']: row['text'] for row in _read_pinned_rows(packet / 'inputs.jsonl', INPUTS_SHA256)}
    pairs = _read_pinned_rows(packet / 'pairs.jsonl', PAIRS_SHA256)
    variants = {variant['id']: variant for pair in pairs if pair['split'] == 'train' for variant in pair['variants']}
    if variants.keys() != MARKS.keys():
        raise ValueError('Training annotation must match the complete sealed training inventory')

    rows = []
    for identifier, mark in sorted(MARKS.items()):
        text = texts[identifier]
        start, end = variants[identifier]['cue']
        cue = text[start:end]
        value = _locate(cue, mark.value, start)
        sufficient = _locate(cue, mark.sufficient, start)[1] if mark.sufficient else None
        contextual = _locate(cue, mark.earliest_contextual, start)[1] if mark.earliest_contextual else None
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
        'training_variants': len(records),
        'sufficient_stops': sum(row['first_sufficient_end'] is not None for row in records),
        'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }, indent=2))


if __name__ == '__main__':
    main()
