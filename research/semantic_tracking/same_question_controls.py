"""Build tokenizer-matched, constructed tests of which question an answer resolves."""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from research.stage2.records import write_jsonl


TOKENIZER_SHA256 = '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42'
SPLIT_SALT = 'same-question-controls-v1'
IDENTITIES = ('A', 'B')


@dataclass(frozen=True)
class Family:
    key: str
    domain: str
    lead: str
    questions: tuple[str, str]
    answers: tuple[str, str]
    tail: str = ''


FAMILIES = (
    Family('museum-crate', 'narrative',
           'A wooden crate and a damaged picture frame arrived in the museum workshop together.',
           ('Where was the wooden crate packed?', 'What was the old frame made of?'),
           ('The wooden crate was packed in Milan.', 'The old frame was made of oak.'),
           'The conservator left both on the padded bench.'),
    Family('street-festival', 'narrative',
           'Before the street festival, Lila checked the sheet for the musicians and the food stalls.',
           ('Which group would open the concert?', 'Who would collect the food vouchers?'),
           ('The brass quartet would open the concert.', 'The school volunteers would collect the food vouchers.'),
           'She folded the sheet and tucked it in her bag.'),
    Family('garden-note', 'narrative',
           'Iris and her uncle left themselves two reminders about their allotment.',
           ('Why did the watering cans have to be moved?', 'When would the new compost arrive?'),
           ('The cans blocked the path to the shed.', 'The new compost would arrive on Friday morning.'),
           'She pinned the reminder beside the garden gate.'),
    Family('attic-inventory', 'narrative',
           'In the attic, Ben found a sketchbook and a bundle of letters with no labels.',
           ('Who had drawn in the sketchbook?', 'From where had the letters been mailed?'),
           ('His aunt Clara had drawn those sketches.', 'The letters had all come from Dublin.'),
           'Ben cleared a space near the window to examine them.'),
    Family('rehearsal-time', 'narrative',
           'The drummer found a note about two changes to Sunday rehearsal.',
           ('At what hour would rehearsal begin?', 'Which room would the group use?'),
           ('The rehearsal would begin at four that afternoon.', 'The group would practice in the small studio.'),
           'She slipped the note into her jacket pocket.'),
    Family('storyboard', 'narrative',
           'A short film needed a different voice for its narrator and a different location for its last shot.',
           ('Who would record the narration?', 'Where would they film the last shot?'),
           ('Maya Bell agreed to record the narration.', 'They would film beside the empty stone fountain.'),
           'The producer added an entry to the shared folder.'),
    Family('dinner-order', 'narrative',
           'Max had an extra place setting and a second pot of soup ready for dinner.',
           ('Which friend was using the extra seat?', 'What kind of soup was in the second pot?'),
           ('Leah was taking the extra seat.', 'The second pot held hot pumpkin soup.'),
           'He moved the plates to the middle of the table.'),
    Family('parcel-next-door', 'narrative',
           'A neighbor asked June to take in a parcel, then handed her an envelope for someone in the building.',
           ('Which apartment was waiting for the parcel?', 'Who was supposed to receive the envelope?'),
           ('The parcel was for the upstairs tenant.', 'The envelope was addressed to Mr Lewis.'),
           'June left both items by the coat rack.'),
    Family('prism-screen', 'experimental_report',
           'The optics team ran one test on the prism and another on the projection screen.',
           ('What material was the prism made from?', 'What color was the replacement screen?'),
           ('The prism was made from clear glass.', 'The replacement screen was a pale blue.'),
           'Both tests were recorded in the same log.'),
    Family('station-clock', 'experimental_report',
           'At the field station, a timing fault and a separate power fault appeared in the log.',
           ('What caused the time stamp to drift?', 'Which component caused the power fault?'),
           ('The station clock had lost its synchronization.', 'A loose connector interrupted the power supply.'),
           'The technician added a line to the maintenance log.'),
    Family('lake-stakes', 'experimental_report',
           'Two researchers adjusted the lake survey after checking the measuring stakes and the shoreline map.',
           ('Which stake needed a replacement?', 'Which side of the lake was missing from the map?'),
           ('The stake by the pier needed replacing.', 'The map omitted the eastern shoreline entirely.'),
           'A complete inventory was scheduled for next week.'),
    Family('tile-finish', 'experimental_report',
           'A tile lab tracked the gloss of its first sample and the size of a second sample.',
           ('What made the first tile appear glossy?', 'Why did the second tile shrink?'),
           ('A glass coating gave the tile its gloss.', 'Water loss made the second tile shrink slightly.'),
           'Both tiles stayed in the same labelled tray.'),
    Family('sound-chamber', 'experimental_report',
           'The acoustics technician compared the chamber walls and the fixture used to hold an instrument.',
           ('What absorbed the sound inside the chamber?', 'How did the fixture hold the instrument?'),
           ('Foam pads absorbed sound inside the chamber.', 'Two wooden clips held the small instrument steady.'),
           'The diagram for the apparatus was ready to file.'),
    Family('dust-trial', 'experimental_report',
           'In the materials test, a yellow sample and a gray sample each had an unexpected residue.',
           ('Where did the yellow sample come from?', 'How long was the gray sample left to settle?'),
           ('The yellow sample came from the rooftop.', 'The gray sample settled for twelve minutes.'),
           'The sample labels were copied into the log.'),
    Family('bell-switch', 'experimental_report',
           'The design team changed one setting for a bell test and another for a lamp test.',
           ('Which switch activated the bell?', 'Did the lamp use a steady or changing current?'),
           ('The lower switch activated the bell.', 'The lamp used a steady current.'),
           'The panel was disconnected when the tests ended.'),
    Family('packed-soil', 'experimental_report',
           'Maia compared drainage across sandy and clay trays before a separate mass check.',
           ('Which tray had slower drainage?', 'What surface was used for the mass check?'),
           ('The clay tray took longer to drain.', 'The mass check used a ceramic plate.'),
           'The remaining instruments were cleared from the bench.'),
    Family('bus-notice', 'operational_notice',
           'The transit office circulated notices for Saturday buses and temporary ticket sales.',
           ('Where should Saturday passengers wait?', 'When will the temporary ticket desk close?'),
           ('Saturday passengers should wait on King Street.', 'The temporary desk closes at six tonight.'),
           'An employee put the notice in the display case.'),
    Family('records-clerk', 'operational_notice',
           'The county is moving its records collection and has opened a new phone queue.',
           ('Where will the historical maps be kept?', 'Who will answer records inquiries this week?'),
           ('The library basement will hold the maps.', 'The duty clerk will answer records inquiries.'),
           'The office posted an update for staff.'),
    Family('courier-route', 'operational_notice',
           'The courier bulletin covers the next delivery to the north warehouse and a return shipment.',
           ('Which entrance can the courier use?', 'When should the return shipment leave?'),
           ('The courier can use the loading gate.', 'The return shipment should leave before lunch.'),
           'A copy of the bulletin went to dispatch.'),
    Family('reading-room', 'operational_notice',
           'The reading room moved its reserved books and revised the process for issuing entry passes.',
           ('Which shelves will hold reserved books?', 'What must visitors show to receive a pass?'),
           ('The wall shelves will hold reserved books.', 'Visitors must show their reader card.'),
           'The desk printed copies of the notice.'),
    Family('canteen-supplies', 'operational_notice',
           'The canteen manager ordered new glasses and scheduled a separate visit from the repair crew.',
           ('How many boxes of glasses are due?', 'Which machine is the crew coming to fix?'),
           ('The delivery includes six small boxes of glasses.', 'The crew is coming to fix the freezer.'),
           'The manager signed the bottom of the form.'),
    Family('sport-hall', 'operational_notice',
           'A sports hall notice covers storage for spare mats and a booking change for the junior team.',
           ('Where should coaches store the spare mats?', 'When does the junior team have practice?'),
           ('The spare mats belong in the west cupboard.', 'The junior team practices at seven on Tuesday.'),
           'The booking sheet was left at reception.'),
    Family('elevator-check', 'operational_notice',
           'The building notice includes one change to delivery hours and one instruction about the elevator check.',
           ('When can deliveries start on Wednesday?', 'Who should log the elevator inspection?'),
           ('Deliveries can resume first thing Wednesday morning.', 'The building caretaker should log the elevator inspection.'),
           'A copy was attached to the notice board.'),
    Family('hillside-post', 'operational_notice',
           'A trail report describes the replacement of a broken sign and a separately scheduled fence repair.',
           ('At which bend will the new sign stand?', 'Which day is the fence repair scheduled?'),
           ('The new sign will stand at Birch Bend.', 'The fence repair is scheduled for next Monday.'),
           'The report is available at the trail entrance.'),
)


def _rank(key):
    return hashlib.sha256(f'{SPLIT_SALT}:{key}'.encode()).digest()


def _prefix_token_count(tokenizer, encoded, text, end):
    prefix_ids = tokenizer.encode(text[:end], add_special_tokens=False).ids
    if encoded.ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError(f'Tokenization changes at character boundary {end}: {text[:end]!r}')
    return len(prefix_ids)


def _build_family(family, split, tokenizer):
    first_question, second_question = family.questions
    first_start = len(family.lead) + 1
    first_end = first_start + len(first_question)
    second_start = first_end + 1
    second_end = second_start + len(second_question)
    prefix = f'{family.lead}\n{first_question}\n{second_question}\n\n'
    inputs, variants, token_geometry = [], [], []
    for identity, answer in zip(IDENTITIES, family.answers, strict=True):
        cue = [len(prefix), len(prefix) + len(answer)]
        text = prefix + answer
        if family.tail:
            text += ' ' + family.tail
        encoded = tokenizer.encode(text, add_special_tokens=False)
        prefix_count, cue_end = [_prefix_token_count(tokenizer, encoded, text, end) for end in cue]
        for end in (first_end, second_end):
            _prefix_token_count(tokenizer, encoded, text, end)
        token_geometry.append((encoded.ids[:prefix_count], cue_end, encoded.ids[cue_end:]))
        identifier = 'sqc-' + hashlib.sha256(text.encode()).hexdigest()[:16]
        inputs.append({'id': identifier, 'text': text})
        variants.append({
            'id': identifier,
            'answer_identity': identity,
            'cue': cue,
            'events': {key: 'resolved' if key == identity else 'unresolved' for key in IDENTITIES},
        })
    if token_geometry[0] != token_geometry[1]:
        lengths = [cue_end - len(before) for before, cue_end, _ in token_geometry]
        raise ValueError(f'Unequal token geometry for {family.key}: {lengths}')
    return inputs, {
        'family_id': family.key,
        'domain': family.domain,
        'source': 'constructed_crossed_answer_control',
        'split': split,
        'shared_prefix_end': len(prefix),
        'anchors': {'A': [first_start, first_end], 'B': [second_start, second_end]},
        'variants': variants,
    }


def build_records(tokenizer):
    """Construct only text inputs and explicitly separate authored scoring metadata."""
    ranks = sorted(family.key for family in FAMILIES)
    if len(set(ranks)) != len(ranks):
        raise ValueError('Duplicate family ID')
    train_count = (7 * len(ranks) + 5) // 10
    training = set(sorted(ranks, key=_rank)[:train_count])

    inputs, pairs = [], []
    for family in FAMILIES:
        split = 'train' if family.key in training else 'test'
        new_inputs, pair = _build_family(family, split, tokenizer)
        inputs.extend(new_inputs)
        pairs.append(pair)
    if len({row['id'] for row in inputs}) != len(inputs):
        raise ValueError('Duplicate authored continuation document')
    return sorted(inputs, key=lambda row: row['id']), sorted(pairs, key=lambda row: row['family_id'])


def write_benchmark(tokenizer_path, output):
    from tokenizers import Tokenizer

    if hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() != TOKENIZER_SHA256:
        raise ValueError('Expected the pinned native Qwen3.5-2B tokenizer.json')
    inputs, pairs = build_records(Tokenizer.from_file(str(tokenizer_path)))
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, records in (('inputs.jsonl', inputs), ('pairs.jsonl', pairs)):
        path = output / name
        write_jsonl(path, records)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        'families': len(pairs),
        'training_families': sum(row['split'] == 'train' for row in pairs),
        'test_families': sum(row['split'] == 'test' for row in pairs),
        'variants': len(inputs),
        'sha256': hashes,
        'tokenizer_sha256': TOKENIZER_SHA256,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_benchmark(args.tokenizer, args.output), indent=2))


if __name__ == '__main__':
    main()
