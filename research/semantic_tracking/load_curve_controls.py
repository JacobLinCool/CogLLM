"""Build held-out, constructed continuation controls with separate scoring metadata."""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from research.stage2.records import write_jsonl


@dataclass(frozen=True)
class Family:
    key: str
    domain: str
    question: str
    setup: str
    opening: str
    opening_phrase: str
    hold: str
    bridge: str
    answer: str
    cue_phrase: str
    unrelated: str
    shift_phrase: str
    unrelated_followup: str


FAMILIES = (
    Family(
        key="dark-stage",
        domain="narrative",
        question="Who left the blue scarf on the chair?",
        setup="Mara was folding the last of the costumes after rehearsal.",
        opening="On a chair by the curtain she found a blue scarf with no name on it.",
        opening_phrase="a blue scarf with no name on it",
        hold="She draped the scarf over the back of the chair and finished folding the costumes.",
        bridge="Mara checked the room for other items left after rehearsal.",
        answer="An entry in the wardrobe list identified the scarf as Inez's.",
        cue_phrase="Inez's",
        unrelated="A flat stretch of coastline is especially vulnerable to rising tides.",
        shift_phrase="stretch of coastline",
        unrelated_followup="Salt water can travel through drainage channels long before it reaches the road.",
    ),
    Family(
        key="paper-lanterns",
        domain="narrative",
        question="Why did the lanterns all go dark at once?",
        setup="From the courtyard, Dev watched a row of paper lanterns light up along the fence.",
        opening="They all went dark at once, although there was hardly any wind.",
        opening_phrase="all went dark at once",
        hold="Dev rested the tray against the wall and listened to the people out in the courtyard.",
        bridge="Dev could still see the indicator on the lighting box at the far end of the fence.",
        answer="Dev's sister reported that the timer had switched the whole string off.",
        cue_phrase="the timer had switched the whole string off",
        unrelated="The oldest map in the archive depicts a ferry crossing no longer in use.",
        shift_phrase="oldest map in the archive",
        unrelated_followup="The cartographer drew the landing as a narrow strip between two islands.",
    ),
    Family(
        key="wedding-train",
        domain="narrative",
        question="Which station should Leo use for the wedding?",
        setup="Leo could take the morning train to either Ashfield or West Cross.",
        opening="His printed invitation gave a street address but omitted the station for the wedding shuttle.",
        opening_phrase="omitted the station for the wedding shuttle",
        hold="He set the invitation on the windowsill beside the timetable, ready to pack both.",
        bridge="His cousin texted to ask what time the first train left.",
        answer="The shuttle operator confirmed that the boarding point was West Cross.",
        cue_phrase="the boarding point was West Cross",
        unrelated="Young pear trees often need support during their first growing season.",
        shift_phrase="Young pear trees",
        unrelated_followup="A loose tie allows the trunk to grow without rubbing against the stake.",
    ),
    Family(
        key="shared-kitchen",
        domain="narrative",
        question="Which bowl should Noah bring upstairs?",
        setup="Noah set a shallow bowl and a deep bowl on the counter in the shared kitchen.",
        opening="His neighbor called down asking for the bowl she had lent him, without describing it.",
        opening_phrase="the bowl she had lent him, without describing it",
        hold="A dish towel lay folded on the counter, so he spread it out beside the two bowls.",
        bridge="His brother offered to carry the bowl upstairs after washing his hands.",
        answer="A photograph from his neighbor showed that the bowl was the shallow one.",
        cue_phrase="the shallow one",
        unrelated="A small satellite was visible just above the horizon before dawn.",
        shift_phrase="small satellite",
        unrelated_followup="Its path crossed the southern sky before it slipped behind a band of clouds.",
    ),
    Family(
        key="reservoir-gauge",
        domain="experimental_report",
        question="Was the overnight drop in the recorded water level real or a gauge error?",
        setup="The morning log showed a drop in the reservoir's water level overnight.",
        opening="Because the automatic gauge had recently been moved, the drop needed independent confirmation.",
        opening_phrase="the drop needed independent confirmation",
        hold="The technician marked the overnight interval in the log and filed the previous day's chart.",
        bridge="A second team compared records from the same overnight interval.",
        answer="A ruler fixed to the reservoir wall showed the same overnight drop.",
        cue_phrase="showed the same overnight drop",
        unrelated="Ticket sales at the museum were highest on the first Sunday of each month.",
        shift_phrase="Ticket sales at the museum",
        unrelated_followup="Most visitors that day used the entrance on the east side of the building.",
    ),
    Family(
        key="east-microphone",
        domain="experimental_report",
        question="Which microphone corresponded to the silent channel?",
        setup="The auditorium test used one microphone at each end of the stage.",
        opening="One channel recorded silence, but the display gave only a channel number.",
        opening_phrase="the display gave only a channel number",
        hold="The reviewer highlighted the silent interval on a printed copy of the recording log.",
        bridge="The technician checked which channel numbers were printed on the recording log.",
        answer="The wiring record assigned that channel to the east microphone, whose cable was unplugged.",
        cue_phrase="the east microphone",
        unrelated="An overnight frost left a thin crust on the empty vegetable beds.",
        shift_phrase="crust on the empty vegetable beds",
        unrelated_followup="By midmorning the beds closest to the wall had begun to thaw.",
    ),
    Family(
        key="glaze-trial",
        domain="experimental_report",
        question="Which change produced the matte finish in the repeat glaze trial?",
        setup="The ceramics team suspected either the drying time or the firing temperature was responsible for a matte finish.",
        opening="They ran a second trial to determine which change actually reproduced the finish.",
        opening_phrase="which change actually reproduced the finish",
        hold="Both repeat-trial tiles were photographed under the same lamp before they were put away.",
        bridge="The comparison chart paired each test result with the setting changed in that test.",
        answer="Only lowering the firing temperature reproduced the matte finish.",
        cue_phrase="lowering the firing temperature",
        unrelated="The new pedestrian bridge contains a deck made from reclaimed beams.",
        shift_phrase="new pedestrian bridge",
        unrelated_followup="The finished deck rests on separate supports at the two banks of the canal.",
    ),
    Family(
        key="scale-calibration",
        domain="experimental_report",
        question="Why did every recorded mass exceed the reference mass by the same amount?",
        setup="Every object on the workshop scale appeared to be exactly four grams too heavy.",
        opening="The team logged the repeated error without identifying what had caused it.",
        opening_phrase="the repeated error without identifying what had caused it",
        hold="The team left the scale on the same workbench for each of the remaining checks.",
        bridge="A calibration check traced what had been included in each measurement.",
        answer="The scale still included the weight of the empty tray in each reading.",
        cue_phrase="included the weight of the empty tray",
        unrelated="In a coastal dialect, the same expression is used for two different winds.",
        shift_phrase="coastal dialect",
        unrelated_followup="Recent recordings preserve both pronunciations in ordinary conversation.",
    ),
    Family(
        key="market-notice",
        domain="operational_notice",
        question="Which entrance will open early for market deliveries?",
        setup="On Saturday the market will have a separate arrival time for suppliers.",
        opening="The earlier notice gave the arrival time but did not name the entrance for deliveries.",
        opening_phrase="did not name the entrance for deliveries",
        hold="Drivers should keep the delivery paperwork with them until a marshal checks it on arrival.",
        bridge="The site plan shows the designated entrance on the edge of the market grounds.",
        answer="Delivery vehicles should enter through the service gate on Alder Street.",
        cue_phrase="the service gate on Alder Street",
        unrelated="The planetarium's spring program includes a session on identifying constellations.",
        shift_phrase="planetarium's spring program",
        unrelated_followup="Visitors can collect a printed star chart outside the main auditorium.",
    ),
    Family(
        key="archive-index",
        domain="operational_notice",
        question="Why can't visitors open the digitized maps through the usual catalog link?",
        setup="The digitized maps remain listed in the library's catalog this week.",
        opening="Several readers reported that the usual link would not open the images.",
        opening_phrase="the usual link would not open the images",
        hold="The call numbers for the printed maps remain the same in this week's catalog.",
        bridge="The library's technical notice explains what happened to the image links.",
        answer="The image server is offline while the library replaces a failed drive.",
        cue_phrase="offline while the library replaces a failed drive",
        unrelated="Late tomatoes are often harvested before their color has fully changed.",
        shift_phrase="Late tomatoes",
        unrelated_followup="A tray on a sunny shelf gives the fruit room to ripen without bruising.",
    ),
    Family(
        key="permit-receipt",
        domain="operational_notice",
        question="What does an applicant have to bring to collect the permit?",
        setup="Vendors who applied online can collect new permits from the town office on Friday.",
        opening="The automatic confirmation did not list the item each applicant must show at collection.",
        opening_phrase="did not list the item each applicant must show",
        hold="The counter opens at nine, and applicants can wait indoors if they arrive early.",
        bridge="The clerk posted a supplementary collection notice beside the entrance to the town office.",
        answer="Bring the original payment receipt to the collection counter.",
        cue_phrase="the original payment receipt",
        unrelated="A glass harmonica uses a series of nested bowls to produce different pitches.",
        shift_phrase="glass harmonica",
        unrelated_followup="A pedal can keep the bowls turning while the player uses both hands.",
    ),
    Family(
        key="bookshop-closure",
        domain="operational_notice",
        question="On which day will the shop stop selling books during the stock count?",
        setup="The bookshop will interrupt sales for one day to finish its annual stock count.",
        opening="The notice on the door left the date of the closure blank.",
        opening_phrase="left the date of the closure blank",
        hold="Online orders will still be accepted through the usual website during the stock count.",
        bridge="The manager posted the date in a separate update for customers.",
        answer="The sales counter will be closed on Thursday.",
        cue_phrase="Thursday",
        unrelated="A train crossing the iron viaduct can be heard from the valley floor.",
        shift_phrase="iron viaduct",
        unrelated_followup="In winter the sound carries across the fields before the train comes into view.",
    ),
)


def _span(sentence, phrase, base):
    start = sentence.index(phrase)
    if sentence.find(phrase, start + 1) != -1:
        raise ValueError(f"Non-unique offset anchor {phrase!r} in {sentence!r}")
    return {"start": base + start, "end": base + start + len(phrase)}


def _assemble(sentences):
    starts, ends = [], []
    text = ""
    for index, sentence in enumerate(sentences):
        if not sentence or sentence != sentence.strip() or "\n" in sentence:
            raise ValueError(f"Expected one authored line per sentence, got {sentence!r}")
        if index:
            text += "\n\n" if index == 3 else " "
        starts.append(len(text))
        text += sentence
        ends.append(len(text))
    return text, starts, ends


def build_records():
    """Return text-only forward inputs alongside an independent scoring inventory."""
    inputs, controls = [], []
    for family_index, family in enumerate(FAMILIES, start=1):
        variants = (
            ("early", (family.answer, family.bridge, family.hold), 2),
            ("abrupt", (family.hold, family.unrelated, family.unrelated_followup), None),
            ("delayed", (family.hold, family.bridge, family.answer), 4),
        )
        family_inputs, family_controls = [], []
        for variant_index, (condition, continuation, cue_index) in enumerate(variants):
            # Rotate output order across families. Record IDs carry no condition names.
            record_id = f"lc-{3 * (family_index - 1) + (variant_index + family_index) % 3 + 1:03d}"
            text, starts, ends = _assemble((family.setup, family.opening, *continuation))
            cue = None if cue_index is None else _span(family.answer, family.cue_phrase, starts[cue_index])
            shift = _span(family.unrelated, family.shift_phrase, starts[3]) if condition == "abrupt" else None
            before_cue = None if cue is None else len(text[:cue["start"]].rstrip())
            family_inputs.append({"id": record_id, "text": text})
            family_controls.append({
                "id": record_id,
                "family_id": family.key,
                "domain": family.domain,
                "condition": condition,
                "question": family.question,
                "source": "constructed_control",
                "offsets": {
                    "opening": _span(family.opening, family.opening_phrase, starts[1]),
                    "common_prefix_end": ends[1],
                    "paragraph_end": ends[2],
                    "continuation_start": starts[3],
                    "before_cue_end": before_cue,
                    "cue": cue,
                    "shift": shift,
                    "sentence_ends": ends,
                    "late_comparison_end": ends[4],
                },
            })

        _validate_family(family_inputs, family_controls)
        inputs.extend(family_inputs)
        controls.extend(family_controls)

    if len({family.key for family in FAMILIES}) != len(FAMILIES):
        raise ValueError("Duplicate family ID")
    return sorted(inputs, key=lambda row: row["id"]), sorted(controls, key=lambda row: row["id"])


def _validate_family(inputs, controls):
    texts = [row["text"] for row in inputs]
    offsets = [row["offsets"] for row in controls]
    early, abrupt, delayed = offsets
    main_prefixes = {text[:item["common_prefix_end"]] for text, item in zip(texts, offsets)}
    if len(main_prefixes) != 1:
        raise ValueError("Comparison family must share an exact common prefix")
    if texts[1][:abrupt["continuation_start"]] != texts[2][:delayed["continuation_start"]]:
        raise ValueError("Unresolved variants must share an exact prefix through the paragraph break")
    for text, item in zip(texts, offsets):
        if text[item["paragraph_end"]:item["continuation_start"]] != "\n\n":
            raise ValueError("Expected the paragraph boundary after sentence 3")
    early_cue = texts[0][early["cue"]["start"]:early["cue"]["end"]]
    delayed_cue = texts[2][delayed["cue"]["start"]:delayed["cue"]["end"]]
    if early_cue != delayed_cue or early_cue in texts[1] or texts[2].count(early_cue) != 1:
        raise ValueError("Expected matched cue wording and no cue in the abrupt variant")
    for item, cue_index in ((early, 2), (delayed, 4)):
        cue = item["cue"]
        preceding, bound = item["sentence_ends"][cue_index - 1:cue_index + 1]
        if not preceding < cue["start"] < cue["end"] <= bound:
            raise ValueError("Cue is outside its precommitted sentence")


def write_benchmark(output):
    inputs, controls = build_records()
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, records in (("inputs.jsonl", inputs), ("controls.jsonl", controls)):
        path = output / name
        write_jsonl(path, records)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"families": len(FAMILIES), "variants": len(inputs), "sha256": hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_benchmark(args.output), indent=2))


if __name__ == "__main__":
    main()
