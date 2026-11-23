"""Lock the equal-word-count Turkish sentence preference assay from Zenodo v5."""

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from decimal import Decimal
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
from statistics import median
from tempfile import TemporaryDirectory
import unicodedata
from urllib.request import urlopen
from xml.etree import ElementTree
from zipfile import ZipFile


RELEASE = "https://zenodo.org/records/22775520"
SOURCE = RELEASE + "/files/"
WORKBOOK = "saha_calismasi_veri_seti_800.xlsx"
QUESTION_CSV = "2_soru_yanitlari-11.csv"
PREFERENCE_CSV = "3_karsilastirma_tercihleri-5.csv"
FILES = {
    WORKBOOK: (454453, "333c2089173f94e9e0afde45fc3e3ddcafcbfcb70ffd963537d03f91aa3f2f7b"),
    QUESTION_CSV: (7160912, "bd9f57888ed1cb602b142d1dff11ec8ced4a8440dcc9f713b5fc70a79dbd0c76"),
    PREFERENCE_CSV: (1153590, "9f4d975848b1684e606d5b721608084e5c8ee2568361392289f2739696cd0cec"),
}
QUESTION_FIELDS = {
    "Katilimci_ID", "Egitim_Durumu", "Task_ID", "Pair_ID", "Cumle_ID", "Cumle_Metni", "Soru_Tipi",
    "Soru_Metni", "Secilen_Sik", "Dogru_Sik_Mi", "Cevaplama_Suresi_sn",
}
PREFERENCE_FIELDS = {
    "Katilimci_ID", "Egitim_Durumu", "Task_ID", "Pair_ID", "Cumle_1", "Cumle_2", "Secilen_Kolay_Cumle",
    "Karar_Suresi_sn",
}
NAMESPACES = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
RELATIONSHIP = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
SPLIT_SEED = "cogllm-turkish-v5-equal-word-v1"


@dataclass(frozen=True)
class Sentence:
    text: str
    words: int
    chars: int
    syllables: int


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def opaque(kind, *values):
    payload = json.dumps([SPLIT_SEED, kind, *values], ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return sha256(payload.encode("utf-8")).hexdigest()[:32]


def normalized_text(text):
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def fetch(output):
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".turkish-source-") as directory:
        staged = Path(directory) / "source"
        staged.mkdir()
        for name, (size, expected) in FILES.items():
            with urlopen(f"{SOURCE}{name}?download=1", timeout=30) as stream:
                data = stream.read(size + 1)
            if len(data) != size or sha256(data).hexdigest() != expected:
                raise ValueError(f"Unexpected source snapshot bytes for {name}")
            with (staged / name).open("xb") as file:
                file.write(data)
        staged.rename(output)


def verify_sources(raw):
    for name, (size, expected) in FILES.items():
        path = raw / name
        if path.stat().st_size != size or digest(path) != expected:
            raise ValueError(f"Expected the exact pinned v5 source file: {name}")


def read_workbook(path):
    with ZipFile(path) as book:
        workbook = ElementTree.fromstring(book.read("xl/workbook.xml"))
        sheet, = [item for item in workbook.findall(".//s:sheet", NAMESPACES)
                  if item.attrib["name"] == "selected_sentences"]
        rels = ElementTree.fromstring(book.read("xl/_rels/workbook.xml.rels"))
        target, = [row.attrib["Target"] for row in rels if row.attrib["Id"] == sheet.attrib[RELATIONSHIP]]
        member = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
        if PurePosixPath(member).parent != PurePosixPath("xl/worksheets"):
            raise ValueError("The selected sentence sheet must be a worksheet in the v5 archive")
        data = ElementTree.fromstring(book.read(member))
        shared = []
        if "xl/sharedStrings.xml" in book.namelist():
            root = ElementTree.fromstring(book.read("xl/sharedStrings.xml"))
            shared = ["".join(node.text or "" for node in item.findall(".//s:t", NAMESPACES)) for item in root]

    def cell_value(cell):
        value = cell.findtext("s:v", namespaces=NAMESPACES)
        kind = cell.attrib.get("t")
        if kind == "s":
            return shared[int(value)]
        if kind == "inlineStr":
            return "".join(node.text or "" for node in cell.findall(".//s:t", NAMESPACES))
        return value

    def cells(row):
        return {cell.attrib["r"].rstrip("0123456789"): cell_value(cell) for cell in row}

    source_rows = iter(data.findall(".//s:sheetData/s:row", NAMESPACES))
    header = {name: column for column, name in cells(next(source_rows)).items()}
    required = ("sid", "sentence", "#word", "#char", "#syl")
    if not all(field in header for field in required):
        raise ValueError("The v5 worksheet needs the declared source text and all three surface counts")
    sentences = {}
    for row in source_rows:
        values = cells(row)
        sid, text, words, chars, syllables = (values.get(header[field]) for field in required)
        if type(sid) is not str or not sid.strip() or sid in sentences or type(text) is not str or not text.strip():
            raise ValueError("A sentence row has a blank or duplicate identifier or an empty stimulus")
        counts = tuple(Decimal(item) for item in (words, chars, syllables))
        if any(not count.is_finite() or count <= 0 or count != int(count) for count in counts):
            raise ValueError(f"Invalid official surface counts for sentence {sid}")
        sentences[sid] = Sentence(text, *map(int, counts))
    if len(sentences) != 800:
        raise ValueError("Unexpected number of selected v5 workbook stimuli")
    return sentences


def read_csv(path, fields):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or len(reader.fieldnames) != len(fields) or set(reader.fieldnames) != fields:
            raise ValueError(f"Unexpected v5 file schema: {path.name}")
        for row in reader:
            if row.keys() != fields or any(value is None for value in row.values()):
                raise ValueError(f"Malformed v5 source row in {path.name}")
            yield row


def trial_id(row):
    key = row["Katilimci_ID"], row["Task_ID"], row["Pair_ID"]
    if any(not value.strip() or value != value.strip() for value in key):
        raise ValueError("Blank or padded participant, task, or pair identifier")
    return key


def read_trial_geometry(raw, sentences):
    trials = defaultdict(set)
    text_ids = {}
    for row in read_csv(raw / QUESTION_CSV, QUESTION_FIELDS):
        key = trial_id(row)
        sid = row["Cumle_ID"]
        text = row["Cumle_Metni"]
        if not sid.strip() or not text.strip() or sid not in sentences or sentences[sid].text != text:
            raise ValueError("A comprehension row lacks an exact original stimulus in the workbook")
        if text_ids.setdefault(text, sid) != sid:
            raise ValueError("One literal sentence resolves to multiple source IDs")
        trials[key].add(sid)
    if len(text_ids) != 787 or len(trials) != 2960 or any(len(ids) != 2 for ids in trials.values()):
        raise ValueError("The v5 question tables disagree with the released stimulus and pair inventory")
    return trials, text_ids


def read_earlier_later(raw, sentences, trials, text_ids):
    earlier_later, graph, degree = {}, defaultdict(set), Counter()
    administered = set()
    for row in read_csv(raw / PREFERENCE_CSV, PREFERENCE_FIELDS):
        key = trial_id(row)
        earlier = text_ids.get(row["Cumle_1"])
        later = text_ids.get(row["Cumle_2"])
        if key in administered or earlier is None or later is None or earlier == later or {earlier, later} != trials[key]:
            raise ValueError("A preference row does not match the two exact stimuli of its unique trial")
        administered.add(key)
        if sentences[earlier].words != sentences[later].words:
            continue
        earlier_later[key] = earlier, later
        graph[earlier].add(later)
        graph[later].add(earlier)
        degree.update((earlier, later))
    if administered != trials.keys() or len(earlier_later) != 2844 or len(graph) != 779:
        raise ValueError("Unexpected v5 equal-word-count cohort")
    if sum(map(len, graph.values())) // 2 != 1589:
        raise ValueError("Unexpected count of unique strict equal-word-count sentence pairs")
    return earlier_later, graph, degree


def connected_components(graph):
    components = {}
    remaining = set(graph)
    while remaining:
        seed = min(remaining)
        stack = [seed]
        members = {seed}
        remaining.remove(seed)
        while stack:
            for neighbor in graph[stack.pop()] & remaining:
                stack.append(neighbor)
                members.add(neighbor)
                remaining.remove(neighbor)
        key = opaque("component", *sorted(members))
        components[key] = members
    return components


def split_components(components, degree):
    """Minimize the count imbalance over cyclic windows of deterministic hash order."""
    ordered = sorted(components)
    nodes = [len(components[key]) for key in ordered]
    rows = [sum(degree[sid] for sid in components[key]) // 2 for key in ordered]
    n_nodes, n_rows = sum(nodes), sum(rows)
    best = None
    for start in range(len(ordered)):
        test_nodes, test_rows = 0, 0
        for width in range(1, len(ordered)):
            end = (start + width - 1) % len(ordered)
            test_nodes += nodes[end]
            test_rows += rows[end]
            distance = abs(4 * test_nodes - n_nodes) * n_rows + abs(4 * test_rows - n_rows) * n_nodes
            candidate = distance, abs(4 * width - len(ordered)), start, width
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise ValueError("Separate sentence sets require at least two connected components")
    _, _, start, width = best
    test = {ordered[(start + i) % len(ordered)] for i in range(width)}
    return test


def author_quality_flags(raw):
    responses = defaultdict(list)
    for row in read_csv(raw / QUESTION_CSV, QUESTION_FIELDS):
        participant, _, _ = trial_id(row)
        label = row["Dogru_Sik_Mi"]
        duration = Decimal(row["Cevaplama_Suresi_sn"])
        if label not in {"Doğru", "Yanlış"} or not duration.is_finite() or duration < 0:
            raise ValueError("Invalid field in the v5 quality-filter source")
        responses[participant].append((duration, label == "Doğru"))
    flags = {
        participant: median(rt for rt, _ in trial_rows) >= 5
        and 10 * sum(correct for _, correct in trial_rows) > 3 * len(trial_rows)
        for participant, trial_rows in responses.items()
    }
    if len(flags) != 1480 or sum(flags.values()) != 1133 or sum(map(len, responses.values())) != 22490:
        raise ValueError("The v5 author quality filter does not reproduce the published denominator")
    return flags


def read_choices(raw, earlier_later, quality):
    choices = {}
    total_retained = 0
    for row in read_csv(raw / PREFERENCE_CSV, PREFERENCE_FIELDS):
        key = trial_id(row)
        participant = key[0]
        earlier, later = row["Cumle_1"], row["Cumle_2"]
        choice = row["Secilen_Kolay_Cumle"]
        if choice not in {earlier, later} or participant not in quality:
            raise ValueError("A released preference is undefined or references neither displayed sentence")
        total_retained += quality[participant]
        if key in earlier_later:
            choices[key] = choice == earlier
    if choices.keys() != earlier_later.keys() or total_retained != 2266:
        raise ValueError("The v5 preference and published quality-filter inventories disagree")
    return choices


def write_jsonl(path, rows, private=False):
    mode = 0o600 if private else 0o644
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def prepare(raw, output):
    if output.exists():
        raise FileExistsError(output)
    verify_sources(raw)
    sentences = read_workbook(raw / WORKBOOK)
    trials, text_ids = read_trial_geometry(raw, sentences)
    earlier_later, graph, degree = read_earlier_later(raw, sentences, trials, text_ids)
    components = connected_components(graph)
    if len(components) != 39 or max(map(len, components.values())) != 51:
        raise ValueError("The equal-word-count source graph differs from the verified v5 inventory")
    test_components = split_components(components, degree)
    component_by_sid = {sid: key for key, members in components.items() for sid in members}
    test_sids = {sid for key in test_components for sid in components[key]}
    fit_sids = graph.keys() - test_sids
    if not fit_sids or not test_sids or test_sids & fit_sids:
        raise ValueError("The equal-word graph did not yield two disjoint sentence sets")
    source_text_index = {normalized_text(sentences[sid].text): sid for sid in sorted(graph)}
    if len(source_text_index) != len(graph):
        raise ValueError("The strict corpus contains a repeated normalized sentence")
    sentence_ids = {sid: opaque("sentence", sid, sentences[sid].text) for sid in graph}
    if len(set(sentence_ids.values())) != len(graph):
        raise ValueError("Two exact sentence stimuli share an opaque identifier")

    # The component assignment is fixed before reading pair choices or response-based filters.
    quality = author_quality_flags(raw)
    choices = read_choices(raw, earlier_later, quality)
    partitions = {"fit": [], "test": []}
    for key, (earlier, later) in earlier_later.items():
        if component_by_sid[earlier] != component_by_sid[later]:
            raise ValueError("An eligible pair crosses the registered sentence holdout boundary")
        split = "test" if earlier in test_sids else "fit"
        identifier = opaque("judgment", *key)
        partitions[split].append((
            {"id": identifier, "earlier_id": sentence_ids[earlier], "later_id": sentence_ids[later]},
            {"id": identifier, "reader_id": opaque("reader", key[0]),
             "earlier_easier": choices[key], "quality_retained": quality[key[0]]},
        ))
    indices = {part: sorted(rows, key=lambda pair: pair[0]["id"]) for part, rows in partitions.items()}
    public_ids = {row["id"] for rows in indices.values() for row, _ in rows}
    if len(public_ids) != len(earlier_later):
        raise ValueError("Two choices share the same opaque identifier")
    readers = {part: {label["reader_id"] for _, label in rows} for part, rows in indices.items()}
    counts = {
        part: {
            "sentences": len(sids),
            "components": len({component_by_sid[sid] for sid in sids}),
            "judgments": len(indices[part]),
            "quality_retained_judgments": sum(label["quality_retained"] for _, label in indices[part]),
            "readers": len(readers[part]),
            "word_count_histogram": dict(Counter(sentences[sid].words for sid in sids)),
        }
        for part, sids in (("fit", fit_sids), ("test", test_sids))
    }
    stimuli, metadata = [], []
    for sid, identifier in sorted(sentence_ids.items(), key=lambda pair: pair[1]):
        source = sentences[sid]
        stimuli.append({"id": identifier, "text": source.text, "stops": [len(source.text)]})
        metadata.append({
            "id": identifier,
            "component_id": component_by_sid[sid],
            "word_count": source.words,
            "char_count": source.chars,
            "syllable_count": source.syllables,
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".turkish-assay-") as directory:
        staged = Path(directory) / "assay"
        (staged / "model_input").mkdir(parents=True)
        files = {
            "model_input/stimuli.jsonl": stimuli,
            "sentence-metadata.jsonl": metadata,
            **{f"{part}.jsonl": (row for row, _ in rows) for part, rows in indices.items()},
            **{f"{part}-truth.jsonl": (label for _, label in rows) for part, rows in indices.items()},
        }
        for name, rows in files.items():
            write_jsonl(staged / name, rows, private=name.endswith("-truth.jsonl"))
        assay = {
            "schema_version": 1,
            "release": RELEASE,
            "release_doi": "10.5281/zenodo.22775520",
            "version": "v5",
            "study": "Reader-based Turkish sentence difficulty",
            "creators": ["Naim Karasekreter", "Özkan Aslan", "Caner Balım", "Taner Sezer",
                         "Özay Karadağ", "Süleyman Yaman"],
            "license": "https://creativecommons.org/licenses/by/4.0/",
            "license_scope": "The authors license the study data and documentation under CC BY 4.0. "
                             "LexiTR compiler Taner Sezer approved redistribution of these sentence excerpts. "
                             "The authors did not independently verify rights in each original source.",
            "modifications": "Retain exact released sentence strings. Select strict equal-word-count pairs and "
                             "assign whole connected components to a held-out set. Replace local identifiers "
                             "with deterministic opaque hashes.",
            "source_sha256": {name: value[1] for name, value in FILES.items()},
            "prepare_sha256": digest(Path(__file__)),
            "source_counts": {"workbook_sentences": 800, "administered_sentences": 787,
                              "comprehension_responses": 22490, "preference_judgments": 2960},
            "strict_equal_word_counts": {
                "judgments": len(choices), "unique_pairs": sum(map(len, graph.values())) // 2,
                "sentences": len(stimuli), "components": len(components),
                "largest_component": max(map(len, components.values())),
            },
            "split": {
                "seed": SPLIT_SEED,
                "target_test_fraction": 0.25,
                "selection": "Sort whole strict-equal-word connected components by the opaque seeded hash "
                             "of their sorted source sentence IDs. Select the contiguous cyclic window "
                             "minimizing normalized absolute deviations from 25% of sentences and judgments. "
                             "Break ties by component-count deviation, starting position and window width.",
                "test_component_ids": sorted(test_components),
                "shared_sentence_ids": 0,
                "shared_normalized_sentences": 0,
                "shared_readers": len(readers["fit"] & readers["test"]),
            },
            "counts": counts,
            "endpoint": "Observed binary choice of the earlier-displayed sentence as easier. Earlier and later "
                        "follow the original Cumle_1 and Cumle_2 fields. Include all valid strict equal-word pairs.",
            "surface_controls": "Use published selected_sentences #word, #char and #syl counts. "
                                "All comparisons have exactly equal #word.",
            "quality_sensitivity": "The original filter retained readers with median comprehension response "
                                   "time >=5 seconds and mean correctness >0.30. Flags are private and do "
                                   "not determine eligibility, split, surface variables or model input.",
            "security": {
                "model_input": "model_input/stimuli.jsonl",
                "local_outcomes": ["fit-truth.jsonl", "test-truth.jsonl"],
            },
            "file_sha256": {name: digest(staged / name) for name in files},
        }
        (staged / "assay.json").write_text(
            json.dumps(assay, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        staged.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    acquire = modes.add_parser("fetch")
    acquire.add_argument("--output", type=Path, required=True)
    intake = modes.add_parser("prepare")
    intake.add_argument("--raw", type=Path, required=True)
    intake.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "fetch":
        fetch(args.output)
    else:
        prepare(args.raw, args.output)


if __name__ == "__main__":
    main()
