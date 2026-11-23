"""Grade-free span features shared by the difficulty and comprehension assays."""

from collections import Counter, defaultdict
from dataclasses import dataclass
import csv
import json
from pathlib import Path

import numpy as np

from research.reading_validation.local_forecast import (
    CACHE,
    CHANNELS,
    CHECKPOINT,
    RESULTS,
    WORD,
    digest,
    rows,
)


PACKET = CACHE / "expository-lora/prepared"
DATA = CACHE / "reading-validation"
OUT = RESULTS / "reading-validation-local"
SPAN_WIDTH = 8
BLOCKS = ("surface", "medical", "lexical", "representation", "shared", "persistent")


def verified_rows(dataset, name):
    path = PACKET / dataset / name
    audit = json.loads((PACKET / "source-audit.json").read_bytes())
    if digest(path) != audit["files_sha256"][f"{dataset}/{name}"]:
        raise ValueError(f"Observation geometry changed: {path}")
    return rows(path)


def row_grams(words):
    return set(zip(*(words[k:] for k in range(5))))


def medical_counts(raw, exclude):
    """Count training words and term spans at their original source coordinates."""
    from research.stage6.medreadme_span_audit import FILES, compact, source_spans

    for name, expected in FILES.items():
        if digest(raw / name) != expected:
            raise ValueError(f"MedReadMe source checksum mismatch for {name}")
    fit_texts = defaultdict(list)
    with (raw / "readability.csv").open(newline="") as stream:
        for item in csv.DictReader(stream):
            if item["split"] == "train":
                fit_texts[item["Source"], item["Side"], compact(item["Sentence"])].append(item["Sentence"])
    document_counts, terms, starts, ends, mentions, term_counts = [Counter() for _ in range(6)]
    labels = [r for r in json.loads((raw / "jargon.json").read_bytes()) if r["split"] == "train"]
    keys = [(*r["belongings"], compact("".join(r["tokens"]))) for r in labels]
    counts = Counter(keys)
    coverage = Counter()
    for key, r in zip(keys, labels, strict=True):
        if len(fit_texts[key]) != 1 or counts[key] != 1:
            coverage["ambiguous_or_unaligned"] += 1
            continue
        (text,) = fit_texts[key]
        matches = list(WORD.finditer(text))
        words = tuple(m.group().casefold() for m in matches)
        if exclude(words):
            coverage["overlap_with_target_material"] += 1
            continue
        coverage["used_sentences"] += 1
        document_counts.update(words)
        for j in range(len(words)):
            term_counts.update(tuple(words[j:e]) for e in range(j + 1, min(j + SPAN_WIDTH, len(words)) + 1))
        selected = set()
        for lo, hi, _ in source_spans(text, r["tokens"], r["entities"]):
            span = [i for i, m in enumerate(matches) if lo <= m.start() and m.end() <= hi]
            if not span or (matches[span[0]].start(), matches[span[-1]].end()) != (lo, hi):
                continue
            selected.update(span)
            phrase = tuple(words[span[0] : span[-1] + 1])
            if len(phrase) <= SPAN_WIDTH:
                terms[phrase] += 1
            starts[words[span[0]]] += 1
            ends[words[span[-1]]] += 1
        mentions.update(words[j] for j in selected)
    return (document_counts, terms, starts, ends, mentions, term_counts), coverage


def source_lexicon():
    target_grams = defaultdict(set)
    target_forms = set()
    for item in rows(DATA / "local-stimuli.jsonl"):
        words = tuple(match.group().casefold() for match in WORD.finditer(item["text"]))
        for gram in row_grams(words):
            target_grams[gram].add(item["id"])
        target_forms.add(words)

    def overlaps(words):
        grams = row_grams(words)
        matches = Counter(key for gram in grams for key in target_grams.get(gram, ()))
        return words in target_forms or (bool(matches) and max(matches.values()) * 4 >= len(grams))

    medical, coverage = medical_counts(CACHE / "medreadme-audit/raw", overlaps)
    complex_path = CACHE / "stage5/complex-l4-input/stimuli.jsonl"
    fit_sentences = {
        tuple(m.group().casefold() for m in WORD.finditer(r["text"]))
        for r in rows(complex_path)
        if r["split"] in ("train", "trial")
    }
    overlap_count = sum(overlaps(words) for words in fit_sentences)
    frequencies = Counter(w for words in fit_sentences if not overlaps(words) for w in words)
    receipt = dict(
        medreadme_training=dict(coverage),
        complex_unique_sentences=len(fit_sentences),
        overlapping_complex_sentences=overlap_count,
        lexical_checkpoint_overlap_qualification="The immutable CompLex head used its published original fit.",
    )
    return Lexicon(frequencies, *medical), receipt


@dataclass
class Lexicon:
    frequencies: Counter
    counts: Counter
    terms: Counter
    begins: Counter
    finishes: Counter
    mentions: Counter
    term_counts: Counter

    def records(self):
        return dict(
            frequencies=self.frequencies,
            counts=self.counts,
            begins=self.begins,
            finishes=self.finishes,
            mentions=self.mentions,
            phrases=[(k, self.terms[k], self.term_counts[k]) for k in sorted(self.terms)],
        )

    def word_scores(self, words):
        columns = []
        for count in (self.mentions, self.begins, self.finishes):
            prior = count.total() / self.counts.total()
            columns.append([(count[w] + 5 * prior) / (self.counts[w] + 5) for w in words])
        return np.column_stack(columns)


def read_gpu(directory, inputs=None, source_seal=None):
    if directory is None:
        return {}
    seal = json.loads((directory / "seal.json").read_bytes())
    expected = json.loads((source_seal or (OUT / "input-seal.json")).read_bytes())
    docs = rows(inputs or (DATA / "local-stimuli.jsonl"))
    runtime = json.loads((directory.parent / "causality.json").read_bytes())
    if runtime.get("checkpoint", runtime.get("checkpoint_sha256")) != CHECKPOINT:
        raise ValueError("GPU extraction used another selected checkpoint")
    if (
        seal["passes"] != len(docs)
        or seal["channels"] != list(CHANNELS)
        or seal["stimuli_sha256"] != expected["stimuli_sha256"]
        or seal["lexical_sha256"] != expected["lexical_sha256"]
        or seal["code_sha256"] != digest(Path(__file__).with_name("local_forecast.py"))
    ):
        raise ValueError("GPU extract disagrees with the sealed inputs")
    out = {}
    for row, info in zip(docs, seal["documents"], strict=True):
        file = directory / info["file"]
        if row["id"] != info["id"] or info["sha256"] != digest(file) or info["id"] in out:
            raise ValueError("GPU extract content mismatch")
        with np.load(file, allow_pickle=False) as arrays:
            item = {name: arrays[name] for name in arrays.files}
        words = np.array([m.span() for m in WORD.finditer(row["text"])], dtype=int)
        if (
            not np.array_equal(item["words"][:, :2], words)
            or len(item["offsets"]) != info["tokens"]
            or any(not np.isfinite(a).all() for a in item.values())
        ):
            raise ValueError("Invalid word alignment or feature")
        out[info["id"]] = item
    return out


def normalize_means(x, lo, hi):
    accum = np.concatenate((np.zeros_like(x[:1]), x.cumsum(axis=0)))
    return (accum[hi + 1] - accum[lo]) / (hi - lo + 1)[:, None]


def disjoint_ranking(coords, scores, limit=None):
    selected = []
    for index in np.argsort(-scores, kind="stable"):
        a, b = coords[index]
        if any(a < coords[j, 1] and coords[j, 0] < b for j in selected):
            continue
        selected.append(int(index))
        if limit is not None and len(selected) >= limit:
            break
    return np.asarray(selected, dtype=int)


@dataclass
class Sentence:
    key: str
    document: str
    coords: np.ndarray
    features: dict
    raw: dict
    sentence: dict


def sentence_features(
    key, document, text, start, end, lex, gpu, sentence_index, version, *, sentence_endpoint=True
):
    all_words = list(WORD.finditer(text))
    indices = np.array(
        [i for i, w in enumerate(all_words) if w.start() >= start and w.end() <= end], dtype=int
    )
    if not len(indices):
        raise ValueError(f"No complete words at {key}")
    tokens = [all_words[i] for i in indices]
    words = [w.group().casefold() for w in tokens]
    spans = np.array([w.span() for w in tokens])
    sizes = spans[:, 1] - spans[:, 0]
    frequency = np.array([np.log1p(lex.frequencies[word]) for word in words])
    medical = lex.word_scores(words)
    pos = np.arange(len(tokens))
    lo = np.repeat(pos, np.minimum(len(pos) - pos, SPAN_WIDTH))
    hi = np.concatenate([np.arange(j, min(j + SPAN_WIDTH, len(pos))) for j in pos])
    widths = hi - lo + 1
    coords = np.column_stack((spans[lo, 0], spans[hi, 1]))
    avg = lambda x: normalize_means(np.atleast_2d(x).T, lo, hi).ravel()
    form = np.column_stack(
        (
            np.log1p(widths),
            np.log1p(coords[:, 1] - coords[:, 0]),
            np.log1p(lo),
            np.log1p(indices[lo]),
            avg(sizes),
            avg(frequency),
            sizes[lo],
            sizes[hi],
            frequency[lo],
            frequency[hi],
            avg([w.group()[0].isupper() for w in tokens]),
            avg([w.group().isupper() for w in tokens]),
            avg([any(c.isdigit() for c in w.group()) for w in tokens]),
        )
    )
    phrases = [tuple(words[a : b + 1]) for a, b in zip(lo, hi, strict=True)]
    external = np.column_stack(
        (
            normalize_means(medical, lo, hi),
            medical[lo, 1],
            medical[hi, 2],
            [np.log1p(lex.terms[p]) for p in phrases],
            [lex.terms[p] / (lex.term_counts[p] + 4) for p in phrases],
        )
    )
    sf = {
        "surface": np.array(
            [
                np.log1p(len(text[start:end])),
                np.log1p(len(words)),
                np.log1p(sentence_index),
                version,
                sizes.mean(),
                frequency.mean(),
                np.quantile(frequency, 0.10),
            ]
        ),
        "medical": np.r_[medical.mean(axis=0), np.quantile(medical[:, 0], 0.90)],
    }
    blocks, raw = dict(surface=form, medical=external), {}
    if gpu is not None:
        native = gpu["words"][indices, 2]
        # Only explicitly certified tokens contribute to a readout. Include coverage as a feature.
        valid = native >= 0
        score = np.zeros(len(words))
        score[valid] = gpu["lexical"][native[valid]]
        blocks["lexical"] = np.column_stack((avg(score), score[lo], score[hi], avg(valid)))
        if sentence_endpoint:
            if not valid.any():
                raise ValueError(f"No certified word for the sentence rating in {key}")
            sf["lexical"] = np.array((score[valid].mean(), np.quantile(score[valid], 0.90), valid.mean()))
        raw["external_lexical"] = avg(score)
        for label, array in [("representation", "projection"), ("shared", "shared")]:
            if array not in gpu:
                continue
            present = np.zeros((len(words), gpu[array].shape[1]))
            present[valid] = gpu[array][native[valid]]
            blocks[label] = normalize_means(present, lo, hi)
            if sentence_endpoint:
                last = np.flatnonzero(gpu["offsets"][:, 1] == end)
                if not len(last):
                    raise ValueError(f"Sentence endpoint lacks a certified token in {key}")
                sf[label] = gpu[array][last[-1]].astype(float)
        if "trace" in gpu:
            native_lo = np.searchsorted(gpu["offsets"][:, 1], spans[:, 0], side="right")
            native_hi = np.searchsorted(gpu["offsets"][:, 0], spans[:, 1], side="left")
            if (native_lo >= native_hi).any():
                raise ValueError("Word has no overlapping original token")
            trace = gpu["trace"]
            change = np.diff(trace[:, 0], prepend=0)
            word_trace = np.array(
                [np.r_[trace[b - 1], change[a:b].max()] for a, b in zip(native_lo, native_hi, strict=True)]
            )
            blocks["persistent"] = np.column_stack((normalize_means(word_trace, lo, hi), word_trace[hi]))
            raw.update(displayed_mass=avg(word_trace[:, 0]), displayed_increase=avg(word_trace[:, -1]))
            sf["persistent"] = np.r_[word_trace.mean(axis=0), word_trace[-1], word_trace.max(axis=0)]
    if any(not np.isfinite(x).all() for x in (*blocks.values(), *sf.values())):
        raise ValueError(f"Nonfinite features for {key}")
    return Sentence(key, document, coords, blocks, raw, sf)


def build(gpu_path=None):
    lex, receipt = source_lexicon()
    gpu = read_gpu(gpu_path)
    if gpu:
        receipt["word_end_coverage"] = {
            study: dict(
                total=sum(len(r["words"]) for k, r in gpu.items() if k.startswith(study + "/")),
                certified=sum(
                    int((r["words"][:, 2] >= 0).sum()) for k, r in gpu.items() if k.startswith(study + "/")
                ),
            )
            for study in ("plaba", "kerwer")
        }
    text = {r["id"]: r["text"] for r in rows(DATA / "local-stimuli.jsonl")}
    out = {}
    for row in verified_rows("plaba", "ratings.jsonl"):
        case = "plaba/" + row["id"]
        mat = text[case]
        for i, node in enumerate(row["ratings"]):
            key = node["anchor_id"]
            out[key] = sentence_features(
                key,
                case,
                mat,
                node["start"],
                node["stop"],
                lex,
                gpu.get(case),
                i,
                float(row["id"].endswith("_simp")),
            )
    # Kerwer matching uses body-text sentences with verbatim counterparts, preserving intervening headings.
    layouts = verified_rows("kerwer", "layout.jsonl")
    source_design = json.loads((RESULTS / "kerwer-transfer-protocol.json").read_bytes())["alignment"][
        "eligible"
    ]["headings"]
    for row in layouts:
        case = "kerwer/" + row["id"]
        mat = text[case]
        source = source_design[row["id"]]
        if [b for a, b in source] != row["matched_stops"]:
            raise ValueError("Kerwer matched sentence coordinates changed")
        for i, (start, stop) in enumerate(source):
            key = case + "@" + str(stop)
            out[key] = sentence_features(key, case, mat, start, stop, lex, gpu.get(case), i, 1.0)
    return out, receipt
