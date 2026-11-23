"""Align original UD lemmas and POS tags to the fixed GUMBridge word candidates."""

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from research.semantic_tracking import gum_bridge_prepare as gum


# The ranker uses this same span rule. Importing it would load the numerical stack.
WORD = re.compile(r"[^\W_]+(?:[’'\-][^\W_]+)*")
GENTLE_AUDIT = Path(__file__).resolve().parents[1] / "results/gum-bridge-gentle-audit.json"
UPOS = {
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART",
    "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
}


def read_word_sentences(path):
    """Read FORM, LEMMA and UPOS with identity keys, without interpreting MISC."""
    document = sentence = None
    columns = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.rstrip("\n")
            if not line:
                if columns:
                    yield document, sentence, columns
                    columns, sentence = [], None
            elif line.startswith("# newdoc id = "):
                document = line.removeprefix("# newdoc id = ")
            elif line.startswith("# sent_id = "):
                sentence = line.removeprefix("# sent_id = ")
            elif not line.startswith("# "):
                identifier, form, lemma, upos, _ = line.split("\t", 4)
                columns.append((identifier, form, lemma, upos))
    if columns:
        yield document, sentence, columns


def index_articles(path):
    articles = {}
    for _, article in gum.jsonl_rows(path):
        key = article["id"]
        if article.keys() != {"id", "text", "stops", "paragraphs"} or key in articles:
            raise ValueError(f"Unexpected grade-free article shape or duplicate ID in {path}")
        articles[key] = article
    return articles


def document_words(identifier, article, sentences, annotations, stats):
    text, sentence_starts, paragraphs = gum.text_layout(sentences, Counter())
    if text != article["text"] or [list(span) for span in paragraphs] != article["paragraphs"]:
        raise ValueError(f"The official text or paragraph geometry differs for {identifier}")
    tokens = {}
    for index, (sentence, offset, source_words) in enumerate(
        zip(sentences, sentence_starts, annotations, strict=True), start=1,
    ):
        # Existing FORM alignment with the reference-annotation column removed.
        tokens_without_misc = tuple((key, form, "") for key, form, _, _ in source_words)
        clean = replace(sentence, tokens=tokens_without_misc)
        positions = gum.align_tokens(clean, offset, stats)
        annotated = [(lemma, upos) for key, _, lemma, upos in source_words if key.isdecimal()]
        for ((left, right), _), (lemma, upos) in zip(positions, annotated, strict=True):
            if not lemma or upos not in UPOS or (left, right) in tokens:
                raise ValueError(f"Unexpected UD word annotation at {identifier}:{right}")
            tokens[left, right] = (upos, None if lemma == "_" else lemma, index)
    if [right for _, right in tokens] != article["stops"]:
        raise ValueError(f"The official token ends differ for {identifier}")
    stops = set(article["stops"])
    features = []
    for match in WORD.finditer(text):
        stats["regex_words"] += 1
        left, right = match.span()
        if right not in stops:
            continue
        stats["regex_words_with_source_stop"] += 1
        annotation = tokens.get((left, right))
        if annotation is None:
            continue
        upos, lemma, sentence = annotation
        features.append({"start": left, "stop": right, "upos": upos, "lemma": lemma, "sentence": sentence})
        stats["singleton_words"] += 1
        stats["singleton_missing_lemma"] += lemma is None
    return features


def write_features(path, expected_source_sha256, articles_path, expected_articles_sha256, output, used_documents):
    if gum.digest(path) != expected_source_sha256 or gum.digest(articles_path) != expected_articles_sha256:
        raise ValueError(f"The CoNLL-U file or official articles changed for {path.name}")
    articles = index_articles(articles_path)
    if used_documents & articles.keys():
        raise ValueError("An official document ID appears in multiple diagnostic datasets")
    used_documents.update(articles)
    tagged_sentences = iter(read_word_sentences(path))
    seen, stats = set(), Counter()
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        for identifier, _, sentences in gum.read_documents(path):
            annotations = []
            for sentence in sentences:
                document_id, sentence_id, words = next(tagged_sentences)
                if (
                    document_id != identifier or sentence_id != sentence.identifier
                    or [(key, form) for key, form, _, _ in words]
                    != [(key, form) for key, form, _ in sentence.tokens]
                ):
                    raise ValueError(f"The CoNLL-U columns do not reproduce official token identities in {identifier}")
                annotations.append(words)
            if identifier not in articles:
                stats["source_documents_outside_fixed_split"] += 1
                continue
            if identifier in seen:
                raise ValueError(f"Repeated official source document ID {identifier}")
            seen.add(identifier)
            stats["official_documents"] += 1
            words = document_words(identifier, articles[identifier], sentences, annotations, stats)
            gum.write_row(stream, {"id": identifier, "words": words})
    if next(tagged_sentences, None) is not None or seen != articles.keys():
        raise ValueError(f"Official source and article membership disagree for {path.name}")
    return {
        "source_file": path.name, "source_sha256": expected_source_sha256,
        "articles_sha256": expected_articles_sha256, "feature_sha256": gum.digest(output),
        "counts": dict(stats),
    }


def prepare(source, assay, gentle, output):
    assay_path = assay / "audit.json"
    gum_audit = json.loads(assay_path.read_text(encoding="utf-8"))
    gentle_audit = json.loads(GENTLE_AUDIT.read_text(encoding="utf-8"))
    if gum_audit["source_revision"] != gum.REVISION:
        raise ValueError("The GUMBridge assay uses a different source revision")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    reports, used_documents = {}, set()
    with TemporaryDirectory(dir=output.parent, prefix=".gum-word-features-") as temporary:
        staged = Path(temporary) / "features"
        staged.mkdir()
        for split in gum.SPLITS:
            name = f"en_gum-ud-{split}.conllu"
            articles = f"model_input/{split}.jsonl"
            reports[split] = write_features(
                source / name, gum.SOURCE_HASHES[name], assay / articles, gum_audit["output_sha256"][articles],
                staged / f"{split}.jsonl", used_documents,
            )
        reports["gentle"] = write_features(
            gentle / gentle_audit["source_file"], gentle_audit["source_sha256"], gentle / "articles.jsonl",
            gentle_audit["derived_sha256"]["articles.jsonl"], staged / "gentle.jsonl", used_documents,
        )
        audit = {
            "schema": 1,
            "assay_sha256": gum.digest(assay_path),
            "gentle_audit_sha256": gum.digest(GENTLE_AUDIT),
            "source_code_sha256": gum.digest(Path(__file__)),
            "alignment_code_sha256": gum.digest(Path(gum.__file__)),
            "paths": reports,
            "word_pattern": WORD.pattern,
            "coordinate_system": "Unicode code points, zero-based and end-exclusive. Sentence ordinal is one-based.",
            "eligibility": "A ranker word must exactly coincide with one integer-ID CoNLL-U token span.",
            "purpose": "The human-annotated word columns are available for an oracle diagnostic.",
        }
        with (staged / "audit.json").open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        staged.rename(output)
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "assay", "gentle", "output"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    print(json.dumps(prepare(**vars(parser.parse_args())), sort_keys=True))
