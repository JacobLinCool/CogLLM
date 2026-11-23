"""Prepare the ReadMe++ English text packet without copying ratings to the GPU."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import unicodedata

from openpyxl import load_workbook


REVISION = "b7b6d0808ad2e12c77867a0563ba7b229cbaf911"
SPLITS = ("test", "val", "train")
HEADERS = ("Domain", "Sub-domain", "Paragraph", "Context", "Sentence", "Rating")


def digest_string(value):
    return sha256(value.encode("utf-8")).hexdigest()


def material_key(text):
    return digest_string(
        " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    )


def workbook_rows(folder, split):
    book = load_workbook(folder / f"{split}.xlsx", read_only=True, data_only=True)
    try:
        values = book.active.values
        if tuple(next(values)) != HEADERS:
            raise ValueError("ReadMe++ worksheet schema changed")
        for row_number, row in enumerate(values, 2):
            if not row or all(cell is None for cell in row):
                continue
            if (
                len(row) != len(HEADERS)
                or not isinstance(row[4], str)
                or not row[4].strip()
            ):
                raise ValueError(f"Missing target sentence in {split} row {row_number}")
            yield row_number, row
    finally:
        book.close()


def prepare(folder, packet, manifest):
    seen_text, seen_paragraph = set(), set()
    records, groups, exclusions = {}, [], Counter()
    for split in SPLITS:
        new_text, new_paragraph = set(), set()
        for row_number, row in workbook_rows(folder, split):
            domain, subdomain, paragraph, _, text = row[:5]
            key = material_key(text)
            paragraph_key = (
                material_key(paragraph)
                if isinstance(paragraph, str) and paragraph.strip()
                else None
            )
            if key in seen_text or (paragraph_key and paragraph_key in seen_paragraph):
                exclusions[split] += 1
                continue
            uid = digest_string(text)
            records[uid] = text
            groups.append(
                dict(
                    id=uid,
                    split=split,
                    row=row_number,
                    domain=domain,
                    subdomain=subdomain,
                    text_key=key,
                    paragraph_key=paragraph_key,
                )
            )
            new_text.add(key)
            if paragraph_key:
                new_paragraph.add(paragraph_key)
        seen_text.update(new_text)
        seen_paragraph.update(new_paragraph)
    packet.parent.mkdir(parents=True, exist_ok=True)
    packet.write_text(
        "".join(
            json.dumps(dict(id=key, text=text), ensure_ascii=False) + "\n"
            for key, text in sorted(records.items())
        ),
        encoding="utf-8",
    )
    metadata = dict(
        revision=REVISION,
        rows=groups,
        exclusions=dict(exclusions),
        stimulus_sha256=sha256(packet.read_bytes()).hexdigest(),
        workbook_sha256={
            s: sha256((folder / f"{s}.xlsx").read_bytes()).hexdigest() for s in SPLITS
        },
    )
    manifest.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.folder, args.packet, args.manifest)
