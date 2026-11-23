"""Seal the public substitutions separately from the readers' preferences."""

import json
import re
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

from research.reading_validation.local_forecast import WORD, digest, rows
from research.reading_validation.nns_exposure import grams, sources
from research.reading_validation.nns_material import (
    ARCHIVE,
    BASE,
    DATA as SELF,
    SOURCE,
    identifier,
)
from research.reading_validation.release_overlap import words
from research.reading_validation.revision_prepare import matches, near_terms

DATA = BASE / "nns-rewrite"
OUT = Path("research/results/reading-validation-nns-rewrite")
GAP = re.compile(r"_{3,}")
MEMBER = "Substitution Ranking/SR_Annotations.txt"


def records(outcome=False):
    if digest(ARCHIVE) != SOURCE:
        raise ValueError("The original user-study archive changed")
    with ZipFile(ARCHIVE) as archive, archive.open(MEMBER) as stream:
        for raw in stream:
            fields = raw.decode().rstrip("\r\n").split("\t")
            if len(fields) != 10:
                raise ValueError("The author preference schema changed")
            public = fields[4:9]
            yield (*public, fields[9]) if outcome else tuple(public)


def focus(doc, start, end):
    return identifier(f"{doc}/{start}/{end}")


def study_geometry():
    sites = defaultdict(set)
    for t, original, pos, a, b in records():
        sites[t, original, int(pos)].update((a, b))
    texts = {}
    variants = {}
    for (t, original, pos), alternatives in sites.items():
        gaps = list(GAP.finditer(t))
        if len(gaps) != 1 or pos >= len(t.split()) or t.split()[pos] != gaps[0].group():
            raise ValueError("Cannot locate original gap at the reported position")
        match = gaps[0]
        a, b = match.span()
        site = identifier(t)
        texts[site] = t
        variants[site] = {}
        for candidate in sorted(alternatives | {original}):
            filled = t[:a] + candidate + t[b:]
            if not WORD.fullmatch(candidate) or (a, a + len(candidate)) not in {
                m.span() for m in WORD.finditer(filled)
            }:
                raise ValueError("Replacement is not a complete printed word")
            identity = identifier(filled)
            variants[site][candidate] = {
                "id": identity,
                "text": filled,
                "start": a,
                "end": a + len(candidate),
                "original": candidate == original,
            }
    return texts, variants


def run():
    texts, variants = study_geometry()
    DATA.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True, parents=True)
    ids = sorted(texts)
    masked = [texts[k] for k in ids]
    originals = [
        next(row["text"] for row in variants[k].values() if row["original"])
        for k in ids
    ]
    parent = list(range(len(ids)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    forms = []
    for i, k in enumerate(ids):
        forms.extend((i, r["text"]) for r in variants[k].values())
        forms.append((i, masked[i]))
    strings = [s for _, s in forms]
    shingles, lookup = near_terms(strings)
    normalized = defaultdict(set)
    windows = defaultdict(set)
    for i, (comp, t) in enumerate(forms):
        u = words(t)
        itself = normalized[u]
        itself.add(comp)
        connected = set(itself)
        for g in grams(u, 12):
            windows[g].add(comp)
            connected.update(windows[g])
        connected.update(forms[j][0] for j in matches(t, shingles, lookup))
        for j in connected:
            a, b = root(comp), root(j)
            parent[max(a, b)] = min(a, b)
    components = {k: ids[root(i)] for i, k in enumerate(ids)}
    print("Checking source-wise material exposure", flush=True)
    prior, _ = sources()
    for role in ("dense", "sparse"):
        prior["self_report_" + role] = {
            r["text"] for r in rows(SELF / (role + ".jsonl"))
        }
    excluded = set()
    audit = {}
    for label, material in prior.items():
        findings = defaultdict(set)
        for t in material:
            u = words(t)
            findings["exact"].update(normalized.get(u, ()))
            for g in grams(u, 12):
                findings["shared_12"].update(windows.get(g, ()))
            findings["near"].update(forms[j][0] for j in matches(t, shingles, lookup))
        hit = set().union(*findings.values())
        excluded.update(components[ids[i]] for i in hit)
        audit[label] = {
            **{k: len(v) for k, v in findings.items()},
            "any": len(hit),
            "source_texts": len(material),
        }
    eligible = sorted(
        set(components.values()) - excluded,
        key=lambda k: identifier("nns-rewrite/split/v1/" + k),
    )
    held = set(eligible[::3])
    roles = {
        k: ("excluded" if f in excluded else "held" if f in held else "train")
        for k, f in components.items()
    }
    public = {}
    for k in ids:
        if roles[k] == "excluded":
            continue
        for r in variants[k].values():
            v = public.setdefault(
                r["id"], {"id": r["id"], "text": r["text"], "spans": set()}
            )
            v["spans"].add((r["start"], r["end"]))
    for v in public.values():
        v["spans"] = sorted(v["spans"])
    packet = DATA / "stimuli.jsonl"
    geometry = DATA / "geometry.json"
    packet.write_text(
        "".join(
            json.dumps(public[k], ensure_ascii=False) + "\n" for k in sorted(public)
        )
    )
    geometry.write_text(
        json.dumps(
            {
                "families": components,
                "roles": roles,
                "variants": {
                    k: {v: focus(r["id"], r["start"], r["end"]) for v, r in vs.items()}
                    for k, vs in variants.items()
                },
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    receipt = {
        "packet_sha256": digest(packet),
        "geometry_sha256": digest(geometry),
        "candidates": len(public),
        "original_contexts": len(originals),
        "training_contexts": sum(v == "train" for v in roles.values()),
        "held_contexts": sum(v == "held" for v in roles.values()),
        "training_families": len(eligible) - len(held),
        "held_families": len(held),
        "excluded_families": len(excluded),
        "audited_sources": audit,
        "release_sha256": SOURCE,
        "protocol_sha256": digest(Path(__file__).with_name("nns-rewrite-protocol.md")),
        "code_sha256": digest(Path(__file__)),
        "outcomes_inspected": False,
    }
    (OUT / "material.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(receipt)


if __name__ == "__main__":
    run()
