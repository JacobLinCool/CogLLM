"""Reuse the certified news word/region measurement on complete original SEAHORSE summaries."""
import argparse
from collections import Counter
import json
from itertools import chain
from pathlib import Path
import re
import unicodedata

import numpy as np
from scipy.special import logsumexp

from research.reading_validation.summeval_controls import lexical
from research.reading_validation.summeval_forecast import open_old
from research.reading_validation.transport_word_archive import article_boundaries
from research.reading_validation.transition_seahorse_data import DEST, PLAN, WORK, ROLES, sha, source_rows
from research.reading_validation.memory_data import h


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def load(path):
    return json.loads(path.read_text())


def norm(text):
    return re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold())


def signatures(text):
    tokens = norm(text)
    return set(chain([h("full/" + " ".join(tokens))],
                     (h("twelve/" + " ".join(tokens[i:i+12])) for i in range(len(tokens)-11))))


def overlap():
    audit = {}
    external = {}
    source_paths = {"earlier_news": WORK/"summeval/local-stimuli.jsonl", "earlier_essay_story": WORK/"cohesion-stimuli.jsonl"}
    for k, src in source_paths.items():
        sig = set()
        for line in src.open():
            r = json.loads(line)
            sig.update(signatures(r["text"]))
        external[k] = sig
    seen = set()
    for role in ROLES:
        articles = set()
        current = set()
        hits = set()
        ext = {name: set() for name in external}
        for row in source_rows(role):
            key = row["gem_id"]
            articles.add(key)
            sig = signatures(row["summary"])
            if seen.intersection(sig):
                hits.add(key)
            current.update(sig)
            for name, index in external.items():
                if index.intersection(sig):
                    ext[name].add(key)
        audit[role] = {"articles": len(articles), "exclude_overlap_with_held_roles": sorted(hits),
                       "overlap_with_earlier_sources": {k: sorted(v) for k,v in ext.items()}}
        seen.update(current)
    save(DEST/"word-exposure-audit.json", audit)
    print({k: (v["articles"], len(v["exclude_overlap_with_held_roles"]), {s:len(z) for s,z in v["overlap_with_earlier_sources"].items()}) for k,v in audit.items()})


def document_features(z, key, details, text, old):
    regions, _ = article_boundaries(z, key, details, text)
    if not regions:
        return None, "no_certified_boundary"
    observed = z[key + ".char"][:, 4].astype(bool)
    if not observed.any():
        return None, "no_observable_word_surprisal"
    surprisal = float(z[key + ".surprisal"][observed].mean())
    form = lexical(text, [text])[0] + [surprisal]
    scaled = np.clip((np.stack(list(regions.values())) - old["english.center"]) / old["english.scale"], -8, 8)
    return np.r_[form, scaled.mean(0), logsumexp(scaled, axis=0) - np.log(len(scaled))], None


def extract(role, folder):
    packet = load(DEST/"exposure-seal.json")
    source = Path(packet["packets"][role]["file"])
    raw = folder/"features.npz"
    report = load(folder/"receipt.json")
    if (sha(PLAN) != packet["protocol"] or report["source"] != packet["packets"][role]["sha256"]
            or sha(source) != report["source"] or sha(raw) != report["file"] or not report["single_forward_per_unique_text"]):
        raise ValueError("Original SEA exposure or native output signature changed")
    text = {r["id"]: r["text"] for r in map(json.loads, source.read_text().splitlines())}
    if set(text) != set(report["articles"]) | set(report["skipped"]):
        raise ValueError("Native forecast omits registered text")
    p, _ = open_old()
    ids, x = [], []
    missed = dict(report["skipped"])
    with np.load(raw, allow_pickle=False) as z:
        for key, d in sorted(report["articles"].items()):
            t = text[key]
            features, reason = document_features(z, key, d, t, p)
            if features is None:
                missed[key] = reason
                continue
            ids.append(key)
            x.append(features)
    data = np.stack(x)
    if not np.isfinite(data).all() or data.shape[1] != 187:
        raise ArithmeticError("Invalid original source news measurements")
    file = WORK/f"seahorse-{role}-document-features.npz"
    np.savez_compressed(file, keys=ids, x=data)
    save(DEST/(role+"-feature-seal.json"), {"file":sha(file), "native":report["file"],"source":report["source"],
                                          "registered":len(text), "included":len(ids),"omitted":missed,
                                          "omission_counts":dict(Counter(missed.values())),
                                          "seconds":report["seconds"],"batches":report["encoder_batches"], "native_revision":report["revision"]})
    print(role,len(ids),len(missed),sha(file),flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("role", choices=("overlap","test","validation",*(f"train{i}" for i in range(4))))
    p.add_argument("folder", nargs="?", type=Path)
    args = p.parse_args()
    if args.role == "overlap":
        overlap()
    else:
        extract(args.role,args.folder)
