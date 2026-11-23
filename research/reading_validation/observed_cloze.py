"""Use directly verified nonempty human completions from the original questionnaire releases."""
from pathlib import Path
from research.backbone_transfer.extract import digest, read_json

BASE = Path("research/results/reading-validation-convergent")


def observe(recovered):
    source = BASE / "independent-source-count-audit.json"
    reg = read_json(BASE / "observed/registration.json")
    if digest(source) != reg["raw_audit"]:
        raise ValueError("Unverified original cloze records")
    audit = read_json(source)
    if (
        audit["word_units"] != 1726
        or audit["cloze_matching_counts"] != 1726
        or audit["ratings_matching_n"] != 1726
    ):
        raise ValueError("Incomplete exact published measurement join")
    counts = {}
    for key, (slots, misses, variance) in recovered.items():
        raw = audit["items"][str(key)]
        hits = slots - misses
        if slots != raw["source_cloze_denominator"] or hits != raw["cloze_source_hits"]:
            raise ValueError(
                "The original received answers disagree with the archived source"
            )
        n = raw["cloze_nonempty"]
        counts[key] = (n, n - hits, variance)
    return counts
