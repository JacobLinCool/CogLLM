"""Map CoheSentia incremental reports and DECOR expert decisions to original boundaries."""
import json
from collections import Counter
from functools import lru_cache

from research.reading_validation.transition_external_prepare import csv_rows
from research.reading_validation.transition_external_source import open_verified


@lru_cache
def collection(name):
    if name.endswith(".json"):
        return json.loads(open_verified(name))
    return csv_rows(name)


def votes(record):
    source = record["source"]
    if source == "cohesentia":
        name, story, index = record["source_ref"]
        trial = collection(name)[story]["IncrementalData"]
        if len(trial["sentences"]) != trial["num_sentences"] or len(trial["reasons"]) != trial["num_annotators"]:
            raise ValueError("Incomplete worker map in public release")
        yes = total = 0
        codes = Counter()
        invalid = 0
        for local_worker, marks in trial["reasons"].items():
            if local_worker not in trial["scores"]:
                raise ValueError("Unable to trace the local rater")
            if any(not str(i).isdigit() or not 0 <= int(i) < len(trial["sentences"]) for i in marks):
                # The public indexing convention is zero based. One out-of-range stop leaves
                # that worker's interpretation of the full story's sentence indices ambiguous.
                invalid += 1
                continue
            reasons = {int(i) for i in marks.get(str(index), [])}
            if reasons and (min(reasons)<1 or max(reasons)>8):
                raise ValueError("Unknown reason code")
            total += 1
            yes += bool(reasons)
            for i in reasons:
                codes[i] += 1
        return yes, total, dict(codes), invalid
    if source == "decor":
        name, index = record["source_ref"]
        row = collection(name)[index]
        label = row["label"]
        if row["essay_id"] != record["group"] or label not in ("0.0","1.0"):
            raise ValueError("DECOR coding does not match the released detection task")
        # The released detection CSV uses 0 for incoherent. Its README specifies that
        # coherent contexts have blank reasons and edits. Check this convention on every row.
        marked = label == "0.0"
        reasons = [row["R"+str(i)] for i in range(1,8)]
        if (marked != bool(row["Rewrite"].strip()) or
                any(x not in ("", "-1.0", "0.0", "1.0") for x in reasons) or
                (not marked and any(x != "" for x in reasons))):
            raise ValueError("Public essay labels and reason/rewrite fields disagree")
        return int(marked), 1, {i: int(v=="1.0") for i,v in enumerate(reasons,1) if v in ("0.0","1.0")}, 0
    raise ValueError("Expert rewrites have no separate post-edit human coherence judgment")
