"""Match fresh one-pass writer outputs with previously sealed official English summary forecasts."""

from argparse import ArgumentParser
import csv
from hashlib import sha256
from io import TextIOWrapper
import json
from pathlib import Path
from shutil import copyfile
from zipfile import ZipFile


DEST = Path("research/results/reading-validation-transition-external/seahorse")
SOURCE = Path(".cache/cogllm/reading-validation/seahorse-writer-stimuli.jsonl")
MAX_ABSOLUTE_DRIFT = .003


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def verify(folder):
    receipt = read(folder / "verification.json")
    locked = read(DEST / "serving-replay.json")
    spec = read(DEST / "serving/spec.json")
    if digest(SOURCE) != locked["native_packet"] or receipt["input_sha256"] != locked["native_packet"]:
        raise ValueError("Source inputs differ from the selections made before the serving run")
    if (receipt["candidate"] != "qwen" or receipt["revision"] != spec["backbone_revision"] or
            receipt["encoder_forwards"] != locked["native_rows"] - 1 or receipt["full_app_forwards"] != 3 or
            receipt["direct_serving_forwards"] != locked["native_rows"] - 4 or
            receipt["empty_abstentions_without_forward"] != 1):
        raise ValueError("Unexpected revision, forward count or application route")
    files = ("native-forecasts.json", "responses.json")
    if receipt["files"] != {name: digest(folder / name) for name in files}:
        raise ValueError("A signed application capture changed during transport")
    with SOURCE.open(encoding="utf8", newline="\n") as stream:
        inputs = [json.loads(line) for line in stream]
    results = read(folder / "native-forecasts.json")
    app = read(folder / "responses.json")
    expected = locked["native_expected"]
    all_ids = {r["id"] for r in inputs}
    if set(results) != all_ids or set(expected) != all_ids - {inputs[0]["id"]}:
        raise ValueError("Expected the synthetic demo and all original-source comparisons")
    cases = []
    largest = max((k for k,v in expected.items() if v["available"]),key=lambda k:expected[k]["probability_difficult"])
    by_id = {r["id"]:r for r in inputs}
    released = DEST / "exposure-seal.json"
    original = Path(".cache/cogllm/reading-validation/seahorse-original.zip")
    if digest(original) != read(released)["source"]:
        raise ValueError("The identified human trial archive has changed")
    identities = {inputs[1]["id"], largest}
    reference = {}
    with ZipFile(original) as archive, archive.open("seahorse_data/test.tsv") as binary:
        for row in csv.DictReader(TextIOWrapper(binary,encoding="utf8",newline=""),delimiter="\t"):
            uid = sha256(row["summary"].encode()).hexdigest()
            if row["worker_lang"]=="en-US" and uid in identities:
                if uid in reference:
                    raise ValueError("Multiple source identities for the selected study example")
                reference[uid] = {"article":row["gem_id"],"system":row["model"],"human_Q1":row["question1"]}
    if set(reference) != identities:
        raise ValueError("Cannot attribute the two original public judgments")
    fixtures = (("synthetic_example",inputs[0]),("licensed_wikilingua_example",inputs[1]),
                ("highest_risk_in_fixed_source_subset",by_id[largest]))
    for number, (name,row) in enumerate(fixtures, start=1):
        record = app[str(number)]
        if record["text"] != row["text"] or record["english"]["summary"] != results[row["id"]]:
            raise ValueError("Full application response lost the original input or its numeric result")
        projection = results[row["id"]]
        if projection["scope"] != spec["scope"] or projection["threshold"] != spec["threshold"]:
            raise ValueError("The serving response acquired an incorrect meaning or threshold")
        cases.append({"fixture": name, "text_sha256": row["id"], "response": projection,
                      "official_human_response":reference.get(row["id"])})
    if {c["response"]["review"] for c in cases}!={False,True}:
        raise ValueError("The two visible author decision states were not both captured")
    max_delta = 0.
    scored = 0
    abstained = 0
    for uid, original in expected.items():
        actual = results[uid]
        if set(actual) != set(original):
            raise ValueError("Serving changed the released response schema")
        for key in original.keys() - {"probability_difficult"}:
            if original[key] != actual[key]:
                raise ValueError(f"Raw input changed the held {key} value on {uid}")
        if not original["available"]:
            if actual["probability_difficult"] is not None or actual["review"] is not None:
                raise ValueError("An unidentified original text acquired a verdict")
            abstained += 1
            continue
        delta = abs(original["probability_difficult"] - actual["probability_difficult"])
        if not 0 <= delta <= MAX_ABSOLUTE_DRIFT:
            raise ArithmeticError(f"Fresh Qwen probability differs by {delta:.6g} for {uid}")
        max_delta = max(max_delta, delta)
        scored += 1
    save = DEST / "writer"
    save.mkdir(exist_ok=True)
    for name in ("responses.json", "verification.json"):
        copyfile(folder / name, save / name)
    report = {
        "registered_input_sha256": digest(SOURCE),
        "native_receipt_sha256": digest(folder / "verification.json"),
        "human_response_source": locked["example"],
        "original_forward_calls": receipt["encoder_forwards"],
        "empty_input_abstentions_without_forward": receipt["empty_abstentions_without_forward"],
        "complete_application_responses": len(app),
        "fresh_held_probabilities": scored,
        "matching_held_abstentions": abstained,
        "same_held_alerts": scored,
        "held_max_absolute_probability_drift": max_delta,
        "fixed_probability_tolerance": MAX_ABSOLUTE_DRIFT,
        "examples": cases,
    }
    (DEST / "serving-native-verification.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{scored} probabilities, {abstained} abstentions, maximum drift {max_delta:.8f}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    verify(parser.parse_args().folder)
