"""Score frozen deployed native sentence understanding at exact paired source offsets."""

import json
from pathlib import Path

from research.reading_validation.local_forecast import digest, rows

PROTOCOL = Path(__file__).with_name("native-heading-protocol.md")


def prepare():
    from collections import defaultdict

    from research.expository_lora.score import ARMS
    from research.reading_validation.local_features import DATA, PACKET, RESULTS, verified_rows

    layout = {r["id"]: r for r in verified_rows("kerwer", "layout.jsonl")}
    inputs = verified_rows("kerwer", "stimuli.jsonl")
    source = json.loads((PACKET / "source-audit.json").read_bytes())["sources"]["kerwer"]
    contract_path = RESULTS / "kerwer-transfer-protocol.json"
    if digest(contract_path) != source["contract_sha256"]:
        raise ValueError("The alignment contract differs from the accepted source")
    by_id = json.loads(contract_path.read_bytes())["alignment"]["eligible"]["headings"]
    matched = defaultdict(dict)
    records = []
    for r in inputs:
        key, text = r["id"], r["text"]
        study = layout[key]
        spans = by_id[key]
        if [b for a, b in spans] != study["matched_stops"] or not set(study["matched_stops"]) <= set(r["stops"]):
            raise ValueError("Paired body stops differ from the certified exposure")
        matched[study["doi"]][study["arm"]] = [text[a:b] for a, b in spans]
        records.append(dict(id=key, text=text, spans=spans))
    if len(records) != 24 or len(matched) != 12 or sum(len(x["spans"]) for x in records) != 362:
        raise ValueError("Source geometry mismatch")
    if any(pair[ARMS[0]] != pair[ARMS[1]] for pair in matched.values()):
        raise ValueError("A registered body sentence differs between arms")
    packet = DATA / "native-heading-stimuli.jsonl"
    packet.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    return packet


def run(runtime, reader, checkpoint, input_path, output_dir):
    import torch
    from research.reading_validation.local_inference import LocalInference, REVISION

    source, out = Path(input_path), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    inputs = rows(source)
    if runtime.revision != REVISION or checkpoint["candidate"] != "qwen" or len(inputs) != 24:
        raise ValueError("Unrecognized heading comparison")
    observer = LocalInference.load()
    calls = []
    records = []
    with runtime.encoder.register_forward_pre_hook(lambda *_: calls.append(1)), torch.inference_mode():
        for row in inputs:
            if set(row) != {"id", "text", "spans"}:
                raise ValueError("Unexpected protected input field")
            spans = [tuple(x) for x in row["spans"]]
            before = len(calls)
            features = runtime.encode(row["text"])
            result = observer.score(row["text"], features, runtime.tokenizer, reader, sentence_spans=spans)
            if len(calls) != before + 1 or spans != [(s["start"], s["end"]) for s in result["sentences"]]:
                raise ValueError("Single-pass or original offset certificate failed")
            scores = [dict(start=s["start"], end=s["end"], rating=s["rating"]) for s in result["sentences"]]
            records.append(dict(id=row["id"], tokens=len(features.token_ids), sentences=scores))
    predictions = out / "native-headings.jsonl"
    predictions.write_text("".join(json.dumps(r) + "\n" for r in records))
    receipt = dict(revision=runtime.revision, passes=len(calls), native_width=reader.input_width,
                   bundle_sha256=observer.bundle_sha256, stimuli_sha256=digest(source),
                   predictions_sha256=digest(predictions), code_sha256=digest(Path(__file__)),
                   spans=sum(len(r["sentences"]) for r in records),
                   certified=sum(s["rating"] is not None for r in records for s in r["sentences"]))
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    print(digest(prepare()))
