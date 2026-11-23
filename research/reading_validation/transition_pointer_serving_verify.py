"""Replay the English writer's single-pass antecedent scorer against blinded research outputs."""

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import torch

from research.backbone_transfer.extract import digest, write_json
from research.reading_validation.antecedent_inference import AntecedentInference, BUNDLE_SHA
from research.token_readout.backbone import Backbone


def run(model, source, references, output, source_sha256, reference_sha256):
    if digest(source) != source_sha256 or digest(references) != reference_sha256:
        raise ValueError("The independent writer verification packet changed")
    runtime = Backbone.load("qwen", model, native_attention=True)
    head = AntecedentInference.load()
    expected = np.load(references)
    materials = list(map(json.loads, source.read_text().splitlines()))
    inputs = forecasts = flags = candidates = calls = 0
    drift = 0.0

    def count(_model, _args):
        nonlocal calls
        calls += 1

    def analyze(text):
        nonlocal inputs
        old = calls
        with head.capture(text, runtime) as pointer:
            f = runtime.encode(text)
            logits = pointer.logits(f)
            links = pointer.links(f)
        if calls != old + 1:
            raise ValueError("The normal pointer invocation issued more than one encoder forward")
        inputs += 1
        return pointer, logits, links

    with torch.inference_mode(), runtime.encoder.register_forward_pre_hook(count):
        for doc in materials:
            call, logits, links = analyze(doc["text"])
            if [(a,b) for a,b,*_ in call.words] != list(map(tuple, doc["words"])) or set(logits) != set(doc["queries"]):
                raise ValueError("Writer and gold benchmark have different visible word positions")
            wanted = []
            for q, score in logits.items():
                f = expected[f"{doc['id']}/{q}"][-1].astype(np.float64)
                if f.shape != score.shape:
                    raise ValueError("Writer dropped an earlier candidate")
                drift = max(drift, float(np.max(abs(f-score))))
                if int(f.argmax()) != int(score.argmax()):
                    raise ValueError("Writer changed the most likely earlier word")
                confidence = 1 / np.exp(f-f.max()).sum()
                if confidence >= head.threshold:
                    wanted.append((*doc["words"][q], *doc["words"][int(f.argmax())]))
                candidates += q
                forecasts += 1
            observed = [tuple(s[k] for k in ("pronoun_start","pronoun_end","antecedent_start","antecedent_end")) for s in links]
            if wanted != observed or drift > .001:
                raise ValueError(f"Writer threshold alignment or logit replay failed: {drift}")
            flags += len(links)
        stem = "Zoë visited Café René with her friend. She smiled."
        variants = [stem, stem+" Then they walked home.", stem+" A truck arrived without warning."]
        traces = [analyze(text) for text in variants]
        short, first, _ = traces[0]
        words = [stem[a:b] for a,b,*_ in short.words]
        if not {"Zoë","Café","René"}.issubset(words):
            raise ValueError("Unicode lexical positions were split")
        changed = 0.0
        for _, other, _ in traces[1:]:
            for q,v in first.items():
                changed = max(changed, float(np.max(np.abs(v-other[q]))))
        if changed > .001:
            raise ValueError("An observed pronoun forecast depends on a later sentence")
        decomposed, _, _ = analyze("Cafe\u0301 opened. It was quiet.")
        if (0,5) not in {tuple(v[:2]) for v in decomposed.words}:
            raise ValueError("A combining mark lost its original text location")
        _, empty, links = analyze("A clear blue morning.")
        if empty or links:
            raise ValueError("A pronoun-free text received a link")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"receipt.json", {"bundle_sha256": BUNDLE_SHA, "source_sha256": source_sha256,
               "reference_sha256": reference_sha256,"verified_docs":len(materials),"heldout_pronouns":forecasts,
               "earlier_candidates":candidates,"threshold_links_all_eligible_forms":flags,"all_flags_and_topwords_match":True,
               "source_threshold":head.threshold,"single_pass_inputs":inputs,"native_encoder_forwards":calls,
               "max_abs_replay_logit":drift,"max_abs_changed_suffix_logit":changed,"zero_pronoun_abstention":True,
               "unicode_offsets_checked":True})


if __name__ == "__main__":
    parser=ArgumentParser(description=__doc__)
    for key in ("model","source","references","output"):
        parser.add_argument("--"+key,type=Path,required=True)
    for key in ("source-sha256","reference-sha256"):
        parser.add_argument("--"+key,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);run(**vars(args))
