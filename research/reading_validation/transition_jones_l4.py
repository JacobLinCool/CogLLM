"""One original-text Qwen forward for each interpretation and reading-time presentation."""

from argparse import ArgumentParser
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from research.backbone_transfer.extract import digest, write_json
from research.reading_validation.antecedent_inference import REVISION, native_signals
from research.reading_validation.transition_external_forecast import token_surprisals
from research.reading_validation.transition_pointer_features import design
from research.reading_validation.transition_pointer_l4 import ARMS, watch
from research.token_readout.backbone import Backbone


@torch.inference_mode()
def run(model, source_model, stimuli, output, input_sha256, protocol_sha256):
    receipt = json.loads((source_model/"receipt.json").read_text())
    source_weights = source_model/"source-weights.npz"
    if (digest(stimuli) != input_sha256 or digest(source_weights) != receipt["weights_sha256"]
            or receipt["weights_sha256"] != "9162700762a7574f45532bac88f8c018b1f2922118ac033bd1206a1edaecfaf5"):
        raise ValueError("Source learner or blinded experimental packet changed")
    data = list(map(json.loads, stimuli.read_text().splitlines()))
    if len(data) != 300 or any(set(r) != {"id","text","words","tokens","queries","pos","pos_last","sentences","token_count"} for r in data):
        raise ValueError("Expected only the original 300 article presentations and blind surface geometry")
    weights = np.load(source_weights)
    runtime = Backbone.load("qwen", model, native_attention=True)
    if runtime.revision != REVISION:
        raise ValueError("Different backbone")
    calls = 0
    def count(_, args):
        nonlocal calls
        calls += 1
    result, surprise = {}, {}
    start = perf_counter()
    with runtime.encoder.register_forward_pre_hook(count):
        for step, row in enumerate(data):
            word_tokens = torch.tensor(row["tokens"], device=runtime.device)
            starts, ends = word_tokens[:, 0], word_tokens[:, 1]
            all_words = torch.arange(len(starts), device=runtime.device)
            with watch(runtime, row["token_count"], ends, starts, ends, False) as cap:
                f = runtime.encode(row["text"])
            native = native_signals(f, cap, all_words, starts, ends)
            surprise[row["id"]] = token_surprisals(runtime, f).astype(np.float32)
            if len(f.token_ids) != row["token_count"] or any(f.offsets[j][1] != b for (_, j), (_, b) in zip(row["tokens"], row["words"], strict=True)):
                raise ValueError("Training and inference word coordinates differ")
            for i, q in enumerate(row["queries"]):
                control = design(row, q, i).astype(float)
                end = min(len(row["words"]), q+25)
                matrix = np.empty((end-q, q, len(control[0])+native.shape[-1]), dtype=float)
                matrix[:, :, :control.shape[1]] = control
                matrix[:, :, control.shape[1]:] = native[q:end, :q]
                z = []
                for arm in ARMS:
                    beta = weights[arm+"__beta"]
                    X = (matrix[:, :, :len(beta)] - weights[arm+"__mean"]) / weights[arm+"__sd"]
                    z.append((X @ beta).astype(np.float32))
                result[row["id"]+"/"+str(q)] = np.stack(z)
            if calls != step+1:
                raise ValueError("Expected exactly one original-text forward per presentation")
            if (step+1) % 100 == 0:
                print(f"Blind physical-event presentation {step+1}/{len(data)}", flush=True)
    output.mkdir(exist_ok=True, parents=True)
    pf, sf = output/"all-earlier-word-logits.npz", output/"surprisal.npz"
    np.savez_compressed(pf, **result)
    np.savez_compressed(sf, **surprise)
    write_json(output/"receipt.json", {"protocol_sha256":protocol_sha256,"source_sha256":input_sha256,"revision":REVISION,
               "source_weights_sha256":digest(source_weights),"arms":ARMS,"native_heads":48,"encoder_forwards":calls,
               "pronoun_forecasts":len(result),"words_after_pronoun":24,"human_outcomes_used":0,
               "forecast_sha256":digest(pf),"surprisal_sha256":digest(sf),"elapsed_seconds":perf_counter()-start})


if __name__ == "__main__":
    p=ArgumentParser(description=__doc__)
    for name in ("model","source-model","stimuli","output"):
        p.add_argument("--"+name,type=Path,required=True)
    for name in ("input-sha256","protocol-sha256"):
        p.add_argument("--"+name,required=True)
    run(**vars(p.parse_args()))
