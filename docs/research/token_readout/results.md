# Initial token-level readout and Gradio validation

The token-level Qwen3.5-2B readout now runs in a Gradio interface that accepts pasted text. Actual L4 requests returned highlighting over the full input, four token-level curves, and inspectable candidate references to earlier text. On the prespecified behavioral endpoints, the frozen readout improved the timing of Frick sentence-level first reports. It fell short of the thresholds for the target effect, the probability assigned to the evidence window, and localization of early and late resolution of the same issue. We selected the frozen readout for the research preview. [Scores and results by family](../../../research/results/token-readout-l4/metrics.json) · [Actual HTTP preview](../../../research/results/token-readout-l4/app/preview.html) · [Checkpoint receipt](../../../research/results/token-readout-l4/candidate.json)

## Three methods on the same split

The 167 stimuli contain 43,975 native tokens. The same readout uses the final layer, layer 12, and visible-prefix relationships from six native full-attention layers. The sentence-level first-report endpoint aggregates tokens within each sentence and retains the original sentence-end response location. LoRA updates only the query output rows and K in layer 24, leaving the output gate at its original value. Both training losses in each arm reached the prespecified eight-epoch training plateau criterion. The frozen arm stopped at epoch 40 and sealed epoch 36, and the LoRA arm stopped and sealed at epoch 38. Each epoch comprised 64 Frick updates and 64 evidence updates. We generated and sealed all holdout predictions before linking them with scoring annotations locally. [Data and protocol](design.md) · [Training history and seal](../../../research/results/token-readout-l4/fit/seal.json)

| Observation | Position baseline | Frozen Qwen | Final-layer Q/K LoRA |
| --- | ---: | ---: | ---: |
| Frick full-document first-report NLL, nats per trial, lower is better | 1.738 | 1.493 | 1.551 |
| Frick target-sentence first-report difference, percentage points, human +55.39 | 0.00 | +19.89 | +19.05 |
| ExpMRC evidence-interval NLL with known questions, nats per question, lower is better | 3.038 | 7.455 | 8.587 |
| ExpMRC ranking of which question has evidence in the shared earlier window, 27 pairs | 50.0% | 63.0% | 59.3% |
| Both early and late resolution separate in their assigned windows, 12 families, threshold of 9 | 0 | 5 | 3 |
| No-cue condition retains at least 80% of its earlier unresolved mass after the topic transition, 12 families | 12 | 10 | 8 |

Frick full-document NLL weights all 384 human trials. The target-sentence difference weights the four stories equally. Both models reproduce the direction of the earlier-context effect at the identical target sentence and underestimate the magnitude by about 35–36 percentage points. With each story weighted equally, LoRA minus frozen full-document NLL was +0.062, with a group-bootstrap 95% interval of [+0.044, +0.092]. These four families are existing development diagnostics. [Original comparison and denominators by story](../../../research/results/token-readout-l4/metrics.json)

On the ExpMRC same-window comparison, the frozen readout ranked 17/27 pairs in the correct direction and LoRA ranked 16/27. Interval errors reflect both premature activation and insufficient probability mass in the annotated support window. Relative to the position baseline, these two components added 1.179 and 3.238 nats per question for the frozen readout and 1.618 and 3.931 for LoRA. The error inside the support window was larger for both models. Some articles also produced substantial premature activation. The annotations locate textual support for given questions. [Independent error decomposition for 54 questions](../../../research/results/token-readout-l4/evidence-timing-diagnostic.json)

The 12 new authored expository triplets agreed at every position in their shared prefixes. For example, the frozen readout separated the early and late conditions from their length-matched controls inside both answer windows for family 3. Neither window reached the 5 percentage point separation threshold for family 1. Both families contained clear answer sentences, and their predicted timing differed. The no-cue condition met the 80% retention threshold across a topic shift in 10 families, and the step-count baseline met it in all 12. Retention alone leaves open whether the model found the answer sentence. The triplets were sealed as model behavior checks before scoring and have no human response data. [Curves and timing by family](../../../research/results/token-readout-l4/metrics.json)

## Gradio validation with the selected checkpoint

We used the frozen checkpoint selected in the comparison for three real HTTP analysis requests. These covered a prefix, its full continuation, and a Unicode passage with combining characters. Each passage used one backbone forward pass. Six cached selections required no additional forward passes. The HTTP checks verified the original text, all rendered highlights, references, and 380 curve points. Across the four channels, the maximum absolute difference over the 24 common prefix tokens was 5.22×10⁻⁸. Every reference into earlier text stayed unchanged. The interface displays the expected unresolved mass of candidates and their opening and resolution flows. Its separate transition score combines the unresolved mass at the previous position with the current difference in representations. References point to candidate spans in text already read. [HTTP receipt](../../../research/results/token-readout-l4/app/verification.json) · [Prefix checks](../../../research/results/token-readout-l4/app/causality.json) · [Preview with original text and curves](../../../research/results/token-readout-l4/app/preview.html)

Training and HTTP validation used the same unpadded, single-document extraction path. The planned right-padding equivalence check remains unverified, and this implementation accepts unpadded inputs. Both L4 sessions were released after about 14 minutes 29 seconds and 2 minutes 53 seconds from allocation. SHA-256 receipts bind the fit, predictions, and interface weights. Independent scoring replayed bit for bit. Token-level issue resolution and interruption still need more reliable position signals and observations closer to reading uninterrupted prose. [Training L4 receipt](../../../research/results/token-readout-l4/fit-run-verification.json) · [Interface L4 receipt](../../../research/results/token-readout-l4/app-run-verification.json)

Start the interface with the sealed research checkpoint in an environment with the same Qwen snapshot and a CUDA L4. [Interface entry point](../../../research/token_readout/app.py) · [Matching checkpoint](../../../research/results/token-readout-l4/fit/frozen.pt)

```sh
python -m research.token_readout.app \
  --runner research.token_readout.runner:load \
  --model /path/to/15852e8c16360a2fea060d615a32b45270f8a8fc \
  --checkpoint research/results/token-readout-l4/fit/frozen.pt
```

[Original research record](../../../research/token_readout/results.md)
