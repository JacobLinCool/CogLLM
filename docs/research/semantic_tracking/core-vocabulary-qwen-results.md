# Qwen end-of-sentence representation improvements to group prediction for sentence-by-sentence word guessing

In 32 held-out target words and 1,603 human trials, adding the causal suffix representation of Qwen3.5-2B reduced the average NLL from 1.86717 (literal baseline) to **1.86328 nats**. The improvement was 0.00388, with a matched 95% confidence interval [0.00209, 0.00571]. Relative to the step-count-only baseline, it also passed the predetermined threshold. Cumulative accuracy rates for sentences 1, 3, and 6 all failed to pass the ±5 percentage point tolerance check. [Fixed Specification](core-vocabulary-qwen-protocol.md) · [Full Results](../../../research/results/core-vocabulary-qwen-l4/results.json)

## sealed snapshot materials and predictions

Each target word has eight possible cue paths drawn via the author's original random design, totaling 768 passages and 69,810 tokens. Each path undergoes one frozen Gradio candidate causal forward pass, extracting the last six 128-dimensional projections. Of the 4,608 end positions, all pass exact character alignment, and all 12 prefix controls pass. The model failed to retrieve the target word or human data. Only the 3,077 trials from the 64 training target words were fitted to the hazard readout. the sealed snapshot was taken before inspecting the scores of the remaining 32 target words. For each path, seven outcome probabilities were computed first, then mixed via equal weighting, and compared against the author's historical group response. [Sentence Extraction and Segmentation](core-vocabulary-marginal-protocol.md) · [Causal Extraction](../../../research/results/core-vocabulary-qwen-l4/extraction-seal.json) · [Predicting Sealed Snapshot](../../../research/results/core-vocabulary-qwen-l4/forecast-seal.json)

| held-out method | NLL (nats / trial) | Improvements and matching interval of 95% after adding Qwen |
| --- | ---: | --- |
| Step count only | 1.86765 | +0.00437 [0.00224, 0.00652] |
| Step count plus visible surface features | 1.86717 | +0.00388 [0.00209, 0.00571] |
| Literal addition of Qwen's final representation | 1.86328 | — |

We matched the target word by author condition, with conditions observed in the readers. When resampling both readers and target words, the improvement interval for Qwen relative to the literal baseline was [0.00177, 0.00602]. Results support a small population prediction gain in sentence-final representations within this resampling design and materials. [Two resampling results](../../../research/results/core-vocabulary-qwen-l4/results.json)

| Cumulative exposure | human accuracy | Qwen accuracy | Model reduced human, cross 95% interval |
| --- | ---: | ---: | --- |
| Sentence 1 | 9.35% | 7.52% | −1.83 percentage points [−5.41, +1.20] |
| Sentence 3 | 35.82% | 34.40% | −1.42 percentage points [−7.20, +4.18] |
| Sentence 6 | 69.89% | 71.43% | +1.54 percentage points [−4.00, +7.26] |

All three bias point estimates fall within the allowable margin, while all intervals exceed it. The eight sampling paths for each held-out word approximate only the author's complete sentence distribution, and individual readers' actual observed text lacks a published trial comparison. These results can validate the group's first-correct response distribution across possible paths, yet lack evidence of individual readers' resolution of questions at specific source positions. [Observation Scope and Source](core-vocabulary-source.md)

## Reproducibility and Resources

We reconstructed the sealed stimuli with `core_vocabulary_marginal_prepare` and checked causal prefixes with `beukers_extract preflight`. The L4 took 29.34 seconds to run the 24 batches of forward passes and readouts. Peak reserved GPU memory was about 7.87 GiB, and 195.68 seconds elapsed from provisioning to verified release. Rerunning fitting, prediction, and scoring with `core_vocabulary_marginal_fit` produced bit-for-bit identical outputs. Independent scalar calculations reproduced each condition's NLL and cumulative response rates. A full local rerun and verification took about 0.41 seconds and peaked at roughly 133 MiB. [Training and scoring](../../../research/semantic_tracking/core_vocabulary_marginal_fit.py) · [Resource use, hashes, and release verification](../../../research/results/core-vocabulary-qwen-l4/run-verification.json)

[Original research record](../../../research/semantic_tracking/core-vocabulary-qwen-results.md)
