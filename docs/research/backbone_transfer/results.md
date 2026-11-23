# A comparison of the three backbones on human reading data held-out

On 2026-11-21, with fixed human data, held-out groups, and readout specifications, the held-out vocabulary error gain was replicated across all three backbones. Sentence-level self-assessed understanding gains were replicated by Qwen and LFM3. All three underestimated the human benefit from vernacular rewriting and failed to replicate the magnitude of the contradictory prior report on human first-pass probability. [Fixed comparison specification](protocol.md)

## human endpoint and common baseline

| held-out endpoint | Common baseline | Qwen3.5-2B | LFM3-1.7B-Base | Gemma 4 E2B |
| --- | ---: | ---: | ---: | ---: |
| CompLex target word difficulty MAE | 0.10959 | 0.08064 | 0.08558 | 0.08716 |
| PLABA-EVAL sentence-level readability NLL | 1.34534 | 1.33344 | 1.32789 | 1.34361 |
| PLABA-EVAL sentence-level self-assessment NLL | 1.34540 | 1.31051 | 1.31057 | 1.34543 |
| Frick's entire first report contradicts NLL. | 1.73313 | 1.73022 | 1.73148 | 1.73206 |
| Frick's target sentence first reported a contradictory increase, human +48.27 percentage points | +0.0393 | +0.0449 | +0.0386 | +0.0390 |

Lower error is preferable. PLABA-EVAL employs a common baseline using sentence length and known paraphrase versions. CompLex and Frick utilize their respective fixed surface feature baselines. The Frick table uses a sentence-level newline reading stop shared across three backbones. The subsequent intervals represent resampled 95% groups of held-out predictions. PLABA-EVAL groups by reader and material into 25 connected groups, CompLex by 1,066 test sentence groups, and Frick resamples both stories and readers. [CompLex score](../../../research/results/backbone-transfer-l4/lexical-sentence/complex/metrics.json) · [PLABA-EVAL score](../../../research/results/backbone-transfer-l4/lexical-sentence/plaba/metrics.json) · [Frick score](../../../research/results/backbone-transfer-l4/frick/separator_end/results.json)

### Target word difficulty

Three groups were fitted with identical ridge readouts on a shared 8,666 training and 1,093 test set. The MAE improvement intervals relative to the surface baseline for all three groups exclude zero. The errors for LFM3 and Gemma relative to Qwen increased by 0.00494 (interval 0.00238 to 0.00746) and 0.00652 (0.00323 to 0.00970), respectively. The token check excluded one training and one test instance in the original split because LFM3's token span exceeded the target word. CompLex's human difficulty rating was obtained after viewing the full sentence. here, the model read to the end of the specified word, measuring whether the representation at that point could read out the rating signal. [Original split and reader-seen text](../stage-5-complex-run.md) · [Shared samples and paired differences](../../../research/results/backbone-transfer-l4/lexical-sentence/complex/metrics.json)

There are 1,050 pairs of same-type and lemma ordering across different sentences. The surface baseline agreement rate is 51.0%, Qwen is 50.9%, LFM3 is 50.1%, and Gemma is 51.8%. These point estimates are close to 50%, and we currently cannot infer from the overall MAE gain that the model can rank same-lemma usages as harder within context. The ordering audit adopted the sentence exclusion and tie-breaking rules from Stage 5, removing a set of same-sentence pairs. The sealed snapshot prediction and MAE are consistent. [Ordering Audit and Replay](../../../research/results/backbone-transfer-l4/run-verification.json)

### Sentence-level scoring and rewriting

All 609 sentences and 1,826 dual-scale ratings were retained. For self-rated comprehension, LFM3 minus Qwen NLL was +0.00007 nats, with interval −0.00591 to +0.00669. Both outperformed the length baseline with known version. Gemma's self-rated comprehension NLL was 0.03492 higher than Qwen's (+0.01867 to +0.05205), and its interval for the difference from the common baseline crossed zero. LFM3 had the lowest point estimate for reading ease NLL, with the difference from Qwen crossing zero. For reading ease measured by ranked probability score, the improvement intervals excluded zero for both Qwen and LFM3 relative to the common baseline. Gemma's interval crossed zero. [Five-fold holdout and both scoring measures](../../../research/results/backbone-transfer-l4/lexical-sentence/plaba/metrics.json)

Across 265 sentence pairs, the human vernacular version improved average readability by 0.537 points and self-reported comprehension by 0.723 points relative to the original. The corresponding model effect sizes were Qwen +0.277/+0.341, LFM3 +0.288/+0.350, and Gemma +0.190/+0.209. For all three models, the difference (model minus human) in both metrics fell within intervals below zero. The baseline for the known version can be directly utilized via rewrite conditions, with an average effect size of +0.496/+0.704, indicating that reproducing the average effect size and accurately predicting the distribution of individual sentences are two distinct validations. [Rewrite effect size and group intervals](../../../research/results/backbone-transfer-l4/lexical-sentence/plaba/metrics.json)

### Semantic contradiction report

20 story sets, 180 readers, 1,800 trial uses the original story family held-out. The probability of human first reporting the contradiction-target sentence increased by 48.27 percentage points (from 39.10 to 57.26). The point estimates for all three backbones were approximately +0.04, comparable in magnitude to the surface baseline, and all fell outside the pre-specified ±10 percentage point tolerance interval. The full-text NLL for Qwen and LFM3 was slightly lower than the surface baseline, while Gemma's improvement interval crossed zero. Considering full-text error across all sentences may be small, yet it could still miss a substantial human response to a specific sentence. [First report of common reading stop and same-sentence report](../../../research/results/backbone-transfer-l4/frick/separator_end/results.json)

LFM3 merges the end-of-sentence tokens of 584 non-final sentences with the original two newlines into a single token, misaligning the original characters reading stop. With the main comparison of the three models shifted to after the original newline, no non-whitespace characters from the next sentence are read. When the original reading stop is retained, both Qwen and Gemma achieve slightly better overall NLL than the surface baseline, yet both still substantially underestimate the target effect. This result only validates the prompted sentence-by-sentence first report. Existing scores cannot validate the token where humans first stalled, the earlier text referenced by the question, or the location of the same question resolution. [Original sentence-end side-by-side comparison](../../../research/results/backbone-transfer-l4/frick/unicode_end/results.json) · [Token-level comparison](../../../research/results/backbone-transfer-l4/frick/results.json)

## Verified execution scope

All three backbones extract the final-layer states using frozen official text weights under a no-reader-background condition. Each backbone reads 9,284 texts in FP32 on Colab L4, performing a single causal forward pass per text. All 10 independent truncation and suffix replacement validations pass. for each text, exactly one reading stop observation is collected, with no autoregressive decoding throughout. The readout head is retrained separately on the CPU using each backbone's original training fold. This round does not compare attention, intermediate logits, backbone fine-tuning, or joint training effects. The held-out set uses existing public materials while preserving each backbone's native tokenizer and hidden width, so differences reflect overall model configuration performance. [Execution Validation](../../../research/results/backbone-transfer-l4/run-verification.json)

| backbone | Pre-training forward and representation extraction | Peak GPU retention | L4 configured to confirmed release |
| --- | ---: | ---: | ---: |
| Qwen3.5-2B | 183.9 seconds | 7.36 GiB | 341 seconds |
| LFM3-1.7B-Base | 175.4 seconds | 6.71 GiB | 340 seconds |
| Gemma 4 E2B | 214.4 seconds | 17.97 GiB | 553 seconds |

Three L4s have been confirmed released. Speed and memory are single measurements of the fixed data and batch for this run. Gemma uses the text-only model from the original checkpoint. LFM3 uses compatible layer retention with original layer order and RoPE parameters, maintaining original hash for official files and weights. The sealed snapshot represents replay scores, with output hashes consistent across runs. The Qwen prior representation, after refitting within the same group, shows CompLex MAE differing by approximately 0.000008 from this run, and PLABA-EVAL's two NLLs differing by approximately 0.000019 and 0.0000002, respectively. [Execution and replay manifest](../../../research/results/backbone-transfer-l4/run-verification.json) · [Qwen sealed snapshot comparison](../../../research/results/backbone-transfer-l4/qwen-replay.json) · [LFM3 load details](lfm3-source.md)

[Original research record](../../../research/backbone_transfer/results.md)
