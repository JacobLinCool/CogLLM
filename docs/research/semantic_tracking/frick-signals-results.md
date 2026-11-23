# per-token vocabulary and attention signal end-of-sentence contradiction report verification

In Frick and Schüler's 20 English story pairs, the probability of a contradictory report at the first keystroke on the same target sentence was 48.27 percentage points higher for inconsistent contexts than for consistent contexts. The readout estimate for the original seven features was 18.95, and 16.36 after adding lexical and attentional signals. The main effect did not pass the target sentence magnitude improvement threshold. [fixed-specs](frick-signals-protocol.md) · [human-observation-and-material-alignment](frick-consistency-source.md) · [sealed-snapshot-results](../../../research/results/frick-signals-l4/results.json)

## held-out results

For each fold, we held out two versions of the same story. All 1,800 human trajectories were retained, with the first keypress position calculated from the second sentence onward, and no report at the end of the story listed as an unobserved event. The condition difference was first estimated within each story, then averaged across 20 story groups using equal weighting. Lower NLL in the table indicates a higher probability that the model assigned to the actual position of the human first report. Intervals were generated via 10,000 cross-validation resamples across stories and readers. [ sealed snapshot prediction](../../../research/results/frick-signals-l4/forecasts.jsonl) · [fitting configuration](../../../research/results/frick-signals-l4/fits.json)

| Readout | The first report of the target sentence (percentage points) | Absolute error improvement relative to the readout, with 95% confidence intervals (percentage points) | The entire NLL (nats/essay) | Relative NLL difference from the readout and 95% confidence interval |
| --- | ---: | ---: | ---: | ---: |
| Original seven features | +18.95 | — | 1.5939 | — |
| Vocabulary and Attention | +16.36 | −2.59 [−5.36, +0.03] | 1.5497 | −0.0441 [−0.0872, +0.0020] |
| Vocabulary only | +15.25 | −3.70 [−6.70, −1.08] | 1.5902 | −0.0037 [−0.0416, +0.0362] |
| Attention only | +20.01 | +1.06 [−0.05, +2.20] | 1.5640 | −0.0299 [−0.0526, −0.0059] |

The primary model met the whole-passage NLL noninferiority criterion against the original readout. The upper bounds on early-report error in the two conditions were +10.94 and +11.56 percentage points, outside the prespecified ±10-point range. In the prespecified comparison adding attention alone, whole-passage NLL was lower than for the original readout, and early reporting met the equivalence criterion in both conditions. The interval for improvement in the target-sentence effect crossed zero, and the predictions still underestimated the human difference by about 28.3 percentage points. These data have already informed several rounds of Frick model comparisons. The attention gain needs validation on new materials. [Criteria and condition-specific distributions](../../../research/results/frick-signals-l4/results.json)

Supplementary diagnosis is restricted to readers who have not yet reported the target sentence in the goal story. The human model and the four readout models all show the same directionality across 20 story sets, with the inconsistent version exhibiting a higher probability of reporting. The conditional report rate for the human model's inconsistent version averages 59.14%, compared to 30.15% for the original seven features, 27.74% when combining both signals, and 30.73% when adding only attention. Even after removing all early reports of the inconsistent version and setting the first report of the target sentence for the consistent version to zero, the fixed attention readout can only achieve a 30.73 percentage point increase in the target sentence probability, which remains lower than the human overall rate of 48.27. Calibration to an earlier position alone cannot correct this gap. The conditional probabilities for each model are derived directly from the sigmoid of the target sentence logits in the sealed snapshot, with the human denominator representing the number of individuals who reached the target sentence in each story version without yet reporting. [Supplementary Diagnosis Method](frick-target-hazard-diagnostic.md#target-sentence-decomposition) · [Current Sealed Snapshot Prediction](../../../research/results/frick-signals-l4/forecasts.jsonl)

## Forward, Coordinate, and Resource

Use the same frozen Qwen3.5-2B candidate with Gradio. Perform causal forward pass once on each of the 40 texts, yielding 9,386 tokens of four signals and 624 end-of-sequence states. Verify that the hidden states extracted during signal acquisition match the original forward pass value-by-value. When substituting unseen suffixes, the four signals from the prefix also match value-by-value. When truncating unseen suffixes, the maximum absolute difference among the four signals does not exceed 0.000013. Independently reconstruct full document coordinates via tokenization and verify that 20 target sentence groups have identical tokens. Each group is fitted with the same original end-of-sequence state. the re-extracted states are used for verification. The original readout is exactly reproduced, and after removing the held-out material's target and report counts, the newly read predictions are also identical. [Forward check](../../../research/results/frick-signals-l4/causality.json) · [Independent coordinate check](../../../research/results/frick-signals-l4/independent-prefix-check.json) · [Readout verification](../../../research/results/frick-signals-l4/axis-audit.json)

L4 forward and readout phases total 10.06 seconds, fully capturing 36.07 seconds, with peak retained display memory at 7.13 GiB. From configuration to confirmed release took 113.73 seconds. Two local CPU fits and evaluations each took approximately 6.5 seconds, with peak RSS around 235 MiB, and the sealed snapshot products were bit-for-bit identical. Additionally, the sealed snapshot logits and per-step reports were recomputed for each group's magnitude and the full NLL using standard library functions. [Capture specifications and hash](../../../research/results/frick-signals-l4/extraction.json) · [Independent evaluation and release manifest](../../../research/results/frick-signals-l4/run-verification.json)

Each token possesses model signals traceable back to the source text. Human responses are positioned after the full sentence is displayed, allowing evaluation of which sentence triggered the first report and its probability, while the intra-sentence word triggering the first detection remains unidentifiable, and neither the report's reference to a prior sentence nor its temporal resolution is queried. This readout is retained as the study result, while Gradio's sentence-level and lexical displays continue to utilize their original readout. [Original study's procedure and measurement](https://doi.org/10.3758/s13421-026-01900-7)

## Reproduce

Prepare human files based on the [source family](frick-consistency-source.md#reproducible-setup). Extract the `prepare-frick-signals` using `cuda_verification_batch.py`, constructing model inputs from the `qwen-packet/stimuli.jsonl` of the original extracted packet and the verified Gradio L4 working directory. The scoring can directly reuse the L4 sealed snapshot. The following operations are performed within the research project directory. install NumPy and the sealed snapshot version of Hugging Face Tokenizers. [L4 execution entry](../../../research/gradio_app/cuda_verification_batch.py) · [signal extraction](../../../research/semantic_tracking/frick_signals_extract.py) · [nested scoring](../../../research/semantic_tracking/frick_signals_score.py)

```sh
frick_cache=.cache/cogllm/semantic-tracking/frick-consistency
frick_old="$frick_cache/l4/cogllm-cuda-checks-20261121T041742Z-f6cc9d6e/outputs/projection"
frick_new="$frick_cache/signals-l4/cogllm-cuda-checks-20261121T072128Z-845a9a12/outputs/frick-signals"
frick_tokenizer="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc/tokenizer.json"
python3 -m research.semantic_tracking.frick_signals_score \
  --assay "$frick_cache/prepared" \
  --archived-extraction "$frick_old" \
  --archived-packet "$frick_cache/qwen-packet" \
  --signals "$frick_new" \
  --signal-stimuli "$frick_cache/signals-input/stimuli.jsonl" \
  --signal-assay research/results/frick-consistency-l4/extractor-preflight.json \
  --signal-preflight research/results/frick-signals-l4/extractor-preflight.json \
  --tokenizer "$frick_tokenizer" \
  --output "$frick_cache/signals-reproduction"
```

[Original research record](../../../research/semantic_tracking/frick-signals-results.md)
