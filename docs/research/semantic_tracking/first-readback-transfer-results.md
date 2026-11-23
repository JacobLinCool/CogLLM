# Only input the first readout held-out results

A set of readouts trained on English-German merger, after incorporating Qwen states, more accurately estimated the first active recall on two unseen materials compared to the position baseline. The Brier difference intervals for both datasets were below zero. The German mean probability passed the pre-specified ±1 percentage point calibration threshold, while the English upper bound was +1.06 percentage points. Overall, neither dataset met the rule requiring both to pass. [Scoring and Pre-specified Thresholds](../../../research/results/first-readback-transfer-l4/scores.json) · [Pre-specified Protocol](first-readback-transfer.md)

| held-out materials | Key readout / valid reading stop | Position Brier | Qwen + Position Brier | Difference and 95% material family interval | Mean probability calibration difference and 90% interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| English, 5 groups | 114／2,717 | 0.03963 | 0.03758 | −0.00205 [−0.00315, −0.00084] | +0.51 [−0.09, +1.06] percentage points |
| German, 16 groups | 381／8,226 | 0.04417 | 0.04188 | −0.00229 [−0.00307, −0.00167] | +0.05 [−0.47, +0.60] percentage points |

The Brier score is the mean squared error between predicted probabilities and binary key observations, where lower is better. Calibration difference is the mean predicted probability minus the human key rate. Combined readouts were fitted with equal weighting across the two studies, with resampling within each study by 10,000 iterations using the source family, with a fixed random seed. The original experiment used a stepwise display of keys, with the rating object being the first forward reading. the next step was either a backward key press or forward. Each study retained its original family held-out, allowing readers to cross materials. Of the 538 designed held-out reading stops, predictions were obtained first, and human data were available at 534 of these locations. [Data Audit and Denominator](../../../research/results/first-readback-transfer-l4/assay.json) · [Original English Study](https://doi.org/10.5070/g6011182) · [Original German Study](https://doi.org/10.5070/G601139)

When displayed with a single token, the German Brier score was −0.00161, with a 95% confidence interval [−0.00393, −0.00023], encompassing 3,313 readouts and 144 no-issues. In English, among 867 readouts, only 15 were no-issues, with a difference of −0.00004 [−0.00013, +0.00005]. When trained exclusively on English, the average predicted probability for German was 11.34%, and the human keypress rate was 4.63%. Readout probabilities varied both within and across studies. Current evidence supports the hypothesis that both groups exhibit reading stops, suggesting confusion may occur prior to keypress, and segments may contain multiple tokens. The initiation and resolution of questions in general texts remain to be directly observed and validated. [Complete scores by granularity and training scope](../../../research/results/first-readback-transfer-l4/scores.json)

L4 employs a frozen Qwen3.5-2B FP32 backbone with a verified Gradio candidate. Across 168 segments totaling 8,163 tokens, all 2,344 original display reading stops passed prefix certification via a single model forward pass per segment, with 12 causal checks passed. The new readout fit and prediction output in the remote stage took 44.2 seconds. backbone forward and extraction took 3.69 seconds, and the instance has been verified released. Predictions from the sealed snapshot were then attached to held-out observations locally. Score reruns took 0.54 seconds, with the maximum RSS at 47.2 MiB. Additionally, 36 metric groups were cross-checked using expanded sequential observations. This readout is retained in the experimental results. [Extraction Log](../../../research/results/first-readback-transfer-l4/extraction.json) · [Causal Check](../../../research/results/first-readback-transfer-l4/causality.json) · [Predicted Sealed Snapshot](../../../research/results/first-readback-transfer-l4/seal.json) · [Verification and Release](../../../research/results/first-readback-transfer-l4/run-verification.json)

Re-running this experiment requires only NumPy, [prepared public data](../../../research/semantic_tracking/first_readback_prepare.py), and [sealed snapshot predictions](../../../research/results/first-readback-transfer-l4/predictions.jsonl).

```sh
python3 -m research.semantic_tracking.first_readback_score \
  --assay-root .cache/cogllm/semantic-tracking/first-readback-v1 \
  --predictions-dir research/results/first-readback-transfer-l4 \
  --output /tmp/cogllm-first-readback-replay.json
```

[Original research record](../../../research/semantic_tracking/first-readback-transfer-results.md)
