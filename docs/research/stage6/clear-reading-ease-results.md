# CLEAR English readability held-out results

On 410 English materials that did not participate in calibration, incorporating fixed Gradio candidate sentence ease into surface features improved the Spearman correlation between discourse ranking and teacher ease estimates from **0.608** to **0.691**. The difference was **+0.083**, with a 95% confidence interval from resampled source pairing of **[+0.051, +0.118]**, passing a pre-specified threshold. Using the current App sentence-level score directly for a count-weighted average, without fitting in CLEAR, yielded a correlation of **0.661**, which is relatively +0.053 [+0.012, +0.096] higher than surface readout. [fixed specification](clear-reading-ease-protocol.md) · [sealed snapshot metric](../../../research/results/clear-reading-ease-l4/metrics.json)

| held-out materials | Number of passages | Directly use App scores. | Surface readout | Surface Augmentation | Surface area-adjusted improvements and 95% confidence intervals |
| --- | ---: | ---: | ---: | ---: | ---: |
| All | 410 | 0.661 | 0.608 | 0.691 | +0.083 [+0.051, +0.118] |
| Expository text | 291 | 0.635 | 0.537 | 0.690 | +0.153 [+0.092, +0.218] |
| Literature | 119 | 0.101 | 0.067 | 0.133 | +0.066 [−0.048, +0.180] |

Two calibration readouts used the same 600 documents, with no cross-splitting between source components. Surface features were the log of body blank word count, average words per sentence, average English word length, and proportion of long words. Relative surface readout held-out RMSE decreased from 0.821 to 0.734, in units of the Bradley–Terry score published by CLEAR. Excluding the single calibration group document sharing 12–29 words with existing materials, and refitting, the correlation gain remained +0.083 [+0.051, +0.118]. Point estimates, both RMSEs, and the primary contrast's 5,000 bootstrap intervals were all independently recalculated. The two full scoring products were identical byte-for-byte. [Data and split](clear-reading-ease-source.md) · [Calibration denominator and weights](../../../research/results/clear-reading-ease-l4/calibration.json) · [Independent check](../../../research/results/clear-reading-ease-l4/independent-check.json)

This endpoint represents pairwise comparisons of difficulty, based on the teacher's review of the full excerpt, for readers ranging from third grade to high school. Published scores can be used to verify discourse ranking. English expository texts show a clear gain, while the gain interval for literature spans zero. Since this data lacks item-level comprehension or difficulty reports from actual students, it cannot be used to calibrate the app's single-sentence color coding or to pinpoint the onset and resolution of questions. This calibration readout is provided solely for verification purposes. [CLEAR original study Study 1](https://link.springer.com/article/10.3758/s13428-022-01802-x)

## Forward and interface validation of the same candidate

The same Qwen3.5-2B candidate was analyzed on a single Colab L4 instance across 1,010 documents. For each document, a single FP32 causal forward pass was executed. all 12,743 end-of-document source coordinates and Qwen token indices matched the pre-sealed snapshot results exactly, with scores derived from the official App's end weights. Three additional prefix contrast checks were performed by truncating unread suffixes. the maximum difference in scores among read sentences was 0.00000015. The model received the main text and localization data, and the teacher estimated that after retrieving predictions, native scoring would be applied locally. [extract manifest ](../../../research/results/clear-reading-ease-l4/extraction.json) · [sentence-wise sealed snapshot ](../../../research/results/clear-reading-ease-l4/forecasts.jsonl) · [prefix verification ](../../../research/results/clear-reading-ease-l4/causality.json)

Two test articles were submitted sequentially using the same model program in Gradio, each with a single forward pass. After replacing the articles, the earlier text pointer was cleared, and both sentences were displayed with background coloration and 18 word underlines. A total of eight subsequent selection and inspection operations were performed with zero forward passes. The original Unicode, marked positions, and the updated "model estimate" and "intra-document relative scale" documentation were verified. [Actual full-text coloring](../../../research/results/clear-reading-ease-l4/preview.html) · [HTTP acceptance](../../../research/results/clear-reading-ease-l4/ui-verification.json)

The forward readout duration was 114.01 seconds, with a total experiment duration of 160.67 seconds. Peak GPU retained memory was approximately 7.09 GiB. From L4 configuration to release confirmation took 219.01 seconds. Based on the 1.94 CU/hour rate shown during the configuration period and account display, an estimated 0.118 CU was consumed. however, the actual account deduction cannot be verified from the precision of this view nor the posting time. [Execute, usage, and release manifest](../../../research/results/clear-reading-ease-l4/run-verification.json)

## Re-run

Rebuild the three data files using the [pinned workbook and preparation code](clear-reading-ease-source.md#formal-material-preparation) and verify their hashes against the original preparation receipt. On the remote host, run `cuda_verification_batch prepare-clear-ease` to bind the verified Gradio candidate to the CLEAR reading stops, then execute the resulting workspace. From the repository root, use the retrieved L4 package below to recompute scores without loading the language model. [Colab entry point](../../../research/gradio_app/cuda_verification_batch.py) · [Scoring entry point](../../../research/stage6/clear_reading_ease_score.py)

```sh
clear_root=.cache/cogllm/clear-reading-ease
clear_l4="$clear_root/l4/cogllm-cuda-checks-20261121T074802Z-8726bc98/outputs/clear-ease"
python3 -m research.stage6.clear_reading_ease_score \
  --prepared "$clear_root/prepared-v1" \
  --forecasts "$clear_l4/forecasts.jsonl" \
  --seal "$clear_l4/seal.json" \
  --output "$clear_root/scoring-reproduction"
```

[Original research record](../../../research/stage6/clear-reading-ease-results.md)
