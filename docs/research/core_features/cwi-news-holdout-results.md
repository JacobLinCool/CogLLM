# Gradio vocabulary score did not pass the professional news held-out

The current Gradio checkpoint achieves a word-ordering agreement of **65.67%** on the CWI 2018 professional news corpus. The baseline for the same sentence, combining word length and reverse word frequency, is **75.25%**, yielding a difference of −9.58 percentage points with a matched 95% confidence interval [−14.79, −4.52]. Of the 979 evaluable candidates, all returned to the original text, and none of the three ordering thresholds were met. [Scheduled threshold](cwi-news-holdout.md) · [Full scoring](../../../research/results/cwi-news-holdout-l4/results.json) · [Data author paper](https://aclanthology.org/W18-0507/)

## Materials and Methods

The official news test split comprises 28 HITs, 177 original sentences, and 2,095 candidate spans. Excluded are sentences overlapping across HITs, those identical to prior CWI material or meeting the fixed-fragment overlap threshold, and candidates not mapping exactly to a single complete interface token. Applying the same rules to the already-identified CompLex train, trial, and test sets revealed no overlaps. The final retained set includes 22 HITs, 100 original sentences, and 979 tokens. The original annotators reviewed the paragraphs. the current release file allows verification of original sentences and HITs but does not provide a verifiable cross-sentence presentation order. [Source, Coordinates, and Overlap Audit](../../../research/results/cwi-news-holdout-l4/assay.json)

Gradio candidates and the verified full-text interface share the same checkpoint, inference script, and CompLex suffix head. Each sentence is run forward once to obtain the interface's vocabulary score rounded to two decimal places. The human score represents the 20 annotators' selections across the two groups in the original study. Comparisons are made across 2,856 word pairs with different rankings within the same sentence. ties receive 0.5 points. Scores are first averaged across sentence pairs, then averaged across sentences within the same HIT, and finally weighted by HIT with equal weighting. [Interface validation](../../../research/results/gradio-combined-overview-l4/run-verification.json) · [Predict sealed snapshot](../../../research/results/cwi-news-holdout-l4/seal.json)

## Comparison

| Readout | Intra-sentence ordering consistency | Difference of Gradio relative to baseline, matched 95% confidence interval |
| --- | ---: | --- |
| Gradio vocabulary score | 65.67% | — |
| Longer | 72.50% | −6.83 percentage points [−13.63, −0.06] |
| Inverse frequency | 67.66% | −1.99 percentage points [−6.72, +2.47] |
| Merging word length with inverse word frequency. | 75.25% | −9.58 percentage points [−14.79, −4.52] |

Of the 2,856 total pairs, 111 pairs had identical interface scores. Assuming these 111 pairs could all be correctly ranked while maintaining all other orderings, the agreement rate would increase to at most 67.34%, which remains below the combined baseline by 7.92 percentage points. When calculated separately for the two annotator groups, Gradio yielded 66.63% and 65.12%, with combined baselines of 75.04% and 75.61%. This result is limited to professionally evaluated candidates and the model's original sentences and cannot estimate individual reader comprehension intensity. [Independent verification and upper bound of ties](../../../research/results/cwi-news-holdout-l4/independent-check.json) · [Grouped results](../../../research/results/cwi-news-holdout-l4/results.json)

`cwi_news_prepare` Reconstruct the original sentence and separately save the scores. L4 exports the two shared held-out Gradio vocabulary scores using `medreadme_lexical_forecast`, then scores them with `cwi_news_score`. After running 100 forward passes, resource allocation to verification and release takes approximately 100 seconds. The two local scoring runs are bitwise identical, each taking about 0.02 seconds, with peak RSS around 100 MiB. An independent program recalculates all denominators, matching effects, and intervals. [Data Preparation](../../../research/core_features/cwi_news_prepare.py) · [Scoring](../../../research/core_features/cwi_news_score.py) · [Execution and Release Manifest](../../../research/results/cwi-news-holdout-l4/run-verification.json)

[Original research record](../../../research/core_features/cwi-news-holdout-results.md)
