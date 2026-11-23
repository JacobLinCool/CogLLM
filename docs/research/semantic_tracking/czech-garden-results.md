# Czech disambiguation readout with held-out results

In the third experiment, which presented sentences word by word, the 24 entirely held-out item groups included 1,281 `qmis` trials with recorded responses. The ambiguous version increased the human rate of accepting the initial misinterpretation at sentence end by **23.23 percentage points**, with a 95% interval of **[16.38, 30.63]**. Combining the ordered word states from the text with the sentence-final state yielded an estimated effect of **7.30 percentage points**, with a 95% interval of **[6.50, 7.99]**. The model effect minus the human effect was −15.93 percentage points, with an interval of [−23.60, −8.85], which did not meet the ±5 percentage point criterion. The readout also missed all three prespecified NLL improvement criteria. [Prespecified protocol](czech_garden_protocol.md) · [Sealed holdout scores](../../../research/results/czech-garden-l4/results.json)

| Readout at the end | NLL (nats/keystrokes, lower is better) | The absolute difference in ambiguity-reduction control, measured in percentage points. |
| --- | ---: | ---: |
| Constant | 0.62927 | 0.00 |
| Literal and 4th word suffix | 0.61622 | +7.58 |
| Append the Qwen end-state. | 0.61246 | +7.66 |
| word-level aggregation with corpus-level ordering | 0.61491 | +7.30 |
| The researcher specified the 7th word state as poor. | 0.61431 | +7.75 |
| Use the author's ambiguity and vitality labels. | 0.59193 | +9.99 |
| human | — | +23.23 |

The ordered-text model's NLL minus the lexical baseline was −0.00131 nats per keypress, with a 95% interval of [−0.00874, +0.00553]. The difference from the sentence-final-state baseline was +0.00245 [−0.00280, +0.00859]. Fitting used only 12 verb groups, 24 items, and 1,380 responses from the second experiment, which presented entire sentences. The third experiment used 12 different verb groups, 24 items, and 159 readers. We retained 'I don't know' as a third response category and dropped five trials with no recorded keypress. In the second experiment, the human effect was +13.00 percentage points on the training items and +17.18 percentage points on the held-out items. The two experiments used different participants and presentation methods. An independent recalculation verified the sealed test probabilities and crossed resampling intervals for all six readouts. [Denominators and source](czech-garden-path-source.md) · [Split](../../../research/results/czech-garden-l4/split.json) · [Paired intervals](../../../research/results/czech-garden-l4/results.json)

## Fixed English disambiguation readout

In 96 Czech sentence pairs, the versions differ only at word 4 and share word 7. The Gradio readout previously successful in English yielded a state-score difference of only **+0.00088** at word 7, with a 95% interval of **[−0.0330, +0.0371]**. The difference at word 7 was **0.0740** below the mean difference at words 5 and 6, with an interval of **[−0.1413, −0.0062]** for this contrast. Neither prespecified criterion was met. At word 4, where different word forms actually appeared, the model's state scores already differed by +0.201. The largest score difference over the shared prefix at words 2 and 3 was about $3.15\times10^{-6}$, within the causal alignment tolerance. [Complete word-level output](../../../research/results/czech-garden-l4/results.json) · [Original study, Experiment Three and Figure Two](https://doi.org/10.1371/journal.pone.0288817)

The author disambiguation position for all sentences is the 7th word. Only 2/192 sentences are fixed to the first position by English alphabetical ordering. Direct specification of the 7th word accounts for 192/192, reflecting the stimulus design. The original study found a difference in reading times at positions 7–9, whereas the current English readout transfer results lack a pre-registered response for the 7th word. The disparate positioning of the endpoints and the morphological differences warrant separate reporting. the reader's own awareness or resolution position remains unmeasured directly. [Position score](../../../research/results/czech-garden-l4/results.json) · [Observation scope of response and word-by-word reading](czech-garden-path-source.md)

## Joint proportion boundary of two end-of-sentence interpretations

Each subject received only one question per topic. Additionally, using all 48 topic sets and two vitality conditions, the hypothetical joint response range was estimated from the "yes" proportions of each question. The goal was the proportion of the same reader answering "yes" to both the true headword and the original misinterpretation when asked separately. Assuming that single-question valid responses from each question represent comparable reader populations under the same condition, and assuming that describing the potential joint responses with these proportions maintains the marginal proportions for each question, the Fréchet formula $\max(0,p_1+p_2-1)$ and $\min(p_1,p_2)$ were calculated within the same topic, vitality, and version, followed by an equal weighting average. [Original study, each person answered only one question per topic](https://doi.org/10.1371/journal.pone.0288817) · [Estimation procedure](../../../research/semantic_tracking/czech_garden_identification.py)

| Experiments and conditions | Estimated joint answer lower bound | Estimated joint answer upper bound |
| --- | ---: | ---: |
| Present the full sentence in its ambiguous form. | 12.75% | 22.21% |
| Full sentence presentation, control version | 3.55% | 7.15% |
| Present word by word, ambiguous version | 17.61% | 28.01% |
| Word-by-word presentation, controlled version | 5.69% | 8.79% |

The word-by-word ambiguity effect shows a margin of **10.40 percentage points**. The estimated lower bound individual 95% bootstrap confidence intervals are [14.58%, 24.44%], and the upper bound is [22.67%, 33.53%]. These are the intervals for each endpoint separately. These aggregations support the evaluation and constraints on the joint distribution hypothesis, yet they cannot identify which reader is likely to simultaneously answer "yes," nor can they determine when the interpretation emerges or disappears during the reading process. [denominator, assumptions, and 10,000 cross-validated bootstrap resamples](../../../research/results/czech-garden-identification.json)

In Colab L4, perform causal forward pass on 192 sentences, extract 2,304 FP32 reading stops, with 12 independent causal checks passing. The L4 forward and readout phases total 2.74 seconds, while the extraction phase takes approximately 43 seconds. Re-run using a local sealed snapshot representation. six score outputs are bit-wise identical, and Qwen was not loaded locally. L4 has been confirmed released. Responses, original stimulus text, and raw representations are retained only in local working cache. [Acceptance and resource manifest ](../../../research/results/czech-garden-l4/run-verification.json) · [ sealed snapshot prediction and coefficients](../../../research/results/czech-garden-l4/forecast-seal.json)

[Original research record](../../../research/semantic_tracking/czech-garden-results.md)
