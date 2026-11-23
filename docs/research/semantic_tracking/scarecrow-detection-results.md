# Self-contradictory word-level localization and annotation degree held-out

On the 109 held-out continuations containing both circled and non-circled words, Qwen3.5-2B showed no improvement in in-sentence localization at the word-level representation. The average in-sentence precision remained approximately **0.326** before and after adding the representation. The negative log-likelihood point estimates for the complete four categories and the severity after circling decreased slightly, with both comparison intervals crossing zero. [prior specification](scarecrow_detection_protocol.md) · [sealed snapshot score](../../../research/results/scarecrow-detection-l4/metrics.json)

| Readout | Intra-document average precision ↑ | Complete four-class NLL ↓ | Post-selection NLL ↓ |
| --- | ---: | ---: | ---: |
| Constant | 0.14149 | 0.065593 | 0.97742 |
| Surface text | 0.32566 | 0.064055 | 0.99458 |
| Surface text + Qwen suffix embeddings | 0.32559 | 0.063992 | 0.99064 |
| Add Qwen subtract text surface, 95% prompt family matching interval | [−0.00485, +0.00565] | [−0.00021, +0.00008] | [−0.01970, +0.01096] |

The mean precision was computed over continuations containing at least one positive label from a qualified annotator, restricted to those with positive and zero votes only. The denominator for the four NLL categories was 226,933 annotator × word pairs, while the denominator for the selection degree was 2,255 votes. The overall selection rate prediction relative to human was −0.174 percentage points, with a 90% confidence interval of [−0.474, +0.081], falling within ±1 percentage point. The 90% confidence interval for the mean selection severity error was [−0.110, +0.167] grade, exceeding ±0.15 grade. [Independent verification](../../../research/results/scarecrow-detection-l4/independent-check.json)

Of the 134 previously unused held-out prompt families, only 2 received positive votes, totaling 9/121,685 votes. Qwen overestimated the selection rate in this group by approximately **0.805 percentage points**. Of the 112 continuations using earlier text to point to a split, 2,246/105,248 received votes. Qwen underestimated by approximately **1.305 percentage points** in this group. The opposite-direction biases make the overall coverage rate appear closer. The first group has too few positive examples, and there is still a lack of independent evidence for the transferability of intra-document positive example localization and severity. [Group denominators and probabilities](../../../research/results/scarecrow-detection-l4/metrics.json)

## Observation and Reproduction

The source annotators can view the full news prompts and continuations. The model received only the publicly released generated continuations, with 205 held-out annotators × masked due to insufficient prior text for word-level prediction. Of 1,229 articles, 983 were fitted and 246 held-out, with identical news prompts or identical continuations appearing in the same fold. The word-end probability represents the posterior probability of being annotated as contradictory. strength reflects the annotators' assessment of the impact of contradiction on text quality or comprehension. The original study did not observe the time to first detection or resolution of the same question. [Original paper §§4.4–4.5](https://aclanthology.org/2022.acl-long.501.pdf#page=6) · [Data sources and masking](../../../research/results/scarecrow-detection-source-audit.json)

The same version of Gradio candidate performed a causal forward pass on 1,229 papers in Colab L4. From 147,461 input tokens, 114,780 precise reading stops were extracted, covering 113,622 word ends. 12 prefix causal checks passed. The forward and readout took 59.93 seconds, while the full extraction took 105.43 seconds. L4 confirmed release. The four core products from two local evaluations matched bit-for-bit, and independent program recalculation of three point estimates showed a maximum difference of $1.59\times10^{-14}$. [Preparation script](../../../research/semantic_tracking/scarecrow_detection_prepare.py) · [Scoring script](../../../research/semantic_tracking/scarecrow_detection_score.py) · [Execution and replay manifest](../../../research/results/scarecrow-detection-l4/run-verification.json)

[Original research record](../../../research/semantic_tracking/scarecrow-detection-results.md)
