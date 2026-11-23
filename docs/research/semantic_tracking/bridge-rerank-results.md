# Cross-span context-aware readout alignment results

Using the same causal Qwen3.5-2B representations, the cosine readout with expanded lexical features assigned higher probability to human-annotated locations than the original cosine readout on both GUMBridge and GENTLE. The per-dimension paired features did not pass this comparison. On GENTLE, NLL for the per-dimension readout was **0.441 nats** higher than for the expanded cosine readout, with a document-paired 95% interval of [0.018, 0.863]. Its top-five hit rate was **25.6 percentage points** lower, with the interval [−48.7, −5.1]. [Fixed comparison protocol](bridge-rerank-protocol.md) · [Full scores and paired intervals](../../../research/results/bridge-rerank/scores.json)

| Readout | GUM test NLL ↓ | First ↑ | The top five ↑ | GENTLE NLL ↓ | First ↑ | The top five ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Literal translation | 4.732 | 3.6% | 8.6% | 4.423 | 0.0% | 2.6% |
| Original cosine | 4.197 | 6.9% | 22.7% | 4.262 | 0.0% | 15.4% |
| Expand the literal | 4.013 | 5.0% | 15.4% | 3.910 | 11.5% | 13.5% |
| Expand literal and cosine | 3.590 | 15.0% | 46.3% | 3.881 | 0.0% | 32.1% |
| Dimension-wise matching | 3.500 | 18.5% | 43.0% | 4.322 | 0.0% | 6.4% |
| Dimension-wise matching and manual part-of-speech tagging | 3.246 | 11.4% | 44.9% | 3.976 | 0.0% | 15.4% |

NLL is the negative log probability, in nats, of the total probability assigned to all acceptable word ends, with a minimum of 0. Hit rate is the proportion of articles where at least one candidate word end is marked, ranging from 0–100%. Both metrics are first averaged within articles, then equal-weighted averaged across articles. Training consists of 114 articles and 530 queries. GUM test comprises 19 articles and 89 queries. GENTLE includes 13 articles and 31 queries. The materials were previously evaluated. this study uses only the GUM training split to fit new coefficients, serving as a retrospective control. [Training weights and material hash](../../../research/results/bridge-rerank/fit.json) · [Scoring script](../../../research/semantic_tracking/bridge_rerank.py)

## The two sources of low hit rates

**The candidate split probability for the current passage.** All positives are in earlier passages. Expanding the relative cosine NLL improvement over the original cosine yields 0.109 nats in the GUM test from increased total probability of earlier passages and 0.498 nats from ranking within earlier passages. In GENTLE, the corresponding values are 0.213 and 0.169 nats. Part of the improvement stems from the cross-passage conditioning itself. Adding cosine to the same expanded literal features in GENTLE results in an additional NLL decrease of only 0.029 nats, with an interval of [−0.399, 0.313]. [Probability decomposition and feature-matched comparison](../../../research/results/bridge-rerank/scores.json)

**Part-of-speech diagnosis can miss misalignments and also overlook manual direction.** Restricting the original cosine candidates to manually annotated nouns, proper nouns, pronouns, and numerals, GENTLE's top hit ranged from 0.0% to 12.8%. This restriction also caused 2 out of 31 queries to lose all acceptable answers, with GUM testing showing 7/89. Using manual part-of-speech as input also reduced the NLL of per-dimension matching, yet GENTLE's top hit remained at 0.0%. [Part-of-speech diagnosis and no-issue target count](../../../research/results/bridge-rerank/scores.json)

App currently, based on IIRC's held-out evidence, exhibits in-scope fragments. Cross-sentence comparison only evaluates the preceding context under known new references. The initial observation of uncertainty and subsequent resolution remain lacking corresponding direct observations. [App Verified Fragment Task](../iirc-anchor-run.md) · [Public Human Report Data Audit](reader-report-source-audit.md)

The same experiment was run twice sequentially on a single-threaded CPU, with weights and scores matching bit-for-bit. The four newly read-out coefficients matched the per-item values before post-hoc diagnostics. Each run took approximately 3.3 seconds, with peak memory usage around 295 MiB. Precomputed representations were used, and the model was not loaded locally. [Re-run entry](../../../research/semantic_tracking/bridge_rerank.py)

[Original research record](../../../research/semantic_tracking/bridge-rerank-results.md)
