# Post-training verification of the MECO passage

The joint readout of Qwen3.5-2B at epoch 31 on Traditional Chinese MECO shows lower post-document understanding accuracy compared to the fixed baseline in zero-shot transfer. The average negative log-likelihood per response is **0.140 nats** higher, with a 95% confidence interval from cross-validation of reader and document as **[0.070, 0.206]**. This set of 12 articles supports the transfer variance. [Score original](../../../research/results/union-observations-formal-l4/meco/scores.json) · [Score specification](formal-meco-plan.md)

| Metrics | CogLLM | OneStop fixed baseline |
| --- | ---: | ---: |
| Average Bernoulli negative log-likelihood, nats/response, lower is better | 0.622 | 0.482 |
| Brier score, range 0–1, lower is better | 0.215 | 0.152 |
| Average predicted accuracy | 56.12% | 80.05% |
| Average predicted accuracy minus human accuracy, in percentage points. | −25.20 | −1.27 |

The dataset contains 60 Hong Kong readers, 12 Traditional Chinese expository texts, and 1,440 post-reading true-or-false responses for the horizontal presentation condition. Human readers answered 1,171 correctly (81.32%). With only the passage as input, the model ran one causal forward pass per text on Colab L4, then used a readout trained on initial-reading correctness in the English OneStop data to produce one probability at article end. The fixed baseline used the 1,208 correct responses out of 1,509 in the OneStop training fold. This evaluation tests the transfer of that post-reading comprehension readout across language, reader population, and question format. On article 6, readers answered 115 of 120 trials correctly (95.83%) and the model assigned the article 56.23%. [Study methods and data](https://doi.org/10.1038/s41597-026-06989-8) · [Results by passage](../../../research/results/union-observations-formal-l4/meco/scores.json)

Before reading the sequential responses, 12 predictions were locked. The initial scoring data key check was halted because the RDA `Session_Name_` encoded readers and presentation direction. the discourse should align with `number` and the actual presentation filename. Post-correction, alignment was verified using synthetic responses in the original design fields. Neither of the two aborts yielded scores, and the corrected prediction file and the original locked file were identical byte-for-byte. For 27,167 documents in the development and training materials, a full-text comparison against a continuous 24-character substring yielded no overlaps. [Original Lock](../../../research/results/union-observations-formal-l4/meco/forecast/forecast-seal.json) · [Alignment Correction Log](../../../research/results/union-observations-formal-l4/meco/adapter-verification.json) · [Text Overlap Check](../../../research/results/union-observations-formal-l4/meco/training-text-overlap.json)

[Original research record](../../../research/union_training/formal-meco-results.md)
