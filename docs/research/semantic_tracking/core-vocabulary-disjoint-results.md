# The new resampling path reproduces the group prediction gain of sentence-by-sentence guessing

Using the same Qwen3.5-2B and fixed readout, each of the eight possible paths for every held-out target was replaced with 32 mutually exclusive new paths. With Qwen included, the held-out NLL decreased from 1.86705 (literal readout) to **1.86230 nats/trial**, an improvement of 0.00475, with a 95% confidence interval for the target word match of [0.00305, 0.00647]. Relative to using only steps, there was also an improvement. The cumulative accuracy deviation for sentences 1, 3, and 6 remained outside the ±5 percentage point acceptance interval. [Fixed specification](core-vocabulary-disjoint-protocol.md) · [Complete held-out results](../../../research/results/core-vocabulary-disjoint-l4/results.json)

| Possible paths / words | Sentence-order NLL | Literal readout of NLL | Add Qwen's NLL | Relative improvement over baseline and target word matching 95% interval |
| --- | ---: | ---: | ---: | ---: |
| Eight Pathways | 1.86765 | 1.86717 | 1.86328 | +0.00388 [0.00209, 0.00571] |
| 32 new pathways | 1.86765 | 1.86705 | 1.86230 | +0.00475 [0.00305, 0.00647] |
| Merge 40 paths | 1.86765 | 1.86706 | 1.86248 | +0.00458 [0.00290, 0.00629] |

Three groups used the same 32 target words not used in fitting, along with 195 historical readers and 1,603 trials. NLL was applied with equal weighting to each target. lower values indicate higher probability of the actual human-observed outcomes. Across 32 new paths, the relative improvement over using only step count was 0.00536 nats/trial, with a target word matching interval of [0.00319, 0.00757]. When re-sampling readers and target words, the relative literal improvement interval was [0.00276, 0.00675]. The difference in literal improvement between new and old paths was +0.00086, with a cross-re-sampling interval of [−0.00012, +0.00185]. Both new and original paths passed relative comparison, and the difference in gain between the two groups remains uncertain. [Matching and Cross-Re-sampling Results](../../../research/results/core-vocabulary-disjoint-l4/results.json)

| Accumulated exposure of new paths | human accuracy | Qwen accuracy | Model reduction human, cross 95% interval (percentiles) |
| --- | ---: | ---: | ---: |
| Sentence 1 | 9.35% | 7.55% | −1.80 [−5.36, +1.21] |
| Sentence 3 | 35.82% | 34.45% | −1.37 [−7.15, +4.22] |
| Sentence 6 | 69.89% | 71.50% | +1.61 [−3.91, +7.31] |

This study examines the impact of possible-path sampling on group prediction. For instance, two possible paths to the same target may be expressed in different wording in the third sentence, yet the trial file only records whether the reader answered correctly at step three. Even if the group prediction holds for both path sets, the reader's correct response cannot be assigned to a specific source text position. [Original file and columns](core-vocabulary-source.md)

## Re-running and resource

The Rust Qwen tokenizer verified 6,144 sentence ends in 1,024 cue texts. The L4 extracted 93,122 tokens in 32 batches. Forward passes and readout took 38.20 seconds, peak GPU reservation was 7.83 GiB, and allocation through verified release took 144.15 seconds. All 12 causal controls passed. Replaying 768 original predictions from the original weights reproduced each value, and reruns of this round's predictions, seal, and scoring were bitwise identical. A separate recomputation used scalar hazard conversion and trial counts to verify the nine NLL values and the cumulative rates for the three path groups. [Seal and release records](../../../research/results/core-vocabulary-disjoint-l4/run-verification.json) · [Replay and scoring code](../../../research/semantic_tracking/core_vocabulary_disjoint_score.py)

[Original research record](../../../research/semantic_tracking/core-vocabulary-disjoint-results.md)
