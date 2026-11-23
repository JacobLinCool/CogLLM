# Frick's prior matching-supervised story held-out results

Supervised by the direct effect of human pairing in the training stories, the model achieved a report difference of **33.29 percentage points** on the current sentence and **23.30 percentage points** on the first report, compared to the same target sentence. The human differences were **52.06** and **48.27 percentage points**, respectively. When fitting with seven features and all reading stop equal weighting, the two effects were **2.92** and **2.22 percentage points**. Pairing supervision reduced the effect difference, but both endpoints only passed the directional threshold. The magnitude remained low, and the overall negative log-likelihood also failed to pass the pre-specified non-inferiority threshold. [Back-analysis specification](frick-pair-protocol.md) · [sealed snapshot results](../../../research/results/frick-pair/results.json)

## The report magnitude of the target sentence

Retain the causal end-state of the frozen Qwen3.5-2B model. For each of the 20 folds, generate two complete story versions with the held-out set, totaling 1,800 human reading trajectories. For each fold, estimate the difference direction and coefficient using only the remaining 19 story pairs. Each version is input once as the full text, with pairing by story composition applied only during held-out scoring. Report the difference as "inconsistent prior" minus "consistent prior". Within each group, first average within story versions, then apply story equal weighting. The 95% confidence intervals reported in this text are derived via story and reader cross-validation. [Pairing design](frick-pair-protocol.md) · [State and model sealed snapshot](../../../research/results/frick-pair/forecast-seal.json)

| Readout or human observation | When the sentence report is poor, by percentage points. | Difference in first reports at this sentence, percentage points |
| --- | ---: | ---: |
| human | +52.06 [42.07, 61.54] | +48.27 [39.10, 57.26] |
| The original prefix Qwen | +0.057 [0.032, 0.084] | +0.051 [0.034, 0.069] |
| The seven features, all observations equal weighting | +2.92 [2.09, 3.84] | +2.22 [1.56, 2.95] |
| Seven features, weighted target sentence | +16.32 [11.94, 21.28] | +8.12 [5.87, 10.60] |
| The seven features, including the matching effect. | +33.29 [25.23, 42.09] | +23.30 [16.33, 31.18] |
| The model's performance decreased by human after adding the matching effect. | −18.77 [−30.18, −7.41] | −24.97 [−35.90, −13.70] |

Compared to the weighted target sentence only, pairing supervision reduced the absolute error for the current sentence and the first report by 16.97 percentage points [12.91, 21.21] and 15.18 percentage points [10.38, 20.63], respectively. The "model minus human" interval at both endpoints still exceeded ±10 percentage points. The specific error of the current sentence can be examined separately for the two versions. The human report rate for the target sentence in the consistent and inconsistent versions was 4.37% and 56.43%, while the paired readout was 19.52% and 52.82%. The magnitude difference was primarily driven by the higher report rate in the consistent version. [Conditional probabilities and paired intervals](../../../research/results/frick-pair/results.json)

## Whole-probability and first-position

negative log-likelihood (NLL) evaluates the probability assigned by the model to the actual occurring report. The minimum is 0, with lower being better. After adding the matching effect, the sentence NLL relative to equal weighting readout increases. the first report NLL relative to equal weighting and the original prefix readout both have confidence intervals crossing zero. The upper bounds of the latter two intervals are both above the +0.05 nats/article non-inferiority limit. Simply weighting the target sentence increases the full-document NLL of the first report, while the matching term decreases relative to this control by 0.1961 nats/article [−0.3000, −0.0770]. [Readout probabilities and matching NLL](../../../research/results/frick-pair/results.json)

| Endpoint | equal weighting readout NLL | Matching readout NLL | Matching equal weighting, 95% interval | Matching reduced prefix Qwen, 95% interval |
| --- | ---: | ---: | ---: | ---: |
| When sentence, nats/sentence | 0.15269 | 0.18595 | +0.03327 [0.02031, 0.04676] | +0.02840 [0.01526, 0.04215] |
| First, nats per article | 1.64464 | 1.65099 | +0.00635 [−0.13555, 0.15385] | −0.07832 [−0.23963, 0.08365] |

The first report shows a clear misalignment in the inconsistent version. Humans had 13.41% of first reports before the target sentence, 51.15% at the target sentence, 1.22% after the target sentence, and 34.225% with no report at the end. The corresponding proportions for paired reads were 26.53%, 31.76%, 5.76%, and 35.95%. Thus, a similar rate of no report at the end can coexist with misaligned report positions. For the consistent version, the model also placed 15.64% of first reports after the target, whereas humans had only 2.31%. After incorporating the paired effect, the first report NLL increased by 0.4203 nats/document in the consistent version and decreased by 0.4076 in the inconsistent version. These two changes nearly cancel out when averaged across the full text. [First report contingency table](../../../research/results/frick-pair/results.json) · [Conditional validation](../../../research/results/frick-pair-independent-check.json)

This is a held-out study of the same set of German stories. Prior results were used to select the direction of comparison. here, the comparison is fixed before computing new results. Since readers report noticing contradictions after the full sentence is presented, both agreement and disagreement fall at the sentence level. Per-token initial question, antecedent, and resolution reading stop require separate observation. [ human button and experimental procedure](frick-consistency-source.md)

## Reproducibility and Quantification

Execute `python3 -m research.semantic_tracking.frick_pair_score`, sequentially providing `--assay`'s original preparation data, `--extraction`'s original L4 projection directory, `--packet`'s original stimulus packet, and the newly created `--output`. The two runs each took 0.693 and 0.692 seconds, with a peak RSS of 133.0 MiB, and all five output files were byte-for-byte identical. This round utilized cached end representations and a single-threaded CPU, requiring no additional model inference. A separate program recomputed 120 sets of fitted training denominators, 20 directions, 3,504 held-out logits, and all sealed snapshot point estimates from the original state and human trial. The maximum difference in point estimates was \(6.66\times10^{-16}\). [Execution Entry](../../../research/semantic_tracking/frick_pair_score.py) · [Predict sealed snapshot](../../../research/results/frick-pair/forecast-seal.json) · [Independent recomputation](../../../research/results/frick-pair-independent-check.json) · [Previous round extraction and results](frick-consistency-results.md)

[Original research record](../../../research/semantic_tracking/frick-pair-results.md)
