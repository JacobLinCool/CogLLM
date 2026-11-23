# Yadav et al. German and Dutch sentence-level difficulty transfer diagnosis

**The PLABA readout can be used to test whether cross-material difficulty direction is retained.** Experiment 1 had native readers view complete single sentences and rate their perceived difficulty. Experiments 2a/2b presented words sequentially in a fixed temporal order, followed by the same rating procedure. The rating scale ranged from 0–100, with higher scores indicating greater difficulty. The authors' R program rescaled the stored continuous 1–9 values to this scale using `(response−1)×12.5`. [Original paper §§3.1.2, 4.1.3](https://sites.socsci.uci.edu/~rfutrell/papers/yadav2025revisiting.pdf) · [Author analysis program](https://osf.io/download/tj2qd/)

## Public data and alignment

The author's OSF project is licensed under CC BY 4.0, containing the author-filtered sequential R data, publication-ready Dutch-Dutchman matching stimulus TSV, and analysis and preprocessing scripts. Data audit outputs only distribution and sample size, retaining file SHA-256 hashes and data sources. [Project License Field](https://api.osf.io/v2/nodes/qbc5f/) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · [Data Audit](../../../research/results/yadav-transfer-audit.json)

| Experiment | Readers (Dutch/German) | Key trial | The aligned stimulus version |
| --- | ---: | ---: | ---: |
| 1 full sentence | 36／40 | 1,701 | 189 |
| 2a/2b word-by-word timed presentation | 134／132 | 5,924 | 191 |

The alignment key comprises language, original item, nested hierarchy, and rewriting condition. In Experiment 1, one version of the trial column contained an additional trailing space. In Experiment 2, the serialized display tokens were rejoined in their original order and matched the author's TSV character-for-character. Underline annotations in word-level stimuli recorded co-presented phrases that were invisible to the reader. when input to a pure text model, these should be restored as spaces. In L3, a few original items lacked retention scores in the German unrewritten condition, and no imputation was applied during matching. [Experiment 1 stimuli](https://osf.io/download/t5d89/) · [Experiment 2 stimuli](https://osf.io/download/kxvf4/) · [Author documentation](https://osf.io/download/6mvkj/) · [Data audit](../../../research/results/yadav-transfer-audit.json)

## human matching results

First, average the reader ratings for each item within the same category. Only include items where all four items in the same category have ratings, and apply equal weighting to average the items. The nested cost is defined as (unrewritten score) minus (rewritten score). a positive value indicates that the unrewritten sentence is more difficult. The unit difference between the two languages remains in the original 0–100 scale. [Distribution and full matching](../../../research/results/yadav-transfer-audit.json)

| Experiments and Hierarchies | Full original question | Dutch nested costs | German nested costs | German language reduction cost |
| --- | ---: | ---: | ---: | ---: |
| 1 L2 | 24 | +2.77 | +15.02 | +12.24 |
| 1 L3 | 21 | +21.83 | +26.37 | +4.54 |
| 2a／2b L2 | 24 | +1.67 | +4.25 | +2.58 |
| 2a／2b L3 | 23 | +14.39 | +23.59 | +9.21 |

Only the original questions with both levels complete: the L3 minus L2 German-Dutch cost difference was −6.57 points in Experiment 1 and +6.60 points in Experiments 2a/2b. This is based on item-level equal weighting. The corresponding three-way interaction obtained via crossed random effects analysis also showed opposite directions, at −8.87 and +4.86, respectively. Both presentation procedures are retained as their respective observational conditions. [Original paper Tables 2, 4](https://sites.socsci.uci.edu/~rfutrell/papers/yadav2025revisiting.pdf) · [Recalculated with pairing](../../../research/results/yadav-transfer-audit.json)

## Model validation scope

First, using the complete single-sentence formulation from Experiment 1, freeze the existing readout. then compare the ranking and average direction across the two versions for each language and level. To compare the German-Dutch cost difference, employ the same set of original items and the same scoring direction, while retaining the baseline comparison grounded in sentence length. Experiment 2a/2b can serve as a diagnostic for different presentation procedures. It reuses most original items, and the model's single-sentence input lacks human timed-presentation information. [Presentation and materials of the two experiments](https://sites.socsci.uci.edu/~rfutrell/papers/yadav2025revisiting.pdf)

The dependency structures of German and Dutch, along with language, native reader groups, and words, are varied. This contrast can check the model's direction on specified materials. identifying the independent causal effect of the structure itself requires corresponding manipulation. The existing PLABA head human training observation is the English self-assessment when the biomedical article full text is visible, whereas here are German-Dutch controlled single sentences and a 0–100 scale of difficulty. Calibration and absolute effect magnitude still require joint measurement. [PLABA sealed snapshot readout range](../../../research/results/causal-plaba-checkpoint/manifest.json) · [audit program](../../../research/stage6/yadav_audit.py)

### Fixed English readout transfer contrast

Formally execute only the 192 published single-sentence versions of Experiment 1. The PLABA head at the end of the sealed snapshot and the sentence-length baseline fitted to the same batch of PLABA human data are considered only for German or Dutch original sentences. During prediction, do not read out reader ratings. The primary comparison uses original sentences and rewritten sentences that share the same language, topic, and level, counting agreement based on the direction of difference for both human and model, and counting ties as half-points. Pairs of original sentences from the same topic across two languages or two levels share a resampling group. Estimate the difference between the end-sentence PLABA and the sentence-length baseline using 10,000 paired resamples, considering an improvement only if the lower bound of the interval is greater than zero. Model readability is scored on a 1–5 scale, with cost calculated as the difference between the rewritten and original sentence readability scores. For humans, maintain the original difficulty on a 0–100 scale, with cost calculated as the difference between the original and rewritten sentence. Absolute error or effect equivalence is not applied to cross-lingual scales. Experiments 2a/2b are retained as data reconstruction results. [Fixed readout](../../../research/results/causal-plaba-checkpoint/manifest.json) · [held-out prediction and rating](../../../research/stage6/yadav_transfer.py)

### readout held-out results

192 versions generated predictions and sealed snapshots, then combined with 1,701 human ratings, including 90 original items × language × level of valid contrasts. The direction consistency rate rose from 21.11% at sentence length baseline to 43.33% at end-of-sentence readout, a difference of 22.22 percentage points (95% confidence interval for the matched 24 original items: [11.36, 32.61]). End-of-sentence readout in this transfer diagnosis improved relative to sentence length results, with only two German conditions estimating positive average nested costs. [per-item prediction](../../../research/results/yadav-transfer-run/predictions.jsonl) · [sealed snapshot information](../../../research/results/yadav-transfer-run/seal.json) · [held-out ratings](../../../research/results/yadav-transfer-run/scores.json)

| Conditions | human difficulty gap (0–100 points) | Readout at the end of the sentence with poor ease (1–5 points) | Sentence-end readout direction consistency rate |
| --- | ---: | ---: | ---: |
| Dutch L2 | +2.77 | −0.075 | 37.50% |
| Dutch L3 | +21.83 | −0.072 | 9.52% |
| German L2 | +15.02 | +0.039 | 62.50% |
| German L3 | +26.37 | +0.027 | 61.90% |

[Original research record](../../../research/stage6/yadav-transfer-audit.md)
