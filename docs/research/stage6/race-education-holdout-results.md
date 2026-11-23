# The educational attainment of English open-book responses in the held-out set

After controlling for education level, the average NLL of selected options decreased from 0.89452 to 0.89285 nats. The improvement was **+0.00167 nats**, with a reader × article matching re-sampling 95% confidence interval of [−0.00411, +0.00718], which did not reach the pre-specified improvement threshold. This analysis used 970 readers across 216 RACE materials, yielding 1,940 open-book responses. Each fold used other readers' OneStopQA selections for fitting. [Fixed Specification](race-education-holdout-protocol.md) · [Full Score](../../../research/results/race-education-holdout/scores.json)

| Readout | Selected option NLL (nats) | Accuracy Brier |
| --- | ---: | ---: |
| Surface features | 1.37645 | 0.43650 |
| Body, title, and options | 1.06891 | 0.31710 |
| Null background, calibrated selection probability | 0.89452 | 0.23006 |
| Education level, position offset | 0.89470 | 0.23014 |
| Education level, position offset, and logits slope | 0.89285 | 0.22821 |

Both scores are lower is better. Per-fold out-of-fold calibration uses OneStopQA out-of-fold logits, without using test RACE responses. Only 938 readers with education records are included, yielding an improvement of +0.00173 nats, 95% interval [−0.00405, +0.00746]. In 64 trials of 32 background-blank readers, the conditional readout distribution matches the background distribution bit-wise. [full_score](../../../research/results/race-education-holdout/scores.json)

This comparison describes the relationship between educational attainment and open-book selection in English. Readers were drawn from the same Prolific study, with training and evaluation folds utilizing non-overlapping reader keys and two distinct materials. The data did not measure concept prior knowledge per article, nor did it measure load sentence-by-sentence during reading. [Original Study §4.3](https://aclanthology.org/2020.acl-main.507.pdf#page=6) · [Source and split manifest](../../../research/results/race-education-holdout/forecast-seal.json)

Retain the existing L4 FP32 causal forward features. The original sealed snapshot RACE probability is replayed bit-wise consistently. all new predictions first attach to the sealed snapshot before appending the answer choice. Recompute the sequential NLL by directly indexing the sealed snapshot probabilities against actual options, matching the original scoring function exactly. The aggregated output from two CPU executions is bit-wise identical, consuming 8.30 seconds and peaking at approximately 273 MiB of memory. [Reproduction code](../../../research/stage6/race_education_holdout.py) · [sealed snapshot manifest](../../../research/results/race-education-holdout/forecast-seal.json)

[Original research record](../../../research/stage6/race-education-holdout-results.md)
