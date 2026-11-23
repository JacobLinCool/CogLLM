# QASPER has evidence of joint calibration with unanswerable questions

After incorporating the localizable evidence and explicitly unanswerable annotations into the temporal calibration, the per-token readout interval NLL on the new QASPER paper decreased from the original **10.153 to 4.370 nats**, with a 95% matched comparison interval of [−7.856, −3.909]. After additionally fitting a fixed tail residual probability parameter, the NLL became 4.385, with a difference of +0.0149 [−0.0210, +0.0580], which did not pass the pre-specified improvement threshold. Lower NLL indicates a higher probability assigned to human annotations. [Fixed analysis specification](qasper-evidence-tail.md) · [Pre-specified comparison](../../../research/results/qasper-evidence-tail-l4/scores.json) · [Supplementary matched comparison](../../../research/results/qasper-evidence-tail-l4/secondary-verification.json)

| per-token readout | held-out NLL | Evidence-based NLL | Explicitly stated unanswerable NLL | Cumulative Brier |
| --- | ---: | ---: | ---: | ---: |
| Original readout | 10.153 | 9.528 | 18.104 | 0.3497 |
| Original ExpMRC calibration | 38.407 | 36.014 | 67.301 | 0.5766 |
| QASPER mixed annotation, calibration time | 4.370 | 4.674 | 1.340 | 0.1554 |
| Similarly, add the tail-end remaining probability parameter. | 4.385 | 4.693 | 1.306 | 0.1551 |

The newly calibrated per-token readout shows an NLL difference of −0.250 [−0.364, −0.138] relative to the same calibration position baseline. Relative to the paragraph interaction readout, the difference is −0.002 [−0.020, +0.018]. Thus, this batch of data supports improved position estimation for in-text information, yet the per-token interaction gain on known issues remains inferior to paragraph interaction. All group monotonic transformations retain strict identical-document pairing cumulative intensity rankings. [Scoring by group](../../../research/results/qasper-evidence-tail-l4/scores.json) · [Same-calibration control](../../../research/results/qasper-evidence-tail-l4/secondary-verification.json)

## Parallel evidence from the same corpus

After the main analysis, we conducted a same-corpora control. Both groups used the same two-parameter model on the same batch of QASPER development materials, with the control group using only localizable evidence. After including unanswerable labels, the NLL of such labels in the held-out set decreased from 2.570 to 1.340, with a paired difference of −1.230 [−1.781, −0.667]. The NLL of all labels was 4.370 versus 4.366, with a paired difference of +0.004 [−0.129, +0.121]. Therefore, the overall improvement after calibrating the new corpus cannot be attributed to the inclusion of unanswerable labels. The additional benefits of these labels are concentrated in the probability assigned by the model to unanswerable cases. [Supplementary Experiments and Intervals](../../../research/results/qasper-evidence-positive-control.json)

## held-out range and recall

Official train: 28 articles, 76 items contained 54 valid human annotations, of which 44 had locatable evidence and 10 were explicitly unanswerable. Official test: 25 articles, 78 items contained 73 items and 165 valid annotations, of which 149 had evidence and 16 were explicitly unanswerable. Annotations missing evidence, lacking text evidence, and unlocatable are counted separately. The two article sets share no identifiers or normalized titles with the 110 papers used in the original QASPER validation. All papers retain their full published text. [held-out and exclusion reasons](../../../research/results/qasper-evidence-tail-l4/assay.json) · [original data and annotation method](https://aclanthology.org/2021.naacl-main.365/)

Four groups shared a frozen Qwen3.5-2B backbone and original readout weights, performing a single model forward pass on each of the 53 full texts. The corpus and stem comprised 136,234 tokens, with all 4,836 scoring reading stops and prefix alignment passing. The forward pass and extraction took 58.8 seconds. After sealing the raw predictions, we fitted and archived locally using train, then proceeded to test. QASPER's fit and score re-runs were bitwise consistent, and all 16 NLLs were recalculated using the direct probability formula. L4 has been released. [L4 forward time](../../../research/results/qasper-evidence-tail-l4/extraction.json) · [fit weights](../../../research/results/qasper-evidence-tail-l4/fit.json) · [release and re-run manifest](../../../research/results/qasper-evidence-tail-l4/run-verification.json)

## Backward chaining applied to Natural Questions

The QASPER time calibration of the sealed snapshot was applied to the already-read Natural Questions, yielding a per-token interval NLL of 3.153, which is lower than the 7.776 from the original readout and the 27.210 from the original ExpMRC calibration. With the addition of the tail-end remaining probability parameter, the value becomes 3.263, a difference of +0.109 [0.048, 0.188]. This control did not use Natural Questions annotations for fitting, yet its prior results were utilized in designing this round of experiments. The endpoints of both corpora are visible in time as evidence of designated questions. Readers' spontaneous queries and when they became aware of the answer remain to be directly verified by the same person's reading stop. [Backward Transfer Results](../../../research/results/qasper-evidence-tail-l4/nq-replay.json) · [Original Natural Questions Results](nq-evidence-transfer-results.md)

In the same hold-out validation, using only QASPER's evidence-annotated matching yielded 3.565 nats. The mixed-annotated difference relative to this same corpus held-out set was −0.411 [−0.600, −0.223], with the no-issue difference being −0.817 [−1.211, −0.397]. [Supplementary hold-out comparison](../../../research/results/qasper-evidence-positive-control-nq-replay.json)

[Original research record](../../../research/semantic_tracking/qasper-evidence-tail-results.md)
