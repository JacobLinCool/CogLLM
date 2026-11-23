# ExpMRC's back-calibration results for the arrival rate

Across 165 articles, 555 specified questions, and 1,110 equal-weighted human references, the calibrated per-token readout yields **3.0760 nats interval NLL, 0.1836 cumulative Brier**. With articles split into five folds of 33 each, only the remaining four folds are used to fit two global parameters. The original analysis's data exclusion, reading stop, and endpoint are retained. Lower values are better for both metrics. [Analysis Specifications](expmrc-calibration.md) · [Summary Results](../../../research/results/expmrc-calibration.json)

| Readout | Interval NLL (nats) | Cumulative Brier |
| --- | ---: | ---: |
| Calibrated per-token | 3.0760 | 0.1836 |
| original per-token | 5.0903 | 0.4497 |
| Original stem | 3.1963 | 0.1942 |
| Post-calibration reduction per-token | −2.0143 [−2.1246, −1.8983] | −0.2661 [−0.2872, −0.2444] |
| Post-calibration reduction of the original stem | −0.1203 [−0.1753, −0.0655] | −0.0107 [−0.0188, −0.0032] |

Following fixed bracketed post-external prediction, the 95% confidence interval derived from 10,000 article resamples. The first reading stop has 4.23% of human references fully covered. average predicted probability is 5.16% calibrated per-token, 0.086% original per-token, and 3.19% original stem. Monotonic transformation preserves rank within each of the 698 original-spec-same-text-same-reading-stop pairs, yielding 435 wins, 263 losses, and 0 draws. The sum of interval arrival probabilities and terminal remaining probabilities across the 555 calibrated trajectories equals 1. [Original scoring results](../../../research/results/expmrc-transfer-l4/scores.json) · [Aggregated results](../../../research/results/expmrc-calibration.json)

The previous transfer experiment has already evaluated the same batch of ExpMRC annotations. the current results represent a back-extrapolated estimate. The evaluation target is the point at which, after reading a given question, the accumulated reading includes the annotated evidence. Absolute performance on the new corpus still requires independent validation.

This machine executes only a two-parameter fit and scoring on a single CPU thread, with a maximum measured RSS of **82.7 MiB**. The aggregated JSON from two executions are bitwise identical, with program timings of 0.41 and 0.42 seconds, respectively. The re-run command for the library directory leverages existing dependency caching, using Python 3.14.7, NumPy 2.5.5, and SciPy 1.17.1. [Execution Program](../../../research/semantic_tracking/expmrc_calibrate.py)

```sh
PYTHONPATH=.cache/cogllm/anglin/deps python3 research/semantic_tracking/expmrc_calibrate.py \
  --assay-root .cache/cogllm/exp-mrc/v1 \
  --forecast-dir research/results/expmrc-transfer-l4 \
  --original-protocol research/semantic_tracking/expmrc-transfer.md \
  --calibration-protocol research/semantic_tracking/expmrc-calibration.md \
  --heads research/results/qasper-factorial-run/heads.npz \
  --output /tmp/expmrc-calibration-replay.json
```

[Original research record](../../../research/semantic_tracking/expmrc-calibration-results.md)
