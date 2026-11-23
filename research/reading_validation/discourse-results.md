# Which connection will readers make between two sentences?

Readers gave English discourse-relation judgments under two response procedures. After training on 5,844 novel and parliamentary items, the causal Qwen readout assigned more probability than text form or the source response frequencies to actual choices in 645 held items from 69 Wikipedia articles. Forecasting which relation people would choose transferred across genres. Forecasting how much readers would disagree about a particular connection proved harder. [Original study and data](https://github.com/merelscholman/DiscoGeM/tree/635515c06259ae40696649cba1d9f097418c8bd4) · [Registered design](discourse-protocol.md) · [Outcomes and intervals](../results/reading-validation-discourse/scores.json)

| Wikipedia response | Votes / distinct worker IDs | Choice log loss, Qwen / form / prior | Disagreement MSE, Qwen / form / prior |
| --- | ---: | ---: | ---: |
| Choice among supplied relations | 6,525 / 63 | 2.433 / 2.544 / 2.532 | 0.01934 / 0.02210 / 0.01940 |
| Free insertion, mapped to a relation | 6,452 / 182 | 1.840 / 1.942 / 1.927 | 0.03150 / 0.03029 / 0.03184 |

Log loss is measured in nats per choice. The disagreement target is the probability that two distinct respondents give different relation labels to the same connection. Training the two response procedures jointly reduced free-response disagreement MSE by 0.00035 relative to separate Qwen readouts [0.00007, 0.00073]. A text-form readout still had smaller error. Random split-half rank correlations between readers' estimates of disagreement had median values of 0.19 and 0.24. This limits the stability of item-by-item evaluations; it supplies no general human ceiling for a different endpoint. [Estimation](discourse_fit.py) · [Scoring](discourse_score.py)

Teachers marked 112 discourse-relation problems among 1,352 sentence boundaries in 100 English essays. Applying the fixed disagreement score there gave average precision of 0.068 [0.051, 0.094] against 0.091 [0.067, 0.129] for text form. A writer can reasonably ask what relation a model expects at a transition. A claim that the transition is defective requires evidence from people who actually evaluated that quality. [Independent essay check](../results/reading-validation-discourse/scores.json)
