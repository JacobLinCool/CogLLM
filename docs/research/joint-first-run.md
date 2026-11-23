# OneStopQA baseline and the first shared-readout experiment

**Incorporating the article improved the human option prediction for OneStopQA. The gain from the first-round shared adapter appeared only at certain endpoints.** Compared to independent adapters, the shared training improved PLABA's "flag uncertain" selection, while the matching intervals for other endpoints all crossed zero. The frozen representation linear readout yielded the minimum estimates in both the sequential sum and sentence-level error listed in this round, and is retained as a reference. [Open-book metric](../../research/results/joint-first-run/onestop-probe.json) · [Shared vs. Independent matching results](../../research/results/joint-first-run/joint-paired.json)

Using Stage 2's fixed five folds and 5,417 existing human responses, first run the OneStopQA single-task comparison, then compare separate and shared training across the three tasks. The loss in this round uses actual trial-level observations. Published aggregate effects are scored on their original scales and reserved for use in training in the next comparison. [Overall training and validation plan](joint-training-plan.md)

## Fixed data and comparison

- Reuse the PLABA end-of-sentence/end-of-paragraph readout state and Zhan free/given pronoun reading stop state on L4. PLABA uses a 1–5 self-rated scale and two structured difficult endpoints supervised, counting only trials lacking a selection as self-rated. Zhan supervises the selection of two candidate references.
- OneStopQA presents the body text, question, and the order of displayed options from the input experiment. A total of 1,872 prompts were generated, with 1,770 question-option combinations, summing to 3,642. All option text uses visible input, and the same question may be selected by multiple readers. Model features are extracted at the end of the prompt via a single causal forward pass, using a fixed Qwen3.5-2B, FP32, and no per-token generation. The body text precedes the question, and the human endpoint is the response when both are visible. [Experiment version](stage-2-data.md#onestopqa-material-version)
- The OneStopQA baselines are the training fold's display position distribution, the normalization-frozen state of question-to-option matching, and the normalization-frozen state of full prompt matching. Both models employ the same four-class linear readout with a fixed L2. The ground truth is used to evaluate correctness only after deriving the four-choice probability. The difference between the body condition and the question condition is matched via 20 held-out source family pairs using random sampling. This comparison measures performance gains after increasing body visibility, excluding data with semantic uncertainty reports or query source segments.
- State readout comparisons freeze linear, each task has its own small residual adapter, and three tasks share identical-sized residual adapters. Task readouts remain separated by measurement. All conditions utilize identical training splits, endpoints, optimization settings, and per-task update opportunities. Joint training computes loss separately by observed scale and divides by sample weights. PLABA's reading and answering must adhere to the original experiment's visible scope, with the first round using only sentence-level review data.

The residual adapter's bottleneck dimension is fixed at 32. the two adapters share initialization and the same number of training iterations. We employ FP32 AdamW, performing 160 full-batch updates per fold with a learning rate of 0.006 and weight decay of 0.01. Gradients are backpropagated based on the average loss across three tasks. PLABA maps scores of 1–5 to 0–1 and applies equal weighting to the self-assessment squared error and the measured checkbox negative log-likelihood. Zhan utilizes binary negative log-likelihood, while OneStop employs four-class negative log-likelihood. A single cross-task aggregated score is not used for evaluation.

The primary comparison is the endpoint held-out error of shared adapters versus per-task independent adapters. PLABA reports two self-assessed MAEs and a checked Brier score, Zhan reports sequential Brier and interaction effects, and OneStopQA reports four-choice NLL, multi-class Brier, and correct/incorrect Brier. All three datasets follow their respective groupings. PLABA simultaneously held-out readers, while the other two use material held-out. Error intervals are computed via 10,000 resampled paired material groups, fixing each fold's held-out predictions and reader samples. PLABA's rewrite difference and Zhan's interaction effects are recalculated using held-out predictions and human groupings.

## OneStopQA held-out results

Of the 1,940 human responses, 1,567 were correct. The in-table NLL is the average negative log-likelihood over actual selected options, measured in nats. The multi-class Brier score is the average sum of squared errors across the four options, while the per-question Brier score considers only the probability of the correct option. Lower values are better.

| Fixed readout visible information | Option NLL ↓ | Multi-class Brier ↓ | Correctness Brier ↓ |
| --- | ---: | ---: | ---: |
| Prior on the training fold display location | 1.3867 | 0.7502 | 0.4663 |
| Question and Actual Display Options | 1.3677 | 0.7406 | 0.4593 |
| Body, title, and actual display options | 0.9844 | 0.5182 | 0.2951 |

The NLL difference added to the main text is −0.3833, with a matched 95% confidence interval [−0.4020, −0.3619]. This represents an open-book comparison of responses using the presentation options from the original study. [metrics, denominator, and interval](../../research/results/joint-first-run/onestop-probe.json)

## Shared and Independent adapters

| held-out endpoint | Fixed linear ↓ | Independent adapter ↓ | Shared adapter ↓ | Shared minus independent paired 95% interval |
| --- | ---: | ---: | ---: | ---: |
| PLABA processing ease, sentence-wise MAE | 0.5824 | 0.6100 | 0.6016 | [−0.0532, +0.0305] |
| PLABA self-assessment, sentence-level mean absolute error (MAE) | 0.6175 | 0.6369 | 0.6177 | [−0.0521, +0.0116] |
| PLABA has doubts, Brier | 0.1918 | 0.2611 | 0.2320 | [−0.0413, −0.0146] |
| PLABA is term-ambiguous, Brier | 0.1963 | 0.2200 | 0.2279 | [−0.0216, +0.0264] |
| Zhan reference selection, Brier | 0.1988 | 0.1999 | 0.2011 | [−0.0075, +0.0106] |
| OneStop option, NLL | 0.7786 | 0.9505 | 0.9467 | [−0.0662, +0.0623] |
| OneStop option, multi-class Brier | 0.4009 | 0.4440 | 0.4444 | [−0.0124, +0.0131] |
| OneStop accuracy, Brier | 0.2003 | 0.2106 | 0.2123 | [−0.0054, +0.0092] |

Three groups used identical optimization settings. The linear readout in the table above employed AdamW, while the OneStopQA single-task baseline readout used fixed L2 and LBFGS. each table's comparison fixed its respective optimization setting. Changes to the shared, relatively independent adapter varied by endpoint. This represents the result of fixing a single group of shared capacity, task weights, and update counts. [Fold-by-fold and step-by-step metrics](../../research/results/joint-first-run/joint-metrics.json) · [Pairwise comparison](../../research/results/joint-first-run/joint-paired.json)

Regarding the effect, PLABA's plain-text rewrite self-assessment average was **0.723** points higher, with linear, independent, and shared readout estimates at **0.658, 0.739, and 0.662**, respectively. Zhan's human interaction effect was **+24.46 percentage points**, with the three groups at **−2.63, +5.99, and +5.07**. These are diagnostic results derived from supervised learning on the held-out data. In the next round, when the effect is incorporated into the loss, matching can be performed using the same data and readout from the current round. [ effect recalculation](../../research/results/joint-first-run/joint-paired.json)

## L4 Resource and Verification

34 formal prompt trials passed, fully extracting 3,642 prompts and 512,429 non-padding tokens. Forward and readout for 456 batches took 209.89 seconds, approximately 2,441 tokens/s. CUDA peak configuration 7.42 GiB, retained 8.90 GiB. Nine prefix checks and three positive controls passed, with maximum relative RMS difference of 6.62×10⁻⁶. Single-task baseline fit 4.15 seconds, three conditions totaling 15 fits for 13.35 seconds. From configuration to issuing stop request approximately 343.72 seconds, confirmed released. Account shows 768.80 to 768.76 CU, representing overall account readout. [Measured extraction](../../research/results/joint-first-run/onestop-extraction.json) · [Prefix check](../../research/results/joint-first-run/onestop-causality.json) · [Execution and release](../../research/results/joint-first-run/manifest.json) · [Usage](../../research/results/joint-first-run/usage.jsonl)

Reconstructing five-fold predictions locally from 15 sets of weights gave a maximum probability or score difference of 6.20×10⁻⁶. Primary metrics agreed when independently recalculated from individual responses, and rerunning the paired comparison gave byte-identical output. In a synthetic CPU check, changing the labels held out for a fold left its predictions unchanged. Changing OneStop training responses left predictions from the other two tasks' separate adapters unchanged and changed predictions from the shared adapter. [Replay and audit](../../research/results/joint-first-run/verification.json) · [Trial-level metric recomputation](../../research/results/joint-first-run/independent-metric-verification.json)

## Parallel-prepared word-level endpoints

In CompLex's fixed release of 10,800 English word and multi-word phrase scores, 10,740 can be uniquely matched to character offsets in the source text. 56 have multiple candidates even after case normalization, 3 have no candidate matching the boundary, and 1 matches only when ignoring case. A total of 1,101 official test inputs and score files have been cross-checked. The official train and test sets overlap across two tasks with 223 character-identical sentences. CogLLM's word-level held-out must be grouped by sentence. [Character Positioning and Version Hashing](../../research/results/joint-first-run/complex-audit.json) · [Re-run Program](../../research/joint/complex_audit.py)

The original measurement requests humans to specify the target vocabulary difficulty when the entire sentence is visible, mapping the five-point scale to 0–1 and aggregating the results. Unrated items are considered unobserved. The fixed release statement is CC BY 4.0. audits separately list upstream sources and biomedical articles that have not yet been fully vetted. This dataset can serve as a subsequent local vocabulary difficulty endpoint, while first-order perplexity, reading stops, and the resolution of the same query still require corresponding temporal observations. This round has completed source and localization audits. [Original Method](https://aclanthology.org/2021.semeval-1.1.pdf#page=3) · [Fixed Release](https://github.com/MMU-TDMLab/CompLex/tree/4b31c2bd34da4c1265f9db3d727d95ee23099f14)

## Re-run the entry point

On this machine, reconstruct the observation counts and no-issue prompts with `python -m research.joint.prepare --data .cache/cogllm/stage2 --output .cache/cogllm/joint/data`. L4 first executes `research.joint.onestop_features`, then uses the same feature to execute `research.joint.onestop_probe` and `research.joint.train`. Training additionally retrieves prior PLABA and Zhan L4 states from Stage 3 and Stage 4. After download, recompute the per-endpoint and literature effect contrasts with `research.joint.compare`. [Data Construction](../../research/joint/prepare.py) · [Extraction](../../research/joint/onestop_features.py) · [Open Baseline](../../research/joint/onestop_probe.py) · [Shared Training](../../research/joint/train.py) · [Held-out Contrast](../../research/joint/compare.py)

[Original research record](../../research/joint-first-run.md)
