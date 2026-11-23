# Stage 4 · State changes around sentence-initial pronouns

**Adding a state-difference readout did not improve reproduction of reference effects.** In five-fold evaluation on Zhan, both weakly supervised heads estimated interactions much smaller than the observed human effect, and the new readout had higher trial-level Brier error. On the external Lam & Hwang data, the new readout had lower trial-level Brier error, while interaction-effect error increased for both fixed pronoun forms. The shared readout remains the reference baseline. [Zhan results](../../research/results/stage4-zhan-update/zhan-score.json) · [External results](../../research/results/stage4-zhan-update/external-score.json)

This round compares the readout of two frozen Qwen3.5-2B models. On the same 72 L4 cache states, we contrast the current single linear head with the linear head that independently adds weighted pronoun-before and after-state differences. For each reading stop, the $h$ is first L2-normalized. The new readout appends a zero vector before the pronoun and $h_1-h_0$ after the pronoun. The new features maintain their original scale, and all weights use existing L2 coefficients. Qwen uses a single forward pass. [Error decomposition and readout formula from the previous round](stage-4-zhan-next-validation.md)

## Pre-specified scoring prior to the results

In the Zhan experiment, we used the original five-fold, 12-group label sampling, three supervision conditions, and loss functions. We primarily compared the absolute error of the material equal-weighted offset between the sequential Brier score and IC-2 minus IC-1 of two weakly supervised heads. We reported the paired 95% material resampling interval for the increase in readout versus the current readout. Additionally, we listed the subject bias difference for the first two material categories prior to intervention, along with the complete direct label results. The data belong to the same study, with the material held-out and readers overlapping. [Stage 4 data and endpoints](stage-4-zhan-weak-supervision.md)

For external evaluation, we fitted both readouts on all stimuli from Zhan Experiment 1 under identical supervision conditions. We fixed both models, the random seeds, and the scoring rules before using the Lam & Hwang outcomes. We report all three supervision conditions and retain the two weakly supervised heads for the primary comparison. The external endpoint uses only the free continuation and supplied-pronoun conditions for the 24 implicit-causality stimulus sets in Lam & Hwang Experiment 1. Using the authors' released fields, we select continuations that refer to either candidate, weight the stimuli equally for the interaction effect, and compute Brier scores per trial. We score each seed before averaging across seeds. The supplied pronoun in the external data follows “because,” while Zhan places it at the start of the next sentence. [External data audit](stage-4-zhan-next-validation.md#accessible-held-out-materials) · [Authors' Appendix A](https://doi.org/10.6084/m9.figshare.27766995.v1)

If the explicit pronoun orthography cannot be determined item-by-item for the original experiment, fix both complete contexts on the same human response, providing "he" for each item and "she" for each item. Additionally, compute the contribution maxima of each orthography for the offset difference and Brier score across materials, and report the range covered by any orthography combination between items. These two analyses do not select orthographies favorable to the data. the range reflects input uncertainty. If the token prefixes of free prompts and explicit pronoun prompts cannot be precisely aligned, record the count and disposition.

Pre-tokenization checks before execution revealed that all 48 explicit versions merge conjunctions and pronouns into a single token. The external primary analysis concatenates the original tokenization's free prompt with separately encoded pronouns. Verification confirms that the decoded text is character-identical, and the free prompt's reading stop can be retrieved from a single forward pass. Additionally, using the default tokenization, two forward passes are performed—one for each of the two prompts (free and given pronouns)—to measure tokenization sensitivity. The results from this approach utilize the final states of both prompts.

## Zhan held-out materials

The Brier score is the mean squared error between predicted probabilities and binary responses, ranging from 0 to 1, where lower is better. The interaction effect is the difference in the subject re-mention shift after the pronoun between IC-2 and IC-1, measured in percentage points. The human interaction effect is **+24.46 percentage points**. [Group-wise and resampling results](../../research/results/stage4-zhan-update/zhan-score.json)

| Supervised conditions | Shared-head Brier ↓ | Poor-state head Brier ↓ | Shared-head interaction effect | State-space head interaction effect |
| --- | ---: | ---: | ---: | ---: |
| A small number of direct | 0.21085 | 0.21238 | −4.09 pp | −6.46 pp |
| A small amount of direct plus weak supervision | 0.21189 | 0.21328 | +1.12 pp | +0.83 pp |
| Complete and direct. | 0.20714 | 0.20814 | −4.18 pp | −6.18 pp |

With weak supervision, the state-difference head minus the shared head had a Brier score difference of +0.00139, with a paired 95% interval across stimuli of [+0.00027, +0.00245]. The difference in absolute interaction-effect error was +0.30 percentage points, with an interval of [−0.49, +1.10]. Before the pronoun, the difference between the two stimulus types went from 15.90 percentage points with the shared head to 15.70 with the state-difference head. The human difference was 67.29 percentage points. These results support retaining the current readout. [Five-fold metrics](../../research/results/stage4-zhan-update/zhan-score.json)

## Lam & Hwang external materials

24 materials yielded 38 readers' **852 existing responses**. The human interaction effect was **+8.65 percentage points**. Below are the primary results using the control lexicon. Both orthographies map to the same set of human responses, reflecting the sensitivity of the original explicit pronoun orthography when each item is unconfirmed. [External metrics](../../research/results/stage4-zhan-update/external-score.json)

| Weakly supervised, given the glyph | Shared-head Brier ↓ | Poor-state head Brier ↓ | Shared-head interaction effect | State-space head interaction effect |
| --- | ---: | ---: | ---: | ---: |
| Each question is given to "he". | 0.24851 | 0.24439 | −1.27 pp | −2.55 pp |
| Each question is given to "her". | 0.25240 | 0.24865 | +0.84 pp | −0.10 pp |

When characters are changed character-by-character, the weak-supervised state-difference head's shared head Brier score reaches an extremum of [−0.00486, −0.00300]. The resampled intervals for fixed "he" and "she" pairings are also each below zero. The absolute error of the interaction effect increases by 1.28 percentage points when fixed to "he" and by 0.94 percentage points when fixed to "she," with both pairing intervals exceeding zero. Under any character-by-character combination, the state-difference head's effect ranges between −3.97 and +1.32 percentage points, all smaller than human +8.65. The sensitivity of the default tokenization also exhibits weak-supervised Brier score reduction and increased error for the two fixed-character effects. [Fixed character and arbitrary combination range](../../research/results/stage4-zhan-update/external-score.json)

Furthermore, the human gap before the external material is 80.67 percentage points, with the two weak-supervised heads estimating 2.45 and 1.99, respectively. The external endpoints cover reference selection following "because," and differ from the material, instruction, punctuation, and Zhan's continuation. The above results measure two readouts of external transfer, yet the causes of the various differences remain unseparated. [External metric](../../research/results/stage4-zhan-update/external-score.json) · [Original material Appendix A](https://doi.org/10.6084/m9.figshare.27766995.v1)

## Resources and Verification

The NVIDIA L4's 175 FP64 fit operations took 59.37 seconds. External extraction was performed with a fixed FP32 Qwen, using 120 input sequences, 888 tokens, and 31 batches. forward and readout times totaled 7.28 seconds, with a CUDA peak of approximately 7.06 GiB. All 48 controlled pronoun versions could be decoded verbatim back to the original text. The maximum relative RMS difference between prefixes and standalone inputs was 2.19×10⁻⁶, and the difference in shared prefixes between the two pronouns was zero. The GPU was stopped after approximately 135 seconds and confirmed released. A single deduction could not be sampled from the account value. [Extraction Audit](../../research/results/stage4-zhan-update/extraction.json) · [Execution and Release](../../research/results/stage4-zhan-update/manifest.json) · [Account Sampling](../../research/results/stage4-zhan-update/usage.jsonl)

Recomputed targets and gradients from 175 weight sets using NumPy, yielding a five-fold maximum probability difference of 3.33×10⁻¹⁶. Human responses were retained locally and utilized for scoring only after the sealed snapshot. Two independent runs produced byte-identical results. [ sealed snapshot and version](../../research/results/stage4-zhan-update/prediction-seal.json) · [audit and hash](../../research/results/stage4-zhan-update/verification.json)

## Reproduce the entry point

1. The current study employed `research.stage4.zhan_update prepare` to construct a full Zhan training design. Using `research.stage4.lam_hwang_materials`, an external stimulus containing only text was derived from the fixed version appendix and author materials.
2. In Colab L4, use `research.stage4.zhan_update fit` to read existing L4 features, five-fold data, and full Zhan design. Use `research.stage4.lam_hwang_features` to extract external states. The training phase includes only Zhan's training summary and external text.
3. Download weights and state. Compare the old fivefold with the new fivefold using `research.stage4.zhan_update_score score-zhan`. Execute `score-external` only after fixing the external prediction with `seal-external`. The sealed snapshot data includes the model, prediction, source trial file, and the SHA-256 hash of the scoring script.

[Training Entry](../../research/stage4/zhan_update.py) · [External Material Construction](../../research/stage4/lam_hwang_materials.py) · [External State Extraction](../../research/stage4/lam_hwang_features.py) · [Scoring Entry](../../research/stage4/zhan_update_score.py)

[Original research record](../../research/stage-4-zhan-update.md)
