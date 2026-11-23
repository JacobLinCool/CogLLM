# Stage 3 Chinese sentence-initial pronoun contrast

Qwen's readout representations improved held-out probability estimates for sequential completion, yet overestimated IC-1 bias and underestimated IC-2, with the ordering of these two biases opposite to human results. This discrepancy can be used to evaluate whether supervised effect supervision can improve performance on difficult conditions while preserving the accuracy of sequential probabilities. [Score Result](../../research/results/stage3-zhan-landmarks/metrics.json)

This experiment tests whether the same causal readout reproduces changes in referent choice between two stops that human readers actually saw. The first stop is the period ending the preceding sentence. The second is an added Mandarin pronoun meaning “he” or “she.” Use Zhan et al.'s Experiment 1, in which both prompts ask for a written explanatory continuation. The study estimates referent choice from the first character mentioned again in the free continuation and from the referent chosen for the supplied pronoun, respectively. [Zhan et al., Experiment 1](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0237012#sec008) · [Original Fig. 1](https://doi.org/10.1371/journal.pone.0237012.g001) · [Stage 2 data](stage-2-data.md)

## pre-readout fixed control

The text introduces two candidate referents of the same gender. The binary endpoint is the probability that an eligible continuation selects the previous sentence's subject. Free continuation and pronoun prompting are separate observation conditions. Each reader saw only one condition per stimulus, and contrasts are calculated by material. The analysis includes 36 stimulus sets, 50 readers, and 1,651 published eligible responses. Following the original study, exclude unclear referents, references to both characters or another person, and invalid referring forms. Both versions remain in the same fold. Five-fold evaluation holds out materials, with the same reader sample appearing across folds. [Zhan et al., Experiment 1 Coding](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0237012#sec008) · [Data splits](stage-2-data.md#split-and-validation-scope)

Extract the final-layer states of the two causal forward passes with longer contexts. Verify tokenization consistency prior to the intervention point and cross-check against truncated prefixes. Both conditions share a linear readout. there are no reader backgrounds, human answers, or author verb types outside the main text. The model estimates only the choice between the subject and another candidate, with the two candidate segments aligned in Stage 2. A small-scale local forward pass is used for design verification. formal training follows the execution plan using Colab L4.

Compare the feature groups using the same five folds. Each readout uses binomial logistic regression with a fixed L2 coefficient of 1. Apply L2 normalization to each sentence-state vector.

| Conditions | Available information |
| --- | --- |
| Prompt type only | Are subject-initial pronouns provided? |
| Fixed end-state + prompt type | Both conditions utilize the same representation prior to intervention, with an additional prompt flag added. |
| The various reading stop states | Free prompts are taken at the end, while pronoun prompts are taken at the position where the pronoun is read. |
| Each reading stop state + cue type | When both state models provide prompt flags, examine the difference in adopting the readout of the pronoun state. |
| Author label alignment | Prompt flags, author verb types, and their interaction |

The primary comparison is the difference in held-out trial-level Brier score between readouts using the state at each observation stop and readouts using a fixed sentence-end state plus prompt type. The Brier score is the mean squared difference between the predicted probability and the binary response. It ranges from 0 to 1, with lower values better. The second comparison subtracts the subject-selection rates in the two conditions for each stimulus and estimates the response shift separately for IC-1 and IC-2. We also report the mean absolute error of the shift across stimuli, in percentage points. In free continuations, IC-1 verbs favor the subject and IC-2 verbs favor the other candidate. The original study reported that supplying a pronoun shifted responses toward the subject in both categories, with a larger shift for IC-2. We estimate intervals with paired resampling stratified by stimulus family, holding the held-out predictions and the reader sample fixed. [Zhan et al., Experiment 1, Interpretation biases](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0237012#sec008)

This design observes how readers select characters under two completion conditions with and without initial pronouns. The model's readout is precise to the supplied pronoun, and the study measures this after completion. To determine whether readers experience confusion at the token or resolve the same question in subsequent text, additional data temporally aligned with events are required.

## The results of this machine

| Conditions | Sequential Brier ↓ | Material-wise shift MAE ↓ | IC-1 exhibits subject bias. | IC-2 exhibits subject bias. |
| --- | ---: | ---: | ---: | ---: |
| human observed mean |  |  | +8.7 pp | +33.1 pp |
| Prompt type only | 0.2471 | 15.6 pp | +22.8 pp | +21.1 pp |
| Fixed end-state + prompt type | 0.2149 | 14.7 pp | +24.7 pp | +24.0 pp |
| The various reading stop states | 0.2072 | 16.1 pp | +25.3 pp | +21.1 pp |
| Each reading stop state + cue type | 0.2074 | 16.1 pp | +25.5 pp | +21.3 pp |
| Author label alignment | 0.1606 | 9.6 pp | +10.9 pp | +31.5 pp |

The primary Brier score was −0.00775. with the reader sample fixed and resampling the held-out prediction material, the 95% interval was [−0.01270, −0.00307]. When both sides provided hint flags, the score was −0.00754 [−0.01236, −0.00301]. Per-condition metrics are retained in the results file. Per-material offset estimates for each reading stop state still exhibit an absolute error of 16.1 percentage points. Author labels align with the original study's verb classification and can be used to measure how much gap remains in the effect. The main readout uses only model states. [Per-fold held-out prediction](../../research/results/stage3-zhan-landmarks/predictions.jsonl) · [Endpoints and intervals](../../research/results/stage3-zhan-landmarks/metrics.json)

## Re-run

Using fixed Qwen3.5-2B weights, a local Apple M5 Pro, MLX 0.32.2, MLX-LM 0.31.3, and Transformers 5.17.0. After converting BF16 weights to FP32, 36 primary forward passes yielded 297 tokens, taking approximately 2.72 seconds. All 36 independent truncated controls passed. the maximum relative RMS difference was 0.00167, below the 0.02 tolerance. Shared tokens and offsets between the two prompts were consistent, with exactly one pronoun token added per group. Feature caching was verified via SHA-256. This represents a local small-scale control run with Colab not initiated. [Feature extraction log](../../research/results/stage3-zhan-landmarks/extraction.json) · [fixed weights version](https://huggingface.co/Qwen/Qwen3.5-2B/tree/15852e8c16360a2fea060d615a32b45270f8a8fc)

```bash
python -m research.stage3.zhan_landmark_features \
  --data .cache/cogllm/stage2/zhan \
  --model /path/to/15852e8c16360a2fea060d615a32b45270f8a8fc \
  --output .cache/cogllm/stage3/zhan-landmarks
python -m research.stage3.zhan_landmark_probe \
  --data .cache/cogllm/stage2/zhan \
  --features .cache/cogllm/stage3/zhan-landmarks/features.npz \
  --output .cache/cogllm/stage3/zhan-landmarks
```

[Original research record](../../research/stage-3-zhan-landmark.md)
