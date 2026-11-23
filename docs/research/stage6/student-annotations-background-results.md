# StudentAnnotations reader background held-out results

Fitted readouts on frozen Qwen3.5-2B causal end states in Colab L4 using a 16-way cross-validation with reader and sentence held-out. Evaluated 3,160 German vocational guide ratings, 192 readers, and 1,143 sentences. Reader demographics were self-reported language groups. The primary group effect used 179 original sentences rated by both groups. Predicted manifest was cross-checked against L4 release. [Original experiment specification](student-annotations-background.md) · [Local sealed snapshot rating](../../../research/results/student-annotations-fit-l4/scores.json) · [Colab execution manifest](../../../research/results/student-annotations-fit-l4/run-verification.json)

| Read out sequentially | Score NLL (nats, lower is better) | Cumulative Ordered Probability Error |
| --- | ---: | ---: |
| Sentence length and language group | 1.23197 | 0.12927 |
| Qwen status and language group main effect | 1.23068 | 0.12845 |
| Qwen state and language group interaction | 1.23469 | 0.12892 |
| Qwen status, background field left blank | 1.23048 | 0.12900 |

The interaction readout minus main effect readout NLL difference was +0.00401 nats, with a 95% cross-fold group interval of [−0.00116, +0.00930], failing to meet the improvement threshold. The main effect readout minus background holdout NLL difference was +0.00020, with a 95% interval of [−0.01624, +0.01722]. This held-out result did not confirm improvement in the single-choice response distribution.

On the same sentence, the other-language group reported higher difficulty than the German-only group, with human mean +0.532 levels, 95% interval [+0.325, +0.749]. The interaction readout estimate was +0.507 levels. the model reduction from human had a 90% interval [−0.218, +0.161] falling within the ±0.5 level equivalence bounds. The sentence-length group's readout estimate was +0.494 levels, also falling within the same range. The interaction readout passed the group direction and average magnitude conditions but failed the sequential NLL condition. The selection of English-article readers for background remained lacking direct held-out validity. [Interval endpoints and denominator](../../../research/results/student-annotations-fit-l4/scores.json)

[Original research record](../../../research/stage6/student-annotations-background-results.md)
