# Validation of answer cues for the same query

In this round of evaluation, when two issues are known, does reading the clue reduce the unsolved score of only the issue for which an answer is obtained? A single-frozen Qwen3.5-2B causal forward pass obtains native token representations, and the readout simultaneously tracks two issues already visible in the body or stem. Test candidates are given as text spans, without automatic hypothesis discovery. The response, test conditions, and scoring location remain in the local scoring file.

## Three levels of evidence

1. **Human first correct.** ACT presents the same reader with the same designated target, revealing cue words in the author's fixed order one by one, and stopping upon the first correct response. The previous sealed snapshot Qwen forward re-matching maintains the original five-word held-out fold. A stepwise likelihood ratio based on a binary risk set compares, a priori, only the step number, text length, and three causal Qwen state changes. The primary metric is the complete 15-step (including right-censoring) held-out NLL, paired and ranked by the same number of steps and the same added token length. The causal comparison between two continuations viewed by the same reader does not occur here. [Source and Observation Window](act-source.md)
2. **Human-annotated text evidence.** ExpMRC retains two questions from the same article, each covering reference evidence in two non-overlapping regions, following the article's segmentation from the previous round. Using the same readout and the same token reading stop, it compares whether human-annotated support belongs to the first or the second question. Both original questions can be presented simultaneously at the top, ordered by a hash of the irrelevant answer time. This readout measures human-annotated visible evidence. question presentation and its word-by-word cognitive state were not observed in the original experiment. [Existing evidence audit](expmrc-transfer.md)
3. **Extension of the Equal-Length Pair Problem.** The self-generated material explicitly presents two unanswered questions. Each extension following A or B is aligned with a fixed tokenizer to share identical prefixes and equal-length answer cues. For A, the extension answering B serves as a no-answer control. for B, the roles are reversed. The material is clustered into held-out sets. Only the full text with both questions visible and one of the two extensions are fed into the model. The authors annotate and pair the two extensions to evaluate model behavior.

## Readout and validation

At the start of each observation window for known problems, the unresolved mass is set to 1. Upon reading the $t$th token, the conditional resolution rate is $h_{r,t}$, and the unresolved load is

$$
S_{r,t}=\prod_{j \le t}(1-h_{r,j}).
$$

Fitted to the within-window first-event probability and right-censored survival probability. only time prior to the scoring window is included within the confirmable prefix range. The start and end of a cue are fixed prior to extraction representation, with both training and held-out splits stratified by topic or source article, and out-of-course anchors determined by given visible questions. The model cannot consult the reference solution to this question.

Using the same set of representations, we compare the number of steps, steps plus the current Qwen representation, steps plus surface overlap between the question and text, and the interaction feature combining known questions with the current Qwen representation. The pure step and current Qwen readout does not receive question identity. thus, two questions at the same version and position should yield identical scores. The question interaction readout allows the same token to produce different resolution rates for two different questions.

Accept the answer window's drop as the metric. For each self-generated family, compute the drop as "number of answered questions minus the other question" across two continuations, then average. ExpMRC employs an early-to-person support window, comparing two questions within the same window, while separately validating the later question's own support window for training. Only the retention quantity based on step count is affected by window length. paired comparisons use identical step counts and windows for both questions. Additionally report the first difference position, premature drop before the answer window, and held-out NLL for solved and unsolved cases. Materials use equal weighting, with 95% paired resampling at the level of topic or article. When the original text shares a prefix, compare the corresponding token ID, original character coordinate, and curve value.

In Colab L4, we extract a single native token representation for each new input. All scoring groups share a sealed snapshot representation, with readout fitted on CPU. The newly extracted representation uses the same 160-dimensional fixed projection, FP32 official weights, and causal acceptance as the first round of the uncurved baseline. Using a single L4 GPU, we maintain the 24-minute hard limit including cleanup. The ACT readout directly reuses the forward representation from the prior sealed snapshot.

[Original research record](../../../research/semantic_tracking/same-question-protocol.md)
