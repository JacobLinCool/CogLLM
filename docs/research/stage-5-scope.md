# Stage 5 · Scope for localization and dataset expansion

Expand the scope of human measurement in the next round. Retain the fixed Qwen3.5-2B and linear readout, first establishing position-aligned word and phrase endpoints. Parallel work involves filling semantic gaps in the report reading stop, earlier text segments, and subsequent updates from existing research. Each line of work has its own human evidence and validation. [Overall Execution Plan](execution-plan.md)

## Why perform localization first?

The current three-task shared adapter can be updated jointly. After incorporating two literature effects, sentence-level self-assessment and sequential reference error improved, while the interval of the difference in the two shared held-out effects still crossed zero. The Zhan shared group was estimated at +24.32 percentage points in training materials, dropping to +8.45 in held-out materials. The linear head remained the lower reference across sequential errors in the first round of shared training. These comparisons support first verifying new measurements against held-out conditions, and leaving representation and training capacity for future testing. [First-round shared results](joint-first-run.md) · [Joint effect control](joint-effects-run.md)

## First-word and phrase experiment

Use the CompLex fixed version. Readers rate the lexical difficulty of a specified word or phrase after viewing the entire sentence. the published value is the mean rating on a 0–1 scale. The original character range is the Unicode half-open interval, and un-scored segments are unobserved. Maintain the fixed sentence held-out, and report words and phrases separately. [Original measurement](https://aclanthology.org/2021.semeval-1.1.pdf#page=3) · [Original data audit](../../research/results/joint-first-run/complex-audit.json)

CPU data construction completed. Proceeded with official testing, followed by adjustment, and finally training. For cross-word and phrase tasks, sentences were grouped after Unicode normalization, case folding, and whitespace unification, removing 461 duplicate scores across split boundaries. Original character counts and offsets were preserved. The final dataset retained 10,279 scores, with zero intersection between sentence groups within each split. [Construction and Data Version](../../research/results/stage5-readiness/preparation.json) · [Re-run Program](../../research/stage5/complex_prepare.py)

| Segmentation | Character segments that can be uniquely identified | Target tokens are aligned and share the same prefix before stopping. |
| --- | ---: | ---: |
| Training | 8,680 | 8,667 |
| Adjust | 503 | 503 |
| Testing | 1,096 | 1,094 |
| In total | 10,279 | 10,264 |

The native Qwen tokenizer exhibits 15 remaining target cases where tokens span non-whitespace annotation boundaries. all three readouts are excluded. For the 10,264 aligned targets, the complete sentence token prefix and independent tokenization to the target character end are identical. Token ID verification was used, with prefix state measured in the L4 model prior to extraction. The formal comparison utilized 9,166 sentences and 5,437 distinct target strings, totaling 333,423 non-padding input tokens. Sentence pairs were scored by pairing identical target strings, with independent counts of human agreements and disagreements. [Token Verification](../../research/results/stage5-readiness/tokenization.json) · [Formal Input Verification](../../research/results/stage5-complex/input-audit.json) · [Prefix Check](../../research/results/stage5-complex/features/causality.json)

| Contrast | The matter to be tested |
| --- | --- |
| Examine only the Qwen status of the target word or phrase | To what extent can independent target strings explain human difficulty. |
| A single causal forward pass per sentence, taking the state at the final target token. | How much signal is available from the prefix when reading the target. Compare the consistency between tokenization and full-sentence prefix after the target stop. |
| The same forward pass target and final state. | The full-sentence visible recall aligns with the CompLex measurement context. |

Primary scores use held-out MAE. additionally report difficulty rankings of the same target across different sentences, and errors for targets seen and unseen during training. Perform paired resampling estimation of MAE differences on held-out sentence groups. Both extraction and fitting utilize Colab L4, initially with stratified-by-length small-batch validation. Both sentence readouts derive position states from a single forward pass. If reproducible localization gains are observed on held-out text, then evaluate whether incorporating this endpoint enables the original three tasks to share signals. [Single forward pass and position definition](mathematical-formulation.md#2-一次因果前向運算)

## Semantic Positioning and Expansion Conditions

We verified the end-of-sentence selection in INQUISITIVE, the DCQA-aligned answer positions, and the TED-Q every-two-sentence pause tracking for interrogative updates. The full INQUISITIVE corpus requires separate application, and TED verbatim transcripts require corresponding authorization for model use. Among the 1,257 TED-Q instances tracking two simultaneous interrogatives with reading stops, 88 cases were identified where only the older interrogative was rated as at least partially answered. This endpoint validates updates within sentence pairs, yet intra-sentence first triggers remain undetectable in this dataset. [Semantic Positioning and Interrogative Update Auditing](stage-5-semantic-run.md)

CompLex has completed three readouts of nested sentence sampling at 25%, 50%, and 100% on L4. Considering only the target, the held-out MAE is lowest across all three data volumes, and the MAE continues to decrease from 50% to 100% across the three readouts. The cross-sentence ranking agreement for the same target using contextual readouts is approximately 51–52%. This lexical score can serve as an auxiliary endpoint for localization work, while the position and update of semantic questions still rely on the independent observation test above. [CompLex Experiment Report](stage-5-complex-run.md)

The Currie cross-sentence contradiction endpoints have been completed at L4. Using 24 sets of near/far contradiction materials and 4,248 existing end-sentence judgments, the complete discourse and last-sentence-only conditions both outperform the training fold average report proportion. The Brier score difference interval between complete discourse and last-sentence-only crosses zero. Contradictory sentences serve as the author's position anchor in the experiment. however, data validation from reader-selected positions and segments is still required to verify semantic localization. [cross-sentence experiment and held-out results](stage-5-semantic-run.md#held-out-results)

The comparison of model scale is scheduled after the data and scoring are fixed and comparable. If increasing the same-endpoint data still yields gains, prioritize expanding the material at that endpoint. If performance plateaus but a clear held-out gap remains, then compare Qwen or the 4B backbone using the same data, steps, loss, and scoring. If the primary gap appears in external studies, prioritize increasing independent material families at the same endpoint. Throughout, adhere to existing human data.

[Original research record](../../research/stage-5-scope.md)
