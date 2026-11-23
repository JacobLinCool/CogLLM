# IIRC localization of earlier anchors

## pre-training design prior to held-out evaluation

Given the reader-selected anchor, the model estimates the support interval preceding it within the passage. The human gold standard comes from the `main` evidence segment selected by IIRC respondents, verbatim matching the original passage. Available data cover 533 training and 65 official dev held-out questions, with 438 and 51 source family articles, respectively. The two sets of passages, segments, and question identifiers have no overlap. The held-out inference reads only the body text, question identifier, and anchor span. [Author Experiment §2.3](https://aclanthology.org/2020.emnlp-main.86.pdf#page=4) · [Data Audit](../../research/results/iirc-seed-evidence-audit.json)

Both groups share candidate spans preceding the anchor span, with each candidate limited to a maximum of 28 consecutive word segments. The form baseline incorporates phrase length, distance from the anchor, relative position within the span, case, punctuation, numbers, and word overlap with the anchor. The Qwen group additionally incorporates fixed Qwen3.5-2B span head and tail representations, along with their respective interactions with anchor representations. Each paragraph undergoes a single FP32 forward pass. The anchor endpoint must pass an independent prefix tokenization check, and candidate representations are derived solely from tokens preceding the anchor start. Both groups are fitted with equal weighting, where the loss is the conditional negative log-likelihood of gold-standard overlapping candidates, and the ridge coefficient is 0.005. [ sealed snapshot experiment implementation](../../research/semantic_tracking/iirc_seed_pointer.py)

The primary metric is the proportion of candidate pairs where the intersection-over-union with the gold character span is at least 0.5 and ranked first. The denominator includes items that failed to authenticate the anchor and thus did not generate a ranking. A 10,000-pair bootstrap resampling was performed by source document. the Qwen reduction from the baseline form was considered passed only if the lower bound of the difference interval exceeded zero. Additionally, the average intersection-over-union of the first-ranked item, the top-five coverage rate, and the upper bound of the candidate set coverage were reported. This dataset contains only one strictly exchangeable dev earlier text span, so the improvement in the primary metric cannot alone prove discrimination across multiple questions within the same span. [Primary Metric Code](../../research/semantic_tracking/iirc_seed_pointer.py) · [Pair Count](../../research/results/iirc-seed-evidence-audit.json)

## held-out results

**The top-ranked primary metric failed to meet the predetermined threshold.** The original IoU range is 0–1. higher values in the table below indicate coverage of more previously labeled earlier text. All 65 held-out anchor spans were certified, with each question having an IoU segment of at least 0.5 in the candidate set. [ sealed snapshot predicted formal score](../../research/results/iirc-seed-pointer-run/scores.json)

| Evaluation | Formal baseline | Add Qwen | Qwen reduction baseline with 95% matched interval |
| --- | ---: | ---: | ---: |
| First candidate coverage | 18.46%（12/65） | 30.77%（20/65） | +12.31 percentage points [−1.56, +26.15] |
| First candidate average F1 score | 0.2065 | 0.3418 | +0.1353 [0.0213, 0.2543] |
| Top-five candidate coverage | 27.69%（18/65） | 53.85%（35/65） | +26.15 percentage points [11.48, 40.33] |

Secondary metrics support the claim that increasing candidate time enables retrieval of more earlier text. The primary result remains unaccepted, and the evaluation fed both groups known anchors. Predicting anchors followed by retrieving earlier text, and determining when readers notice the solution, still require separate validation. [Full metrics and acceptance](../../research/results/iirc-seed-pointer-run/scores.json)

## Execution and Reproducibility

Local Metal extraction with FP32 Qwen covered 489 paragraphs and 135,576 tokens. Training used 531 questions. One question could not be included because tokenization could not certify the anchor endpoint, and another lacked a candidate that met the overlap threshold. The maximum absolute projection difference was 0 across all three independent truncated-prefix checks. Extraction, fitting, and sealing took 98.04 seconds. [Run and sealed weights](../../research/results/iirc-seed-pointer-run/manifest.json)

Using `iirc_seed_pointer` `prepare`, `forecast`, and `score` in sequence as fixed inputs, sealed snapshot predictions, and held-out gold standards. Independently re-read the authors' public repository to verify 598 anchor spans and segments across both groups, recalculating metrics and matching intervals. Data construction and sealed snapshot score outputs were rerun bit-for-bit consistent. [ sealed snapshot weights](../../research/results/iirc-seed-pointer-run/heads.npz) · [Independent verification](../../research/results/iirc-seed-pointer-run/verification.json)

## The subsequent evidence interval in the same paragraph

Additionally, after cross-checking the 260/22 anchor spans against the original author files, the support segments were derived from 239/19 training and held-out source families. The dev evidence endpoint appeared 91–953 characters after the anchor span, with a median of 271.5. No in-document exchange pairs met the strict criteria in this dev set. All selected titles had at least one subsequent support segment satisfying the rules. thus, the interval distribution is conditioned on the presence of subsequent evidence. [Official file reconstruction](../../research/results/iirc-seed-later-audit.json)

Reusable scoring paths backtrack from the anchor end, identifying prefixes eligible for exact tokenization verification at fixed intervals of every 16 tokens. Each reading stop estimates the probability interval where the evidence fully appears, with the final cell retaining the probability exceeding the current observation range. Model candidates are scored based on the anchor and the projected state of the prefix at that time, contrasting against a form using only the tokens passed and the interval length. Synthetic weights are used to verify probability normalization, the negative log score of interval and right-censoring, and the consistency of prior probabilities when truncation and rewriting occur. Later evidence remains human held-out, unmodeled or unscored, where intervals in the data correspond to the reading stop at which the supporting text first becomes visible. [Interval API](../../research/semantic_tracking/iirc_seed_later.py) · [Synthetic Check](../../research/results/iirc-seed-later-synthetic-checks.json)

[Original research record](../../research/iirc-seed-pointer-run.md)
