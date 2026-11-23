# How well does CogLLM compare two revisions of the same source?

People can prefer the wording of one revision and the information in another. CogLLM correctly separated those two preferences in 50.7% of such comparisons on new English texts, compared with 37.4% for a trained lexical editing model. When people preferred the same revision on both dimensions, CogLLM selected it on both dimensions only 23.2% of the time. The model often anticipates a trade-off. Its accuracy depends on whether readers actually experienced one. [Registered contrast](revision-contrast-protocol.md) · [All contrast estimates and intervals](../results/reading-validation-revision/contrasts.json)

## What the readers rated

Eleven judges in SimpDA 2022 rated six candidate revisions of each of 60 English sources on simplicity, preservation of meaning and fluency. Each candidate received three judgments per aspect, giving 1,080 ratings per aspect. We compare two drafts of the same source using their average human scores. Each source receives equal weight, ties in human scores are excluded, and ties in model scores receive half credit. The intervals below resample sources and judges in 10,000 paired draws. Source-only intervals and comparisons against the preferences of individual judges are also available. The three grades describe complete drafts. None supplies a time or word at which reading became difficult. [Original study](https://aclanthology.org/2023.acl-long.905/) · [Evaluation design](revision-protocol.md) · [Scores](../results/reading-validation-revision/scores.json)

| Human preference on new sources | Eligible pairs / sources | Source-relative CogLLM (95% interval) | Supervised lexical editing model | Choose shorter draft | Choose easier draft by absolute ReadMe++ rating |
| --- | ---: | ---: | ---: | ---: | ---: |
| Simplicity | 893 / 60 | 61.4% [53.2, 66.7] | 51.5% | 68.0% | 68.1% |
| Meaning preservation | 844 / 60 | 58.9% [51.4, 66.0] | 58.5% | 41.0% | 41.7% |
| Fluency | 790 / 59 | 56.0% [48.7, 65.8] | 55.4% | 55.9% | 50.4% |

On the primary simplicity outcome, Qwen features gained 9.9 percentage points over the supervised editing model [2.7, 13.7]. Choosing the shorter draft outperformed CogLLM by 6.5 points [0.2, 11.7]. The absolute ReadMe++ model also did better on average. With word counts within 10% of one another, CogLLM chose the simpler draft 46.5% of the time [38.2, 56.6] in 307 pairs, compared with 53.3% for shorter length and 63.7% for ReadMe++. With exactly equal lengths there were 59 pairs in 38 sources, and CogLLM reached 48.6% [28.2, 64.7]. The proposed training did not improve the primary decision over length. [Group and individual estimates](../results/reading-validation-revision/scores.json) · [Length-matched analysis](../results/reading-validation-revision/contrasts.json)

For meaning preservation, a separately trained source-relative score gained 17.2 points [6.3, 29.4] over simply applying the ReadMe++ ease ordering. A frequency-weighted source-word coverage rule reached 57.6%, and the gain over the trained editing model was 0.4 points [−1.8, 4.8]. For fluency, the corresponding gain over that editing model was 0.6 points [−5.6, 9.1]. Against another judge's own simplicity preferences where they rated both revisions, CogLLM agreed 60.8% of the time, shorter length 64.0%, and the other available judges 63.8%. With only three judges per draft, individual preferences and the average panel ordering provide materially different targets. [Rater and model comparisons](../results/reading-validation-revision/scores.json)

## When simplicity and preservation disagree

| Which drafts the panel preferred | Pairs / sources | CogLLM chose both preferences | Supervised editing model chose both | Using ReadMe++ ease for both |
| --- | ---: | ---: | ---: | ---: |
| Different drafts | 441 / 59 | 50.7% [40.6, 59.4] | 37.4% | 0.0% |
| Same draft | 398 / 60 | 23.2% [14.5, 34.2] | 23.0% | 56.5% |

On the first row the paired gain over the supervised editing model was 13.3 points [4.8, 19.3]. A single ordering used for two objectives cannot select opposite winners except by giving partial credit to ties. A trained comparator with two scores is therefore essential. The second row shows the cost of learning opposing directions too readily. Membership in either row is known only after humans rate both drafts. These estimates do not license an automatic rewrite decision in advance. [Predictions frozen before evaluation](../results/reading-validation-revision/forecast/forecast-seal.json) · [Joint-preference results](../results/reading-validation-revision/contrasts.json)

## What joint training contributed

Development used 4,196 original–revision pairs across 359 source families from ASSET, WikiDA and SimpEval-past. Graded judgments and paired votes had separate observation models. SimpEval-past's composite quality rating had its own head. Pair comparisons derived from a set of grades were grouped with those grades when weighting the studies. They are dependent observations. Each fit omitted eligible source families together across corpora. All 60 evaluation source IDs survived the registered exact and near-overlap screen against development materials. Public pretraining exposure remains unknown. [Source revisions and exclusions](../results/reading-validation-revision/inputs.json) · [Losses and weights](revision-estimator.md) · [ASSET](https://aclanthology.org/2020.acl-main.424/) · [WikiDA release](https://github.com/feralvam/metaeval-simplification/tree/7703c461858cb2557ff1d8ce0a20c63d8a3d280f)

Source-relative features measured added and deleted words, potential omissions of protected information, latent alignment between words, and changes in the absolute difficulty score. The out-of-fold mixture put weights of 1.000, 0.885 and 1.000 on the full model for simplicity, preservation and fluency. Holding features, data, individual observation weights and ridge penalty fixed, the shared model differed from independent heads by −0.56 [−1.62, 0.44], −0.44 [−1.51, 0.44] and +0.32 [−1.47, 1.45] percentage points, respectively. A separately prespecified length or coverage prior reached 62.4% simplicity and 58.5% meaning agreement. The observed gain over a fitted lexical-edit baseline does not establish a gain from sharing parameters across tasks. [Sharing specification](revision-sharing-protocol.md) · [Prior sensitivity](revision-prior-protocol.md) · [All comparator estimates](../results/reading-validation-revision/scores.json)

## Reproduction and access

The Colab NVIDIA L4 encoded 4,968 distinct passages and 141,936 model tokens in 311 batches and 70.7 seconds. Each of the 103,901 word spans used in the neural alignment had an exact token-end match. Three separately checked passages and their shortened prefixes required six technical forwards. Across native unbatched and prefix checks, the largest Euclidean norm of the state difference divided by the norm of the reference state was 1.18 × 10⁻⁶. Each new draft takes one backbone pass when the original has already been encoded. Comparing an uncached original and a single draft takes two passage encodings. The maximum absolute training gradient over 24 fold fits and full refits was below 3.1 × 10⁻⁷, with at most 101 optimizer updates. [Extraction, convergence and checksum verification](../results/reading-validation-revision/execution.json)

The study-specific protocols, scoring programs and source revisions were locked before linking any model prediction to the new human scores. The researchers saw published system-level averages while verifying the response scale. No model feature identifies the originating system. Raw passage files are available from the original authors. The repository stores input checksums, trained heads and forecasts with content hashes. The remote Qwen state archive resides in the working cache. [Analysis lock and disclosure](../results/reading-validation-revision/analysis-lock.json) · [Numerical forecasts](../results/reading-validation-revision/forecast/forecasts.jsonl) · [Cross-fitted development results](../results/reading-validation-revision/development-scores.json)

From the repository root, after acquiring the pinned source files and the L4 feature directory, the following commands reproduce the numerical fit and evaluations. Use a scratch copy of the workspace when recreating the published score files.

```sh
python -m research.reading_validation.revision_prepare
python -m research.reading_validation.revision_fit --gpu /path/to/revision-forecast --out /tmp/cogllm-revision-replay
python -m research.reading_validation.revision_score --forecast /tmp/cogllm-revision-replay
python -m research.reading_validation.revision_contrasts
python -m research.reading_validation.revision_development
```

The last two analyses read the repository's sealed forecast directory. To check numerical parity, compare the fresh per-candidate logits and coefficient arrays with the sealed forecast artifacts. [Extraction code](revision_forecast.py) · [Estimator](revision_fit.py) · [Evaluation code](revision_score.py)
