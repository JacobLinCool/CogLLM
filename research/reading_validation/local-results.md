# What can CogLLM locate when people read scientific text?

A frozen Qwen3.5-2B readout can rank which of two designated terms a researcher will find less familiar in a scientific abstract. A reader's own rating supplied before reading was substantially more informative. In biomedical abstracts, a separate phrase head retrieves many expressions that readers explicitly named once the relevant sentence has been supplied. Retrieval across the entire abstract is less reliable. These measurements concern familiarity, retrospective difficulty reports and self-rated understanding. None timestamps the moment an unaided reader got stuck.

[Figure and caption](../results/reading-validation-local/figure/caption.md) · [Vector figure](../results/reading-validation-local/figure/term-familiarity-and-localization.pdf) · [Numerical source data](../results/reading-validation-local/figure/source-data.csv)

## Transfer to an independent study of term familiarity

Guo et al. collected before- and after-reading ratings and requests for supplementary information from eleven computer science researchers. We froze the PLABA-trained span model and the independent CompLex head. We then encoded 100 abstracts in 100 single Qwen forwards on an NVIDIA L4. We locked predictions before joining the responses. Of 961 source–term entries, 581 had a unique eligible occurrence within one sentence, 185 had multiple occurrences, 144 were absent, and 51 did not meet word-boundary requirements. Eligible terms covered 94 abstracts and 6,391 anonymous reader–term responses. Eighty-three abstracts had at least one informative within-person comparison for the primary outcome. None overlapped a training text under the specified five-word screening rule. The distributed CSV stores the five ordered response categories as 0 through 4. We mapped them to the original questionnaire's 1 through 5 scale before computing changes. [Original study](https://aclanthology.org/2024.naacl-long.255/) · [Forecast and overlap lock](../results/reading-validation-local/jargon/prediction-lock.json) · [Protocol](local-jargon-protocol.md)

| Within the same reader and abstract | Form and position | PLABA local phrase head | Frozen CompLex lexical head | Local minus form, paired 95% interval |
|---|---:|---:|---:|---:|
| Which term was less familiar after reading? | 56.1% | 65.5% | 70.5% | +9.4 points [5.2, 13.5] |
| Which term was less familiar before reading? | 57.6% | 66.0% | 69.5% | +8.5 [4.1, 12.7] |
| For which term did this reader want a definition? | 58.4% | 68.5% | 71.3% | +10.1 [5.2, 15.3] |
| For which term did this reader want background? | 52.7% | 58.9% | 63.4% | +6.1 [1.7, 10.9] |

The primary comparison included 10,694 unequal term pairs across 773 reader–abstract combinations. Each abstract received equal weight and intervals resampled both readers and abstracts. The simpler CompLex head surpassed the local phrase model by 4.9 percentage points [0.8, 9.4] on this endpoint. A reader's own prereading rating ranked postreading familiarity correctly in 86.1% [83.4, 88.6] of comparable instances. In a subsequent descriptive check, the mean rating of the other ten researchers predicted the held-out reader's ordering in 81.5% of instances. Familiarity increased in 2,207 responses, decreased in 354, and stayed constant in 3,830. These are two survey times for a designated term. The data cannot establish where in the intervening abstract familiarity changed. [All estimators, counts and intervals](../results/reading-validation-local/jargon/scores.json) · [Other-reader comparison](../results/reading-validation-local/jargon/human-reference.json)

A secondary analysis matched pairs of terms on the same reader's five-category prereading rating within the same abstract. Of 7,866 matching pairs, 1,838 received different postreading ratings. On these pairs the lexical head ranked the term that remained less familiar correctly in 58.2% [52.0, 66.0] of cases, the local model in 56.2% [50.3, 63.1], and form in 50.8%. Matching the initial category reduces observable differences in prior knowledge. The ordinal scale leaves room for differences within each category. This analysis was specified after the primary results. [Matched-start protocol](local-context-change.md) · [Paired estimates](../results/reading-validation-local/jargon/matched-start.json)

In a separate calibration experiment we simultaneously held out the reader and one fifth of the abstracts. For postreading familiarity, the negative log likelihood was 1.575 for surface features, 1.490 for the combined textual readouts, 1.062 for the reader's prereading rating, and 1.058 for text plus the prior. The paired increment from adding text to the prior was −0.004 nats [−0.020, 0.012]. A term-specific prior improved prediction of subsequent requests for a definition and background. Knowing the reader's term rating is an effective form of personalization for these outcomes. Whether the text adds predictive information after a reliable prior can be collected remains uncertain. [Method](local-knowledge-protocol.md) · [Crossed holdout results](../results/reading-validation-local/jargon/crossfit-scores.json)

## Where readers explicitly named difficult wording

PLABA-EVAL yielded 94 distinct source spans in 70 sentences, 27 article variants and 19 article families from 42 readers. We used an existing positive-only audit. Readers could inspect the full article and choose what to mention. A hit meant overlap of at least half the union of predicted and named Unicode character intervals. Given a sentence, the full model proposed up to five non-overlapping phrases of one through eight consecutive words. All eligible phrases competed within the held-out sentence. The 12 folds held out connected reader and article groups. Earlier project work had used PLABA, so these estimates are retrospective. [Original source](https://github.com/SDS-NLP/PLABA-EVAL/tree/153082e7d58b833d5e2f481b92fdcf9248c60d11) · [Observation audit](../results/plaba-explicit-span-audit.json) · [Statistical protocol](local-protocol.md)

| Ranking applied to the supplied sentence | Named spans recovered among five suggestions |
|---|---:|
| Current checkpoint's displayed unresolved level | 21/94 |
| Largest positive single-step increase in that level | 51/94 |
| Trained frequency, length and position | 64/94 |
| Surface plus a MedReadMe-trained terminology lexicon | 69/94 |
| Local span head incorporating frozen Qwen and CompLex | 75/94 |
| Local span head also given persistent signals | 76/94 |

When the interface underlines only the highest-ranked phrase, the local head retrieved 45/94 named spans (47.9%), surface features retrieved 25/94 (26.6%), and the persistent level retrieved 7/94 (7.4%). The paired top-one gain over surface features was 21.3 percentage points [15.9, 26.7]. Requiring exact boundaries reduced the local top-one count to 30/94.

The local model's gain over the form model was 11.7 percentage points [5.6, 19.7]. The single additional hit from the persistent features had an interval including zero. Within entire abstracts the original local ranker retrieved 26/94, versus 19/94 for form, with a difference interval of −4.8 to +19.8 points. A subsequent sentence-selection model, specified after viewing the first result, retrieved 28/94 against 19/94 for its corresponding form comparator, +9.6 points [4.1, 16.3]. Its improvement over the original local model was 2.1 points [−6.2, 11.6]. False-positive rates require labels for all passages a reader did and did not find difficult. Optional written comments provide only named positives. [Sentence results](../results/reading-validation-local/localization/scores.json) · [Article retrieval protocol](local-article-protocol.md) · [Article results](../results/reading-validation-local/article-localization/scores.json)

## Sentence ratings and section headings

On PLABA we fit a small regularized ordinal head to 609 sentence-level human response distributions and 265 alignments between original and plain-language versions. Across five folds, the 25 connected article/reader groups stayed intact. The mean human ease gain was +0.537 points on the 1–5 scale. The full local sentence head predicted +0.497 and the preceding frozen sentence model +0.415. For self-rated understanding the corresponding gains were +0.723, +0.626 and +0.558. Observed improvements in held-out ordinal negative log likelihood over the previous frozen head were small, with intervals spanning zero for both outcomes. [Sentence and paired rewrite results](../results/reading-validation-local/ratings/scores.json)

We also assessed the effect of section headings on comprehensibility in the previously examined Kerwer corpus. There were 1,283 ratings from 166 readers across 12 sources. Mean self-rated comprehensibility increased by 0.359 points [0.062, 0.676] on the 1–8 scale when headings preceded the otherwise matched body text. A PLABA model using 32-dimensional Qwen features predicted +0.042 [0.019, 0.066], leaving a shortfall of 0.316 [0.029, 0.623]. The surface-only contrast was zero. This comparison used matched states for 181 sentence pairs. The separate deployed 2,048-dimensional rating head requires native terminal states that were unavailable in this Kerwer cache. [Heading study](https://doi.org/10.1525/collabra.18898) · [Matched-body estimates](../results/reading-validation-local/headings/scores.json)

## Independent readers assessing words in isolation

We assessed the archived CompLex target-only head on word complexity ratings from eleven fluent English L2 readers and eleven deaf or hard-of-hearing adults. The separately collected studies use the same word inventory. Before examining ratings we selected 2,440 available forms, including 184 forms that occurred in neither the training nor trial CompLex target vocabulary, even as components of multiple-word targets. On these novel forms, the mean within-reader Spearman correlation was 0.424 [0.294, 0.533] for the L2 cohort and 0.308 [0.222, 0.393] for the DHH cohort. The corresponding trained word-length and subtitle-frequency correlations were 0.363 and 0.278. Paired advantages over that comparator had intervals spanning zero for both groups. The eligible novel sample contained 1,213 L2 and 1,953 DHH ratings. Five DHH values in the entire matched inventory fell outside the published response levels and were excluded as invalid. These results estimate shared word rankings among the represented readers. They provide no estimate of effects across measured language proficiency levels. [L2 study](https://aclanthology.org/D18-1410/) · [DHH release](https://github.com/oliveralonzo/DHH-lexical-dataset) · [Forecast lock](../results/reading-validation-local/word-readers/prediction-lock.json) · [Scoring results](../results/reading-validation-local/word-readers/scores.json)

## Using the readout

The English adapter accepts the original text, the existing frozen backbone's token features, its tokenizer, and the selected reader. It returns half-open Unicode offsets for up to five individually ranked, nonoverlapping phrases per sentence. Its sentence-level estimates of ease and self-rated understanding use the PLABA 1–5 scale. The phrase score is a ranking value within a completed sentence. Use that score for retrospective review and preserve the separate causal trace when asking about an unresolved issue during reading. Replaying the exported coefficients against archived L4 features reproduced all 3,044 checked top offsets and all 609 sentence estimates exactly. Deterministic sentence detection recovered 602 of the 609 author-supplied intervals. Source-defined spans can be supplied when available. In a separate writer-interface replay the deployed adapter processed three real passages (75 tokens) in three backbone calls on an NVIDIA L4. [Writer verification](writer_verify.py) · [Inference adapter](local_inference.py) · [Model and resource identity](../results/reading-validation-local/serving/manifest.json) · [Replay validation](../results/reading-validation-local/serving/replay.json)

## Reproduction from the shared experiment archives

Run the following from the repository root with NumPy and SciPy. The fitters validate the checksums of the source studies and Qwen feature archive on entry. These example paths identify the two archived NVIDIA L4 runs used above. Native inference ran once per unique article and no decoder tokens were generated. PLABA and Kerwer shared the first 102-document pass; Guo occupied 100 separate document passes. [Selected readout identity](../results/adaptive-joint-fixed-retention-l4/seal.json) · [PLABA and Kerwer input hashes](../results/reading-validation-local/input-seal.json) · [Guo source and stimulus hashes](../results/reading-validation-local/jargon/source-seal.json)

```sh
validation=.cache/cogllm/reading-validation/l4
local_gpu="$validation/cogllm-reading-validation-20261122T153541Z-eec8fa/extraction/local-forecast"
guo_gpu="$validation/cogllm-reading-transfer-20261122T155723Z-7b357d/extraction/jargon-forecast"
result=.cache/cogllm/reading-validation/reproduction
python -m research.reading_validation.local_rank --gpu "$local_gpu" --out "$result/localization"
python -m research.reading_validation.local_ratings --gpu "$local_gpu" --out "$result/ratings"
python -m research.reading_validation.local_article --gpu "$local_gpu" --out "$result/article-localization"
python -m research.reading_validation.local_heading --gpu "$local_gpu"
python -m research.reading_validation.local_export --gpu "$local_gpu"
python -m research.reading_validation.local_provenance --gpu "$guo_gpu"
python -m research.reading_validation.local_jargon score
python -m research.reading_validation.local_knowledge
python -m research.reading_validation.local_context_change
python -m research.reading_validation.local_human_reference
python -m research.reading_validation.local_word_readers score
```

The provenance replay writes the two independent lexical forecasts to temporary directories and verifies byte-for-byte identity against the original pre-outcome predictions. The outcome readers were corrected to match the Guo release’s category indexes and to discard the DHH release’s out-of-scale values. Both original forecast functions match their pre-outcome code. [Replay receipt](../results/reading-validation-local/forecast-replay.json)

The isolated-word test reuses the historical target-only features and does not run a new backbone. Obtain each author release from the revisions in the word-reader input receipt and place the two TSV files and subtitle-frequency file in the local data cache. Keep volunteer response files local. The figure renderer consumes only the stored numerical summaries. [Word-reader inputs](../results/reading-validation-local/word-readers/prediction-lock.json) · [Figure renderer](../results/reading-validation-local/figure/render.py)
