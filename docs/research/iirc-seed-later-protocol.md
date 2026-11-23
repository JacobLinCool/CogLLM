# IIRC protocol for later support in the same paragraph

Given an original passage and a known anchor within it, estimate when the selected supporting text chosen by the respondent is fully present within the same paragraph. IIRC, the link was selected by the questioner, and then another batch of respondents marked the supporting text. This experiment uses the evidence location published by the author `main` as the measurement endpoint. The moment the questioner first generated a query or realized the query was answered was not recorded. [IIRC §§2.2–2.3](https://aclanthology.org/2020.emnlp-main.86.pdf#page=4) · [Author file independent reconstruction](../../research/results/iirc-seed-later-audit.json)

## Data and prefix

Retain a single retrievable link and a segment of supporting text that can be verbatim traced back to the original passage and appears after the anchor. Inference conditions include the original passage's plain text, the provided anchor coordinates, the current reading stop, and the identifier for each path. Forward input contains only the original text. Readout is based on coordinate-indexed states. identifiers are used solely for data alignment. Question content, answer, link title, and evidence location do not enter inference. If pure-text anchor detection is later integrated, its false negatives and matching errors must be separately accounted for. [Data Preparation](../../research/semantic_tracking/iirc_seed_later_assay.py) · [Inclusion Rules](../../research/results/iirc-seed-evidence-audit.json)

| Split | Number of problems | Original passage and article family count | Original token count | Non-repeated reading stop counts |
| --- | ---: | ---: | ---: | ---: |
| Training | 260 | 239 | 69,070 | 3,447 |
| official dev held-out | 22 | 19 | 5,018 | 263 |

The two partitioned article families, paragraph identifiers, normalized paragraph text, and question identifiers exhibit no overlap. All anchors and at least one subsequent reading stop pass the tokenization check. The training set contains 14 shared anchors, where inputs under the same anchor may be identical across questions. The formal held-out set comprises 0 strict same-paragraph exchange pairs, allowing estimation of overall localization performance but precluding testing of item-specificity across different questions within the same paragraph. [Author Data Audit](../../research/results/iirc-seed-later-audit.json)

Each path extends backward from the anchor span's end, inserting a nominal readout stop every 16 Qwen tokens and including end-of-paragraph candidates. Prefixes that fail validation are excluded from the table. Each retained readout stop is validated against the token, character offset, and boundary of the independent sub-tokenization. Each original segment undergoes forward pass extraction exactly once, allowing the union of readout stops required by multiple anchor spans within the same segment to be exported. The Qwen3.5-2B model is frozen, and the 128-dimensional FP32 unit vectors for the required readout stops are stored. Additional prefix sampling checks are performed to verify causal consistency during truncation, right-side padding, and changes to the subsequent text. [readout stop construction](../../research/semantic_tracking/iirc_seed_later.py) · [extraction](../../research/semantic_tracking/iirc_linked_extract.py)

## Readout and validation

Three groups share the same reading stop, discrete event model, and L2 penalty. The position baseline uses the reading stop index, the number of tokens passed through the anchor, and the stride. The second group additionally incorporates the current reading stop representation, while the third group further adds the element-wise product of the current and anchor vectors. Since the representation in the second group also includes the read context, the difference in estimates between the two groups explicitly captures the gain of the anchor vector.

The main analysis compares the interval negative log-likelihood and cumulative Brier score of complete paragraphs. Earlier reading stops that reach the selected support text sooner correspond to earlier event intervals. If the support has not yet fully appeared within the observation window, the interval is capped at the right of the final reading stop. Samples themselves are filtered based on the eventual presence of later support text. all 282 complete paragraph observations yielded events. Additionally, the last reading stop within 64 tokens following the anchor is used as a fixed truncation control. First, we verify that prefix probabilities align with complete predictions sharing the same reading stop, then report the observed and capped scores under the truncation condition. This control applies to shorter observation windows conditioned on the eventual presence of later support text.

Adopting the fixed comparison threshold from the linked-evidence experiment, annotations are appended only after the held-out prediction of the sealed snapshot. Pairwise resampling across article families is conducted 10,000 times, with the upper bound of the candidate relative-position baseline likelihood-ratio difference interval required to be ≤−0.02 nats, and the cumulative Brier difference upper bound required to be ≤+0.01. The upper bound of the candidate relative-to-current representation baseline likelihood-ratio difference must be <0. Prefix control is fixed and reports only descriptive results. The 19 held-out article families constrain the precision of effect estimates. [Fixed thresholds and scoring](iirc-linked-evidence-protocol.md#metrics-and-acceptance-thresholds)

## Execution and Progress

The machine has been populated with data using the specified Qwen tokenizer. Packet `model-inputs.tar.gz` contains six members with SHA-256 hash `e345442379ee14c5e478921314329ecf9caf9e57dc3aca4a16ce9e5157232d3b`. Upon decompressing to `inputs`, the same L4 extractor is applied. The held-out actual labels `dev/locked-intervals.jsonl` are retained on the machine. [Preparation, fitting, and scoring routine](../../research/semantic_tracking/iirc_seed_later_assay.py)

```sh
python -m research.semantic_tracking.iirc_linked_extract --inputs inputs --output extraction
python -m research.semantic_tracking.iirc_seed_later_assay fit --inputs inputs --features extraction/features.npz --extraction extraction/extraction.json --output fit
python -m research.semantic_tracking.iirc_seed_later_assay score --inputs inputs --outcomes prepared/dev/locked-intervals.jsonl --forecasts fit/predictions.jsonl --fit-seal fit/fit-manifest.json --output scores.json
```

CPU synthetic testing first validates three groups of fitted, fixed-truncated right-censored scores and prediction completeness. Actual data are re-validated using 3,710 reading stops via the extraction program's tokenization rules, with data packet reconstruction verified bit-for-bit. Based on prior same-type extraction measurements of 2,419 tokens/s, the pre-run estimate for forward and readout is approximately 31 seconds. [This pre-check](../../research/results/iirc-seed-later-assay-preflight.json) · [Previous extraction measurement](../../research/results/iirc-anchor-run/extraction.json)

## Colab L4 held-out results

Fixed Qwen at L4 to process 258 documents, 74,088 tokens, and 3,710 causal reading stops, with forward and readout latency of 31.10 seconds. For 22 held-out problems, the NLL for position baseline, body state, and body plus explicit anchor span intervals were 2.6161, 2.5422, and 2.6185 nats, respectively, where lower is better. The difference in NLL for explicit anchor span relative to body-only was +0.0762 nats, with a matched 95% confidence interval [−0.0160, +0.1645]. The NLL difference relative to position was +0.0024 nats, with a 95% confidence interval [−0.0987, +0.0907], and the upper bound of the Brier score also exceeded the non-inferiority threshold. None of the three pre-specified conditions were met. [Extract Record](../../research/results/iirc-seed-later-l4/extraction.json) · [held-out score](../../research/results/iirc-seed-later-l4/scores.json)

In the 64-token truncation contrast following the held-out anchor span, 8 items were fully read with evidence marked, and 14 items were right-censored. The explicit anchor's NLL difference relative to the body text was +0.0750 nats, with a 95% confidence interval of [+0.0078, +0.1418]. This data measures only the post-hoc support of the given anchor when it becomes fully visible. [Truncation Score](../../research/results/iirc-seed-later-l4/scores.json)

[Original research record](../../research/iirc-seed-later-protocol.md)
