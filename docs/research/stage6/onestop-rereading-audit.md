# OneStop: Initial and Repeated Reading

The OneStop general reading group viewed comprehension questions only after completing each paragraph. The study had the same reader read two articles again, retaining the difficulty version of each paragraph while swapping the questions for each paragraph. After answering the questions during the initial reading, the reader viewed correctness feedback. This condition measures the re-reading experience encompassing prior reading, response, and feedback. [Original study Methods, Trial structure, and Randomization](https://pmc.ncbi.nlm.nih.gov/articles/PMC12727756/#Sec11)

Organize the public release files by trial, yielding 180 readers, 30 articles, 9,718 initial reads, and 1,944 re-reads. Pair each reader, article, segment, and difficulty version, resulting in 1,944 pairs that share identical release segment strings but differ in question ID. Exclude 3 pairs with truncated stems or options via author error checking. The accuracy difference below is re-read minus initial read. The right column uses only segment versions × question ID groups where both rounds have responses, averaging the accuracy difference for each group with equal weighting. [Initial Read Report](https://osf.io/download/xkgfz/) · [Re-read Report](https://osf.io/download/dwfk4/) · [Author Error Check](https://lacclab.github.io/OneStop-Eye-Movements/known_issues.html) · [Pairing and Recalculation](../../../research/results/onestop-rereading-audit.json)

| Article batch | Readers | Paragraph matching | Direct matching difference (percentage points) | Item number and version standardization difference (percentage points) |
| --- | ---: | ---: | ---: | ---: |
| All 30 papers | 180 | 1,941 | +5.00 | +3.77 |
| Batch 1 · 10 articles | 60 | 646 | −0.15 | −0.73 |
| Batch 2, 10 papers | 60 | 648 | +7.72 | +7.52 |
| Batch 3, 10 papers | 60 | 647 | +7.42 | +4.51 |

The direct-matching human accuracy rates were 79.24% for first reads and 84.23% for re-reads. The standardized coverage included 648 two-round matching pairs, encompassing 1,294 first reads and all 1,941 re-reads following matching. The two difference metrics used different sample ranges, both descriptive statistics. The prior OneStopQA readout utilized a separate Prolific open-book test, with material indexing up to item 20. This study features sequential responses, source text, and clear reuse terms, allowing retention of items 21–30 and the third reader cohort in formal model comparisons. [Existing OneStopQA materials](../stage-2-data.md#onestopqa-material-version) · [Original study Usage Notes](https://pmc.ncbi.nlm.nih.gov/articles/PMC12727756/#:~:text=The%20underlying%20textual%20materials)

For the same passage, certainty in readout from single-input text yields identical values for first and second read responses. To test whether reader conditions can be leveraged, the model must receive the state of having read the same article outside the passage and retain responses for evaluation. Subsequent comparisons must control for question ID and paragraph version, and include a simple model of overall read-difference learning as a baseline. [Reproduction script](../../../research/stage6/onestop_rereading_audit.py)

The second streaming run took approximately 17 seconds, with a peak resident memory of 44.8 MiB, and the two outputs were byte-for-byte identical. The downloaded trial artifacts were sourced from public repositories, and only the aggregated output was retained. [Data hashing and re-computation](../../../research/results/onestop-rereading-audit.json)

## The third-order re-reading prior

We fitted the baseline on batches 1–2 and predicted the third batch using only initial-reading versus rereading status. With articles weighted equally, human comprehension accuracy increased by 4.00 percentage points on rereading and the status prior predicted an increase of 3.78 percentage points. The Brier score fell from 0.14996 to 0.14956, a difference of −0.00040 with a 95% article-bootstrap interval of [−0.00106, +0.00027], which failed the improvement criterion. The Brier score is the mean squared difference between the estimated probability of a correct answer and the observed binary outcome, with 0 best. The mean increase closely matched the human result, while the trial-level gain remains uncertain. The prior provides a baseline for later text- and reader-background readouts. [Scoring protocol](onestop-rereading-baseline.md) · [Full results](../../../research/results/onestop-rereading-baseline.json)

[Original research record](../../../research/stage6/onestop-rereading-audit.md)
