# ARAU prior positioning held-out results

On 20 PEAR transcripts and 302 scorable new referring expressions held out from fitting, adding Qwen3.5-2B cosine similarity increased the top-five candidate hit rate **from 23.3% to 34.4%**, a gain of 11.1 percentage points with an article-paired 95% interval of [5.9, 16.9]. Negative log likelihood (NLL) for positive labels fell **from 4.596 to 4.217 nats**, a difference of −0.379 nats with interval [−0.460, −0.301]. The transcript count, top-five hit rate, and NLL all met criteria fixed before scoring. [Prespecified protocol](arrau-bridge-protocol.md) · [Scores and paired intervals](../../../research/results/arrau-bridge-l4/scores.json)

| Materials | Article/Query | Distance from top five ↑ | The top five characters preceding ↑ | Lexical + Qwen top-five hit rate ↑ | Literal and Qwen preferred ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| PEAR in its entirety | 20／302 | 17.6% | 23.3% | 34.4% | 9.5% |
| PEAR cross-sentence | 20／259 | 13.6% | 23.0% | 35.4% | 9.6% |
| PEAR within the same sentence | 14／25 | 67.9% | 28.6% | 31.0% | 6.0% |
| GNOME entirety | 5／521 | 12.6% | 24.7% | 34.4% | 11.0% |
| GNOME cross-sentence | 5／377 | 3.3% | 15.0% | 26.6% | 6.6% |
| GNOME in the same sentence | 5／143 | 37.4% | 52.6% | 55.7% | 22.8% |

Hit rate ranges from 0 to 100%, with higher values better. Average across queries within each article, then weight articles equally. A top-five hit means that at least one of the five highest-ranked word endings matches a human-annotated antecedent ending. NLL has a minimum of 0 nats. Lower values indicate that the human target received higher probability. The 19 scorable annotations whose spans could not be assigned to a single sentence remain in their respective aggregate rows. [Breakdowns and per-article differences](../../../research/results/arrau-bridge-l4/scores.json)

The improvement was concentrated in cross-sentence queries. In PEAR, the cross-sentence difference in top-five hit rate was +12.4 percentage points, with an interval of [6.5, 18.9]. For PEAR's 25 within-sentence queries, the distance readout had a higher top-five hit rate than either fitted readout. The cross-sentence results pointed in the same direction in the five GNOME documents, and overall NLL fell from 4.987 to 4.538 nats. Four GNOME documents were medical texts and the other was about art history. The genres and number of documents limit the scope of the written-text validation. [Results by stratum](../../../research/results/arrau-bridge-l4/scores.json) · [Original genres and annotations](arrau-bridge-intake.md)

The improvement in cross-sentence performance is measured when competing against both in-sentence and cross-sentence candidates, without utilizing sentence boundaries, paragraph boundaries, or manually defined part-of-speech tags. The primary PEAR result still achieves only a 9.5% first-pass hit rate. Query timestamps are manually annotated by human reference annotators. This evidence supports searching for earlier text candidates following a given phrase, as readers' natural reading processes remain uncertain regarding when confusion arises and when it is resolved. [Readout features and data boundaries](arrau-bridge-protocol.md)

## Reproducibility and resource

Reading out only the GUM-trained split fit: of 3,407 annotated items across 177 documents, 3,366 were evaluable. Of ARRAU's 855 annotations, both ends matched Qwen prefix reading stops. 32 original annotations had antecedents ending outside the scoring-eligible word range, leaving 823. This represents word-candidate alignment exclusion, with all 35,517 source word positions extracted. The legacy GUM test on 30 documents yielded a secondary diagnostic of NLL improvement of 0.214 nats and a 5.15 percentage point increase in top-five hits. [ sealed snapshot coefficient](../../../research/results/arrau-bridge-l4/fit.json) · [legacy GUM diagnostic](../../../research/results/arrau-bridge-l4/gum-diagnostic.json) · [held-out coordinates and exclusion](../../../research/results/arrau-bridge-l4/scores.json)

For each of the 25 complete texts, perform a causal forward pass using the same L4 configuration, totaling 42,939 Qwen tokens. 12 independent causal comparisons all passed. The forward pass and readout phase took 19.4 seconds, the extraction phase took 92.3 seconds, and the content batch peak retention was approximately 8.76 GiB. The L4 workspace release has been verified. The account shows 735.01 CU prior to this start and 734.94 CU after release, with only this configuration's activity present in the concurrent snapshot. The CPU held-out score was identical bit-by-bit in both cases, approximately 1.6 seconds each, with a peak of approximately 197 MiB. [Extraction and Benchmarking](../../../research/results/arrau-bridge-l4/extraction.json) · [Causal Verification](../../../research/results/arrau-bridge-l4/causality.json) · [Release Log](../../../research/results/arrau-bridge-l4/run-verification.json) · [Account Snapshot](../../../research/results/arrau-bridge-l4/usage.jsonl)

[Original research record](../../../research/semantic_tracking/arrau-bridge-results.md)
