# The first correct response position of ACT's human aligns with the clues

The German ACT main experiment by Löffler and Topolinski released results from **100 readers × 20 wordlists**. Each wordlist contained 15 fixed-numbered cues. upon seeing each word, readers could input a guess or skip. The trial ended immediately upon a correct response. Two pilot studies, each with 100 participants, assessed cue proximity and were not included in the first-response analysis. [Original Study, Pre-study, Sample, Materials and Procedure](https://reference-global.com/article/10.5334/joc.254?tab=article) · [Author Open Data](https://osf.io/et5kb/)

## Verified materials

We extracted 20 answer headings and 300 numbered cues from the OSF word-list PDF. The journal appendix and OSF PDF differ at the byte level. An item-by-item comparison confirmed identical headings, word forms, and order. Each model input retains the original spelling and order of cues, joined by line breaks. We exclude headings, answers, relatedness ratings, and trial data. In list 7, the appendix gives the seventh and eighth cues the same relatedness rating. We preserve the order from their numbering, which agrees across the two sources. This tie lies outside the primary six-step analysis. The public files provide cue numbers and steps, without trial-level screen recordings. [Journal appendix](https://doi.org/10.5334/joc.254.s1) · [Original appendix archived by PMC](https://pmc-oa-opendata.s3.amazonaws.com/PMC9838243.1/joc-6-1-254-s1.pdf) · [Original OSF word lists](https://files.us.osf.io/v1/resources/et5kb/providers/osfstorage/6284cc9c47df854ff26ef906)

The German instruction page first presents an example with known answers. The formal word list order is randomized independently for each reader. Within each word list, cues are presented in a fixed order, each displayed for two seconds. Omitted steps still count as one cue exposure. The author's response file `clues_*` and `unsolved_*` can be linked back to the 20 original word lists. The file does not retain readers' step-by-step raw guesses or personalized word list orders. [German instruction original file](https://files.osf.io/v1/resources/et5kb/providers/osfstorage/6380c985f284991927ddcce1) · [response original file](https://files.us.osf.io/v1/resources/et5kb/providers/osfstorage/62666ba85bb1ac1b2d460e95)

| Scoring range | Pinpointable first correct response | Failed to answer correctly by the due date. | Authors marked as correct with missing steps |
| --- | ---: | ---: | ---: |
| The first six exposures | 355 | 1,644 | 1 |
| All fifteen exposures | 1,321 | 678 | 1 |

Trials that answered correctly after step six were treated as right-censored at step six in the main analysis. Trials with missing step counts were independently retained and could not be recovered from the individual overview. the denominator for both analyses was 1,999. The first correct response event corresponds to the cue word. There are different observable granularities for which sub-tokens a word is split into, and when spontaneous questions in ordinary text are formed. [Stimulus and Outcome Reconstruction](../../../research/semantic_tracking/act_prepare.py) · [Endpoint and Model Comparison](act-transfer-protocol.md)

The journal article is marked CC BY 4.0, while the open metadata of the independent OSF data repository is unlicensed. Reproducibility is ensured via the original file hash. Raw human responses, de-identified trials, and the full vocabulary are retained in the local reproducibility folder. [Article licensing and data availability](https://pmc.ncbi.nlm.nih.gov/articles/PMC9838243/) · [OSF project data](https://osf.io/et5kb/)

[Original research record](../../../research/semantic_tracking/act-source.md)
