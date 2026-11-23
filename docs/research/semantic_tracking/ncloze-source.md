# nCloze's token-by-token generation and original span coordinates

Public data from Ondov et al. allow the recovery of 7,707 four-option responses mapped to original text blanks, used options, and item conditions. After excluding the tenth blank using the authors' parsing program, 7,307 responses remain, covering 12 presentation units. Readers answered multiple embedded menus within the same censored article, with text before and after the blank remaining readable. This data enables measurement of option judgment for specified blanks. To locate the prefix at which readers first achieved understanding, however, exposure and response order per blank is still required. [Paper §4.2–4.6, Figure 2](https://aclanthology.org/2024.naacl-long.220.pdf#page=7) · [Counts and file hashes](../../../research/results/ncloze-source-audit.json)

## Data merging

| Unit verification | Quantity and Results |
| --- | --- |
| Presentation unit | There are six `high` units, each containing one article. There are six `middle` units, each containing one set of consecutive `<br/>` separators between two articles, totaling 18 source articles. |
| Submit response | 200 submissions, 199 distinct worker IDs, 3 HITs. Each corresponds to two distinct presentation units, one teacher version and one experiment version. |
| Space concatenation | 7,707 A–D selections fully correspond to 1,155 presentation units × conditions × blank combinations. All 400 submitted article segments possess complete blank indexing. |
| Exclude check position | After excluding the index of each submission block `9`, 7,307 responses and 1,095 item variants remain. The teacher version includes 3,644 responses corresponding to 219 blanks. |

`exp-data.json` provides the text with gaps, options, and answer keys for each presentation unit. The `Answer` field in `results.json` wraps JSON in XML. Each response key combines the presentation unit, condition, and blank index. In all 400 blocks, the serialized key order differs from the order of the blanks in the text. Submission data record assignment acceptance and submission times and retain one final selection per blank. The paper describes randomizing the order of the teacher and experiment versions. The response fields identify the two versions and their articles, while the interaction history for each blank remains unavailable. [Stimulus and response files](https://github.com/ondovb/nCloze/tree/10260b8e2dd20aa93b5f6fdcdb83ba2bb8154ceb/Data) · [Parser, lines 17–42 and 70–101](https://github.com/ondovb/nCloze/blob/10260b8e2dd20aa93b5f6fdcdb83ba2bb8154ceb/Data/parseResults.py#L17-L101)

## Matchable conditions

By backfilling each `_` using the scoring key, a half-open character range `[start, end)` can be established for the original text. This range is calculated based on the published source string. If `<br/>` is converted to a display newline, the coordinates of subsequent spaces must be synchronized accordingly. The backfilled terms are applicable for coordinate reconstruction. the stimulus presented to the reader during response includes the article with spaces and the options. [Material Original File](https://github.com/ondovb/nCloze/blob/10260b8e2dd20aa93b5f6fdcdb83ba2bb8154ceb/Data/exp-data.json) · [Figure 2](https://aclanthology.org/2024.naacl-long.220.pdf#page=8)

| Conditions | Selections excluding the tenth column | Space and option verification |
| --- | ---: | --- |
| Teacher CLOTH | 3,644 | The teacher specifies the blank. For the 231 items, all four options are mutually distinct. |
| CDGP mechanical space selection | 862 | Shared 231 spaces with two other mechanical methods. The correct string for 87 items appears in multiple options. |
| T5-multi mechanical space selection | 927 | The same set of blanks. Of the items, 66 have duplicate options, including 20 with a duplicated correct-answer string. |
| nCloze-m mechanical space selection | 970 | The same set of blanks. For question 231, all four options are mutually distinct. |
| nCloze-r selects the blank based on context | 904 | Another set of blanks. The four options for question 231 are all distinct. |

The 12 text units and 231 spaces after three mechanical backfills are verbatim identical in their original coordinates. Excluding index `9`, and requiring each of the three conditions to have four distinct option strings, the remaining 12 units yield **92 common spaces**, corresponding to 1,159 responses. Since the three experimental conditions were answered by different readers, these pairings are suitable for comparing different option sets at the same position. The teacher version and generated version are verbatim identical in 10 units, and the 12 units are identical after NFKC normalization, case folding, and whitespace normalization. The teacher version and nCloze-r have their own space positions. [Condition Definitions §4.3–4.5](https://aclanthology.org/2024.naacl-long.220.pdf#page=7) · [Counting and Alignment](../../../research/results/ncloze-source-audit.json)

## Reproducibility and Scope of Application

The author parser expected the material file to be `compiled.json`, but the published file was `exp-data.json`. When the author program generates incorrect answer hints, it backfills other blanks using the current blank's correct answer. This coordinate check uses each answer individually. Section 4.6 of the paper requires both attention checks to be correct and the teacher version score to exceed 25%. The published program's analysis only checks the experimental version index `9` and accepts submissions with a teacher version score exactly at 25%. To reproduce the paper's accuracy analysis, one must confirm the actual options presented in the interface for both versions of the attention checks and first fix the filtering criteria. This count treats published submissions as the population, using index `9` to distinguish check positions, without calculating option accuracy. [Author Parser](https://github.com/ondovb/nCloze/blob/10260b8e2dd20aa93b5f6fdcdb83ba2bb8154ceb/Data/parseResults.py#L25-L139) · [Paper §4.6](https://aclanthology.org/2024.naacl-long.220.pdf#page=7)

The author's repository MIT license covers software and related documentation. Materials originate from CLOTH's collected third-party exam questions. response originals retain MTurk work and submission IDs. The verified release does not contain explicit authorization for either of these two categories. Consequently, the count files retain only coverage and hash. The single, fixed-position selection in the main study experiment is reproducible, and the NVS and health domain trials in §5 do not appear in this batch of release files. [MIT license](https://github.com/ondovb/nCloze/blob/10260b8e2dd20aa93b5f6fdcdb83ba2bb8154ceb/LICENSE) · [CLOTH original study §3.1](https://aclanthology.org/D18-1257.pdf#page=3) · [author release list](https://github.com/ondovb/nCloze/blob/10260b8e2dd20aa93b5f6fdcdb83ba2bb8154ceb/README.md)

[Original research record](../../../research/semantic_tracking/ncloze-source.md)
