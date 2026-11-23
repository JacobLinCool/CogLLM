# ContractNLI's hypothesis evidence localization

ContractNLI can evaluate whether a model, upon reading a contract, can locate sentences or list items that support or refute **pre-specified propositions**. The original study instructed annotators to review full non-disclosure agreements, select relevant sections, and then mark them as supporting, refuting, or not mentioned. The released dataset retains one merged annotated document per paper. Recorded metadata includes the original response order, the time of first contradiction discovery, the annotator's identity, and whether the same annotator subsequently resolved the ambiguity. However, only agreement with offline evidence annotation can be evaluated. the legal enforceability of the contract extends beyond this observational endpoint. [Original §2, Appendix A.1.1, Fig. 5](https://aclanthology.org/2021.findings-emnlp.164.pdf) · [Author Field Description](https://stanfordnlp.github.io/contract-nli/)

## Source and Split

The official ZIP is hosted on the Stanford NLP project page and includes the full text, evidence annotations, and source documents. The table below lists the SHA-256 hashes of the original members within the ZIP, recalculated after extraction. Verification is limited to byte ranges of three JSON files and the license file, totaling approximately 2.44 MB of compressed data. The authors released the package under CC BY 4.0 and explicitly documented the sources for SEC EDGAR files and PDFs obtained via web search in the paper. The experiments in this note are restricted to documents tagged with `sec-text` or `sec-html` and having URLs matching `www.sec.gov`. [Official ZIP](https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip) · [Release License](https://stanfordnlp.github.io/contract-nli/#license) · [Original Table 1, Ethical Consideration, Appendix A.1](https://aclanthology.org/2021.findings-emnlp.164.pdf)

| Members | Byte | SHA-256 |
| --- | ---: | --- |
| `contract-nli/train.json` | 7,608,211 | `dbceb356cd6203b35b27be94a5fa85e499a81c34c42c89ad53060b39f0257ba5` |
| `contract-nli/dev.json` | 1,183,107 | `310af7d661d2ab50ee3700169cef524c75f39fb296bbf5a515c229eb0f42e68e` |
| `contract-nli/test.json` | 2,260,203 | `460267b56052a2dc5aead98eb35eadef9e6734d5723d37b4a9790e410f812387` |
| `contract-nli/LICENSE` | 18,656 | `9e5f1b3c610b9c2da5c313bf81d577a7d1acec686bdb0384edefa6df0f90cd94` |

Of the 607 official documents, the `id` are mutually exclusive across the three splits. the source URLs and verbatim text of the EDGAR subset are also mutually exclusive across splits. After normalizing the text with Unicode NFKC, merging whitespace, and `casefold`, a set of cross-split duplicates was identified, used for training `538` and testing `537`. If the official test set is retained, removing training `538` preserves the verified test denominator, reducing the training file count to 161. The training set also contains normalized duplicates `515` and `548`, as well as pairs of files sharing source URLs: `436` and `437`, and `468` and `469`. Documents sharing the same source or same text should be grouped during sampling and resampling. These checks cover exact string matching and specified normalization, but do not yet cover all rewritten contract templates. [Data fields and three official splits](https://stanfordnlp.github.io/contract-nli/)

| Official segmentation | All files | EDGAR text/HTML | EDGAR supports | EDGAR refutes | EDGAR not mentioned |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 423 | 83 / 79 | 1,334 | 307 | 1,113 |
| Verification | 61 | 12 / 11 | 194 | 41 | 156 |
| Testing | 123 | 24 / 23 | 394 | 93 | 312 |

The in-table judgment unit is "document × hypothesis". The EDGAR test set comprises 47 documents and 799 judgments. There are 883 supporting and 172 refuting evidence spans. There are 249 and 42 multi-span judgments, respectively. Authors require annotations that are independently readable and concise. even when evidence repeats, it must still be annotated. Multiple spans may be mutually complementary text or repeated evidence. Existing annotations can only be used as a set of selected positions and cannot be used to determine whether any single span within the set is sufficient to support or refute. [Annotation definition §2.1](https://aclanthology.org/2021.findings-emnlp.164.pdf)

## Evidence pairing with assessable evidence

For each document, the `spans` provides a half-open character interval `[start, end)`, and each merged annotation selects one interval via indexing. All 10,319 JSON judgments can be linked back to a defined proposition, with the selected index and character interval falling within the range. The EDGAR test corpus contains a total of 667,386 characters, offering 4,318 candidate positions. Each document has between 13 and 218 candidates, with no overlapping intervals and all endpoints distinguishable. The actual tokenizer used must also verify that each character endpoint aligns precisely with a token boundary. [Author JSON field specification](https://stanfordnlp.github.io/contract-nli/#dataset-specification)

**Compare different hypotheses within the same file.** For all unordered hypothesis pairs in a single document that are supported by evidence, retain the complete interval set for each. A pair of intervals is counted as "non-overlapping" only if the intersection between every pair of intervals from the two sets is empty. This comparison can test whether readout shifts when hypotheses change, moving scores to different positions. [Author-specified specification](https://stanfordnlp.github.io/contract-nli/#dataset-specification)

| Test set pairs | Evidence-based matching | The complete evidence set is non-overlapping. | Number of covered files |
| --- | ---: | ---: | ---: |
| Support and Support | 1,706 | 1,548 | 45 |
| Support and Refutation | 806 | 718 | 45 |
| Rebuttal and rebuttal | 71 | 60 | 24 |

Among 718 comparable support–rebuttal pairs, 570 pairs had evidence from at least one side spanning more than two intervals. In textual order, 470 pairs had all rebuttal evidence preceding all support evidence, 182 pairs had all support evidence preceding all rebuttal evidence, and 66 pairs had the two sets of evidence interleaved. These order descriptions refer to text coordinates. If checking whether both sets of evidence were read, the stopping position must include each set's final marked interval and can only be interpreted as having seen the full set of marked intervals. [Evidence span rule §2.1](https://aclanthology.org/2021.findings-emnlp.164.pdf)

**Cross-file comparison of two judgments for the same proposition.** The EDGAR test set contains ten propositions that simultaneously exhibit support and refutation, covering 207 supportive and 93 refuting instances. Cross-matching supportive and refuting documents per proposition yields 1,158 pairs. Since a single document may appear in multiple pairings, 1,158 represents the number of constructible comparisons. Each of the six propositions has at least two supportive documents and two refuting documents. [Official test file and proposition definition](https://stanfordnlp.github.io/contract-nli/)

| Hypothesis ID | Support | Refute |
| --- | ---: | ---: |
| `nda-1` | 12 | 13 |
| `nda-2` | 1 | 38 |
| `nda-3` | 29 | 1 |
| `nda-4` | 35 | 7 |
| `nda-5` | 34 | 1 |
| `nda-7` | 30 | 10 |
| `nda-16` | 13 | 1 |
| `nda-17` | 8 | 6 |
| `nda-19` | 29 | 3 |
| `nda-20` | 16 | 13 |

These data support evaluating proposition evidence location at the document level, and comparing support and rebuttal locations across documents after fixing the proposition. If testing reading traces, the "specified proposition" should be treated as a known reading task. Separate observational data are required for the occurrence and disappearance of reader reports and questions during natural reading of identical texts. [Original §2.1, Appendix A.1.1](https://aclanthology.org/2021.findings-emnlp.164.pdf)

[Original research record](../../../research/semantic_tracking/contractnli-evidence-source.md)
