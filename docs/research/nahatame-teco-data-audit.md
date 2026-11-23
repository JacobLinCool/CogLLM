# TECO document ratings, comprehension, and reader background

The data from Nahatame, Kimura, and Ogiso (2026) encompass 41 Japanese university-level and graduate-level English learners and 30 EIKEN expository texts. After reading each text, participants provided an overall difficulty rating on a 1–9 scale, where 9 indicates the most difficult. This is a single-item subjective rating based on the entire text, distinct from the sentence-level readability assessed by PLABA-EVAL and self-reported comprehension. For each text, trial comprehension questions were presented after each presentation paragraph to confirm that readers had engaged with the content. [2026 paper methods](https://link.springer.com/article/10.1007/s11145-026-10888-0) · [original experiment, §2.2–2.3](https://www.jstage.jst.go.jp/article/arele/35/0/35_17/_pdf/-char/en#page=5)

| Field | The meaning of the data | Audit results |
| --- | --- | --- |
| `Subjective` | Readers rate each passage individually, with 1 indicating the easiest and 9 indicating the most difficult. | 1,208 records. The recalculated mean scores of the 30 articles are consistent with the published data. |
| `acc` | Proportion of correctly answered passages. Each passage contains 3–6 true/false items, totaling 125 items. | 1,216 records, including 1,208 with scores. All proportions are consistent with the number of items in the published materials. The observed proportion average is 0.888, with 742 out of 1,216 achieving full marks. |
| `pro` | Prior to the reading experiment, general English reading ability was measured using an independent 19-item scoring scale. | 41 readers can be aligned by ID. Observed scores range from 6–19, with a mean of 12.85. Prior knowledge regarding the article topic was not measured. |
| `order` | The actual presentation order in which readers read each paper | Each reader's ordering covered 1–30. One person each in the A and B groups had a first-half arrangement that differed from the typical arrangement for their group. |

30 materials were concatenated by ID. During comparison, the PDF article body, the text spreadsheet titles and abbreviations, and the word data `textid` and `position` starting from 1 were used. After Unicode NFKC normalization, case folding, and retention of alphanumeric characters, the 30 article bodies and word-by-word sequences were fully identical. Article 27 used an abbreviated title in the PDF but matched the body text. The language model should reconstruct the body text in the original order. The answer key and solution can be configured separately according to the presentation flow of existing research. [2026 data](https://osf.io/t75nh/) · [TECO data](https://osf.io/wrvj3/) · [material PDF](https://osf.io/tj3wf/) · [word order](https://osf.io/3vb9s/) · [text spreadsheet](https://osf.io/jz5wu/) · [trial and presentation order](https://osf.io/2g4pn/) · [reader ability](https://osf.io/mfuyt/) · [field definitions](https://osf.io/hqu6s/)

This data can assess whether English reading ability contributes to reproducing the full-score distribution or full-response performance. When generalizing to new articles and new readers, assessment must be held-out separately, with variance distinguished at the article and reader levels. Existing `acc` measures capture per-article response outcomes, while per-item selection, error semantics, and difficulty locations require corresponding human observations for validation. Public readability metrics and GPT-4.1 scores can serve as baselines, with their data sources recorded separately from human ratings. [Paper Methods and Limitations](https://link.springer.com/article/10.1007/s11145-026-10888-0)

Verify using Python's `openpyxl` and Poppler's `pdftotext`. The program checks the SHA-256 hash of the fixed source. The reconstructed text can be written to local cache, while the public feed contains only counts and summaries without personally identifiable information. [Audit Program](../../research/stage5/nahatame_audit.py) · [Feed Results](../../research/results/nahatame-teco-audit.json)

```sh
python3 -B research/stage5/nahatame_audit.py .cache/cogllm/nahatame-t75nh/raw
```

[Original research record](../../research/nahatame-teco-data-audit.md)
