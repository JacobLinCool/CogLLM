# Sentence-by-sentence contradiction judgments between Frick and Schüler

The public data from the first experiment in Frick and Schüler (2026) link **the first keypress by a reader reporting an inconsistency in a story** to the sentence displayed at the time. PsychoPy displayed whole sentences one at a time. Readers pressed `K` if a sentence was consistent with the preceding text and `D` if it was inconsistent. The next sentence appeared after the keypress. The instructions directed readers to press `K` on the first sentence, so first reports are counted from the second sentence onward. Each keypress concerns the current sentence's consistency with earlier text. A `K` after a `D` evaluates a different sentence and supplies no measure of whether the earlier inconsistency was resolved. Readers saw the entire sentence before responding. Neither a triggering word within the sentence nor an unsolicited report during natural reading was directly observed. [Original paper, Experiment 1, measures and procedure](https://doi.org/10.3758/s13421-026-01900-7)

## Source and Alignment

| Author Profile | Verified content |
| --- | --- |
| [`data_final_experiment_1.xlsx`](https://osf.io/download/hp9sg/) | 27,917,882 bytes. SHA-256 `439a63a228d06257661c76ed5b9b96ae3c8f57c4ac08b393844a9ef9cf517ba4`. A single worksheet contains 133 columns, 72,723 data rows and a header, of which 56,160 rows are story presentation rows. |
| [`analysis_experiment_1_R1.Rmd`](https://osf.io/download/gbpkh/) | Lines 21, 29–71, 75–100 record the spreadsheet, raw fields, word order, four conditions, and timing conversion. The authors selected target sentences in lines 108–129 and filtered them by response time. |
| [`Experimental_Texts_Experiment_1.pdf`](https://osf.io/download/3p2d7/) | 294,477 bytes. SHA-256 `c249217876c2c46a98a706babd3bc9644592e6bdb47abc4ca72c4fc677e13b5d`. Supplementary materials are listed by story, with consistent and inconsistent versions. precise sentence boundary presentation can be sourced from the `Satz` column of the spreadsheet. |

We took each row where `Satz` was neither empty nor `NA`, padded `ID` on the left with zeros to at least six characters, and split it into the two-digit story number, two-digit condition, two-digit sentence position, and optional `t` or `s` suffix. The suffixes identify the target and following sentence, respectively. In all 56,160 records, `LoopSentences.thisN` equaled the sentence position minus one, `StoryKey.keys` was `k` or `d`, and `StoryKey.rt` was numeric. Grouping by the worksheet's `participant`, story, and condition yielded 180 readers and 3,600 complete sentence-by-sentence trajectories, with 900 per condition. Each reader read five different stories in each of the four conditions. Sentence positions were consecutive and unique within stories. Each story version and sentence position had a unique `Satz` string. [Authors' worksheet](https://osf.io/download/hp9sg/) · [Authors' analysis](https://osf.io/download/gbpkh/)

The text-only conditions `01` and `02` contain 1,800 story trajectories and 28,140 sentence-level judgments. `Bild` is `trans.png` throughout. Each of the 20 pairs of text-only versions differs in just one displayed sentence, placed 7–10 sentences before the target. The target and subsequent sentences are identical word for word. Each reader saw only one version of each story. This supports direct comparison of responses to the same target after different preceding text and preserves sentence-by-sentence reports from the same reader within a version. [Original paper, Experiment 1, Materials](https://doi.org/10.3758/s13421-026-01900-7) · [Author worksheet](https://osf.io/download/hp9sg/)

## Pure text observation

| Estimator | Consistent `01` | Inconsistency `02` |
| --- | ---: | ---: |
| All story trials in the target sentence `D` | 42／900（4.7%） | 505／900（56.1%） |
| Target sentences retained according to the author's timing rules `D` | 42／890（4.7%） | 505／889（56.8%） |
| First `D` before the target sentence | 117／900 | 120／900 |
| First `D` in the target sentence | 28／900 | 458／900 |
| First `D` after the target sentence | 20／900 | 11／900 |
| No `D` from sentence two through the end of the passage | 735／900 | 311／900 |

In the table, the first `D` excluded the first sentence, while all other keys were retained intact. The authors used `1000 × StoryKey.rt + 100` ms for the target sentence, removing target responses below 500 ms. A total of 21 pure-text trials were excluded, with original records all being `K`. The first `D` calculated from the second sentence onward could occur earlier than the researcher's scheduled contradiction, without asking these reports to point to which preceding proposition. Additionally, 3,600 post-story memory test keys were included. These probes were administered after all stories were read to measure recognition of key information. [Original Experiment 1, Data cleaning and inspection, and Additional measure](https://doi.org/10.3758/s13421-026-01900-7) · [Author's spreadsheet](https://osf.io/download/hp9sg/) · [Author's analysis](https://osf.io/download/gbpkh/)

If subsequently used for model evaluation, the primary endpoint can be the difference in `D` probability under consistent versus inconsistent contexts for the same target sentence. Each material is input with text up to the current sentence only. the two versions are held out by story, while keys, conditions, and `t`／`s` are reserved solely for scoring. Since the target sentences are identical across both conditions, the model is expected to yield the same result for this pairing regardless of whether the `D` is earlier or later. however, both initial indication and final reporting must be handled based on actual observations. These data derive from existing prompt-consistency judgment experiments and are applicable to similar narratives and tasks.

The paper is licensed under CC BY 4.0. The `node_license` field in the public OSF project is empty, and there is no `license` association. Reuse conditions for the worksheet and independent stimulus files remain to be verified. 12 stimuli are adapted from Cook (2014). This round has verified the source list, file hashes, and the aforementioned sentence-by-sentence records. [Paper Rights Statement](https://link.springer.com/article/10.3758/s13421-026-01900-7#rightslink) · [OSF Project Metadata](https://api.osf.io/v2/nodes/pg67h/) · [Data List](https://api.osf.io/v2/nodes/pg67h/files/osfstorage/666ad95f77ff4c59e8e047fb/) · [Stimulus List](https://api.osf.io/v2/nodes/pg67h/files/osfstorage/666ad9726b6c8e2f2304d4d1/)

## reproducible setup

Place the above spreadsheet and PDF in the `--raw` directory. Construct a retention of the original presentation string, inserting two newlines between sentences, and save the Unicode exclusive reading stop at the end of each sentence. A total of 40 full texts and 332 deduplicated sentence pairs sum to 372 records. The model input contains only text, text-derived IDs, and reading stops. The additionally stored data includes story pairing, conditions, and reader trajectories represented by code. A fixed tokenizer passed all prefix checks for the 956 reading stops. The two data construction runs were bitwise identical. [Construction script](../../../research/semantic_tracking/frick_consistency_prepare.py) · [Source and output hash](../../../research/results/frick-consistency-source-audit.json)

```sh
python3 -m research.semantic_tracking.frick_consistency_prepare \
  --raw .cache/cogllm/semantic-tracking/frick-consistency/raw \
  --output .cache/cogllm/semantic-tracking/frick-consistency/prepared
```

[Original research record](../../../research/semantic_tracking/frick-consistency-source.md)
