# Stage 2 human observation data

The initial dataset comprises **366 stimulus versions and 6,586 observations**, derived from converting existing open research trials. PLABA-EVAL provides sentence-level and full-text self-assessment and open-book responses, Zhan offers Chinese reference selection, and OneStopQA provides English open-book understanding. Each study retains its own task, visible text, and scale. [Stage 1 signal selection](stage-1/data-selection.md) · [output audit log](../../research/results/stage2-audit.json)

| Corpus | Stimulus version | human observation | reproducible primary research statistics |
| --- | ---: | ---: | --- |
| PLABA-EVAL | 78 | 2,995 | 1,826 sentence-level ratings, 234 full-text ratings, 935 open-book responses. Across 265 aligned sentence pairs from 39 source families, the vernacular average self-assessment was **0.723 points** higher, on a 1–5 scale. |
| Zhan Experiment 1 | 72 | 1,651 | The data contain 780 free prompts and 871 prompts that supply a pronoun. Giving each reader equal weight, the rate of rementioning the subject in the free-prompt condition was **77.68%** for IC-1 (SE 3.25 percentage points) and **11.72%** for IC-2 (SE 1.99 percentage points). |
| OneStopQA Prolific | 216 | 1,940 | 1,567/1,940 correct on the written test (**80.77%**). Advanced: 774/970. Basic: 793/970. |

## Reconstruction

In the project root directory, execute with Python 3.11 or a later version. The converter uses only standard libraries and CPU.

```bash
python3 -m research.stage2 \
  --cache-dir .cache/cogllm/raw \
  --output-dir .cache/cogllm/stage2-rebuild
```

The script downloads and verifies the source files from a fixed version or release ID using SHA-256. When the source is ready, add `--offline`. The existing output directory must be replaced with a new path, and the source cache should be placed outside the output directory. If the original file version does not match or is missing, the script reports the reason. output data is written to the designated directory only after verification. The first run successfully retrieved all eight original files from public sources. Running offline twice confirmed that the hashes of the nine output files and the total audit file are consistent. Rebuilding with local cache takes approximately 0.20 seconds per run. three JSONL files and five-fold name lists sum to approximately 7.0 MB. [Conversion Entry](../../research/stage2/__main__.py) · [Source Version, License, and Hash](../../research/stage2/sources.json)

## Output and training interface

- **`stimuli.jsonl`** Preserve the exact verbatim text, document/stimulus family, language, and verified anchor spans. The offset is a half-open interval of Python Unicode code points `[start, end)`. PLABA-EVAL employs sentence boundaries. Zhan's `np1` and `np2` represent two candidate roles. a prompt character is additionally marked when a supply pronoun is present. OneStopQA includes only segments observed in the experiment.
- **`trials.jsonl`** Preserve stimulus, reader ID, source file, and 1-based data rows starting after the header, task, visible scope, target segment, and observation. Main text input reads `text[:visible_end]`. the research task can read the task's `task_context`. Pure main-text models use optional pre-reading context. these three batches have no directly accessible pre-test context. Author correct answers, responses, and evidence sentences remain in the observation column. subject ID, version, and Zhan's verb type are usable for grouping and for clearly designed comparisons.
- **Observed state.** `observed` There are observed values. PLABA-EVAL has 452 selections of at least one item flagged as ambiguous, and 406 selections explicitly marked as unambiguous. The remaining 968 items originate from the free-text interface, with a structured type of `unobserved`. `missing` indicates that the value corresponding to that field could not be used for analysis, such as when a response points to a non-existent span. The missing mask should be generated based on the observed field and task. Options or evidence sentences belong to the response and material information of that item, and scores should be retained at the corresponding item endpoint.
- **`splits.json` and `audit.json`** retain the development fivefold, signal usage, raw source, per-task denominator, recomputed results, and output SHA-256. Raw and sequential data remain in the local ignore directory, while the project retains the code, source lock, and audit log excluding sequential responses.

### PLABA-EVAL evidence sentence correction

The original open-book sorting baseline used single-sentence indices from the answer sheet. Of 935 answer records, 932 retained indexable sentences, with **213 pointing to sentences outside the question table evidence scope**. Additionally, 3 old indices could not align. Stage 2 recorded the evidence list per `mcq_items.csv`. 932 records had evidence, including 126 pointing to multiple sentences, while retaining old indices for traceability. [Historical Baseline Record](training-protocol.md#已完成的首輪基線)

We used a newer causal end-ordering readout to correct the evidence list for open-book sorting. Among 932 scoreable responses, 133 were incorrect. The average understanding score per question was 0.557 AUROC. Relative to the same-evidence sentence-length baseline, the difference was −0.008 (95% CI [−0.067, +0.061]). Relative to the whole-model score, the difference was +0.005 (95% CI [−0.022, +0.029]). The data do not support treating the current understanding self-assessment as a direct proxy for local incorrectness risk. This comparison followed a reader-and-document-isolated development fold and fixed readout, without using response re-fitting. [Scoring Rubric](stage6/plaba-corrected-evidence.md) · [Actual Results](../../research/results/plaba-corrected-evidence-scores.json)

The 11 response strings with inconsistent answer keys have been marked with `metadata.selected_text_matches_item_option`. When exporting by publication source, correctness can be retained. however, when exporting with option text as the target, these columns must be masked. The original display order of PLABA-EVAL was not obtained. export is performed in publication order. Of the 158 items, 153 have the correct answer as A, and option position heuristics must be controlled prior to evaluation.

### OneStopQA material version

The fixed HF Parquet contains 1,458 columns, comprising paragraphs, questions, options, and author evidence segments. Precise matching is performed by difficulty category, original paragraph, question stem, and source option order. in the human trial, only **32** of the 648 question versions map uniquely to the same material. Consequently, the actual text presented for this batch of author experiment TSVs is saved with option order per `new_options`. The STARC phrase was not received in the human trial. The SHA-256 of the fixed HF archive is stored in the source lock. [HF fixed material](https://huggingface.co/datasets/malmaud/onestop_qa/tree/1df9aea8c9532045bc55c001d7c588372de2d723) · [author human trial](https://github.com/berzak/onestop-qa/tree/7a058755e1ff0d3f835a67109c669c7e294d4239/human_experiments)

## Split and validation scope

| Corpus | Develop five-fold cross-validation | Cross-fold readership |
| --- | --- | ---: |
| PLABA-EVAL | Matched by 39 sources and a shared reader for cross-document scoring, merged into 25 connected components. | 0 |
| Zhan Experiment 1 | 36 original stimulus sets, with the free/pronoun cue condition collapsed within each set. | 50 |
| OneStopQA Prolific | 20 source articles, all paragraphs, titles, and advanced/basic versions folded. | 847 |

A fivefold cross-validation was conducted with family as the grouping unit. PLABA-EVAL simultaneously validated unseen readers. The analysis by Zhan and OneStopQA can validate unseen materials, with readers spanning folds. The entire research family held-out remains pending matching endpoints, visible conditions, and available versions. Stage 2 has provided the data required for the initial Colab L4 implementation. model training, per-token alignment, and output validation belong to Stage 3. [Execution Plan](execution-plan.md#階段交付與驗收)

## Engineering Acceptance

Verdict: PASS

### Preference Fit

- The eight source families are fixed at public release, requiring only Python standard libraries for conversion. Sequential data and conversion scripts are stored separately.

### Correctness

- 366 stimulus versions, 6,586 observations, developed five-fold and main denominator for programmatic verification. Re-verified PLABA-EVAL's evidence index and option text, Zhan's prompt types and candidate positions, and OneStopQA's option ordering.

### Evidence Run

- Download the original file via the network. Two offline reruns produced bit-for-bit identical outputs. Python 3.11.14 and 3.14.4 yielded identical audit files. Additionally, test for missing original files, SHA-256 mismatches, and existing output directories. [audit file](../../research/results/stage2-audit.json)

### Scope/Architecture

- The acceptance scope covers the parsing, offset, observation, and splitting of the initial three datasets. Actual model training and external research held-out are to be validated separately.

### Required Fixes

- None.

[Original research record](../../research/stage-2-data.md)
